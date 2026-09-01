"""Predicted-physics features for the Tc head (the pf ablation ladder).

One frozen forward pass of a multitask checkpoint over the transfer index turns
the encoder's own predictions into STATIC head inputs, cached to disk:

  group P (per-atom, consumed before pooling — see FineTune `pred_features`):
    magmom_abs   |m̂_i|            (sign-invariant; the global flip is arbitrary)
    mom_align    neighbor-mean of m̂_i·m̂_j / (|m̂_i||m̂_j| + eps) — the LOCAL
                 AFM-order field, invariant under the global spin flip
    f_norm       |F̂_i|            (how strained the encoder thinks the site is)
    f_c, f_ab    |F̂·ĉ| and the in-plane residual — the layered-material split
                 (rotation-invariant: ĉ rotates with the structure)
    f_radial     neighbor-mean of F̂_i·r̂_ij — signed bond-stretching component
                 (apical relaxation tendency); bend character = residual vs f_norm
    site_energy  per-atom energy head output ê_i (site stability/frustration)
    dos_q1..q4   per-atom electronic DOS in 4 coarse bins of the ±1 eV window

  group G raw material (global, stored for the later G derivation step):
    energy, bandgap, stress (3,3), dos (n_energy,), eph_a2f (n_phonon,)
    + the raw signed per-atom magmoms (staggered indicator needs them).

Direction handling: raw Cartesian components are frame-dependent (the head has
no equivariance) — only invariant reductions are emitted. Padding slots in the
neighbor list self-loop with a zero image, so the bond vector is exactly 0 and
`|d| > eps` is the real-neighbor mask (see _assemble_sample).

The pass mirrors train._validate_mt: cart/strain leaves + torch.enable_grad in
eval so the conservative-autograd forces exist. Per-atom energy / DOS head
outputs (pre-pooling) are captured with forward hooks — no readout duplication.

Cache: torch.save at `cache_path` (default derived from checkpoint+index under
database/datafiles/MP/pred_cache/), keyed by (version, checkpoint, index); a
mismatch rebuilds. Bump _VERSION whenever the feature definitions change — the
stale-cache-column trap (packed_v4 NaN DOS) must not reappear here.
"""

import hashlib
import os

import torch

_VERSION = 1

PER_ATOM_NAMES = ["magmom_abs", "mom_align", "f_norm", "f_c", "f_ab",
                  "f_radial", "site_energy", "dos_q1", "dos_q2", "dos_q3", "dos_q4"]

# Tasks the group-P derivation consumes. forces implies energy (conservative).
_REQUIRED_TASKS = {"energy", "forces", "magmom", "dos"}


def default_cache_path(checkpoint_path, index_path):
    key = hashlib.sha1(f"{checkpoint_path}|{index_path}".encode()).hexdigest()[:12]
    return os.path.join("database", "datafiles", "MP", "pred_cache", f"pf_v{_VERSION}_{key}.pt")


@torch.no_grad()
def _derive_per_atom(forces, mag, site_e, pa_dos, cart, lattice, seg, nbr_idx, jimage):
    """The 11 group-P channels (N, 11) from raw predictions + batch geometry."""
    d_img = torch.einsum("nmk,nkc->nmc", jimage.to(cart.dtype), lattice[seg])
    d = cart[nbr_idx] + d_img - cart.unsqueeze(1)                    # (N, M, 3)
    dn = d.norm(dim=-1)                                              # (N, M)
    real = dn > 1e-6                                                 # pad slots: |d| == 0
    cnt = real.sum(1).clamp(min=1)
    rhat = d / dn.clamp_min(1e-8).unsqueeze(-1)
    f_rad = (torch.einsum("nc,nmc->nm", forces, rhat) * real).sum(1) / cnt
    c_vec = lattice[seg][:, 2, :]
    c_hat = c_vec / c_vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    f_norm = forces.norm(dim=-1)
    f_c = (forces * c_hat).sum(-1).abs()
    f_ab = (f_norm ** 2 - f_c ** 2).clamp(min=0).sqrt()
    mj = mag[nbr_idx]                                                # (N, M)
    # eps=1e-2 mu_B damps the sign product where moments are numerically ~0
    align = ((mag.unsqueeze(1) * mj)
             / (mag.abs().unsqueeze(1) * mj.abs() + 1e-2) * real).sum(1) / cnt
    dos_q = pa_dos.view(pa_dos.shape[0], 4, -1).mean(-1)             # (N, 4)
    return torch.cat([torch.stack([mag.abs(), align, f_norm, f_c, f_ab,
                                   f_rad, site_e], dim=1), dos_q], dim=1)


def build_pred_cache(model, ds, collate, device, batch_size=32, num_workers=4):
    """Run the frozen checkpoint over the full dataset; return the cache dict."""
    from torch.utils.data import DataLoader
    if model.tasks is None or not (_REQUIRED_TASKS <= set(model.tasks)):
        raise ValueError(f"pred_features needs tasks >= {sorted(_REQUIRED_TASKS)}; "
                         f"checkpoint has {sorted(model.tasks or [])}")
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
                        num_workers=num_workers, pin_memory=(str(device) != "cpu"))
    feats, mag_raw, glob = {}, {}, {}
    fsum_max = fsum_acc = n_struct = 0.0
    grabbed = {}
    hooks = [model.heads[k].register_forward_hook(
                 (lambda key: lambda _m, _i, o: grabbed.__setitem__(key, o))(k))
             for k in ("energy", "dos")]
    try:
        for inp, _t, _l, cids in loader:
            # collate layout (collate_pool + geom): [0..5] graph tensors, [6] seg,
            # [7] n_crystals (int), [8] frac, [9] lattice, [10] nbr_jimage —
            # matching model.forward's positional signature exactly.
            inp = tuple(x.to(device) if torch.is_tensor(x) else x for x in inp)
            seg = inp[6].long()
            B = int(inp[7])
            frac, lattice, jimage = inp[8], inp[9], inp[10]
            cart = torch.einsum("ni,nij->nj", frac, lattice[seg]).detach().requires_grad_(True)
            strain = torch.zeros(B, 3, 3, device=cart.device, dtype=cart.dtype,
                                 requires_grad=True)
            grabbed.clear()
            with torch.enable_grad():   # forces need a graph even in eval
                out = model(*inp, cart=cart, strain=strain)
            forces = out["forces"].detach()
            mag = out["magmom"].detach().squeeze(-1)
            site_e = grabbed["energy"].detach().squeeze(-1)
            pa_dos = grabbed["dos"].detach()
            pa = _derive_per_atom(forces, mag, site_e, pa_dos, cart.detach(),
                                  lattice, seg, inp[2], jimage).cpu()
            # translation-invariance check: conservative forces sum to ~0 per cell
            fs = torch.zeros(B, 3, device=forces.device).index_add_(0, seg, forces)
            fs = fs.norm(dim=-1)
            fsum_acc += float(fs.sum()); fsum_max = max(fsum_max, float(fs.max()))
            n_struct += B
            for i, cid in enumerate(cids):
                m = seg == i
                feats[cid] = pa[m.cpu()]
                mag_raw[cid] = mag[m].cpu()
                # store whatever global heads this checkpoint has (stress/eph_a2f
                # are optional — rungs 27-33 lack a2f; group G derivation checks)
                g = {"energy": float(out["energy"][i].detach()),
                     "dos": out["dos"][i].detach().cpu()}
                if "bandgap" in out:
                    g["bandgap"] = float(out["bandgap"][i].detach())
                for k in ("stress", "eph_a2f"):
                    if k in out:
                        g[k] = out[k][i].detach().cpu()
                glob[cid] = g
    finally:
        for h in hooks:
            h.remove()
    print(f"[pf] pred cache built: {len(feats)} structures; "
          f"|sum F| mean {fsum_acc / max(n_struct, 1):.2e} max {fsum_max:.2e} eV/A")
    _sanity_print(feats, glob, model.n_phonon if "eph_a2f" in model.tasks else None)
    return {"version": _VERSION, "per_atom_names": list(PER_ATOM_NAMES),
            "feats": feats, "magmom_raw": mag_raw, "global": glob}


def _sanity_print(feats, glob, n_phonon):
    """Domain-shift gate: per-channel spread + a2f-derived lambda quartiles.
    A collapsed distribution here means the head gets a constant, not a feature."""
    mat = torch.cat(list(feats.values()))
    for j, name in enumerate(PER_ATOM_NAMES):
        col = mat[:, j]
        print(f"[pf]   {name:12s} mean {col.mean():+.3f}  std {col.std():.3f}  "
              f"p5 {col.quantile(0.05):+.3f}  p95 {col.quantile(0.95):+.3f}")
    if n_phonon is None:
        return
    # lambda = 2 * sum a2f(w)/w dw on the fixed 0-60 THz grid (skip bin 0 — the
    # soft-mode smearing tail the 12 K cutoff excludes in the target build)
    w = (torch.arange(n_phonon, dtype=torch.float32) + 0.5) * (60.0 / n_phonon)
    dw = 60.0 / n_phonon
    lam = torch.tensor([float(2.0 * (g["eph_a2f"][1:] / w[1:]).sum() * dw)
                        for g in glob.values()])
    q = [float(lam.quantile(x)) for x in (0.05, 0.25, 0.5, 0.75, 0.95)]
    print(f"[pf]   a2f-derived lambda quartiles p5/p25/p50/p75/p95: "
          + "/".join(f"{v:.3f}" for v in q))


THZ_TO_K = 47.9924       # h/k_B: 1 THz in Kelvin
A2F_WMAX_THZ = 60.0      # the fixed spectrum grid both phonon targets share
DOS_WINDOW_EV = 2.0      # electronic DOS window: E_F ± 1 eV

GLOBAL_NAMES = (["g_bandgap", "g_energy", "g_pressure", "g_vonmises",
                 "g_mean_absm", "g_staggered", "g_nef", "g_dos_slope"]
                + [f"g_dosb{i+1}" for i in range(8)]
                + ["g_lambda", "g_area", "g_logwlog", "g_logw2",
                   "g_logtc_ad10", "g_logtc_ad13"]
                + [f"g_a2fb{i+1}" for i in range(16)])


def _allen_dynes_tc(lam, wlog_K, w2_K, mu):
    """Allen-Dynes Tc (K) with the f1/f2 strong-coupling corrections. Returns 0
    where the exponent denominator is non-positive (lambda too small for this
    mu* — the physically correct limit, and the numerical guard)."""
    den = lam - mu * (1.0 + 0.62 * lam)
    if den <= 0 or lam <= 0 or wlog_K <= 0:
        return 0.0
    r = max(w2_K / max(wlog_K, 1e-12), 1.0)
    f1 = (1.0 + (lam / (2.46 * (1.0 + 3.8 * mu))) ** 1.5) ** (1.0 / 3.0)
    l2 = 1.82 * (1.0 + 6.3 * mu) * r
    f2 = 1.0 + ((r - 1.0) * lam ** 2) / (lam ** 2 + l2 ** 2)
    import math
    return f1 * f2 * (wlog_K / 1.2) * math.exp(-1.04 * (1.0 + lam) / den)


def derive_global_features(cache):
    """Group G: the ~38-dim global feature vector per structure from the cached
    raw predictions (see GLOBAL_NAMES). ω_log / ω̄₂ / Tc_AD enter as log1p —
    their 30x dynamic range would otherwise dominate the z-scored block."""
    import math
    glob, mag_raw = cache["global"], cache["magmom_raw"]
    _probe = next(iter(glob.values()))
    missing = [k for k in ("bandgap", "stress", "eph_a2f") if k not in _probe]
    if missing:
        raise ValueError(f"pred_features['global'] needs checkpoint heads {missing} "
                         "— use an a2f-bearing checkpoint (rungs 35+)")
    n_e = _probe["dos"].numel()
    n_p = _probe["eph_a2f"].numel()
    de = DOS_WINDOW_EV / n_e
    w = (torch.arange(n_p, dtype=torch.float64) + 0.5) * (A2F_WMAX_THZ / n_p)
    dw = A2F_WMAX_THZ / n_p
    c = n_e // 2
    out = {}
    for cid, g in glob.items():
        s = g["stress"].double()
        press = float(-s.trace() / 3.0)
        dev = s - s.trace() / 3.0 * torch.eye(3, dtype=s.dtype)
        vm = float((1.5 * (dev * dev).sum()).sqrt())
        m = mag_raw[cid].double()
        sabs = float(m.abs().sum())
        stag = float((sabs - abs(float(m.sum()))) / sabs) if sabs > 1e-6 else 0.0
        dos = g["dos"].double()
        nef = float(dos[c - 2:c + 2].mean())
        slope = float((dos[c + 2:c + 6].mean() - dos[c - 6:c - 2].mean()) / (8 * de))
        dosb = dos.view(8, -1).mean(1)
        a2f = g["eph_a2f"].double()
        # bin 0 skipped everywhere: the extraction's ~12 K soft-mode cutoff
        lam = float(2.0 * (a2f[1:] / w[1:]).sum() * dw)
        area = float(a2f.sum() * dw)
        if lam > 1e-6:
            wlog_K = math.exp(float((2.0 / lam) * (w[1:].log() * a2f[1:] / w[1:]).sum() * dw)) * THZ_TO_K
            w2_K = math.sqrt(max(float((2.0 / lam) * (w[1:] * a2f[1:]).sum() * dw), 0.0)) * THZ_TO_K
        else:
            wlog_K = w2_K = 0.0
        a2fb = a2f.view(16, -1).mean(1)
        vec = ([float(g["bandgap"]), float(g["energy"]), press, vm,
                float(m.abs().mean()), stag, nef, slope]
               + [float(x) for x in dosb]
               + [lam, area, math.log1p(wlog_K), math.log1p(w2_K),
                  math.log1p(_allen_dynes_tc(lam, wlog_K, w2_K, 0.10)),
                  math.log1p(_allen_dynes_tc(lam, wlog_K, w2_K, 0.13))]
               + [float(x) for x in a2fb])
        out[cid] = torch.tensor(vec, dtype=torch.float32)
    return list(GLOBAL_NAMES), out


def load_or_build_pred_cache(model, ds, collate, device, checkpoint_path, index_path,
                             cache_path=None):
    """Load the cache when (version, checkpoint, index) match; otherwise build+save."""
    path = cache_path or default_cache_path(checkpoint_path, index_path)
    key = {"version": _VERSION, "checkpoint": checkpoint_path, "index": index_path}
    if os.path.exists(path):
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if all(cache.get("key", {}).get(k) == v for k, v in key.items()):
            print(f"[pf] pred cache loaded: {path} ({len(cache['feats'])} structures)")
            return cache
        print(f"[pf] pred cache stale ({path}) — rebuilding")
    cache = build_pred_cache(model, ds, collate, device)
    cache["key"] = key
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(cache, tmp)
    os.replace(tmp, path)   # atomic: a killed build never leaves a half-written cache
    print(f"[pf] pred cache saved: {path}")
    return cache
