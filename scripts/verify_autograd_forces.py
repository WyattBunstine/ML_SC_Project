#!/usr/bin/env python3
"""Verify the GPS conservative-autograd forces/stress are correct (CPU, float64).

The must-pass gate for the differentiable-geometry multitask model: it proves
forces = -dE/dcart and stress = dE/dstrain are computed correctly BEFORE any cluster
re-pack / training. Checks, on a synthetic periodic crystal built directly (no dataset):
  1. Finite-difference forces  -dE/dr  vs  autograd forces            (~1e-6)
  2. Force sum-to-zero  Sum_i F_i ~ 0    (translation invariance)
  3. Rotation equivariance: rotate the cell -> energy invariant, forces rotate
  4. Stress symmetry + finite-difference stress vs autograd stress
  5. Double-backward: a force/stress loss backprops to model parameters
  6. Multi-head output shapes (energy/forces/stress/magmom/bandgap/dos)

    python scripts/verify_autograd_forces.py     # exit 0 = pass, 1 = fail
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("models/common", "models/GPSTransformer"):
    sys.path.insert(0, os.path.join(ROOT, _p))

import torch  # noqa: E402
from model import GPSCrystalNet  # noqa: E402

DT = torch.float64
TASKS = {"energy", "forces", "stress", "magmom", "bandgap", "dos"}


def build_crystal(N=4, seed=0, cell=5.0):
    """A small cubic cell; each atom neighbors every other via the min-image edge
    (with the periodic offset, so some bonds genuinely cross the cell boundary)."""
    g = torch.Generator().manual_seed(seed)
    lattice = (torch.eye(3, dtype=DT) * cell).unsqueeze(0)            # (1,3,3)
    frac = torch.rand(N, 3, generator=g, dtype=DT)
    seg = torch.zeros(N, dtype=torch.long)
    M = N - 1
    idx = torch.zeros(N, M, dtype=torch.long)
    jimg = torch.zeros(N, M, 3, dtype=DT)
    for i in range(N):
        for s, j in enumerate(j for j in range(N) if j != i):
            idx[i, s] = j
            jimg[i, s] = -(frac[j] - frac[i]).round()                # min-image offset
    atom_fea = torch.rand(N, 14, generator=g, dtype=DT)
    return lattice, frac, seg, idx, jimg, atom_fea


def static_features(cart, lattice, seg, idx, jimg):
    """Reference nbr_fea (col0=|d|, col1=|d|/sumR, cols2-6 nonzero) + (M,M) angle-cos
    matrix with a 2.0 sentinel on the diagonal — matching the real pipeline's layout."""
    img = torch.einsum("nmk,nkc->nmc", jimg, lattice[seg])
    d = cart[idx] + img - cart.unsqueeze(1)
    dist = d.norm(dim=-1)
    N, M = idx.shape
    nbr_fea = torch.full((N, M, 7), 0.3, dtype=DT)                    # cols 2-6 = real weights
    nbr_fea[..., 0] = dist
    nbr_fea[..., 1] = dist / 2.0                                     # sum_radii = 2
    dn = d / dist.unsqueeze(-1).clamp_min(1e-12)
    cos = torch.einsum("nak,nbk->nab", dn, dn)
    # Production fills the (M,M) matrix from angle_triplets: MOST off-diagonal pairs stay
    # the 2.0 sentinel, only a subset is real. Mimic PARTIAL coverage so the recompute's
    # masking path (keep sentinels sentinel, recompute only real pairs) is exercised.
    keep = torch.full((M, M), 2.0, dtype=DT)
    for a, b in ((0, 1), (1, 2)):           # only these off-diagonal pairs are "real"
        keep[a, b] = keep[b, a] = 0.0
    real = (keep.abs() <= 1.0).unsqueeze(0)
    return nbr_fea, torch.where(real, cos, torch.full_like(cos, 2.0))


def main():
    torch.manual_seed(0)
    lattice, frac, seg, idx, jimg, atom_fea = build_crystal(N=4)
    cart0 = torch.einsum("ni,nij->nj", frac, lattice[seg])
    nbr_fea, nbr_angle = static_features(cart0, lattice, seg, idx, jimg)
    N = frac.shape[0]
    poly_fea = torch.zeros(N, 0, 7, dtype=DT)
    poly_idx = torch.zeros(N, 0, dtype=torch.long)

    model = GPSCrystalNet(14, 7, poly_fea_len=7, atom_fea_len=16, n_conv=2,
                          h_fea_len=16, n_h=2, n_heads=2, use_poly_edges=False,
                          gps_global=False, use_angle_bias=True,
                          shell_aggregation="attention", tasks=TASKS, n_energy=8,
                          differentiable_geometry=True).double()
    model.train()

    def run(cart, strain):
        return model(atom_fea, nbr_fea, idx, poly_fea, poly_idx, nbr_angle, seg, 1,
                     frac_coords=frac, lattice=lattice, nbr_jimage=jimg,
                     cart=cart, strain=strain)

    # Warm-train so the geometric response is non-trivial: an untrained net has ~1e-3
    # forces that finite differences can't resolve against E's float64 round-off
    # (ΔE = F·eps sinks below ~1e-13·|E|). A few steps fitting random force/energy
    # targets give O(0.1-1) forces. (The autograd machinery is identical either way —
    # sum-to-zero/equivariance/stress-FD already confirm it; this just makes the
    # force-FD numerically resolvable.)
    opt = torch.optim.Adam(model.parameters(), lr=0.004)
    gen = torch.Generator().manual_seed(3)
    tgt_f = (torch.rand(N, 3, generator=gen, dtype=DT) - 0.5) * 0.2
    for _ in range(60):
        c = cart0.clone().requires_grad_(True)
        s = torch.zeros(1, 3, 3, dtype=DT, requires_grad=True)
        loss = (run(c, s)["forces"] - tgt_f).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    cart = cart0.clone().requires_grad_(True)
    strain = torch.zeros(1, 3, 3, dtype=DT, requires_grad=True)
    out = run(cart, strain)

    ok = True

    # ---- 0. SHIPPED forces/stress correctness (the guard the gradchecks alone miss):
    # out['forces'] MUST equal -dE_total/dcart and out['stress'] = dE_total/dstrain / V.
    # The gradchecks only validate dE/dcart, dE/dstrain on an energy-only model; this ties
    # them to the ACTUAL readout values, catching a sign flip, wrong grad-input, or a
    # missing/incorrect volume division. (energy is intensive E_total/N, so E_total=energy*N.) ----
    E_total = (out["energy"] * N).sum()
    g_cart, g_strain = torch.autograd.grad(E_total, (cart, strain), retain_graph=True)
    vol = torch.det(lattice).abs().item()
    f_match = (out["forces"].detach() + g_cart).abs().max().item()      # forces == -dE/dcart
    s_match = (out["stress"].detach() - g_strain / vol).abs().max().item()  # stress == dE/dstrain/V
    shipped_ok = f_match < 1e-10 and s_match < 1e-10
    print(f"  shipped     |F+dE/dcart|={f_match:.2e}  |S-dE/dstrain/V|={s_match:.2e}  (<1e-10) {shipped_ok}")
    ok &= shipped_ok

    # ---- 6. shapes ----
    shapes = {"energy": (1,), "forces": (N, 3), "stress": (1, 3, 3),
              "magmom": (N, 1), "bandgap": (1,), "dos": (1, 8)}
    shape_ok = all(tuple(out[k].shape) == s for k, s in shapes.items())
    print(f"  shapes      {{k: tuple(v.shape)}} match={shape_ok}")
    ok &= shape_ok

    # ---- 1. forces = -dE/dcart, verified by torch.autograd.gradcheck (rigorous FD
    # with proper eps/tolerance). The model's out["forces"] IS -autograd.grad(E, cart),
    # so a passing gradcheck of dE/dcart certifies the reported forces. ----
    F = out["forces"].detach()

    def E_of_cart(cf):
        return run(cf.view(N, 3), strain)["energy"]

    # gradcheck differentiates E_of_cart; do it with an ENERGY-ONLY task set so the
    # model's INTERNAL autograd.grad(E, cart) (force computation) doesn't run and tangle
    # gradcheck's own backward. forces = -dE/dcart, so this certifies the reported forces.
    saved_tasks = model.tasks
    try:
        model.tasks = {"energy"}           # energy-only: no internal grad to tangle gradcheck
        fd_ok = torch.autograd.gradcheck(E_of_cart, cart0.flatten().clone().requires_grad_(True),
                                         eps=1e-6, atol=1e-7, rtol=1e-5, raise_exception=False)
    finally:
        model.tasks = saved_tasks
    print(f"  gradcheck-F gradcheck(dE/dcart)={fd_ok}  |F|max={F.abs().max().item():.3f}")
    ok &= bool(fd_ok)

    # ---- 2. force sum-to-zero ----
    sz = F.sum(0).abs().max().item()
    sz_ok = sz < 1e-7
    print(f"  sum-zero    max|Sum_i F_i| = {sz:.2e}  (<1e-7) {sz_ok}")
    ok &= sz_ok

    # ---- 3. rotation equivariance ----
    q, _ = torch.linalg.qr(torch.rand(3, 3, generator=torch.Generator().manual_seed(7),
                                      dtype=DT))
    R = q * torch.det(q)                                             # proper rotation
    cart_r = (cart0 @ R.T).clone().requires_grad_(True)
    lat_r = lattice @ R.T
    out_r = model(atom_fea, nbr_fea, idx, poly_fea, poly_idx, nbr_angle, seg, 1,
                  frac_coords=frac, lattice=lat_r, nbr_jimage=jimg,
                  cart=cart_r, strain=strain)
    e_inv = abs((out_r["energy"] - out["energy"]).abs().max().item())
    f_equiv = (out_r["forces"].detach() - F @ R.T).abs().max().item()
    rot_ok = e_inv < 1e-9 and f_equiv < 1e-8
    print(f"  rotation    dE={e_inv:.2e} (inv)  |F(Rx)-F@R^T|={f_equiv:.2e} (equiv) {rot_ok}")
    ok &= rot_ok

    # ---- 4. stress symmetry + gradcheck(dE/dstrain) ----
    S = out["stress"].detach()[0]
    sym = (S - S.T).abs().max().item()

    def E_of_strain(sf):
        return run(cart0.clone().requires_grad_(True), sf.view(1, 3, 3))["energy"]

    try:
        model.tasks = {"energy"}
        stress_fd_ok = torch.autograd.gradcheck(
            E_of_strain, torch.zeros(9, dtype=DT).requires_grad_(True),
            eps=1e-6, atol=1e-7, rtol=1e-5, raise_exception=False)
    finally:
        model.tasks = saved_tasks
    stress_ok = sym < 1e-9 and bool(stress_fd_ok)
    print(f"  stress      sym={sym:.2e} (<1e-9)  gradcheck(dE/dstrain)={stress_fd_ok} -> {stress_ok}")
    ok &= stress_ok

    # ---- 5. double-backward: force/stress loss -> model params ----
    cart2 = cart0.clone().requires_grad_(True)
    strain2 = torch.zeros(1, 3, 3, dtype=DT, requires_grad=True)
    o2 = run(cart2, strain2)
    loss = o2["forces"].pow(2).mean() + o2["stress"].pow(2).mean() + o2["energy"].pow(2).mean()
    model.zero_grad()
    loss.backward()
    gnorm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    db_ok = torch.isfinite(loss).item() and gnorm > 0
    print(f"  dbl-backward loss={loss.item():.4f} param-grad-norm={gnorm:.2e} finite&nonzero={db_ok}")
    ok &= db_ok

    print("verify_autograd_forces: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
