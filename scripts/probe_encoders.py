"""Frozen transfer probe across a set of pretrained encoders — a low-noise ranking.

For each checkpoint: freeze the encoder, build its per-atom embedding cache once,
train the head (hyperparameters sourced from HeadMain.DEFAULTS, deepsets pooling)
warmup-only on the cached train
embeddings (no encoder fine-tuning), and evaluate on TEST. Because the encoder is
frozen and the head trains on fixed cached tensors, this is near-deterministic —
it sidesteps the ~0.5-0.7 K CUDA fine-tuning noise, so encoder DIFFERENCES are
readable. Ranks "what pretraining choice moves the current head" with zero new
pretraining. Reports SC-only MAE / cuprate MAE / MSLE per encoder (mean over seeds).

  PYTHONHASHSEED=0 python scripts/probe_encoders.py --out model_data/hpo/zoo.csv
"""
import argparse
import glob
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
for _p in (os.path.join(_ROOT, "models", "common"), os.path.join(_ROOT, "models", "GPSTransformer"),
           os.path.join(_ROOT, "models", "head")):
    sys.path.insert(0, _p)

from embed_cache import build_embed_cache, cat_cached  # noqa: E402
from HeadData import assemble, make_splits, parent_composition_groups  # noqa: E402
from HeadMain import DEFAULTS  # noqa: E402
from HeadModel import TcHead  # noqa: E402

ENCODERS = {
    "05_energy_only": "model_data/2026-06-30/gps_mt_05_energy_only_r4/*/result_model_best.pth.tar",
    "04_forces_w10": "model_data/2026-06-26/gps_mt_04_dos_full/*/result_model_best.pth.tar",
    "06_forces_w2": "model_data/2026-06-30/gps_mt_06_forces_w2/*/result_model_best.pth.tar",
    "07_no_dos": "model_data/2026-07-02/gps_mt_07_forces_w2_no_dos/*/result_model_best.pth.tar",
    "08_no_magmom": "model_data/2026-07-02/gps_mt_08_forces_w2_no_magmom/*/result_model_best.pth.tar",
    "10_valence_only": "model_data/2026-07-14/gps_mt_10_valence_only/*/result_valence_only_model_best.pth.tar",
    "11_dos_only": "model_data/2026-07-14/gps_mt_11_dos_ef1_only/*/result_dos_ef1_only_model_best.pth.tar",
    "09_valence_dos": "model_data/2026-07-14/gps_mt_06_forces_w2_valence/*/result_valence_model_best.pth.tar",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="database/datafiles/MP/SC_MP_V4_doped.pickle")
    ap.add_argument("--descriptors", default="database/datafiles/MP/descriptors_doped.pickle")
    ap.add_argument("--metadata", default="database/datafiles/MP/3DSC_MP.csv")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--warmup-max", type=int, default=150)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--out", default="model_data/hpo/encoder_zoo.csv")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    data = assemble(args.index, "unused", args.descriptors, args.metadata,
                    pooling="meanmax", require_embed=False)
    ids = list(data["ids"])
    grp = parent_composition_groups(ids)
    split = make_splits(data, 123, 0.1, 0.2, groups=grp)
    id2split = dict(zip(ids, split)); id2row = {i: r for r, i in enumerate(ids)}
    is_sc = data["label"] == 1
    phys_t = torch.as_tensor(data["phys"].astype(np.float32), device=device)
    tc_t = torch.as_tensor(data["tc"], device=device).float()
    fam = data["family"]

    from data import load_cif_dataset_from_args, dataset_ids, collate_pool_geom
    from model import GPSCrystalNet
    from torch.utils.data import DataLoader, Subset

    def build_cache(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        a = ckpt.get("args", {})
        ds = load_cif_dataset_from_args(args.index, a)
        sa, sn, _, sp, _, _ = ds[0][0][:6]
        model = GPSCrystalNet.from_args(a, (sa.shape[-1], sn.shape[-1], sp.shape[-1]))
        model.load_state_dict(ckpt["state_dict"]); model.to(device).eval()
        ds_ids = dataset_ids(ds)
        pos = {s: [i for i, c in enumerate(ds_ids) if id2split.get(c) == s and is_sc[id2row[c]]]
               for s in ("train", "val", "test")}
        cache = {}
        for s in ("train", "val", "test"):
            loader = DataLoader(Subset(ds, pos[s]), batch_size=64,
                                collate_fn=collate_pool_geom, num_workers=4)
            cache.update(build_embed_cache(model, loader, device))
        edim = model.atom_fea_len
        del model; torch.cuda.empty_cache()
        return cache, {s: [ds_ids[i] for i in pos[s]] for s in pos}, edim

    def fwd(head, cache, bids):
        h, seg = cat_cached(cache, bids, device)
        rows = [id2row[c] for c in bids]
        _lg, z = head(h, phys_t[rows], seg=seg, n=len(bids))
        return z, rows

    def train_eval(cache, ids_by, edim, seed):
        tr, va, te = ids_by["train"], ids_by["val"], ids_by["test"]
        tr_rows = [id2row[c] for c in tr]
        torch.manual_seed(seed)
        # Head hyperparameters sourced from HeadMain.DEFAULTS (single source of the
        # tuned protocol) — hardcoding them here left the zoo ranking measured under
        # a head no config uses once DEFAULTS moved (pool_dim 32->64 already did).
        head = TcHead(edim, phys_t.shape[1], DEFAULTS["pca_k"], DEFAULTS["hidden"],
                      DEFAULTS["dropout"], pooling="deepsets",
                      pool_dim=DEFAULTS["pool_dim"]).to(device)
        head.fit_target(tc_t[tr_rows].cpu()); head.phys_std.fit(phys_t[tr_rows].cpu()); head.to(device)
        y_z = head.target_to_z(tc_t)
        opt = torch.optim.AdamW(head.parameters(), lr=DEFAULTS["tc_lr"],
                                weight_decay=DEFAULTS["tc_weight_decay"])
        rng = random.Random(seed); best, bad, best_state = np.inf, 0, None
        for ep in range(args.warmup_max):
            head.train(); order = tr[:]; rng.shuffle(order)
            for b0 in range(0, len(order), 64):
                opt.zero_grad(); z, rows = fwd(head, cache, order[b0:b0 + 64])
                F.l1_loss(z, y_z[rows]).backward(); opt.step()
            head.eval(); num = den = 0.0
            with torch.no_grad():
                for b0 in range(0, len(va), 256):
                    z, rows = fwd(head, cache, va[b0:b0 + 256]); num += (z - y_z[rows]).abs().sum().item(); den += len(rows)
            v = num / max(den, 1)
            if v < best - 1e-6: best, bad, best_state = v, 0, {k: t.clone() for k, t in head.state_dict().items()}
            else: bad += 1
            if bad >= args.patience: break
        # best_state stays None if no epoch ever improved (NaN val from a bad
        # checkpoint, or --warmup-max 0) — keep the last weights rather than crash
        # the whole zoo sweep, matching FineTune.fit_phase's guard.
        if best_state is not None:
            head.load_state_dict(best_state)
        head.eval()
        with torch.no_grad():
            zs = torch.cat([fwd(head, cache, te[b0:b0 + 256])[0] for b0 in range(0, len(te), 256)])
        k = head.z_to_kelvin(zs).cpu().numpy()
        rows = [id2row[c] for c in te]; t = data["tc"][rows]; f = fam[rows]
        return dict(sc_mae=float(np.abs(t - k).mean()),
                    cuprate_mae=float(np.abs(t[f == "Cuprate"] - k[f == "Cuprate"]).mean()),
                    msle=float(np.mean((np.log1p(t) - np.log1p(np.maximum(k, 0))) ** 2)))

    rows = []
    for tag, pat in ENCODERS.items():
        hits = glob.glob(pat)
        if not hits:
            print(f"  SKIP {tag}: no checkpoint", flush=True); continue
        cache, ids_by, edim = build_cache(sorted(hits)[-1])
        res = [train_eval(cache, ids_by, edim, s) for s in range(args.n_seeds)]
        agg = {k: np.mean([r[k] for r in res]) for k in res[0]}
        sd = {k: np.std([r[k] for r in res]) for k in res[0]}
        row = dict(encoder=tag, node_dim=edim, **{k: round(agg[k], 3) for k in agg},
                   sc_mae_sd=round(sd["sc_mae"], 3),
                   # protocol columns: rows from different head protocols must be
                   # distinguishable in the shared CSV (dropout drifted 0.15->0.2
                   # unnoticed once; review 2026-07-28)
                   dropout=DEFAULTS["dropout"], tc_lr=DEFAULTS["tc_lr"],
                   pool_dim=DEFAULTS["pool_dim"], hidden=DEFAULTS["hidden"])
        rows.append(row)
        pd.DataFrame([row]).to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
        print(f"[{tag:16s}] SC-MAE {agg['sc_mae']:.2f}±{sd['sc_mae']:.2f}  "
              f"cuprate {agg['cuprate_mae']:.2f}  MSLE {agg['msle']:.3f}", flush=True)
    print("\n=== ranked by SC-MAE ===")
    print(pd.DataFrame(rows).sort_values("sc_mae").to_string(index=False))


if __name__ == "__main__":
    main()
