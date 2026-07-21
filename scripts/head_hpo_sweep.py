"""Stage-A head HPO: random search over TcHead + optimizer hyperparameters.

Every candidate head trains WARMUP-ONLY (frozen encoder) against ONE shared
per-atom embedding cache, built once — the cache is head-independent, so a
candidate costs ~1-2 GPU-minutes instead of a ~35-minute full fine-tune. This
is the missing fairness step vs the (heavily tuned) XGBoost baseline: our head
hyperparameters were hand-set once (2026-06-29) and never swept.

Selection is on VAL in the target loss space only; test is never read here.
Stage B (top-k configs -> full two-phase protocol, multi-seed) confirms before
anything touches a reported number.

Resumable: rows append to the leaderboard CSV per config; a restart skips the
first N already-scored configs (the sampler is seeded, so config i is stable).

  PYTHONHASHSEED=0 python scripts/head_hpo_sweep.py --target msle \
      --split-group chemsys --n-configs 80 --out model_data/hpo/head_msle.csv
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
for _p in (os.path.join(_ROOT, "models", "common"),
           os.path.join(_ROOT, "models", "GPSTransformer"),
           os.path.join(_ROOT, "models", "head")):
    sys.path.insert(0, _p)

from models.head.embed_cache import build_embed_cache, cat_cached  # noqa: E402
from models.head.FineTune import VALID_LOSSES, reg_loss  # noqa: E402
from HeadData import assemble, chemsys_groups, make_splits, parent_composition_groups  # noqa: E402
from HeadModel import TcHead  # noqa: E402

# Bump whenever sample_config's axes/ranges change: the sampler is seeded, so any
# change remaps which config index i produces which hyperparameters. Resume
# refuses to append to a CSV written under a different version (see main).
SAMPLER_VERSION = 2


def sample_config(rng):
    # v2 (2026-07-21): added pool_agg/pool_rank/head_arch axes + pool_dim 128.
    return {
        "hidden": rng.choice([32, 64, 128, 192]),
        "pool_dim": rng.choice([16, 32, 64, 128]),
        "pool_agg": rng.choice(["meanmax", "mean", "max"]),
        "pool_rank": rng.choice([0, 16, 32]),          # 0 = full-rank g
        "head_arch": rng.choice(["concat", "concat", "struct_only", "resid"]),
        "dropout": round(rng.uniform(0.05, 0.35), 3),
        "tc_lr": float(np.exp(rng.uniform(np.log(3e-4), np.log(1e-2)))),
        "tc_weight_decay": float(np.exp(rng.uniform(np.log(1e-5), np.log(3e-3)))),
        "batch_size": rng.choice([32, 64, 128]),
        "patience": rng.choice([10, 20, 30]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="model_data/2026-07-14/gps_mt_06_forces_w2_valence/"
                    "gps_mt_06_forces_w2_valence_2026-07-14_13-46-58/result_valence_model_best.pth.tar")
    ap.add_argument("--index", default="database/datafiles/MP/SC_MP_V4_doped.pickle")
    ap.add_argument("--descriptors", default="database/datafiles/MP/descriptors_doped.pickle")
    ap.add_argument("--metadata", default="database/datafiles/MP/3DSC_MP.csv")
    ap.add_argument("--split-group", default="chemsys", choices=["chemsys", "parent_comp", "parent"])
    # Loss spaces shared with FineTune.reg_loss — stage A can select under any
    # objective stage B trains with (previously z-space msle/l1 only, so mse_k/wl1_k
    # configs were tuned under an objective the sweep never scored).
    ap.add_argument("--target", default="msle", choices=list(VALID_LOSSES))
    ap.add_argument("--n-configs", type=int, default=80)
    ap.add_argument("--warmup-max", type=int, default=120)
    ap.add_argument("--sample-seed", type=int, default=7)
    ap.add_argument("--split-seed", type=int, default=123)
    ap.add_argument("--out", default="model_data/hpo/head_sweep.csv")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ---- static tables + leak-free grouped split (identical to FineTune) ----
    data = assemble(args.index, "unused_embed_dir", args.descriptors, args.metadata,
                    pooling="meanmax", require_embed=False)
    ids = list(data["ids"])
    grp = (chemsys_groups(ids) if args.split_group == "chemsys"
           else parent_composition_groups(ids) if args.split_group == "parent_comp" else None)
    split = make_splits(data, args.split_seed, 0.1, 0.2, groups=grp)
    id2split = dict(zip(ids, split))
    id2row = {i: r for r, i in enumerate(ids)}
    is_sc = data["label"] == 1

    # ---- frozen encoder + ONE embedding cache over train+val ----
    from data import load_cif_dataset_from_args, dataset_ids, collate_pool_geom
    from model import GPSCrystalNet
    from torch.utils.data import DataLoader, Subset
    ckpt = torch.load(args.checkpoint, map_location=device)
    a = ckpt.get("args", {})
    ds = load_cif_dataset_from_args(args.index, a)
    sa, sn, _, sp, _, _ = ds[0][0][:6]
    model = GPSCrystalNet.from_args(a, (sa.shape[-1], sn.shape[-1], sp.shape[-1]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    ds_ids = dataset_ids(ds)
    assert set(ds_ids) == set(id2split), "dataset/static-table id mismatch"

    def positions(s):
        return [i for i, cid in enumerate(ds_ids) if id2split.get(cid) == s and is_sc[id2row[cid]]]

    cache = {}
    for s in ("train", "val"):
        dl = DataLoader(Subset(ds, positions(s)), batch_size=64,
                        collate_fn=collate_pool_geom, num_workers=4)
        cache.update(build_embed_cache(model, dl, device))
    print(f"cache built: {len(cache)} structures", flush=True)
    enc_dim = model.atom_fea_len
    del model
    torch.cuda.empty_cache()

    phys_t = torch.as_tensor(data["phys"].astype(np.float32), device=device)
    tc_t = torch.as_tensor(data["tc"], device=device).float()
    tr_ids = [ds_ids[i] for i in positions("train")]
    va_ids = [ds_ids[i] for i in positions("val")]
    tr_rows = [id2row[c] for c in tr_ids]
    # Kelvin-space loss normalizers, train-fit — same definition as FineTune.run.
    _tc_tr = tc_t[tr_rows].cpu()
    k_var = max(float(_tc_tr.var()), 1.0)
    k_wmean = max(float((1.0 + _tc_tr).mean()), 1.0)

    def forward(head, bids):
        h, seg = cat_cached(cache, bids, device)
        rows = [id2row[c] for c in bids]
        _lg, z = head(h, phys_t[rows], seg=seg, n=len(bids))
        return z, rows

    def run_config(cfg, seed=0):
        torch.manual_seed(seed)
        rank = int(cfg.get("pool_rank", 0)) or None    # 0 = full-rank g
        head = TcHead(enc_dim, phys_t.shape[1], 64, int(cfg["hidden"]), float(cfg["dropout"]),
                      pooling="deepsets", pool_dim=int(cfg["pool_dim"]), pool_rank=rank,
                      head_arch=str(cfg.get("head_arch", "concat")),
                      pool_agg=str(cfg.get("pool_agg", "meanmax"))).to(device)
        head.fit_target(tc_t[tr_rows].cpu())
        head.phys_std.fit(phys_t[tr_rows].cpu()); head.to(device)
        y_z = head.target_to_z(tc_t)
        opt = torch.optim.AdamW(head.parameters(), lr=cfg["tc_lr"],
                                weight_decay=cfg["tc_weight_decay"])
        rng = random.Random(seed)
        best, bad, best_ep = np.inf, 0, -1
        bs = int(cfg["batch_size"])
        for ep in range(args.warmup_max):
            head.train()
            order = tr_ids[:]; rng.shuffle(order)
            for b0 in range(0, len(order), bs):
                bids = order[b0:b0 + bs]
                opt.zero_grad()
                z, rows = forward(head, bids)
                # Shared loss (FineTune.reg_loss): stage A trains and selects in the
                # SAME space stage B will optimize, for every supported target.
                loss = reg_loss(z, y_z[rows], tc_t[rows], args.target,
                                head.tc_mean, head.tc_std, k_var, k_wmean)
                loss.backward(); opt.step()
            head.eval(); num = den = 0.0
            with torch.no_grad():
                for b0 in range(0, len(va_ids), 256):
                    bids = va_ids[b0:b0 + 256]
                    z, rows = forward(head, bids)
                    num += float(reg_loss(z, y_z[rows], tc_t[rows], args.target,
                                          head.tc_mean, head.tc_std,
                                          k_var, k_wmean)) * len(bids)
                    den += len(bids)
            v = num / max(den, 1)
            if v < best - 1e-6:
                best, bad, best_ep = v, 0, ep
            else:
                bad += 1
            if bad >= int(cfg["patience"]):
                break
        return best, best_ep

    rng = random.Random(args.sample_seed)
    configs = [sample_config(rng) for _ in range(args.n_configs)]
    done = 0
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        # Resume skips by row count, which is only valid if the existing rows came
        # from the SAME sampler version and sweep flags — otherwise config_idx maps
        # to different hyperparameters and the leaderboard mixes incomparable
        # objectives/encoders. Refuse instead of silently mixing.
        stale = []
        if "sampler_version" not in prev.columns or \
                not (prev["sampler_version"] == SAMPLER_VERSION).all():
            stale.append(f"sampler_version != {SAMPLER_VERSION}")
        for col, cur in (("target", args.target), ("split_group", args.split_group),
                         ("checkpoint", os.path.basename(args.checkpoint))):
            if col in prev.columns and not (prev[col] == cur).all():
                stale.append(f"{col} != {cur!r}")
        if stale:
            sys.exit(f"refusing to resume {args.out}: existing rows don't match this "
                     f"invocation ({'; '.join(stale)}) — use a fresh --out")
        done = len(prev)
        print(f"resuming: {done} configs already scored", flush=True)
    for i in range(done, len(configs)):
        cfg = configs[i]
        v, ep = run_config(cfg, seed=0)
        row = {**cfg, "config_idx": i, "val": v, "best_epoch": ep, "target": args.target,
               "split_group": args.split_group, "checkpoint": os.path.basename(args.checkpoint),
               "sampler_version": SAMPLER_VERSION}
        pd.DataFrame([row]).to_csv(args.out, mode="a", header=not os.path.exists(args.out),
                                   index=False)
        print(f"[{i + 1}/{len(configs)}] val={v:.4f} @ep{ep}  {cfg}", flush=True)
    lb = pd.read_csv(args.out).sort_values("val")
    print("\n=== top 10 ===")
    print(lb.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
