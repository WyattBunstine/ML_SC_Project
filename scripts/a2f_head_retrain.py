"""Head-only MSE retrain of the a2f head on the FROZEN rung-38 encoder + DS-B
validation — the step-1 experiment that isolates the spectrum loss from the
encoder (rung 42 showed the full-retrain damages the dome-carrying latents:
L-dome .515->.378 while the lambda quartiles decompressed).

1. DS-B accuracy of the FULL checkpoints (38 L1 / 42 MSE): per-material
   lambda / omega_log rank-corr + MAE vs the Cerqueira labels. CAVEAT printed:
   both pretrains trained on the full 8,253-row pack, so DS-B rows were seen —
   the 38-vs-42 comparison is still like-for-like.
2. Cache per-atom h for all e-ph structures from the FROZEN rung-38 encoder.
3. Retrain ONLY the a2f head (same arch) with MSE on the binned spectrum,
   trained on DS-A rows only (DS-B = genuinely unseen test), K bootstrapped
   seeds (BETE-style), ensemble = mean prediction.
4. Report: DS-B lambda/omega accuracy of the ensemble, spectrum effective
   rank, lambda quartiles — next to the 38/42 numbers.

Usage: python scripts/a2f_head_retrain.py [--seeds 5] [--epochs 300]
Ensemble weights -> model_data/<date>/a2f_head_mse/.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
for _p in ("models/common", "models/GPSTransformer", "models/head", "scripts"):
    sys.path.insert(0, os.path.join(_ROOT, _p))

from models.head.pred_features import _a2f_moments  # noqa: E402

CKPT38 = ("model_data/2026-08-31/gps_mt_38_w_magmom2/"
          "gps_mt_38_w_magmom2_2026-08-31_23-04-44/result_t38_model_best.pth.tar")
CKPT42 = ("model_data/2026-09-03/gps_mt_42_bete/"
          "gps_mt_42_bete_2026-09-03_11-03-25/result_t42_model_best.pth.tar")
EPH_IDX = "database/datafiles/EPH_Cerqueira/EPH_index.pickle"
DSB_IDS = "database/datafiles/EPH_Cerqueira/dsb_ids.csv"


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def eff_rank(mat, q=0.99):
    s = np.linalg.svd(mat - mat.mean(0), compute_uv=False)
    var = s ** 2 / (s ** 2).sum()
    return int((var.cumsum() < q).sum()) + 1


def build_loader(args_dict, device):
    from data import load_cif_dataset_from_args, collate_pool_geom
    from torch.utils.data import DataLoader
    idx = pd.read_pickle(EPH_IDX).copy()
    idx["value"] = 0.0                     # dummy target; a2f comes from graphs
    tmp = "/tmp/claude-1000/-home-wyatt-PycharmProjects-ML-SC-Project/b20ac24d-1c36-4209-bbc5-41e7bd3ae040/scratchpad/eph_index_val0.pickle"
    idx.to_pickle(tmp)
    ds = load_cif_dataset_from_args(tmp, args_dict, target_column="value")
    loader = DataLoader(ds, batch_size=32, shuffle=False,
                        collate_fn=collate_pool_geom, num_workers=6,
                        pin_memory=(str(device) != "cpu"))
    return ds, loader


def load_model(ckpt_path, device):
    from data import load_cif_dataset_from_args  # noqa: F401 (path setup)
    from model import GPSCrystalNet
    ck = torch.load(ckpt_path, map_location=device)
    a = ck["args"]
    ds, loader = build_loader(a, device)
    sa, sn, _, sp, _, _ = ds[0][0][:6]
    model = GPSCrystalNet.from_args(a, (sa.shape[-1], sn.shape[-1], sp.shape[-1]))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ds, loader


def targets_from_graphs():
    idx = pd.read_pickle(EPH_IDX)
    tg = {}
    for _, r in idx.iterrows():
        try:
            g = json.load(open(r["graph_path"]))
        except Exception:
            continue
        if g.get("a2f") and len(g["a2f"]) == 256:
            tg[str(r["id"])] = np.asarray(g["a2f"], np.float32)
    return tg


@torch.no_grad()
def full_model_a2f(model, loader, device):
    out = {}
    for inp, _t, _l, cids in loader:
        inp = tuple(x.to(device) if torch.is_tensor(x) else x for x in inp)
        o = model(*inp)
        for i, c in enumerate(cids):
            out[str(c)] = o["eph_a2f"][i].detach().cpu().numpy()
    return out


def score(pred, truth_idx, dsb, tag):
    lam_p, lam_t, wl_p, wl_t = [], [], [], []
    for cid in dsb:
        if cid not in pred:
            continue
        lam, wlog, _w2, _a = _a2f_moments(torch.as_tensor(pred[cid]))
        row = truth_idx.loc[cid]
        lam_p.append(lam); lam_t.append(row.eph_lambda)
        wl_p.append(wlog); wl_t.append(row.eph_wlog)
    lam_p, lam_t = np.array(lam_p), np.array(lam_t)
    wl_p, wl_t = np.array(wl_p), np.array(wl_t)
    mat = np.stack([pred[c] for c in dsb if c in pred])
    print(f"  {tag:16s} n={len(lam_p):4d}  lambda: rho {spearman(lam_p, lam_t):+.3f} "
          f"MAE {np.abs(lam_p - lam_t).mean():.3f} (pred med {np.median(lam_p):.3f} "
          f"true med {np.median(lam_t):.3f}) | omega_log: rho {spearman(wl_p, wl_t):+.3f} "
          f"MAE {np.abs(wl_p - wl_t).mean():.0f}K | spec eff-rank(99%) {eff_rank(mat)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    truth = pd.read_pickle(EPH_IDX).set_index("id")
    dsb = [c for c in pd.read_csv(DSB_IDS)["mat_id"].astype(str) if c in truth.index]
    tg = targets_from_graphs()
    print(f"{len(tg)} structures with baked a2f; DS-B {len(dsb)}")
    print("== DS-B accuracy, FULL checkpoints (CAVEAT: DS-B seen in pretraining; "
          "38-vs-42 comparison is like-for-like)")
    m38, ds, loader = load_model(CKPT38, dev)
    p38 = full_model_a2f(m38, loader, dev)
    score(p38, truth, dsb, "38 full (L1)")
    m42, _, loader42 = load_model(CKPT42, dev)
    p42 = full_model_a2f(m42, loader42, dev)
    score(p42, truth, dsb, "42 full (MSE)")
    del m42, p42, loader42

    # ---- frozen-38 h cache ----
    from embed_cache import build_embed_cache
    h = build_embed_cache(m38, loader, dev)          # cid -> (n_atoms, 128) on dev
    del m38
    ids = [c for c in h if str(c) in tg]
    dsb_set = set(dsb)
    train_ids = [c for c in ids if str(c) not in dsb_set]
    rng = np.random.RandomState(0)
    rng.shuffle(train_ids)
    val_ids = train_ids[: len(train_ids) // 10]
    tr_ids = train_ids[len(train_ids) // 10:]
    test_ids = [c for c in ids if str(c) in dsb_set]
    print(f"head retrain: train {len(tr_ids)} (DS-A) / val {len(val_ids)} / "
          f"DS-B test {len(test_ids)} (UNSEEN by this head)")
    Y = {c: torch.as_tensor(tg[str(c)], device=dev) for c in ids}

    def make_head(seed):
        torch.manual_seed(seed)
        return torch.nn.Sequential(
            torch.nn.Linear(128, 128), torch.nn.Softplus(),
            torch.nn.Linear(128, 128), torch.nn.Softplus(),
            torch.nn.Linear(128, 256), torch.nn.Softplus()).to(dev)

    def batch_pred(head, cids):
        hh = torch.cat([h[c] for c in cids])
        seg = torch.repeat_interleave(
            torch.arange(len(cids), device=dev),
            torch.tensor([h[c].shape[0] for c in cids], device=dev))
        out = head(hh)
        pooled = torch.zeros(len(cids), 256, device=dev).index_add_(0, seg, out)
        cnt = torch.bincount(seg, minlength=len(cids)).clamp(min=1).unsqueeze(1)
        return pooled / cnt

    heads = []
    for seed in range(a.seeds):
        head = make_head(seed)
        rs = np.random.RandomState(seed)
        boot = [tr_ids[i] for i in rs.randint(0, len(tr_ids), len(tr_ids))]
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        best, best_state, bad = float("inf"), None, 0
        for ep in range(a.epochs):
            head.train()
            rs.shuffle(boot)
            for b0 in range(0, len(boot), 256):
                cids = boot[b0:b0 + 256]
                opt.zero_grad()
                loss = ((batch_pred(head, cids)
                         - torch.stack([Y[c] for c in cids])) ** 2).mean()
                loss.backward()
                opt.step()
            head.eval()
            with torch.no_grad():
                v = float(((batch_pred(head, val_ids)
                            - torch.stack([Y[c] for c in val_ids])) ** 2).mean())
            if v < best - 1e-9:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in head.state_dict().items()}
            else:
                bad += 1
                if bad >= 30:
                    break
        head.load_state_dict(best_state)
        heads.append(head.eval())
        print(f"  seed {seed}: best val MSE {best:.3e} @ep{ep - bad}")

    with torch.no_grad():
        ens = {}
        for b0 in range(0, len(test_ids), 256):
            cids = test_ids[b0:b0 + 256]
            preds = torch.stack([batch_pred(hd, cids) for hd in heads]).mean(0)
            for i, c in enumerate(cids):
                ens[str(c)] = preds[i].cpu().numpy()
    print("== DS-B accuracy, retrained head (DS-B genuinely unseen):")
    score(ens, truth, dsb, "38h+MSE ens")
    lam = np.array([_a2f_moments(torch.as_tensor(v))[0] for v in ens.values()])
    q = np.quantile(lam, [0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"  ensemble DS-B lambda quartiles: {'/'.join(f'{x:.3f}' for x in q)}")
    out_dir = "model_data/2026-09-03/a2f_head_mse"
    os.makedirs(out_dir, exist_ok=True)
    torch.save([hd.state_dict() for hd in heads], os.path.join(out_dir, "ensemble.pt"))
    print(f"ensemble saved -> {out_dir}/ensemble.pt")


if __name__ == "__main__":
    main()
