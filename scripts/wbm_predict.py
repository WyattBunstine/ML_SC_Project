"""Matbench Discovery, energy-only track: predict WBM formation energies with a
trained MPtrj rung and score the discovery metrics.

Prediction is direct on the DFT-RELAXED structure (the benchmark's RS2RE-style
entry point — we never relax). Stability follows the benchmark's own
convention: the model's formation-energy RESIDUAL is added to the DFT hull
distance, so
    e_above_hull_pred = e_above_hull_true + (e_form_pred - e_form_true)
which keeps the MP convex hull fixed and asks only whether the model would have
called each material stable. Metrics are reported over all rows and over the
`unique_prototype` subset the benchmark headlines.

  python scripts/wbm_predict.py [--checkpoint ...] [--pack ...] [--batch-size 256]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
for _p in ("models/common", "models/GPSTransformer"):
    sys.path.insert(0, os.path.join(_ROOT, _p))
from data import load_cif_dataset, collate_pool_multitask  # noqa: E402
from model import GPSCrystalNet  # noqa: E402

WBM = os.path.join(_ROOT, "database", "datafiles", "WBM")
CKPT = ("model_data/2026-09-08/gps_mt_50_fe_full/gps_mt_50_fe_full_2026-09-08_11-36-24/"
        "result_t50_model_best.pth.tar")


def metrics(df, tag):
    """Discovery metrics at the 0 eV/atom stability threshold."""
    t, p = df.e_above_hull_true.values, df.e_above_hull_pred.values
    ts, ps = t < 0, p < 0
    tp = int((ts & ps).sum()); fp = int((~ts & ps).sum()); fn = int((ts & ~ps).sum())
    tn = int((~ts & ~ps).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    base = ts.mean()                                   # stable fraction of the pool
    err = df.e_form_pred - df.e_form_true
    ss = float(((df.e_form_true - df.e_form_true.mean()) ** 2).sum())
    return {
        "subset": tag, "n": len(df),
        "MAE_eV_atom": float(err.abs().mean()), "RMSE": float(np.sqrt((err ** 2).mean())),
        "R2": float(1 - (err ** 2).sum() / ss) if ss > 0 else float("nan"),
        "F1": f1, "precision": prec, "recall": rec,
        "accuracy": (tp + tn) / max(len(df), 1),
        "DAF": prec / base if base > 0 else float("nan"),
        "n_stable_true": int(ts.sum()), "n_predicted_stable": int(ps.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--pack", default=os.path.join(WBM, "wbm_pack_v45"))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(WBM, "wbm_predictions.csv"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location=dev)
    args = dict(ck["args"])
    print(f"[wbm] checkpoint {os.path.basename(os.path.dirname(a.checkpoint))} "
          f"(tasks {args.get('tasks')})", flush=True)
    ds = load_cif_dataset(
        a.pack, max_num_nbr=args.get("max_num_nbr", 14),
        max_num_poly_nbr=args.get("max_num_poly_nbr", 16),
        graph_cache_size=args.get("graph_cache_size", 4096),
        target_column=args.get("target_column"),
        use_bond_angles=args.get("use_bond_angles", False),
        use_poly_edges=args.get("use_poly_edges", True), build_angle_bias=True,
        use_rich_node_features=args.get("use_rich_node_features", False),
        use_valence_features=args.get("use_valence_features", False),
        use_cf_features=args.get("use_cf_features", False),
        use_bvs_features=args.get("use_bvs_features", False),
        mask_oxidation_feature=args.get("mask_oxidation_feature", False),
        mask_geometry_features=args.get("mask_geometry_features", False),
        use_poly_node_summary=args.get("use_poly_node_summary", False),
        use_dihedrals=args.get("use_dihedrals", False), multitask=True,
        n_energy=args.get("n_energy", 256), dos_per_atom=args.get("dos_per_atom", True),
        frame_subsample=1)
    s0 = ds[0][0]
    model = GPSCrystalNet.from_args(args, (s0[0].shape[-1], s0[1].shape[-1], s0[3].shape[-1]))
    model.load_state_dict(ck["state_dict"])
    model = model.to(dev).eval()
    loader = torch.utils.data.DataLoader(
        ds, batch_size=a.batch_size, shuffle=False, collate_fn=collate_pool_multitask,
        num_workers=a.workers, pin_memory=(dev.type == "cuda"))
    ids, preds = [], []
    with torch.no_grad():
        for k, (inp, _t, _m, cids) in enumerate(loader):
            inp = tuple(x.to(dev) if torch.is_tensor(x) else x for x in inp)
            out = model(*inp)                         # cart=None -> energy head only
            preds.append(out["energy"].detach().float().cpu().numpy().ravel())
            ids += [str(c) for c in cids]
            if (k + 1) % 100 == 0:
                print(f"[wbm] {len(ids)}/{len(ds)}", flush=True)
    pred = pd.DataFrame({"material_id": ids, "e_form_pred": np.concatenate(preds)})
    summ = pd.read_csv(os.path.join(WBM, "wbm-summary.csv.gz"))
    summ["material_id"] = summ.material_id.astype(str)
    df = pred.merge(summ[["material_id", "formula", "e_form_per_atom_mp2020_corrected",
                          "e_above_hull_mp2020_corrected_ppd_mp", "unique_prototype"]],
                    on="material_id", how="inner")
    df = df.rename(columns={"e_form_per_atom_mp2020_corrected": "e_form_true",
                            "e_above_hull_mp2020_corrected_ppd_mp": "e_above_hull_true"})
    df = df.dropna(subset=["e_form_true", "e_above_hull_true"])
    # benchmark convention: shift the DFT hull distance by the model's residual
    df["e_above_hull_pred"] = df.e_above_hull_true + (df.e_form_pred - df.e_form_true)
    df.to_csv(a.out, index=False)
    rep = pd.DataFrame([metrics(df, "all"),
                        metrics(df[df.unique_prototype], "unique_prototype")])
    pd.set_option("display.width", 200)
    print("\n" + rep.round(4).to_string(index=False))
    print(f"\n[wbm] {len(df)} predictions -> {a.out}")


if __name__ == "__main__":
    main()
