"""Nickelate zero-shot holdout scorer: the 52-row holdout (43 oxide nickelates +
9 Ni-pnictide-oxides, zero train exposure; ids from the RUN's own config.json
holdout_ids_csv) read out of a head run's predictions.csv.

Reports, over the holdout rows found: n, the SC rows' (tc>0) true-vs-predicted
Pearson r, MAE and median predicted/true ratio (the "scale" the head reaches),
the tc=0 rows' mean prediction (false-positive pressure), and the per-row table
for the SC rows. The 2026-08-20 rung-20 verdict was: every SC row predicted
~0.5-2 K vs true 13.5-30 K (scale ratio ~0.05) — the d9 transfer FAILS.

Usage:
    python scripts/nickelate_holdout.py <run_dir> [--rows]
"""
import json
import os
import sys

import numpy as np
import pandas as pd


def main():
    run = sys.argv[1].rstrip("/")
    show_rows = "--rows" in sys.argv
    cfg = json.load(open(os.path.join(run, "config.json")))
    hold = set(pd.read_csv(cfg["holdout_ids_csv"])["id"])
    p = pd.read_csv(os.path.join(run, "predictions.csv"))
    m = p[p["id"].isin(hold)].copy()
    sc = m[m["tc_true_K"] > 0]
    zero = m[m["tc_true_K"] <= 0]
    name = os.path.basename(run)[:44]
    if len(sc) >= 3:
        r = np.corrcoef(sc["tc_true_K"], sc["tc_head_K"])[0, 1]
        mae = (sc["tc_true_K"] - sc["tc_head_K"]).abs().mean()
        ratio = float(np.median(sc["tc_head_K"] / sc["tc_true_K"]))
        pred_mean, true_mean = sc["tc_head_K"].mean(), sc["tc_true_K"].mean()
    else:
        r = mae = ratio = pred_mean = true_mean = float("nan")
    print(f"{name:44s} n={len(m):2d}/{len(hold)} SC={len(sc):2d} r_sc {r:6.3f} "
          f"mae {mae:6.2f} pred/true {ratio:5.2f} (mean pred {pred_mean:5.1f}K vs true "
          f"{true_mean:5.1f}K) | tc=0 rows {len(zero):2d} mean pred {zero['tc_head_K'].mean():5.2f}K")
    if show_rows and len(sc):
        cols = ["id", "family", "tc_true_K", "tc_head_K"]
        print(sc.sort_values("tc_true_K", ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
