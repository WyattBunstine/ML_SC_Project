"""Per-SC-class scoring of a head run's predictions.csv: MAE / signed bias /
count on Tc>0 rows per family (Cuprate, Ferrite=iron-based, Heavy_fermion,
Oxide, Chevrel, Other), plus all-rows MAE per family.

Usage: python scripts/family_stats.py <run_dir_or_csv> [...]"""
import os
import sys

import numpy as np
import pandas as pd

FAMS = ["Cuprate", "Ferrite", "Heavy_fermion", "Oxide", "Chevrel", "Other"]

def stats(csv):
    d = pd.read_csv(csv)
    row = {}
    for f in FAMS:
        g = d[d.family == f]
        p = g[g.tc_true_K > 0]
        row[f] = (len(p),
                  float((p.tc_true_K - p.tc_head_K).abs().mean()) if len(p) else np.nan,
                  float((p.tc_head_K - p.tc_true_K).mean()) if len(p) else np.nan)
    row["ALL"] = (int((d.tc_true_K > 0).sum()),
                  float((d.tc_true_K - d.tc_head_K).abs().mean()),
                  float(np.corrcoef(d.tc_true_K, d.tc_head_K)[0, 1]))
    return row

if __name__ == "__main__":
    hdr = f"{'run':38s}" + "".join(f"{f[:9]:>16s}" for f in FAMS) + f"{'ALL(mae,r)':>18s}"
    print(hdr)
    for arg in sys.argv[1:]:
        csv = arg if arg.endswith(".csv") else os.path.join(arg, "predictions.csv")
        r = stats(csv)
        tag = os.path.basename(os.path.dirname(csv) if csv.endswith(".csv") else arg)[:38]
        cells = "".join(f"  {r[f][1]:5.1f}({r[f][0]:3d})" if r[f][0] else f"  {'--':>10s}"
                        for f in FAMS)
        print(f"{tag:38s}{cells}   {r['ALL'][1]:5.2f},{r['ALL'][2]:.3f}")
