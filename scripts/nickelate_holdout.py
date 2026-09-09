"""Nickelate zero-shot holdout scorer: the 52-row holdout (43 oxide nickelates +
9 Ni-pnictide-oxides, zero train exposure; ids from the RUN's own config.json
holdout_ids_csv) read out of a head run's predictions.csv.

Reports, over the holdout rows found and per SUBSET — "d9" = the cuprate
analogs (infinite-layer ANiO2 + doped 112s + the quintuple-layer A6Ni5O12,
nominal Ni valence <= 1.35 from the formula), "oxide-other" = the rest of the
oxide nickelates (RP bilayer/trilayer, 214s, NiO3 parents), "pnictide" = the
Ni-pnictide-oxides — n, the SC rows' (tc>0) true-vs-predicted Pearson r, MAE
and median predicted/true ratio (the "scale" the head reaches), the tc=0 rows'
mean prediction (false-positive pressure), and the per-row table for SC rows.

THE BAR TO MEET (user, 2026-09-09): the 2026-07-14 V4 zero-shot run
(model_data/2026-07-14/gps_tc_v4_nickelate_holdout_2026-07-14_14-16-13,
forces-w2 encoder, 3DSC-only training, 43-row oxide holdout / 17 SC) nailed the
d9 family — NdNiO2 13.2 vs 13.5 K, Nd0.9Sr0.1NiO2 12.8/9, Nd6Ni5O12 14.9/15,
the Nd-112 doping series 11.7-12.8 vs 9-13.5, LaNiO2 4-5 vs 11-15 — with
non-SC parents at 0.42 K mean; SC-17 MAE 14.3 (dragged by the RP misses:
La3Ni2O7 80 K -> 0.01). The 09-valence encoder (07-15) then fixed LaNiO2
(10.99) at SC-17 MAE 13.4, Spearman 0.47. The 2026-08-20 rung-20 run FAILED it:
every SC row 0.5-2 K (scale 0.13).

Usage:
    python scripts/nickelate_holdout.py <run_dir> [--rows]
"""
import json
import os
import sys

import numpy as np
import pandas as pd


_PN = {"P", "As", "Sb", "Bi"}
_A3 = {"La", "Nd", "Pr", "Sm", "Eu", "Gd", "Y", "Ce"}
_A2 = {"Sr", "Ca", "Ba"}


def subset(cid):
    """d9 / oxide-other / pnictide from the id's formula prefix (before '-MP-')."""
    from pymatgen.core import Composition
    try:
        c = Composition(cid.split("-MP-")[0]).get_el_amt_dict()
    except Exception:  # noqa: BLE001
        return "unknown"
    if _PN & set(c):
        return "pnictide"
    ni, o = c.get("Ni", 0.0), c.get("O", 0.0)
    if ni <= 0 or o <= 0:
        return "unknown"
    a3 = sum(v for k, v in c.items() if k in _A3)
    a2 = sum(v for k, v in c.items() if k in _A2)
    ni_val = (2.0 * o - 3.0 * a3 - 2.0 * a2) / ni
    return "d9" if ni_val <= 1.35 else "oxide-other"


def _line(tag, m):
    sc = m[m["tc_true_K"] > 0]
    zero = m[m["tc_true_K"] <= 0]
    if len(sc) >= 3:
        r = np.corrcoef(sc["tc_true_K"], sc["tc_head_K"])[0, 1]
        mae = (sc["tc_true_K"] - sc["tc_head_K"]).abs().mean()
        ratio = float(np.median(sc["tc_head_K"] / sc["tc_true_K"]))
        pm, tm = sc["tc_head_K"].mean(), sc["tc_true_K"].mean()
    else:
        r = mae = ratio = pm = tm = float("nan")
    z = zero["tc_head_K"].mean() if len(zero) else float("nan")
    return (f"  {tag:12s} n={len(m):2d} SC={len(sc):2d} r_sc {r:6.3f} mae {mae:6.2f} "
            f"pred/true {ratio:5.2f} (mean pred {pm:5.1f}K vs true {tm:5.1f}K) | "
            f"tc=0 n={len(zero):2d} mean pred {z:5.2f}K")


def main():
    run = sys.argv[1].rstrip("/")
    show_rows = "--rows" in sys.argv
    cfg = json.load(open(os.path.join(run, "config.json")))
    hold = set(pd.read_csv(cfg["holdout_ids_csv"])["id"])
    p = pd.read_csv(os.path.join(run, "predictions.csv"))
    m = p[p["id"].isin(hold)].copy()
    m["subset"] = m["id"].map(subset)
    print(f"{os.path.basename(run)[:60]}  holdout {len(m)}/{len(hold)} rows")
    print(_line("ALL", m))
    for tag in ("d9", "oxide-other", "pnictide"):
        sub = m[m["subset"] == tag]
        if len(sub):
            print(_line(tag, sub))
    if show_rows:
        sc = m[m["tc_true_K"] > 0]
        cols = ["subset", "id", "tc_true_K", "tc_head_K"]
        print(sc.sort_values(["subset", "tc_true_K"], ascending=[True, False])[cols].to_string(index=False))


if __name__ == "__main__":
    main()
