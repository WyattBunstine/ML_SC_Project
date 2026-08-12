"""Dome metrics for la_series fine-tune runs, one table row per run dir.

Reproduces the plot_lsco_dome selection (is_lsco + charge-balance Cu oxidation)
and the established metric conventions:
  r_all   — Pearson(true, pred) over ALL holdout rows ("dome r" in the memos:
            09=.791 / 12=.774 / 15=.777 / 16=.811 / 17=.748 / 18=.817)
  r_lsco  — Pearson over the 59 LSCO variants only
  onset   — predicted-dome guide interpolated at Cu=2.0 (sharp insulator onset
            is LOW: 09-era ~5.4 K sharp vs disorder-flattened 13.3 K)
  peak    — guide maximum and its Cu position
  mae_all — MAE over all holdout rows (the "holdout MAE" in the memos)

Usage: python scripts/dome_stats.py <run_dir_or_predictions.csv> [...]
"""
import re
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from pymatgen.core import Composition

OX = {"La": 3, "Sr": 2, "Ba": 2, "Ce": 4, "Nd": 3, "Eu": 3, "Gd": 3, "Sm": 3,
      "Pr": 3, "Ca": 2, "K": 1, "Na": 1, "Li": 1, "Zn": 2, "Ni": 2, "O": -2}


def cu_oxidation(formula):
    try:
        c = Composition(formula).get_el_amt_dict(); cu = c.get("Cu", 0)
        if cu <= 0:
            return None
        return -sum(OX.get(e, 0) * n for e, n in c.items() if e != "Cu") / cu
    except Exception:
        return None


def is_lsco(f):
    try:
        c = Composition(f).get_el_amt_dict(); cu = c.get("Cu", 0)
        if cu <= 0 or "La" not in c or "O" not in c:
            return False
        if any(c.get(t, 0) > 0.3 * cu for t in ("Fe", "Ni", "Co", "Mn", "Ru", "Mo", "Ti", "V", "Cr")):
            return False
        a = sum(c.get(e, 0) for e in ("La", "Sr", "Ba", "Ca", "Nd", "Ce", "K", "Na", "Eu", "Gd", "Sm", "Pr", "Y"))
        return abs(a - 2 * cu) < 0.15 * cu and abs(c.get("O", 0) / cu - 4) < 0.6
    except Exception:
        return False


def dome_stats(pred_csv):
    d = pd.read_csv(pred_csv)
    d["formula"] = d.id.map(lambda i: re.split(r"-MP-|-ICSD-", str(i))[0])
    r_all = float(np.corrcoef(d.tc_true_K, d.tc_head_K)[0, 1])
    mae_all = float((d.tc_true_K - d.tc_head_K).abs().mean())
    l = d[d.formula.map(is_lsco)].copy()
    l["cu"] = l.formula.map(cu_oxidation)
    l = l.dropna(subset=["cu"]).sort_values("cu")
    r_lsco = float(np.corrcoef(l.tc_true_K, l.tc_head_K)[0, 1]) if len(l) > 2 else float("nan")
    BINW = 0.02
    edges = np.arange(np.floor(l.cu.min() / BINW) * BINW, l.cu.max() + BINW, BINW)
    g = l.groupby(pd.cut(l.cu, edges, labels=(edges[:-1] + edges[1:]) / 2),
                  observed=True).tc_head_K.mean().dropna()
    bx, by = g.index.astype(float).values, g.values
    gx = np.linspace(bx.min(), bx.max(), 300)
    guide = np.clip(gaussian_filter1d(np.interp(gx, bx, by),
                                      sigma=0.03 / ((gx.max() - gx.min()) / (len(gx) - 1))), 0, None)
    return {"n_lsco": len(l), "r_all": r_all, "r_lsco": r_lsco, "mae_all": mae_all,
            "onset": float(np.interp(2.0, gx, guide)),
            "peak": float(guide.max()), "peak_cu": float(gx[guide.argmax()])}


if __name__ == "__main__":
    import os
    print(f"{'run':44s} {'r_all':>6s} {'r_lsco':>7s} {'onset':>7s} {'peak':>13s} {'MAE':>6s}")
    for arg in sys.argv[1:]:
        csv = arg if arg.endswith(".csv") else os.path.join(arg, "predictions.csv")
        s = dome_stats(csv)
        tag = os.path.basename(os.path.dirname(csv) if csv.endswith(".csv") else arg)[:44]
        print(f"{tag:44s} {s['r_all']:6.3f} {s['r_lsco']:7.3f} {s['onset']:6.2f}K "
              f"{s['peak']:6.1f}K @{s['peak_cu']:.2f} {s['mae_all']:6.2f}")
