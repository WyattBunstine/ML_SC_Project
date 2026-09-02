"""Family-holdout dome metrics + figure (the family analog of dome_stats /
plot_lsco_dome, which are LSCO-hardwired via is_lsco + the La-era OX table).

The held-out family comes from the RUN's own config.json (holdout_ids_csv) —
no per-family flags to keep in sync. Metrics mirror the dome_stats
conventions: same 0.02 Cu-ox binning + Gaussian guide through the ensemble
predictions, r/MAE over the family rows found in predictions.csv, guide value
at Cu=2.0 (reported as "at2.0" — the sharp-onset reading is La-specific: YBCO
chains put the undoped parent nearer 2.3, so interpret per family), guide peak
and its position.

Usage:
    python scripts/family_dome.py <run_dir> [--plot out.png [title]]
"""
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from pymatgen.core import Composition

# dome_stats' charge-balance table + the block cations of the held-out families
# (nominal states: Bi3+/Tl3+ rock-salt layers, Hg2+ dumbbells, Pb2+ substituent)
OX = {"La": 3, "Sr": 2, "Ba": 2, "Ce": 4, "Nd": 3, "Eu": 3, "Gd": 3, "Sm": 3,
      "Pr": 3, "Ca": 2, "K": 1, "Na": 1, "Li": 1, "Zn": 2, "Ni": 2, "O": -2,
      "Y": 3, "Bi": 3, "Tl": 3, "Hg": 2, "Pb": 2, "Al": 3, "Ti": 4, "B": 3}


def cu_oxidation(formula):
    try:
        c = Composition(formula).get_el_amt_dict()
        cu = c.get("Cu", 0)
        if cu <= 0:
            return None
        return -sum(OX.get(e, 0) * n for e, n in c.items() if e != "Cu") / cu
    except Exception:
        return None


def family_dome(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    hold = set(pd.read_csv(cfg["holdout_ids_csv"])["id"].astype(str))
    d = pd.read_csv(os.path.join(run_dir, "predictions.csv"))
    f = d[d.id.astype(str).isin(hold)].copy()
    if not len(f):
        raise SystemExit(f"{run_dir}: no holdout rows in predictions.csv")
    f["formula"] = f.id.map(lambda i: re.split(r"-MP-|-ICSD-", str(i))[0])
    f["cu"] = f.formula.map(cu_oxidation)
    f = f.dropna(subset=["cu"]).sort_values("cu")
    r_fam = float(np.corrcoef(f.tc_true_K, f.tc_head_K)[0, 1]) if len(f) > 2 else float("nan")
    mae = float((f.tc_true_K - f.tc_head_K).abs().mean())
    BINW = 0.02
    edges = np.arange(np.floor(f.cu.min() / BINW) * BINW, f.cu.max() + BINW, BINW)
    g = f.groupby(pd.cut(f.cu, edges, labels=(edges[:-1] + edges[1:]) / 2),
                  observed=True).tc_head_K.mean().dropna()
    bx, by = g.index.astype(float).values, g.values
    gx = np.linspace(bx.min(), bx.max(), 300)
    guide = np.clip(gaussian_filter1d(np.interp(gx, bx, by),
                                      sigma=0.03 / ((gx.max() - gx.min()) / (len(gx) - 1))), 0, None)
    return f, gx, guide, {
        "n_fam": len(f), "r_fam": r_fam, "mae_fam": mae,
        "at2.0": float(np.interp(2.0, gx, guide)) if gx.min() <= 2.0 <= gx.max() else float("nan"),
        "peak": float(guide.max()), "peak_cu": float(gx[guide.argmax()])}


def plot(f, gx, guide, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for _, r in f.iterrows():
        ax.plot([r.cu, r.cu], [r.tc_true_K, r.tc_head_K], color="0.8", lw=0.7, zorder=1)
    ax.scatter(f.cu, f.tc_true_K, s=28, c="tab:blue", label="actual", zorder=3)
    ax.scatter(f.cu, f.tc_head_K, s=34, facecolors="none", edgecolors="tab:red",
               label="predicted", zorder=3)
    ax.plot(gx, guide, color="tab:red", lw=1.5, alpha=0.6, label="predicted dome (guide)")
    ax.axvline(2.0, color="0.6", ls=":", lw=0.8)
    ax.set_xlabel("formal Cu oxidation state (charge balance)")
    ax.set_ylabel("$T_c$ (K)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    run_dir = sys.argv[1]
    f, gx, guide, s = family_dome(run_dir)
    tag = os.path.basename(os.path.normpath(run_dir))[:44]
    print(f"{tag:44s} n={s['n_fam']:3d} r_fam {s['r_fam']:6.3f} mae {s['mae_fam']:6.2f} "
          f"at2.0 {s['at2.0']:6.2f}K peak {s['peak']:6.1f}K @{s['peak_cu']:.2f}")
    if "--plot" in sys.argv:
        i = sys.argv.index("--plot")
        out = sys.argv[i + 1]
        title = sys.argv[i + 2] if len(sys.argv) > i + 2 else "family holdout dome"
        plot(f, gx, guide, out, title)
