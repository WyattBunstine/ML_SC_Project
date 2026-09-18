"""Two (or more) single-series valence-dome holdouts on one axis, points only.

    python scripts/plot_matthias_combined.py <out.png> <label>=<run_dir> [...]

No smoothed guides and no envelopes: the marks are the data. Colour carries the
SERIES (fixed palette order); fill carries measured vs predicted, so the two are
never distinguished by colour alone. A thin connector pairs the two readings of
one material.
"""
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matthias_dome import ea  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def rows(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    hold = set(pd.read_csv(cfg["holdout_ids_csv"])["id"].astype(str))
    d = pd.read_csv(os.path.join(run_dir, "predictions.csv"))
    f = d[d.id.astype(str).isin(hold)].copy()
    f["formula"] = f.id.map(lambda i: re.split(r"-MP-|-ICSD-", str(i))[0])
    f["ea"] = f.formula.map(ea)
    return f.dropna(subset=["ea"]).sort_values("ea")


def main():
    out = sys.argv[1]
    specs = [s.split("=", 1) for s in sys.argv[2:]]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    for k, (label, run) in enumerate(specs):
        f = rows(run)
        c = SERIES[k % len(SERIES)]
        for _, r in f.iterrows():
            ax.plot([r.ea, r.ea], [r.tc_true_K, r.tc_head_K], color=c, lw=0.6, alpha=0.30, zorder=1)
        ax.scatter(f.ea, f.tc_true_K, s=34, c=c, edgecolors=SURF, linewidths=0.7,
                   label=f"{label} — measured", zorder=3)
        ax.scatter(f.ea, f.tc_head_K, s=38, facecolors="none", edgecolors=c, linewidths=1.1,
                   label=f"{label} — predicted", zorder=3)
        pos = f.tc_true_K > 0
        r_ = np.corrcoef(f.tc_true_K[pos], f.tc_head_K[pos])[0, 1] if pos.sum() > 2 else float("nan")
        print(f"{label}: n={len(f)} r={r_:.3f} MAE={np.abs(f.tc_true_K - f.tc_head_K).mean():.2f} K")
    ax.set_ylim(0, max(ax.get_ylim()[1], 13) * 1.16)   # headroom so the legend never sits on data
    for x in (4.7, 6.5):
        ax.axvline(x, color="#8d8c87", lw=0.8, ls=":", zorder=0)
        ax.text(x + 0.04, ax.get_ylim()[1] * 0.055, f"Matthias {x}", fontsize=8, color=INK2, va="bottom")
    ax.set_xlabel("valence electrons per atom (e/a)", color=INK2)
    ax.set_ylabel("$T_c$ (K)", color=INK2)
    ax.set_title("Valence-dome holdouts: each series removed from training entirely (rung 51, msle)",
                 loc="left", fontsize=11, color=INK)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color("#c9c8c2")
    ax.tick_params(colors=INK2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper center",
              ncol=2, bbox_to_anchor=(0.5, 1.0), columnspacing=2.6, handletextpad=0.5)
    fig.tight_layout(); fig.savefig(out, dpi=160, facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    main()
