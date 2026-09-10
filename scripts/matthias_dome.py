"""Matthias valence-dome metrics + figure: T_c vs valence electrons per atom for
a held-out set of ALLOY SYSTEMS (the e/a analog of dome_stats/family_dome).

The held-out ids come from the run's own config.json (holdout_ids_csv), so
there are no per-arm flags to keep in sync. Reports r / MAE over the holdout,
the binned max-T_c envelope (the Matthias curve), and where the predicted
envelope peaks — the question being whether a model that never saw these
alloy systems reproduces the 4.7 / 6.5 e/a humps.
    python scripts/matthias_dome.py <run_dir> [--plot out.png [title]]
"""
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pymatgen.core import Composition

VAL = {"Sc": 3, "Y": 3, "La": 3, "Lu": 3, "Ti": 4, "Zr": 4, "Hf": 4, "V": 5, "Nb": 5, "Ta": 5,
       "Cr": 6, "Mo": 6, "W": 6, "Mn": 7, "Tc": 7, "Re": 7, "Fe": 8, "Ru": 8, "Os": 8,
       "Co": 9, "Rh": 9, "Ir": 9, "Ni": 10, "Pd": 10, "Pt": 10}
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def ea(f):
    try:
        c = Composition(f).get_el_amt_dict()
    except Exception:  # noqa: BLE001
        return None
    return sum(VAL[e] * a for e, a in c.items()) / sum(c.values()) if set(c) <= set(VAL) else None


def envelope(x, y, lo=3.75, hi=8.5, w=0.25):
    edges = np.arange(lo, hi + w, w)
    b = pd.cut(x, edges, labels=(edges[:-1] + edges[1:]) / 2)
    g = pd.DataFrame({"b": b, "y": y}).groupby("b", observed=True).y.max().dropna()
    return g.index.astype(float).values, g.values


def stats(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    hold = set(pd.read_csv(cfg["holdout_ids_csv"])["id"].astype(str))
    d = pd.read_csv(os.path.join(run_dir, "predictions.csv"))
    f = d[d.id.astype(str).isin(hold)].copy()
    if not len(f):
        raise SystemExit(f"{run_dir}: no holdout rows in predictions.csv")
    f["formula"] = f.id.map(lambda i: re.split(r"-MP-|-ICSD-", str(i))[0])
    f["ea"] = f.formula.map(ea)
    f = f.dropna(subset=["ea"]).sort_values("ea")
    pos = f.tc_true_K > 0
    r = float(np.corrcoef(f.tc_true_K[pos], f.tc_head_K[pos])[0, 1]) if pos.sum() > 2 else float("nan")
    tx, ty = envelope(f.ea, f.tc_true_K)
    px, py = envelope(f.ea, f.tc_head_K)
    s = {"n": len(f), "n_pos": int(pos.sum()), "r_pos": r,
         "mae": float((f.tc_true_K - f.tc_head_K).abs().mean()),
         "mae_pos": float((f.tc_true_K[pos] - f.tc_head_K[pos]).abs().mean()) if pos.any() else float("nan"),
         "true_peak_K": float(ty.max()), "true_peak_ea": float(tx[ty.argmax()]),
         "pred_peak_K": float(py.max()), "pred_peak_ea": float(px[py.argmax()]),
         "env_r": float(np.corrcoef(np.interp(tx, px, py), ty)[0, 1]) if len(tx) > 2 else float("nan")}
    return f, (tx, ty, px, py), s


def guide(x, y, binw=0.05, sigma_ea=0.15):
    """Binned mean + Gaussian smoothing, the dome_stats/family_dome convention
    (there on Cu oxidation, here on e/a)."""
    from scipy.ndimage import gaussian_filter1d
    edges = np.arange(np.floor(x.min() / binw) * binw, x.max() + binw, binw)
    g = pd.DataFrame({"b": pd.cut(x, edges, labels=(edges[:-1] + edges[1:]) / 2), "y": y}) \
        .groupby("b", observed=True).y.mean().dropna()
    bx, by = g.index.astype(float).values, g.values
    if len(bx) < 3:
        return bx, by
    gx = np.linspace(bx.min(), bx.max(), 400)
    step = (gx.max() - gx.min()) / (len(gx) - 1)
    return gx, np.clip(gaussian_filter1d(np.interp(gx, bx, by), sigma_ea / step), 0, None)


def plot(f, curves, out, title):
    """Same anatomy as the cuprate dome figures (plot_lsco_dome / family_dome):
    actual filled, predicted open, a grey connector per material, and a smoothed
    guide through each — with e/a on the x-axis instead of Cu oxidation. The
    max-Tc envelope (Matthias's own construction) is kept as a faint dashed line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tx, ty, px, py = curves
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    for _, r in f.iterrows():
        ax.plot([r.ea, r.ea], [r.tc_true_K, r.tc_head_K], color="0.82", lw=0.7, zorder=1)
    ax.scatter(f.ea, f.tc_true_K, s=26, c="#2a78d6", edgecolors=SURF, linewidths=0.6, label="actual", zorder=3)
    ax.scatter(f.ea, f.tc_head_K, s=30, facecolors="none", edgecolors="#eb6834", linewidths=0.9, label="predicted", zorder=3)
    ax_, ay_ = guide(f.ea.values, f.tc_true_K.values)
    px_, py_ = guide(f.ea.values, f.tc_head_K.values)
    ax.plot(ax_, ay_, color="#2a78d6", lw=1.7, alpha=0.75, label="actual dome (guide)", zorder=2)
    ax.plot(px_, py_, color="#eb6834", lw=1.7, alpha=0.85, label="predicted dome (guide)", zorder=2)
    ax.plot(tx, ty, color="#2a78d6", lw=1.0, ls="--", alpha=0.35, label="actual max-$T_c$ envelope", zorder=1)
    ax.plot(px, py, color="#eb6834", lw=1.0, ls="--", alpha=0.4, label="predicted max-$T_c$ envelope", zorder=1)
    for x in (4.7, 6.5):
        ax.axvline(x, color="#8d8c87", lw=0.8, ls=":", zorder=0)
    ax.text(4.75, ax.get_ylim()[1] * 0.97, "Matthias peaks 4.7 / 6.5", fontsize=8, color=INK2, va="top")
    ax.set_xlabel("valence electrons per atom (e/a)", color=INK2)
    ax.set_ylabel("$T_c$ (K)", color=INK2)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(colors=INK2)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right", ncol=1)
    fig.tight_layout(); fig.savefig(out, dpi=150, facecolor=SURF)


if __name__ == "__main__":
    run = sys.argv[1]
    f, curves, s = stats(run)
    print(f"{os.path.basename(os.path.normpath(run))[:46]:46s} n={s['n']:4d} (pos {s['n_pos']:4d}) "
          f"r {s['r_pos']:6.3f} mae {s['mae']:5.2f} (pos {s['mae_pos']:5.2f}) | "
          f"true peak {s['true_peak_K']:5.1f}K @{s['true_peak_ea']:.2f} | pred peak {s['pred_peak_K']:5.1f}K @{s['pred_peak_ea']:.2f} | env_r {s['env_r']:6.3f}")
    if "--plot" in sys.argv:
        i = sys.argv.index("--plot")
        plot(f, curves, sys.argv[i + 1], sys.argv[i + 2] if len(sys.argv) > i + 2 else "Matthias valence dome — holdout")
