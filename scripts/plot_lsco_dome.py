"""LSCO zero-shot doping-dome figure (matches the original 'doping series' style):
per-dopant colored, actual (filled) vs predicted (open) circles joined by a gray connector,
electron/hole-doped shaded regions -- PLUS a Gaussian-weighted guide-to-the-eye curve
through the predictions (the 'predicted dome').

Data reconstructed from a fine-tune run's predictions.csv (id, tc_true_K, tc_head_K):
LSCO variants = the mp-1077929-parented test rows; formal Cu oxidation = charge-balance
residual (La3+/Sr2+/Ba2+/O2-/Ce4+/... -> Cu; 2 = undoped).

Usage: python scripts/plot_lsco_dome.py [predictions.csv] [out.png]
"""
import sys, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pymatgen.core import Composition

PRED = sys.argv[1] if len(sys.argv) > 1 else \
    "model_data/2026-07-09/gps_tc_3dsc_doped_ft_forces2_oxifix_2026-07-09_11-36-22/predictions.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/figures/tc_vs_cu_oxidation_mp1077929_dome.png"

OX = {"La": 3, "Sr": 2, "Ba": 2, "Ce": 4, "Nd": 3, "Eu": 3, "Gd": 3, "Sm": 3,
      "Pr": 3, "Ca": 2, "K": 1, "Na": 1, "Li": 1, "Zn": 2, "Ni": 2, "O": -2}
# dopant -> color, matching the original figure's tab10 assignment
DOP_COLOR = {"Ba": "tab:blue", "Ca": "tab:orange", "Ce": "tab:green", "K": "tab:purple",
             "Li": "tab:brown", "Na": "tab:gray", "Sr": "tab:olive", "none": "tab:cyan",
             "Eu": "tab:pink", "Zn": "tab:red", "Ni": "tab:pink"}
DOP_ORDER = ["Ba", "Ca", "Ce", "K", "Li", "Na", "Sr", "none"]

def cu_oxidation(formula):
    try:
        c = Composition(formula).get_el_amt_dict(); cu = c.get("Cu", 0)
        if cu <= 0:
            return None
        return -sum(OX.get(e, 0) * n for e, n in c.items() if e != "Cu") / cu
    except Exception:
        return None

def dopant(formula):
    els = set(Composition(formula).get_el_amt_dict()) - {"La", "Cu", "O"}
    for d in ["Ba", "Sr", "Ca", "Ce", "K", "Li", "Na", "Eu", "Zn", "Ni"]:
        if d in els:
            return d
    return "none"                                    # O-excess only

def is_lsco(f):
    """La2CuO4 (K2NiF4 '214') family, provenance-independent (MP/NEMAD/ICSD): La-Cu-O with
    Cu the only major TM, the 214 A-site (La + A-dopants ~ 2*Cu), O ~ 4*Cu (allows +/-d)."""
    try:
        c = Composition(f).get_el_amt_dict(); cu = c.get("Cu", 0)
        if cu <= 0 or "La" not in c or "O" not in c:
            return False
        if any(c.get(t, 0) > 0.3 * cu for t in ("Fe", "Ni", "Co", "Mn", "Ru", "Mo", "Ti", "V", "Cr")):
            return False                               # a different active-TM cuprate
        a = sum(c.get(e, 0) for e in ("La", "Sr", "Ba", "Ca", "Nd", "Ce", "K", "Na", "Eu", "Gd", "Sm", "Pr", "Y"))
        return abs(a - 2 * cu) < 0.15 * cu and abs(c.get("O", 0) / cu - 4) < 0.6
    except Exception:
        return False

d = pd.read_csv(PRED)
d["formula"] = d.id.map(lambda i: re.split(r"-MP-|-ICSD-", str(i))[0])
l = d[d.formula.map(is_lsco)].copy()                 # 214 family, any provenance
l["cu"] = l.formula.map(cu_oxidation); l["dop"] = l.formula.map(dopant)
l = l.dropna(subset=["cu"]).sort_values("cu")

fig, ax = plt.subplots(figsize=(13, 8))
# shaded doping regions + undoped line + region labels (as in the original)
ax.axvspan(1.5, 2.0, color="aliceblue", zorder=0)
ax.axvspan(2.0, 2.7, color="mistyrose", alpha=0.6, zorder=0)
ax.axvline(2.0, color="0.4", ls="--", lw=1.2, zorder=1)
import matplotlib.transforms as mtransforms
_blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
ax.text(1.66, 0.03, "electron-doped", color="steelblue", ha="center",
        fontsize=15, transform=_blend)
ax.text(2.37, 0.03, "hole-doped", color="firebrick", ha="center",
        fontsize=15, transform=_blend)

# per-material: gray connector + actual (filled) & predicted (open) circles, colored by dopant
for r in l.itertuples():
    col = DOP_COLOR.get(r.dop, "k")
    ax.plot([r.cu, r.cu], [r.tc_true_K, r.tc_head_K], color="0.7", lw=0.8, zorder=1)
    ax.scatter(r.cu, r.tc_true_K, s=55, facecolor=col, edgecolor=col, zorder=3)
    ax.scatter(r.cu, r.tc_head_K, s=55, facecolor="none", edgecolor=col, linewidth=1.6, zorder=3)

# guide to the eye = mean predicted Tc in fixed-width Cu-oxidation bins, plotted at each
# (non-empty) bin center. Chosen over a polynomial fit: the sparse electron-doped / over-
# doped tails make any low-order polynomial overshoot or wiggle; binning stays faithful.
BINW = 0.02
edges = np.arange(np.floor(l.cu.min() / BINW) * BINW, l.cu.max() + BINW, BINW)
centers = (edges[:-1] + edges[1:]) / 2
l["_b"] = pd.cut(l.cu, edges, labels=centers)
g = l.groupby("_b", observed=True).tc_head_K.mean().dropna()
bx, by = g.index.astype(float).values, g.values
# light smoothing: interpolate the bin means onto a fine grid + small Gaussian filter
# (SMOOTH_EV wide) -> rounds the piecewise-linear kinks without polynomial overshoot.
SMOOTH_EV = 0.03
gx = np.linspace(bx.min(), bx.max(), 300)
sigma_pts = SMOOTH_EV / ((gx.max() - gx.min()) / (len(gx) - 1))
guide = np.clip(gaussian_filter1d(np.interp(gx, bx, by), sigma=sigma_pts), 0, None)
ax.plot(gx, guide, "-", color="0.15", lw=2.4, zorder=4)

ax.set_xlabel("formal Cu oxidation state  (2 = undoped)", fontsize=16)
ax.set_ylabel("$T_c$ (K)", fontsize=16)
ax.tick_params(axis="both", labelsize=14)
ax.set_title("La$_2$CuO$_4$ family (mp-1077929) doping series — actual (●) vs predicted (○), test set",
             fontsize=16)
ax.set_ylim(bottom=-2)

# legend: dopant colors (present) + actual/predicted/guide marker styles, 2 columns
handles = [Line2D([], [], marker="o", ls="", mfc=DOP_COLOR[dp], mec=DOP_COLOR[dp],
                  label=f"{dp}-doped") for dp in DOP_ORDER if (l.dop == dp).any()]
for dp in ["Eu", "Zn", "Ni"]:
    if (l.dop == dp).any():
        handles.append(Line2D([], [], marker="o", ls="", mfc=DOP_COLOR[dp], mec=DOP_COLOR[dp], label=f"{dp}-doped"))
handles += [Line2D([], [], marker="o", ls="", mfc="0.3", mec="0.3", label="actual"),
            Line2D([], [], marker="o", ls="", mfc="none", mec="0.3", label="predicted"),
            Line2D([], [], color="0.15", lw=2.4, label="predicted dome (guide)")]
ax.legend(handles=handles, ncol=2, loc="upper right", fontsize=12, framealpha=0.9)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}  ({len(l)} LSCO variants, {BINW}-wide bins ({len(bx)}) + {SMOOTH_EV} eV smooth, "
      f"peak {guide.max():.1f} K @ Cu={gx[guide.argmax()]:.2f})")
