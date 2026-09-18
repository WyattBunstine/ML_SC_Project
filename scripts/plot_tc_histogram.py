"""Stacked histogram of superconductors by T_c, colour = family (V11 by default).

    python scripts/plot_tc_histogram.py [--index SC_MP_V11.pickle] [--bin 2.5] [--out docs/figures/dataset/tc_histogram_v11.png]
"""
import argparse, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _ROOT)
from models.head.HeadData import load_family_metadata  # noqa: E402
MP = os.path.join(_ROOT, "database", "datafiles", "MP")
GROUP = {"Other": "conventional", "Chevrel": "conventional", "Carbon": "conventional",
         "Ferrite": "iron-based", "Cuprate": "cuprate", "Oxide": "other oxide", "Heavy_fermion": "heavy fermion"}
ORDER = ["conventional", "iron-based", "cuprate", "other oxide", "heavy fermion"]
COLOR = {"conventional": "#2a78d6", "iron-based": "#eb6834", "cuprate": "#1baf7a", "other oxide": "#eda100", "heavy fermion": "#8f6bd6"}
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(MP, "SC_MP_V11.pickle"))
    ap.add_argument("--meta", default=os.path.join(MP, "3DSC_MP_v11.csv"))
    ap.add_argument("--bin", type=float, default=2.5)
    ap.add_argument("--out", default=os.path.join(_ROOT, "docs", "figures", "tc_histogram_v11.png"))
    ap.add_argument("--log", action="store_true", help="log count axis (stacked segments then misrepresent proportions)")
    ap.add_argument("--font-scale", type=float, default=1.0, help="multiply every font size (figure grows with it)")
    a = ap.parse_args()
    v = pd.read_pickle(a.index); meta = load_family_metadata(a.meta)
    v = v.merge(meta, left_on="id", right_index=True, how="left")
    v["group"] = v.family.map(GROUP).fillna("conventional")
    sc = v[v.tc > 0]
    edges = np.arange(0, np.ceil(sc.tc.max() / a.bin) * a.bin + a.bin, a.bin)
    counts = {g: np.histogram(sc[sc.group == g].tc, bins=edges)[0] for g in ORDER}
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fs = a.font_scale
    fig, ax = plt.subplots(figsize=(14 * max(1, fs / 2.2), 6 * max(1, fs / 2.2))); fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    bottom = np.zeros(len(edges) - 1); w = a.bin * 0.92
    for g in ORDER:
        ax.bar(edges[:-1] + a.bin / 2, counts[g], width=w, bottom=bottom, color=COLOR[g], edgecolor=SURF, linewidth=0.4,
               label=f"{g} ({int(counts[g].sum()):,})", zorder=3)
        bottom += counts[g]
    ax.set_xlabel(f"$T_c$ (K), {a.bin:g} K bins", color=INK2, fontsize=11 * fs); ax.set_ylabel("superconductors", color=INK2, fontsize=11 * fs)
    ax.set_title(f"Superconductors by $T_c$ and family ({len(sc):,} with $T_c$ > 0)", loc="left", fontsize=11 * fs, color=INK)
    ax.set_xlim(0, edges[-1])
    if a.log:
        ax.set_yscale("log"); ax.set_ylim(0.8, bottom.max() * 1.6)
    else:
        ax.set_ylim(0, bottom.max() * 1.06)
    ax.grid(color=GRID, lw=0.6, which="major", axis="y"); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=10 * fs, length=4 * fs, width=max(0.8, 0.8 * fs / 2))
    ax.legend(frameon=False, fontsize=9 * fs, labelcolor=INK2, loc="upper right", title="family (count)", title_fontsize=9 * fs)
    fig.text(0.008, 0.012, ("log count axis: stacked segments are not proportional. " if a.log else "")
             + f"First bin is (0, {a.bin:g}] K — T_c = 0 rows excluded.", fontsize=8 * fs, color=INK2)
    fig.tight_layout(rect=(0, 0.03 * min(fs, 2), 1, 1)); fig.savefig(a.out, dpi=int(150 / max(1, fs / 2.2)), facecolor=SURF)
    print("wrote", a.out)
    tot = sum(counts[g] for g in ORDER)
    print("bins with the most rows:", ", ".join(f"{edges[i]:.1f}-{edges[i+1]:.1f} K: {int(tot[i])}" for i in np.argsort(-tot)[:6]))


if __name__ == "__main__":
    main()
