"""Predicted-vs-experimental T_c parity panels for one encoder's transfer runs.

    python scripts/plot_parity.py <probe_tag> <dome_tag> <fam_tag> <nick_tag> <out.png> [title]
e.g. python scripts/plot_parity.py 51phdos 51phdos st_51p_l 51p docs/figures/parity_51.png "rung 51"

Panels: (a) broad probe test set (msle protocol) by family, (b) La-series dome
holdout, (c) unseen cuprate families (structure holdout: Bi / Hg / YBa-123 /
T'), (d) nickelate zero-shot holdout by subset. Same axes per panel, 1:1 line,
thin marks with a surface ring, legend for >=2 series, sparse direct labels.
Categorical hues are the validated reference palette in fixed order; "Other"
is neutral. Static PNG (paper figure convention of docs/figures/).
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from nickelate_holdout import subset as nick_subset  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER, SURFACE, INK, INK2, GRID = "#b8b7b2", "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def latest(name):
    return sorted(glob.glob(f"model_data/*/gps_tc_{name}_2*"))[-1]


def stats(t, p):
    t, p = np.asarray(t, float), np.asarray(p, float)
    pos = t > 0
    r = np.corrcoef(t[pos], p[pos])[0, 1] if pos.sum() > 2 else float("nan")
    return f"n={len(t)}  MAE {np.abs(t - p).mean():.1f} K  |  T$_c$>0: n={int(pos.sum())}, MAE {np.abs(t[pos] - p[pos]).mean():.1f} K, r {r:.2f}"


def panel(ax, groups, lim, title, note, labels=None):
    ax.set_facecolor(SURFACE)
    ax.plot([0, lim], [0, lim], color="#8d8c87", lw=1, ls="--", zorder=1)
    for name, t, p, color, z in groups:
        ax.scatter(t, p, s=26, c=color, edgecolors=SURFACE, linewidths=0.8, alpha=0.9, zorder=z, label=f"{name} ({len(t)})")
    for txt, x, y in (labels or []):
        ax.annotate(txt, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=7.5, color=INK2,
                    arrowprops=dict(arrowstyle="-", color="#8d8c87", lw=0.6))
    ax.set_xlim(-2, lim); ax.set_ylim(-2, lim)
    ax.set_xlabel("experimental $T_c$ (K)", color=INK2); ax.set_ylabel("predicted $T_c$ (K)", color=INK2)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.text(0.02, 0.97, note, transform=ax.transAxes, fontsize=8, color=INK2, va="top")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=INK2, labelsize=8.5)
    if len(groups) > 1:   # upper-left, under the stats line: the one region every panel leaves empty
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.93), labelcolor=INK2, markerscale=1.2)


def main():
    probe, dome, fam, nick, out = sys.argv[1:6]
    title = sys.argv[6] if len(sys.argv) > 6 else probe
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11.5))
    fig.patch.set_facecolor(SURFACE)
    # (a) broad probe by family, fixed hue order; Other neutral and underneath
    p = pd.read_csv(latest(f"probe_{probe}") + "/predictions.csv")
    fams = ["Cuprate", "Ferrite", "Heavy_fermion", "Oxide", "Chevrel", "Carbon"]
    groups = [("Other", p[p.family == "Other"].tc_true_K, p[p.family == "Other"].tc_head_K, OTHER, 2)]
    groups += [(f.replace("_", " "), p[p.family == f].tc_true_K, p[p.family == f].tc_head_K, SERIES[i], 3)
               for i, f in enumerate(fams) if (p.family == f).any()]
    lim = max(p.tc_true_K.max(), p.tc_head_K.max()) * 1.05
    panel(axes[0, 0], groups, lim, "(a) broad probe test set (chemsys split)", stats(p.tc_true_K, p.tc_head_K))
    # (b) La-series dome holdout vs the rest of that run's test set
    R = latest(f"la_series_{dome}"); cfg = json.load(open(R + "/config.json"))
    hold = set(pd.read_csv(cfg["holdout_ids_csv"])["id"].astype(str))
    d = pd.read_csv(R + "/predictions.csv"); h = d[d.id.isin(hold)]; rest = d[~d.id.isin(hold)]
    panel(axes[0, 1], [("other test rows", rest.tc_true_K, rest.tc_head_K, OTHER, 2),
                       ("La$_{2-x}$(Sr,Ba,Ce)$_x$CuO$_4$ holdout", h.tc_true_K, h.tc_head_K, SERIES[0], 3)],
          lim, "(b) La-series doping holdout (parent_comp split)", stats(h.tc_true_K, h.tc_head_K))
    # (c) unseen families, structure holdout
    groups, labels = [], []
    for i, (f, nm) in enumerate((("bi", "Bi"), ("hg", "Hg"), ("yba", "YBa-123"), ("tprime", "T$'$ (electron-doped)"))):
        R = latest(f"fam_{f}_{fam}"); cfg = json.load(open(R + "/config.json"))
        hold = set(pd.read_csv(cfg["holdout_ids_csv"])["id"].astype(str)); d = pd.read_csv(R + "/predictions.csv"); d = d[d.id.isin(hold)]
        groups.append((nm, d.tc_true_K, d.tc_head_K, SERIES[i], 3))
    allt = np.concatenate([g[1] for g in groups]); allp = np.concatenate([g[2] for g in groups])
    panel(axes[1, 0], groups, max(allt.max(), allp.max()) * 1.05, "(c) unseen cuprate families (structure holdout)", stats(allt, allp))
    # (d) nickelates by subset
    R = latest(f"nickelate_{nick}"); cfg = json.load(open(R + "/config.json"))
    hold = set(pd.read_csv(cfg["holdout_ids_csv"])["id"].astype(str)); d = pd.read_csv(R + "/predictions.csv"); d = d[d.id.isin(hold)].copy()
    d["sub"] = d.id.map(nick_subset)
    groups = [(nm, d[d["sub"] == k].tc_true_K, d[d["sub"] == k].tc_head_K, SERIES[i], 3)
              for i, (k, nm) in enumerate((("d9", "d9 infinite-layer / A$_6$Ni$_5$O$_{12}$"), ("oxide-other", "RP / 214 / other oxides"), ("pnictide", "Ni pnictide-oxides")))]
    for f_ in ("La3Ni2O7", "LaNiO2-", "NdNiO2-", "Nd6Ni5O12"):
        m = d[d.id.str.startswith(f_)]
        if len(m):
            labels.append((f_.rstrip("-"), float(m.tc_true_K.iloc[0]), float(m.tc_head_K.iloc[0])))
    panel(axes[1, 1], groups, max(d.tc_true_K.max(), d.tc_head_K.max()) * 1.08, "(d) nickelate zero-shot holdout", stats(d.tc_true_K, d.tc_head_K), labels)
    fig.suptitle(f"predicted vs experimental $T_c$ — {title}", fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
