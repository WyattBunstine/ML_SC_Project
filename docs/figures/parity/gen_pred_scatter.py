#!/usr/bin/env python3
"""Publication parity plot: predicted vs reported T_c on the held-out test set.

Reads a T_c-head run's predictions.csv (test rows only; tc_head_K is the
seed-ensemble mean) and writes tc_pred_vs_reported_<tag>.pdf/.png here.

Default run: the rung-20 dome champion fine-tune
(gps_tc_v4_la_series_20cfbvsfnd, 2026-08-13: parent-grouped test split,
n=1223, MAE 5.32 K).  Override with --run <dir> --tag <name>.

Design notes (dataviz-skill procedure):
- log(1+T_c) axes: the target spans 0-127 K with median 0.76 K and 536
  exact zeros; linear axes pile everything into the origin corner.
- categorical colour = SC family, fixed order, CVD-validated (Machado
  protan/deutan dE in OKLab): worst adjacent pair passes; the one 6-8
  all-pairs band (Oxide-Fe-based) is covered by the marker-shape
  secondary encoding.  Chevrel (15) and Carbon (5) fold into Other.
- marker shapes double-encode family; white 0.35pt edges separate
  overlapping marks; cuprates drawn last (the family the paper is about).
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_RUN = ("../../model_data/2026-08-13/"
               "gps_tc_v4_la_series_20cfbvsfnd_2026-08-13_07-49-44")

# family -> (display name, colour, marker, z-order); fixed legend order
FAMILIES = {
    "Other":         ("Other",         "#4C72B0", "o", 1),
    "Oxide":         ("Oxide",         "#55A868", "s", 2),
    "Cuprate":       ("Cuprate",       "#C03A3E", "^", 5),
    "Ferrite":       ("Fe-based",      "#E69F00", "D", 3),
    "Heavy_fermion": ("Heavy fermion", "#CC79A7", "v", 4),
}
FOLD_INTO_OTHER = {"Chevrel", "Carbon"}

INK, GRID = "#2B2B2B", "#DDDDDD"
TICKS_K = [0, 1, 2, 5, 10, 20, 50, 100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN, help="head-run dir with predictions.csv")
    ap.add_argument("--tag", default="r20", help="output filename tag")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    run = args.run if os.path.isabs(args.run) else os.path.join(here, args.run)
    df = pd.read_csv(os.path.join(run, "predictions.csv"))
    df["family"] = df["family"].where(~df["family"].isin(FOLD_INTO_OTHER), "Other")

    t, p = df["tc_true_K"].to_numpy(), df["tc_head_K"].clip(lower=0).to_numpy()
    pos = t > 0
    mae = np.abs(t - p).mean()
    mae_pos = np.abs(t - p)[pos].mean()
    r = np.corrcoef(t, p)[0, 1]

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8.0,
        "axes.labelsize": 9.0, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    })
    fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=300)

    lim = (-0.12, np.log1p(145))
    ax.plot(lim, lim, ls="--", lw=0.8, color="#999999", zorder=0)

    for fam, (label, color, marker, z) in FAMILIES.items():
        sub = df[df["family"] == fam]
        ax.scatter(np.log1p(sub["tc_true_K"]), np.log1p(sub["tc_head_K"].clip(lower=0)),
                   s=13, marker=marker, facecolor=color, edgecolor="white",
                   linewidth=0.35, alpha=0.85, zorder=z,
                   label=f"{label} ({len(sub)})")

    tickpos = np.log1p(TICKS_K)
    for a in (ax.set_xticks, ax.set_yticks):
        a(tickpos)
    ax.set_xticklabels(TICKS_K)
    ax.set_yticklabels(TICKS_K)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Reported $T_c$ (K)")
    ax.set_ylabel("Predicted $T_c$ (K)")
    ax.grid(True, color=GRID, linewidth=0.5, zorder=-5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(colors=INK)

    ax.text(0.03, 0.97,
            f"$n$ = {len(df)} (held-out test)\n"
            f"MAE = {mae:.1f} K\n"
            f"MAE ($T_c$>0) = {mae_pos:.1f} K\n"
            f"$r$ = {r:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
            color=INK, linespacing=1.45)

    leg = ax.legend(loc="lower right", fontsize=7.0, frameon=True,
                    borderaxespad=0.4, handletextpad=0.15, labelspacing=0.35,
                    framealpha=0.88, edgecolor="none", facecolor="white")
    for h in leg.legend_handles:
        h.set_alpha(1.0)

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        out = os.path.join(here, f"tc_pred_vs_reported_{args.tag}.{ext}")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
        print("wrote", out)


if __name__ == "__main__":
    main()
