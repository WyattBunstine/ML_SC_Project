"""V8 (expansion) vs V9 (expansion + Cu-oxide negatives) parity, points only.

    python scripts/plot_v8v9_parity.py <out.png>

One panel per protocol; colour is the training index, fill is nothing else -
both series are open marks so neither hides the other. Holdout panels show only
the held-out rows.
"""
import glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
SERIES = ["#2a78d6", "#eb6834"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def rows(tag, holdout_only):
    R = sorted(glob.glob(f"model_data/*/gps_tc_{tag}_2*"))[-1]
    p = pd.read_csv(os.path.join(R, "predictions.csv"))
    if holdout_only:
        cfg = json.load(open(os.path.join(R, "config.json")))
        h = set(pd.read_csv(cfg["holdout_ids_csv"])["id"].astype(str))
        p = p[p.id.astype(str).isin(h)]
    return p


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/figures/parity/v8v9_parity.png"
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    panels = [("probe", False, "(a) broad probe test set"),
              ("ladome", True, "(b) La-series dome holdout"),
              ("nickelate", True, "(c) nickelate zero-shot holdout")]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    fig.patch.set_facecolor(SURF)
    for ax, (arm, ho, title) in zip(axes, panels):
        ax.set_facecolor(SURF)
        lim = 0
        for k, v in enumerate(("v8", "v9")):
            p = rows(f"{v}_{arm}_51", ho)
            lim = max(lim, p.tc_true_K.max(), p.tc_head_K.max())
            pos = p.tc_true_K > 0
            r = np.corrcoef(p.tc_true_K[pos], p.tc_head_K[pos])[0, 1] if pos.sum() > 2 else float("nan")
            mae = float((p.tc_true_K - p.tc_head_K).abs().mean())
            ax.scatter(p.tc_true_K, p.tc_head_K, s=26, facecolors="none", edgecolors=SERIES[k],
                       linewidths=0.9, alpha=0.75, zorder=3,
                       label=f"{v.upper()}  n={len(p)}  MAE {mae:.2f} K  r {r:.2f}")
        lim *= 1.05
        ax.plot([0, lim], [0, lim], color="#8d8c87", lw=1, ls="--", zorder=1)
        ax.set_xlim(-lim * 0.02, lim); ax.set_ylim(-lim * 0.02, lim)
        ax.set_xlabel("experimental $T_c$ (K)", color=INK2)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    axes[0].set_ylabel("predicted $T_c$ (K)", color=INK2)
    fig.suptitle("V8 (expansion) vs V9 (+ curated Cu-oxide $T_c$=0 negatives), rung 51",
                 fontsize=12, color=INK, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=160, facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    main()
