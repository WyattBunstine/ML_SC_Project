"""Per-epoch formation-energy learning curves for the fe ablation ladder.

One line per rung, train and validation in separate panels on a shared log
scale, so the ladder's ordering (composition -> node scalars -> bonds ->
angles -> polyhedra -> all frames) is visible epoch by epoch.

  python scripts/plot_fe_training_curves.py [out.png]
"""
import glob
import os
import sys

import pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
# (run tag, legend label) in ladder order — the fe_parity ablation figure's rungs
RUNGS = [
    ("43_fe_element", "43  composition only (n_conv 0)"),
    ("44_fe_node", "44  + node scalars"),
    ("45_fe_bonds", "45  + bond message passing"),
    ("27_t_energy", "27  + bond angles"),
    ("46_fe_poly", "46  + polyhedral edges"),
    ("50_fe_full", "50  = 46, every MPtrj frame"),
]


def load(tag):
    d = sorted(glob.glob(f"model_data/*/gps_mt_{tag}/gps_mt_{tag}_2*"))
    if not d:
        return None
    f = glob.glob(os.path.join(d[-1], "*epoch_log.csv"))
    return pd.read_csv(f[0]) if f else None


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/figures/fe_training_curves.png"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    fig.patch.set_facecolor(SURF)
    for ax, col, title in ((axes[0], "train_energy_mae", "(a) training"),
                           (axes[1], "val_energy_mae", "(b) validation")):
        ax.set_facecolor(SURF)
        for k, (tag, label) in enumerate(RUNGS):
            df = load(tag)
            if df is None or col not in df:
                print(f"  {tag}: no {col}")
                continue
            y = df[col].dropna()
            ax.plot(df.epoch[y.index], y, color=SERIES[k], lw=2.0,
                    label=label if ax is axes[0] else None, zorder=3)
            if ax is axes[1]:            # direct label at the curve's end, fanned
                # out so the four converged rungs (45/27/46/50, all ~0.04) stay legible
                dy = {0: 0, 1: 0, 2: -11, 3: 0, 4: 11, 5: -22}.get(k, 0)
                ax.annotate(label.split()[0], (df.epoch[y.index].iloc[-1], y.iloc[-1]),
                            xytext=(7, dy), textcoords="offset points", fontsize=8.5,
                            color=SERIES[k], va="center",
                            arrowprops=(dict(arrowstyle="-", color=SERIES[k], lw=0.6,
                                             shrinkA=0, shrinkB=2) if dy else None))
        ax.set_yscale("log")
        ax.set_xlabel("epoch", color=INK2)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.grid(color=GRID, lw=0.6, which="both"); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c9c8c2")
        ax.tick_params(colors=INK2)
        ax.set_xlim(-2, 108 if ax is axes[1] else 102)
    axes[0].set_ylabel("formation-energy MAE (eV/atom)", color=INK2)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")
    fig.suptitle("Formation-energy learning curves across the ablation ladder",
                 fontsize=12, color=INK, x=0.008, ha="left")
    fig.text(0.008, 0.015,
             "Rungs 43-46 and 27 subsample MPtrj 1-in-3; rung 50 uses every frame, so one of its "
             "epochs is 2.6x the gradient steps.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.035, 1, 0.945))
    fig.savefig(out, dpi=160, facecolor=SURF)
    print("wrote", out)
    for tag, label in RUNGS:
        df = load(tag)
        if df is not None:
            print(f"  {label:38s} epochs {len(df):3d}  final train {df.train_energy_mae.iloc[-1]:.4f}  "
                  f"best val {df.val_energy_mae.min():.4f} @ep{int(df.val_energy_mae.idxmin())}")


if __name__ == "__main__":
    main()
