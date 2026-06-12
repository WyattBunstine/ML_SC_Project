import colorsys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_RESULTS = "CNN/test_result.csv"

# Per-run line styles (cycled if there are more runs than styles). Combined with
# the per-(metric,run) color below, this gives runs a second visual cue beyond
# shade, so overlapping curves stay distinguishable.
_LINESTYLES = ["-", "--", "-.", ":"]


def plot_results(results_file=DEFAULT_RESULTS):
    """Scatter-plot CNN test predictions vs. targets.

    Accepts either format, sniffed from the first line:
    - the headerless (cif_id, target, prediction) CSV written by CGCNNMain.py's
      test pass;
    - a T_c-head run's ``predictions.csv`` (header with ``tc_true_K`` plus one
      ``tc_<series>_K`` column per prediction series) -> toggleable multi-series
      scatter, see :func:`plot_head_predictions`.
    """
    with open(results_file) as f:
        header = f.readline()
    if "tc_true_K" in header:
        return plot_head_predictions(results_file)

    data = pd.read_csv(results_file, header=None)

    mse = np.sum(np.abs(data[1] - data[2])) / len(data[1])

    plt.scatter(data[1], data[2], label="MAE: " + str(mse))
    plt.plot([np.min(data[1]), np.max(data[1])], [np.min(data[1]), np.max(data[1])], color='black', linestyle='--')
    #plt.xlim([0, 100])
    plt.xlabel("target")
    plt.ylabel("prediction")
    plt.legend()
    plt.show()


# Family display order: conventional-leaning first, then oxide, then the
# unconventional families — so the legend reads along the mechanism axis.
_FAMILY_ORDER = ["Other", "Chevrel", "Carbon", "Oxide",
                 "Cuprate", "Ferrite", "Heavy_fermion"]
_SERIES_MARKERS = ["o", "^", "s", "D", "v", "P"]


def plot_head_predictions(predictions_file):
    """Pred-vs-true scatter for a T_c-head run's ``predictions.csv``.

    Encoding: **color = superconductor family, marker = prediction series**
    (probe, head, and any future ``tc_<series>_K`` column — discovered from the
    header, nothing hardcoded). Check boxes toggle each series and each family
    independently (a point shows when both its series and its family are on),
    plus a linear/log axis switch — symlog, so the T_c = 0 rows stay visible.
    Per-series test MAE is shown in the legend; the dashed line is y = x.
    """
    from matplotlib.lines import Line2D
    from matplotlib.widgets import CheckButtons

    df = pd.read_csv(predictions_file)
    target_all = df["tc_true_K"].to_numpy()
    pred_cols = [c for c in df.columns
                 if c.startswith("tc_") and c.endswith("_K") and c != "tc_true_K"]
    if not pred_cols:
        raise ValueError(f"no tc_<series>_K prediction columns in {predictions_file}")

    fam_col = df["family"] if "family" in df.columns else pd.Series(["all"] * len(df))
    families = ([f for f in _FAMILY_ORDER if f in set(fam_col)]
                + sorted(set(fam_col) - set(_FAMILY_ORDER)))
    colors = plt.get_cmap("tab10").colors
    fam_color = {f: colors[i % len(colors)] for i, f in enumerate(families)}

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    fig.subplots_adjust(left=0.30, right=0.78)

    # One scatter per (series, family): visibility = series on AND family on.
    names, series_mae, points = [], {}, {}
    hi = float(max(target_all.max(), df[pred_cols].to_numpy().max(), 1.0)) * 1.05
    for si, col in enumerate(pred_cols):
        name = col[len("tc_"):-len("_K")]
        names.append(name)
        series_mae[name] = float(np.abs(df[col] - target_all).mean())
        marker = _SERIES_MARKERS[si % len(_SERIES_MARKERS)]
        for fam in families:
            m = (fam_col == fam).to_numpy()
            if not m.any():
                continue
            points[(name, fam)] = ax.scatter(
                target_all[m], df[col].to_numpy()[m], s=18, alpha=0.6,
                marker=marker, color=fam_color[fam], linewidths=0)

    ax.plot([0, hi], [0, hi], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("experimental T_c (K)")
    ax.set_ylabel("predicted T_c (K)")
    run_dir = os.path.basename(os.path.dirname(os.path.abspath(predictions_file)))
    ax.set_title(f"T_c head test predictions — {run_dir}", fontsize="medium")

    # Two-part legend outside the axes: series (marker, with MAE) + families (color).
    handles = [Line2D([], [], linestyle="none", marker=_SERIES_MARKERS[si], color="0.35",
                      label=f"{n} — MAE {series_mae[n]:.2f} K")
               for si, n in enumerate(names)]
    handles += [Line2D([], [], linestyle="none", marker="o", color=fam_color[f],
                       label=f"{f} (n={int((fam_col == f).sum())})")
                for f in families]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize="small", frameon=True)

    # Toggle stacks on the left: series, families, axis scale.
    series_ax = fig.add_axes([0.03, 0.74, 0.20, 0.04 + 0.045 * len(names)])
    series_ax.set_title("series", fontsize="small")
    series_checks = CheckButtons(series_ax, names, [True] * len(names))

    fam_ax = fig.add_axes([0.03, 0.30, 0.20, 0.04 + 0.045 * len(families)])
    fam_ax.set_title("family", fontsize="small")
    fam_checks = CheckButtons(fam_ax, families, [True] * len(families))

    scale_ax = fig.add_axes([0.03, 0.16, 0.20, 0.08])
    scale = CheckButtons(scale_ax, ["log axes"], [False])

    def refresh(_label=None):
        s_on = dict(zip(names, series_checks.get_status()))
        f_on = dict(zip(families, fam_checks.get_status()))
        for (name, fam), sc in points.items():
            sc.set_visible(s_on[name] and f_on[fam])
        fig.canvas.draw_idle()

    def on_scale(_label):
        kind = "symlog" if scale.get_status()[0] else "linear"
        extra = {"linthresh": 1.0} if kind == "symlog" else {}
        ax.set_xscale(kind, **extra)
        ax.set_yscale(kind, **extra)
        fig.canvas.draw_idle()

    series_checks.on_clicked(refresh)
    fam_checks.on_clicked(refresh)
    scale.on_clicked(on_scale)
    ax._head_widgets = (series_checks, fam_checks, scale)  # keep refs until show()

    plt.show()
    return fig


def _autoscale_to_visible(ax, lines):
    """Rescale the y-axis to the currently-visible lines only.

    Epoch-log columns span wildly different scales (loss ~1e-2, lr ~1e-3,
    epoch_time_sec ~1e2, is_best in {0,1}). Without this, one large-scale column
    flattens the rest; hiding it via the checkboxes then rescales so the
    remaining curves fill the axes.
    """
    ys = [np.asarray(ln.get_ydata(), dtype=float)
          for ln in lines.values() if ln.get_visible()]
    vals = np.concatenate(ys) if ys else np.array([])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    lo, hi = float(vals.min()), float(vals.max())
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    pad = 0.05 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)


def _run_labels(paths):
    """Short, unique display name per epoch-log path.

    Strips the ``_epoch_log.csv`` / ``.csv`` suffix; if two files collapse to the
    same name, disambiguates by prefixing the parent directory (e.g. two
    ``mpnn_result_epoch_log.csv`` in different run dirs).
    """
    def base(p):
        b = os.path.basename(p)
        for suf in ("_epoch_log.csv", ".csv"):
            if b.endswith(suf):
                return b[: -len(suf)]
        return b

    names = [base(p) for p in paths]
    counts = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    out = []
    for p, n in zip(paths, names):
        if counts[n] > 1:
            parent = os.path.basename(os.path.dirname(os.path.abspath(p)))
            out.append(f"{parent}/{n}")
        else:
            out.append(n)
    return out


def _line_color(metric_idx, n_metrics, run_idx, n_runs):
    """Color for one (metric, run) curve.

    Hue encodes the metric (so e.g. val_loss is the same hue across runs), while
    saturation/value shift per run — giving each cell its own exact color while
    keeping metrics visually grouped by hue. The toggle table tints each cell with
    this color and the legend groups by run, so the reader can map any line back to
    both its metric (hue) and its run (shade + linestyle).
    """
    hue = (metric_idx / max(1, n_metrics)) % 1.0
    if n_runs <= 1:
        sat, val = 0.65, 0.85
    else:
        f = run_idx / (n_runs - 1)        # 0 (first run) .. 1 (last run)
        val = 0.95 - 0.45 * f             # later runs darker
        sat = 0.55 + 0.30 * f             # later runs more saturated
    return colorsys.hsv_to_rgb(hue, sat, val)


def _grouped_legend(ax, lines, labels, metrics):
    """(Re)build the legend with only currently-visible lines, grouped under a bold
    per-run header. A run whose lines are all hidden drops out entirely (header
    included); if nothing is visible, the legend is removed. Call again after any
    toggle to keep it in sync.
    """
    from matplotlib.lines import Line2D

    handles, leg_labels, is_header = [], [], []
    for rj, run_name in enumerate(labels):
        visible = [(mi, lines[(rj, mi)]) for mi in range(len(metrics))
                   if (rj, mi) in lines and lines[(rj, mi)].get_visible()]
        if not visible:
            continue
        handles.append(Line2D([], [], linestyle="none", marker="", color="none"))
        leg_labels.append(f"▸ {run_name}")
        is_header.append(True)
        for mi, ln in visible:
            handles.append(ln)
            leg_labels.append(f"    {metrics[mi]}")
            is_header.append(False)

    existing = ax.get_legend()
    if not handles:                       # nothing visible -> no legend
        if existing is not None:
            existing.remove()
        return None

    leg = ax.legend(handles, leg_labels, loc="upper left",
                    bbox_to_anchor=(1.01, 1.0), fontsize="small",
                    frameon=True, handlelength=2.2, borderaxespad=0.0)
    for txt, hdr in zip(leg.get_texts(), is_header):
        if hdr:
            txt.set_fontweight("bold")
    return leg


def _build_toggle_table(fig, ax, lines, labels, metrics, rect):
    """Clickable table of toggles: rows = metrics, columns = runs.

    Each cell is filled with its line's color when the line is visible (white when
    hidden), and edged with that color always; clicking a cell toggles its line.
    Cells for a metric a given run doesn't have are hatched/greyed and inert.
    ``rect`` is the [left, bottom, width, height] figure-fraction box for the table.
    """
    from matplotlib.patches import Rectangle

    n_runs, n_metrics = len(labels), len(metrics)
    tax = fig.add_axes(rect)
    tax.set_xlim(0, n_runs)
    tax.set_ylim(0, n_metrics)

    cells = {}
    for mi in range(n_metrics):
        y = n_metrics - 1 - mi          # metric 0 at the top
        for rj in range(n_runs):
            ln = lines.get((rj, mi))
            rect = Rectangle((rj + 0.06, y + 0.12), 0.88, 0.76)
            if ln is None:
                rect.set_facecolor((0.92, 0.92, 0.92))
                rect.set_edgecolor((0.8, 0.8, 0.8))
                rect.set_hatch("xx")
            else:
                col = ln.get_color()
                rect.set_edgecolor(col)
                rect.set_linewidth(1.6)
                rect.set_facecolor(col if ln.get_visible() else "white")
            tax.add_patch(rect)
            cells[(rj, mi)] = rect

    tax.set_xticks([rj + 0.5 for rj in range(n_runs)])
    tax.set_xticklabels(labels, fontsize="small", rotation=30, ha="right")
    tax.set_yticks([n_metrics - 1 - mi + 0.5 for mi in range(n_metrics)])
    tax.set_yticklabels(metrics, fontsize="small")
    tax.tick_params(length=0)
    tax.set_title("toggle  (row = metric, col = run)", fontsize="small")
    for spine in tax.spines.values():
        spine.set_visible(False)

    def on_click(event):
        if event.inaxes is not tax or event.xdata is None or event.ydata is None:
            return
        rj, row = int(event.xdata), int(event.ydata)
        if not (0 <= rj < n_runs and 0 <= row < n_metrics):
            return
        mi = n_metrics - 1 - row
        ln = lines.get((rj, mi))
        if ln is None:
            return
        vis = not ln.get_visible()
        ln.set_visible(vis)
        cells[(rj, mi)].set_facecolor(ln.get_color() if vis else "white")
        _autoscale_to_visible(ax, lines)
        _grouped_legend(ax, lines, labels, metrics)   # drop hidden lines from legend
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    # Keep references so the axes/handler aren't garbage-collected before show().
    ax._toggle_table = (tax, cells)


def plot_epoch_logs(epoch_log_files):
    """Overlay per-epoch stats from one or more runs, with a toggle table.

    Each file is an ``*_epoch_log.csv`` (header row with an ``epoch`` column plus
    one numeric column per metric — train_loss, val_mae, lr, ...). Every metric of
    every run is drawn as a line on shared axes; a clickable table on the left
    (rows = metrics, columns = runs) toggles individual lines, and the legend is
    grouped by run. Metrics are colored by hue and runs by shade + linestyle.

    Accepts a single path or a list of paths.
    """
    if isinstance(epoch_log_files, str):
        epoch_log_files = [epoch_log_files]
    if not epoch_log_files:
        raise ValueError("plot_epoch_logs: no epoch-log files given")

    runs = []
    for path in epoch_log_files:
        df = pd.read_csv(path)  # header row expected
        x = df["epoch"] if "epoch" in df.columns else pd.Series(range(len(df)), name="epoch")
        cols = [c for c in df.columns
                if c != "epoch" and pd.api.types.is_numeric_dtype(df[c])]
        runs.append((df, x, cols))
    labels = _run_labels(epoch_log_files)

    # Union of metric columns across runs, preserving first-seen order. Different
    # trainers expose slightly different stats (e.g. MPNN adds val_bal_mae), so a
    # run simply has no line/cell for a metric it didn't log.
    metrics = []
    for _, _, cols in runs:
        for c in cols:
            if c not in metrics:
                metrics.append(c)
    if not metrics:
        raise ValueError("no numeric stat columns to plot in any of: "
                         + ", ".join(epoch_log_files))

    n_runs, n_metrics = len(runs), len(metrics)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    # Lay out the toggle table on the left (its width grows with the number of run
    # columns) with room to its left for the metric row labels, and leave the right
    # margin for the run-grouped legend.
    tax_left = 0.13                                   # space for metric row labels
    tax_w = min(0.05 + 0.06 * n_runs, 0.42)           # ~one column-width per run
    fig.subplots_adjust(left=tax_left + tax_w + 0.07, right=0.80)

    lines = {}
    for rj, (df, x, cols) in enumerate(runs):
        ls = _LINESTYLES[rj % len(_LINESTYLES)]
        for mi, metric in enumerate(metrics):
            if metric not in cols:
                continue
            color = _line_color(mi, n_metrics, rj, n_runs)
            (ln,) = ax.plot(x, df[metric], color=color, linestyle=ls, label=metric)
            lines[(rj, mi)] = ln

    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.set_title(f"Epoch logs — {n_runs} run(s)")
    _autoscale_to_visible(ax, lines)
    _grouped_legend(ax, lines, labels, metrics)
    _build_toggle_table(fig, ax, lines, labels, metrics,
                        rect=[tax_left, 0.10, tax_w, 0.78])

    plt.show()
    return fig


def plot_epoch_log(epoch_log_file):
    """Backward-compatible single-file wrapper for :func:`plot_epoch_logs`."""
    return plot_epoch_logs([epoch_log_file])


if __name__ == "__main__":
    plot_results()
