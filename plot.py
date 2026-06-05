import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_RESULTS = "CNN/test_result.csv"


def plot_results(results_file=DEFAULT_RESULTS):
    """Scatter-plot CNN test predictions vs. targets.

    ``results_file`` is the headerless (cif_id, target, prediction) CSV written by
    CGCNNMain.py's test pass.
    """
    data = pd.read_csv(results_file, header=None)

    mse = np.sum(np.abs(data[1] - data[2])) / len(data[1])

    plt.scatter(data[1], data[2], label="MAE: " + str(mse))
    plt.plot([np.min(data[1]), np.max(data[1])], [np.min(data[1]), np.max(data[1])], color='black', linestyle='--')
    #plt.xlim([0, 100])
    plt.xlabel("target")
    plt.ylabel("prediction")
    plt.legend()
    plt.show()


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


def plot_epoch_log(epoch_log_file):
    """Plot every per-epoch statistic vs. epoch, with interactive column toggles.

    ``epoch_log_file`` is an ``*_epoch_log.csv`` written by the trainers (a header
    row with an ``epoch`` column plus one column per metric, e.g. train_loss,
    val_mae, lr, epoch_time_sec, is_best). Every numeric column other than
    ``epoch`` is drawn as a line; a checkbox panel on the left toggles each line
    on/off, and the y-axis auto-rescales to whatever is visible.
    """
    from matplotlib.widgets import CheckButtons

    df = pd.read_csv(epoch_log_file)  # header row expected
    x = df["epoch"] if "epoch" in df.columns else pd.Series(range(len(df)), name="epoch")
    cols = [c for c in df.columns
            if c != "epoch" and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        raise ValueError(f"no numeric stat columns to plot in {epoch_log_file} "
                         f"(columns: {list(df.columns)})")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(left=0.30)  # room for the checkbox panel

    lines = {}
    for c in cols:
        (ln,) = ax.plot(x, df[c], label=c)
        lines[c] = ln
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.set_title(f"Epoch log — {epoch_log_file}")
    ax.legend(loc="upper right", fontsize="small")
    _autoscale_to_visible(ax, lines)

    # Interactive column toggles. Labels are colored to match their line.
    rax = fig.add_axes([0.02, 0.20, 0.24, 0.60])
    rax.set_title("show columns", fontsize="small")
    check = CheckButtons(rax, cols, [True] * len(cols))
    for text, c in zip(check.labels, cols):
        text.set_color(lines[c].get_color())

    def _toggle(label):
        ln = lines[label]
        ln.set_visible(not ln.get_visible())
        _autoscale_to_visible(ax, lines)
        fig.canvas.draw_idle()

    check.on_clicked(_toggle)
    # Keep a reference so the widget isn't garbage-collected before show().
    ax._check_widget = check
    plt.show()
    return fig


if __name__ == "__main__":
    plot_results()
