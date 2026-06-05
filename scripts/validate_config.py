#!/usr/bin/env python3
"""Validate a training config before it is submitted (especially for remote SLURM
runs, where a typo otherwise costs a queue wait + a wasted allocation).

Detects the config type and checks, for whichever it is:
  * required keys present; enum values valid; numeric ranges sane
  * referenced files exist (the dataset/index pickle, atom_init, sampled graphs)
  * the regression target is resolvable, and that log1p is not applied to a target
    with negative values (e.g. formation energies) which would produce NaNs

  - "MPNN"     config: has `index_path` -> CNN/MPNN/MPNNMain.py + CIFDataV4
  - "Original" config: has `dataset`   -> CNN/CGCNNMain.py + CIFData

Exit code 0 = OK (warnings allowed), 1 = one or more errors (printed). Paths in a
config are resolved relative to the current working directory (the project root).

Usage:
    validate_config.py CONFIG.json [--cpus N]
"""
import argparse
import json
import math
import os
import sys

KNOWN_TARGETS = ("tc", "e_above_hull", "formation_energy_per_atom")

# Valid enum values, mirrored from CNN/MPNN/MPNNMain.py + MPNNModel.py and
# CNN/CGCNNMain.py. Kept here so a bad value is caught before a job is submitted.
MPNN_ENUMS = {
    "task": {"regression", "classification"},
    "target_transform": {"none", "log1p"},
    "optim": {"SGD", "Adam", "AdamW"},
    "edge_aggregation": {"ecn_weighted", "attention"},
    "atom_pooling": {"mean", "mean_max", "attention", "set2set"},
    "selection_metric": {"auc", "fbeta", "f1", "recall", "precision", "accuracy"},
}
ORIG_ENUMS = {
    "task": {"regression", "classification"},
    "optim": {"SGD", "Adam"},          # CGCNNMain raises on anything else
}


def _check_common_numeric(cfg, err):
    """Range checks shared by both config types."""
    def num(key, lo=None, hi=None, integer=False, allow_eq_hi=False):
        if key not in cfg:
            return
        v = cfg[key]
        if integer and not (isinstance(v, int) and not isinstance(v, bool)):
            err.append(f"{key}={v!r} must be an integer")
            return
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            err.append(f"{key}={v!r} must be numeric")
            return
        if lo is not None and v < lo:
            err.append(f"{key}={v} must be >= {lo}")
        if hi is not None and (v > hi or (not allow_eq_hi and v == hi)):
            rel = "<=" if allow_eq_hi else "<"
            err.append(f"{key}={v} must be {rel} {hi}")

    num("epochs", lo=1, integer=True)
    num("batch_size", lo=1, integer=True)
    num("learning_rate", lo=0)            # > 0 enforced below
    if cfg.get("learning_rate", 1) == 0:
        err.append("learning_rate must be > 0")
    num("val_ratio", lo=0, hi=1)
    num("test_ratio", lo=0, hi=1)
    num("num_workers", lo=0, integer=True)
    num("weight_decay", lo=0)
    num("graph_cache_size", lo=0, integer=True)
    num("dropout", lo=0, hi=1)            # p in [0,1); p=1 would zero everything

    vr, tr = cfg.get("val_ratio"), cfg.get("test_ratio")
    if isinstance(vr, (int, float)) and isinstance(tr, (int, float)) and vr + tr >= 1:
        err.append(f"val_ratio + test_ratio = {vr + tr} must be < 1 (no training data left)")


def _check_enums(cfg, enums, err):
    for key, valid in enums.items():
        if key in cfg and cfg[key] not in valid:
            err.append(f"{key}={cfg[key]!r} is not one of {sorted(valid)}")


def _require(cfg, keys, err):
    for k in keys:
        if k not in cfg:
            err.append(f"missing required key: '{k}'")


def _parse_ratio(v):
    """Mirror MPNNMain._parse_ratio: None/inf/'none'/<=0 -> inf (no non-SC)."""
    if v is None:
        return math.inf
    if isinstance(v, str):
        if v.strip().lower() in ("inf", "infinity", "none", "null", ""):
            return math.inf
        try:
            v = float(v)
        except ValueError:
            return math.inf
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.inf
    return f if f > 0 else math.inf


def _resolve_target(df, cfg, structural_cols, err):
    """Resolve the regression target column the way CIFDataV4/CIFData do.
    Returns the column name, or None (and appends an error)."""
    cols = list(df.columns)
    named = [c for c in cols if c not in structural_cols and not df[c].isna().all()]
    tcol = cfg.get("target_column")
    if tcol is not None:
        if tcol not in cols:
            err.append(f"target_column '{tcol}' not in dataset columns {cols}")
            return None
        return tcol
    for c in ("value", "tc"):
        if c in cols and not df[c].isna().all():
            return c
    err.append("no usable default target ('value'/'tc' absent or empty); "
               f"set 'target_column' to one of {named or KNOWN_TARGETS}")
    return None


def _check_target_and_log1p(df, cfg, target_col, err, *, is_mpnn):
    """log1p must not be applied to a target with negative values."""
    if target_col is None:
        return
    series = df[target_col].dropna()
    if series.empty:
        err.append(f"target column '{target_col}' has no non-null values")
        return
    if is_mpnn and cfg.get("target_transform", "none") == "log1p":
        mn = float(series.min())
        if mn < 0:
            err.append(
                f"target_transform 'log1p' cannot be used with target '{target_col}': "
                f"its minimum value is {mn:.4g} (log1p needs values >= 0; e.g. formation "
                f"energies are negative). Use target_transform 'none'.")


def _sample_files_exist(paths, err, what):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        shown = ", ".join(missing[:3])
        err.append(f"{len(missing)}/{len(paths)} sampled {what} not found (e.g. {shown})")


def validate_mpnn(cfg, err, warn, cpus):
    _require(cfg, ["index_path", "out_file", "epochs", "batch_size",
                   "learning_rate", "val_ratio", "test_ratio"], err)
    _check_enums(cfg, MPNN_ENUMS, err)
    _check_common_numeric(cfg, err)

    index_path = cfg.get("index_path")
    if not index_path:
        return
    if not os.path.exists(index_path):
        err.append(f"index_path not found: {index_path}")
        return

    import pandas as pd
    try:
        df = pd.read_pickle(index_path)
    except Exception as e:  # noqa: BLE001
        err.append(f"could not read index_path {index_path}: {e}")
        return
    if not hasattr(df, "columns"):
        err.append(f"index_path {index_path} is not a DataFrame (got {type(df).__name__})")
        return

    target_col = _resolve_target(df, cfg, {"id", "value", "graph_path", "label"}, err)
    _check_target_and_log1p(df, cfg, target_col, err, is_mpnn=True)

    # Classification, or any finite SC:non-SC ratio, needs non-SC (label==0) rows.
    ratio = _parse_ratio(cfg.get("SC_to_non_SC_ratio"))
    needs_nonsc = cfg.get("task") == "classification" or math.isfinite(ratio)
    if needs_nonsc:
        if "label" not in df.columns:
            err.append("task/ratio needs non-SC rows but the index has no 'label' column")
        elif not (df["label"] == 0).any():
            err.append("task is classification (or SC_to_non_SC_ratio is finite) but the "
                       "index has no non-SC rows (label==0); build with --nonsc-source")

    # Spot-check that referenced graph files exist (they get synced to the cluster).
    if "graph_path" in df.columns and len(df):
        sample = df["graph_path"].dropna().head(5).tolist()
        _sample_files_exist(sample, err, "graph files")

    _check_workers_vs_cpus(cfg, cpus, warn)


def validate_orig(cfg, err, warn, cpus):
    _require(cfg, ["dataset_rd", "dataset", "atom_init", "out_file", "epochs",
                   "batch_size", "learning_rate", "val_ratio", "test_ratio"], err)
    _check_enums(cfg, ORIG_ENUMS, err)
    _check_common_numeric(cfg, err)

    root = cfg.get("dataset_rd", "")
    pickle_path = os.path.join(root, cfg.get("dataset", ""))
    atom_init_path = os.path.join(root, cfg.get("atom_init", ""))
    if cfg.get("dataset") and not os.path.exists(pickle_path):
        err.append(f"dataset not found: {pickle_path}")
    if cfg.get("atom_init") and not os.path.exists(atom_init_path):
        err.append(f"atom_init not found: {atom_init_path}")

    if cfg.get("dataset") and os.path.exists(pickle_path):
        import pandas as pd
        try:
            df = pd.read_pickle(pickle_path)
        except Exception as e:  # noqa: BLE001
            err.append(f"could not read dataset {pickle_path}: {e}")
            return
        if not hasattr(df, "columns"):
            err.append(f"dataset {pickle_path} is not a DataFrame (got {type(df).__name__})")
            return
        target_col = _resolve_target(df, cfg, {"id", "value", "struc_dict", "label"}, err)
        # The original CGCNN has no target_transform, so no log1p check; still
        # verify the target column has data.
        _check_target_and_log1p(df, cfg, target_col, err, is_mpnn=False)

    _check_workers_vs_cpus(cfg, cpus, warn)


def _check_workers_vs_cpus(cfg, cpus, warn):
    nw = cfg.get("num_workers")
    if cpus is not None and isinstance(nw, int) and nw + 1 > cpus:
        warn.append(f"num_workers ({nw}) + 1 exceeds requested cpus ({cpus}); "
                    f"workers may oversubscribe cores — set cpus >= {nw + 1}")


def main():
    ap = argparse.ArgumentParser(description="Validate a training config before submission.")
    ap.add_argument("config")
    ap.add_argument("--cpus", type=int, default=None,
                    help="resolved SLURM cpus-per-task, to cross-check num_workers")
    a = ap.parse_args()

    if not os.path.exists(a.config):
        print(f"validate_config: config not found: {a.config}", file=sys.stderr)
        return 1
    try:
        with open(a.config) as f:
            cfg = json.load(f)
    except json.JSONDecodeError as e:
        print(f"validate_config: invalid JSON in {a.config}: {e}", file=sys.stderr)
        return 1
    if not isinstance(cfg, dict):
        print(f"validate_config: {a.config} must be a JSON object", file=sys.stderr)
        return 1

    err, warn = [], []
    if "index_path" in cfg:
        kind = "MPNN"
        validate_mpnn(cfg, err, warn, a.cpus)
    elif "dataset" in cfg:
        kind = "Original CGCNN"
        validate_orig(cfg, err, warn, a.cpus)
    else:
        print("validate_config: cannot tell config type — expected 'index_path' "
              "(MPNN) or 'dataset' (original CGCNN).", file=sys.stderr)
        return 1

    for w in warn:
        print(f"  [warn] {w}", file=sys.stderr)
    if err:
        print(f"validate_config: {len(err)} problem(s) in {a.config} ({kind}):", file=sys.stderr)
        for e in err:
            print(f"  [error] {e}", file=sys.stderr)
        return 1
    print(f"validate_config: {a.config} ({kind}) OK"
          + (f" ({len(warn)} warning(s))" if warn else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
