#!/usr/bin/env python3
"""Validate a training config before it is submitted (especially for remote SLURM
runs, where a typo otherwise costs a queue wait + a wasted allocation).

Detects the config type and checks, for whichever it is:
  * required keys present; enum values valid; numeric ranges sane
  * referenced files exist (the dataset/index pickle, atom_init, sampled graphs)
  * the regression target is resolvable, and that log1p is not applied to a target
    with negative values (e.g. formation energies) which would produce NaNs

  - "GPS"      config: has `index_path` + architecture="gps" -> models/GPSTransformer/gps_main.py
  - "MPNN"     config: has `index_path` -> models/MPNN/MPNNMain.py + CIFDataV4
  - "Original" config: has `dataset`   -> models/CGCNNMain.py + CIFData

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
# configs write site-specific data roots as ${ML_SC_DATA} (models/common/cfg_paths.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.common.cfg_paths import expand_config_paths  # noqa: E402

KNOWN_TARGETS = ("tc", "e_above_hull", "formation_energy_per_atom", "energy_per_atom")

# Valid enum values, mirrored from models/MPNN/MPNNMain.py + MPNNModel.py and
# models/CGCNNMain.py. Kept here so a bad value is caught before a job is submitted.
MPNN_ENUMS = {
    "task": {"regression", "classification"},
    "target_transform": {"none", "log1p"},
    "optim": {"SGD", "Adam", "AdamW"},
    "edge_aggregation": {"ecn_weighted", "attention", "set_transformer"},
    "poly_fusion": {"sum", "gate"},
    "split_by": {"frame", "material"},
    "atom_pooling": {"mean", "mean_max", "attention", "set2set"},
    "selection_metric": {"auc", "fbeta", "f1", "recall", "precision", "accuracy"},
}
ORIG_ENUMS = {
    "task": {"regression", "classification"},
    "optim": {"SGD", "Adam"},          # CGCNNMain raises on anything else
}
# GPS shares the v4-graph data path with MPNN but its trainer is regression-only
# (gps_main.py exits otherwise) and GPSCrystalNet restricts atom_pooling to
# mean/mean_max (its _POOLINGS). Stricter than MPNN_ENUMS for the shared keys.
GPS_ENUMS = {
    "task": {"regression"},
    "target_transform": {"none", "log1p"},
    "optim": {"SGD", "Adam", "AdamW"},
    "split_by": {"frame", "material"},
    "atom_pooling": {"mean", "mean_max"},
    "shell_aggregation": {"attention", "mean"},
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


def _check_atom_init_coverage(df, atom_init_path, err):
    """Fail if any element in the dataset lacks an embedding in atom_init.json.

    The ORIG loader asserts `atom_type in atom_init` per atom (data.py
    get_atom_fea), so an uncovered element (e.g. the actinides in the MP energy
    data when atom_init only spans Z<=84) crashes mid-training. Element symbols
    are pulled straight from each struc_dict's site species — a cheap full-pass
    dict walk (~0.25s for ~50k rows), so this catches rare heavy elements that
    sampling a few rows would miss. Report the missing elements and the fix.
    """
    try:
        with open(atom_init_path) as f:
            covered = {int(k) for k in json.load(f)}
    except Exception as e:  # noqa: BLE001 — unreadable atom_init already flagged upstream
        return
    # Collect unique element symbols across all structures (no pymatgen Structure
    # build — just walk the dict), then resolve symbol -> Z once.
    symbols = set()
    for sd in df["struc_dict"]:
        if not isinstance(sd, dict):
            continue
        for site in sd.get("sites", []):
            for sp in site.get("species", []):
                el = sp.get("element")
                if el:
                    symbols.add(el)
    if not symbols:
        return
    from pymatgen.core.periodic_table import Element
    missing = sorted(sym for sym in symbols if Element(sym).Z not in covered)
    if missing:
        max_z = max(Element(s).Z for s in missing)
        err.append(
            f"atom_init ({atom_init_path}) is missing embeddings for elements "
            f"present in the dataset: {', '.join(missing)} — the ORIG loader will "
            f"assert mid-training. Regenerate with coverage through these, e.g. "
            f"`python main.py build-db --kind atom-init --max-z {max_z + 1}`.")


def validate_mpnn(cfg, err, warn, cpus):
    _require(cfg, ["index_path", "out_file", "epochs", "batch_size",
                   "learning_rate", "val_ratio", "test_ratio"], err)
    _check_enums(cfg, MPNN_ENUMS, err)
    _check_common_numeric(cfg, err)

    index_path = cfg.get("index_path")
    if not index_path:
        return
    # Masked-union (gps_main): a LIST of packs trained together. Each pack is
    # validated independently; >1 pack requires multitask (mirrors gps_main's
    # "masked-union requires multitask" exit).
    if isinstance(index_path, list):
        # gps_main infers multitask from a non-empty `tasks` list (there is no
        # separate `multitask` key); a >1-pack union requires it.
        if len(index_path) > 1 and not cfg.get("tasks"):
            err.append("index_path is a list (masked-union) but `tasks` is empty; "
                       "a multi-pack union requires a multitask `tasks` config")
        for ip in index_path:
            _validate_one_index(ip, cfg, err, warn, masked_union=bool(cfg.get("tasks")))
        _check_workers_vs_cpus(cfg, cpus, warn)
        return
    _validate_one_index(index_path, cfg, err, warn)
    _check_workers_vs_cpus(cfg, cpus, warn)


def _validate_one_index(index_path, cfg, err, warn, masked_union=False):
    """masked_union: this pack is one member of a multitask union. A member that
    lacks the scalar `target_column` is legal there — the data layer masks its
    rows for that target (`_select_target_key(required=not multitask)`) instead
    of raising — so that case is a note, not an error."""
    if not os.path.exists(index_path):
        # Cluster-built datasets (e.g. MPtrj via `deploy.sh build-mptrj` /
        # `pack-mptrj`) exist ONLY on the cluster, so a locally-missing index is
        # a warning, not an error. The trade-off: a typo'd path now queues a job
        # that dies in seconds at dataset load instead of being caught here. The
        # deeper index checks below are skipped.
        warn.append(f"index_path not found locally: {index_path} — OK if it is a "
                    "cluster-built dataset that exists on the remote; otherwise fix the path")
        return

    import pandas as pd
    # A directory is a packed dataset (MPNNPack): validate against its meta
    # table, which carries the same id/label/target columns as an index pickle.
    if os.path.isdir(index_path):
        if not os.path.exists(os.path.join(index_path, "pack_header.json")):
            err.append(f"index_path {index_path} is a directory but not a packed "
                       "dataset (no pack_header.json)")
            return
        index_path = os.path.join(index_path, "meta.pickle")
    try:
        df = pd.read_pickle(index_path)
    except Exception as e:  # noqa: BLE001
        err.append(f"could not read index_path {index_path}: {e}")
        return
    if not hasattr(df, "columns"):
        err.append(f"index_path {index_path} is not a DataFrame (got {type(df).__name__})")
        return

    # mp_id is the material-grouping key for split_by="material", not a target.
    tcol = cfg.get("target_column")
    if masked_union and tcol is not None and tcol not in df.columns:
        warn.append(f"union member {index_path} has no '{tcol}' column; its rows are "
                    "masked for that target (spectral/other tasks still train on it)")
    else:
        target_col = _resolve_target(df, cfg, {"id", "value", "graph_path", "label", "mp_id"}, err)
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


def validate_gps(cfg, err, warn, cpus):
    # Reuse every MPNN index/target/file check (GPS reads the same v4-graph store),
    # then add the GPS-only model constraints that GPSCrystalNet asserts at
    # construction — caught here so they fail locally, before a queue wait.
    validate_mpnn(cfg, err, warn, cpus)
    _check_enums(cfg, GPS_ENUMS, err)

    # atom_feat_len must be divisible by both the local (set_transformer_heads) and
    # the global (gps_global_heads, default = local) attention head counts.
    d = cfg.get("atom_feat_len", 128)
    heads = cfg.get("set_transformer_heads", 8)
    if isinstance(d, int) and isinstance(heads, int) and heads > 0 and d % heads != 0:
        err.append(f"atom_feat_len ({d}) must be divisible by set_transformer_heads ({heads})")
    if cfg.get("gps_global", True):
        gh = cfg.get("gps_global_heads", heads)
        if isinstance(d, int) and isinstance(gh, int) and gh > 0 and d % gh != 0:
            err.append(f"atom_feat_len ({d}) must be divisible by gps_global_heads ({gh})")

    # Size-grouped batching knobs (optional): a max_atoms cap below batch_size's
    # nominal load just forces tiny batches — warn rather than error.
    map_ = cfg.get("max_atoms_per_batch")
    if map_ is not None and (not isinstance(map_, int) or map_ <= 0):
        err.append(f"max_atoms_per_batch={map_!r} must be a positive integer (or omitted)")
    spf = cfg.get("size_pool_factor")
    if spf is not None and (not isinstance(spf, int) or spf < 1):
        err.append(f"size_pool_factor={spf!r} must be an integer >= 1 (or omitted)")
    if cfg.get("use_dist_bias") and not cfg.get("gps_global", True):
        err.append("use_dist_bias=true requires gps_global=true (the distance bias "
                   "is applied to the global attention)")


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
        # The ORIG loader builds each crystal from row['struc_dict'] at load time
        # (models/OriginalCGCNN/data.py). A V4 graph pickle (built for the MPNN model)
        # carries 'graph_path' instead and has no 'struc_dict', so it would pass
        # target validation but KeyError at runtime. Catch the model/dataset
        # mismatch here, with a hint at the likely cause.
        if "struc_dict" not in df.columns:
            hint = (" (this looks like an MPNN graph pickle — use the MPNN config "
                    "with 'index_path', or build a struc_dict dataset for ORIG)"
                    if "graph_path" in df.columns else "")
            err.append(f"dataset {pickle_path} has no 'struc_dict' column required by "
                       f"the ORIG model; columns: {list(df.columns)}{hint}")
            return
        target_col = _resolve_target(df, cfg, {"id", "value", "struc_dict", "label"}, err)
        # The original CGCNN has no target_transform, so no log1p check; still
        # verify the target column has data.
        _check_target_and_log1p(df, cfg, target_col, err, is_mpnn=False)
        # Every element in the dataset must have an atom_init embedding, else the
        # loader asserts mid-training (e.g. actinides absent from a Z<=84 file).
        if os.path.exists(atom_init_path):
            _check_atom_init_coverage(df, atom_init_path, err)

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
            cfg = expand_config_paths(json.load(f))
    except json.JSONDecodeError as e:
        print(f"validate_config: invalid JSON in {a.config}: {e}", file=sys.stderr)
        return 1
    if not isinstance(cfg, dict):
        print(f"validate_config: {a.config} must be a JSON object", file=sys.stderr)
        return 1

    err, warn = [], []
    if "index_path" in cfg:
        if str(cfg.get("architecture", "")).strip().lower() == "gps":
            kind = "GPS"
            validate_gps(cfg, err, warn, a.cpus)
        else:
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
