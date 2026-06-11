#!/usr/bin/env python3
"""reorg_runs.py — tidy a model_data/ tree and emit a comparison index.

Two jobs, run identically on the cluster and locally (stdlib only, so the
cluster login node's bare python3 can run it):

  1. reorg : move FLAT run dirs  model_data/<tag>_<date>_<time>[_suffix]/
             into nested         model_data/<date>/<tag>/<same-name>/
             Idempotent: dirs already living under a <date>/ group are left
             alone, so re-running is a no-op. The leaf dir name is preserved,
             so the SAME move happens on both remote and local — after which
             `deploy.sh fetch` (a plain rsync mirror) re-pairs them with no
             re-download.

  2. index : walk every run dir (flat or nested) and write model_data/index.csv
             — one row per run with the key metrics pulled from metadata.json +
             *_epoch_log.csv, so runs are comparable without opening folders.

Usage:
  reorg_runs.py --root model_data                # DRY RUN: print the move plan + write index
  reorg_runs.py --root model_data --apply        # actually move, then write index
  reorg_runs.py --root model_data --index-only    # just (re)write index.csv
"""
import argparse
import csv
import json
import os
import re
import shutil

# <tag>_<YYYY-MM-DD>_<HH-MM-SS>[_<freeform suffix>]
# tag is non-greedy so it ends at the FIRST date token (tags never contain one).
RUN_RE = re.compile(
    r"^(?P<tag>.+?)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})(?P<suffix>_.*)?$"
)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ARCHIVE_DIRNAME = ".archive"


def parse_run_name(name):
    """(date, tag) for a flat run dir name, or None if it isn't one."""
    m = RUN_RE.match(name)
    if not m:
        return None
    return m.group("date"), m.group("tag")


# --------------------------------------------------------------------------- #
# 1. reorg
# --------------------------------------------------------------------------- #
def plan_moves(root):
    """List of (src_abs, dst_abs) for flat run dirs that need nesting."""
    moves = []
    for name in sorted(os.listdir(root)):
        src = os.path.join(root, name)
        if not os.path.isdir(src):
            continue
        # Already-organized date groups and the archive are left untouched.
        if DATE_RE.match(name) or name == ARCHIVE_DIRNAME:
            continue
        parsed = parse_run_name(name)
        if parsed is None:
            print(f"  ?? skip (not a recognized run dir): {name}")
            continue
        date, tag = parsed
        dst = os.path.join(root, date, tag, name)
        moves.append((src, dst))
    return moves


def do_reorg(root, apply):
    moves = plan_moves(root)
    if not moves:
        print(">> reorg: nothing to do (already organized).")
        return
    verb = "moving" if apply else "would move"
    for src, dst in moves:
        rel_src = os.path.relpath(src, root)
        rel_dst = os.path.relpath(dst, root)
        if os.path.exists(dst):
            print(f"  !! dst exists, skipping: {rel_dst}")
            continue
        print(f"  {verb}: {rel_src}  ->  {rel_dst}")
        if apply:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
    print(f">> reorg: {len(moves)} run(s) {'moved' if apply else 'to move (dry run)'}.")


# --------------------------------------------------------------------------- #
# 2. index
# --------------------------------------------------------------------------- #
def _find_run_dirs(root):
    """Every leaf run dir (contains metadata.json, config.json, or an epoch log),
    skipping the archive."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        if ARCHIVE_DIRNAME in dirnames:
            dirnames.remove(ARCHIVE_DIRNAME)  # don't descend into archived runs
        is_run = (
            "metadata.json" in filenames
            or "config.json" in filenames
            or any(f.endswith("_epoch_log.csv") for f in filenames)
        )
        if is_run:
            out.append(dirpath)
            dirnames[:] = []  # a run dir has no sub-runs
    return sorted(out)


def _best_row(epoch_log):
    """The selected epoch's row (last is_best==1, else the min-val_loss row),
    plus the number of epochs logged. Returns (best_dict, n_epochs).

    n_epochs is the row COUNT, not max(epoch): the epoch column is 0-indexed, so
    a finished 500-epoch run ends at epoch 499 — counting rows gives 500 and
    compares cleanly against the configured target regardless of 0/1 indexing."""
    try:
        with open(epoch_log, newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return None, None
    if not rows:
        return None, None

    def fnum(r, k):
        try:
            return float(r.get(k, ""))
        except (TypeError, ValueError):
            return None

    n_epochs = len(rows)

    best = [r for r in rows if str(r.get("is_best", "")).strip() in ("1", "1.0", "True", "true")]
    if best:
        return best[-1], n_epochs
    # Fallback: lowest val_loss.
    scored = [(fnum(r, "val_loss"), r) for r in rows]
    scored = [(v, r) for v, r in scored if v is not None]
    if scored:
        return min(scored, key=lambda x: x[0])[1], n_epochs
    return rows[-1], n_epochs


def _row_for_run(root, run_dir):
    rel = os.path.relpath(run_dir, root)
    info = {
        "rel_path": rel, "date": "", "run_tag": "", "run_name": os.path.basename(run_dir),
        "target": "", "task": "", "params": "", "target_epochs": "",
        "epochs_trained": "", "best_epoch": "", "val_loss": "", "val_mae": "",
        "val_bal_mae": "", "complete": "",
    }
    # date / tag from the nested path when present, else from the dir name.
    parts = rel.split(os.sep)
    if len(parts) >= 3 and DATE_RE.match(parts[0]):
        info["date"], info["run_tag"] = parts[0], parts[1]
    else:
        parsed = parse_run_name(os.path.basename(run_dir))
        if parsed:
            info["date"], info["run_tag"] = parsed

    meta_path = os.path.join(run_dir, "metadata.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            info["task"] = meta.get("task", "") or ""
            info["target"] = (meta.get("dataset") or {}).get("target_column", "") or ""
            info["params"] = (meta.get("model_size") or {}).get("total_params", "") or ""
            info["target_epochs"] = (meta.get("training") or {}).get("epochs", "") or ""
            if not info["run_tag"]:
                info["run_tag"] = meta.get("model_type", "") or ""
        except (OSError, ValueError):
            pass

    # The baseline CGCNN metadata stores only the pickle name, not the target
    # column — fall back to the copied config.json so its rows aren't blank.
    if not info["target"]:
        cfg_path = os.path.join(run_dir, "config.json")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path) as f:
                    info["target"] = json.load(f).get("target_column", "") or ""
            except (OSError, ValueError):
                pass

    logs = [f for f in os.listdir(run_dir) if f.endswith("_epoch_log.csv")]
    if logs:
        best, n_epochs = _best_row(os.path.join(run_dir, sorted(logs)[0]))
        if n_epochs is not None:
            info["epochs_trained"] = n_epochs
            if info["target_epochs"] != "":
                try:
                    info["complete"] = "yes" if n_epochs >= int(info["target_epochs"]) else "no"
                except (TypeError, ValueError):
                    pass
        if best:
            info["best_epoch"] = best.get("epoch", "")
            for k in ("val_loss", "val_mae", "val_bal_mae"):
                v = best.get(k, "")
                # Trim float noise for readability.
                try:
                    info[k] = f"{float(v):.5g}"
                except (TypeError, ValueError):
                    info[k] = v or ""
    return info


COLUMNS = [
    "date", "run_tag", "run_name", "target", "task", "params",
    "epochs_trained", "target_epochs", "complete",
    "best_epoch", "val_loss", "val_mae", "val_bal_mae", "rel_path",
]


def write_index(root):
    runs = _find_run_dirs(root)
    rows = [_row_for_run(root, d) for d in runs]
    # Sort newest first by (date, run_name).
    rows.sort(key=lambda r: (r["date"], r["run_name"]), reverse=True)
    out = os.path.join(root, "index.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})
    print(f">> index: wrote {len(rows)} run(s) -> {os.path.relpath(out, root)}")


# --------------------------------------------------------------------------- #
# 3. distribute SLURM logs into their run dirs
# --------------------------------------------------------------------------- #
def distribute_logs(logs_dir, root):
    """Move fetched SLURM logs (<job>.out/.err) into the model_data run dir each
    belongs to, so a run's stdout lives next to its checkpoints/metadata.

    The mapping comes from the log content: MPNNMain/CGCNNMain print
    "Run output dir: model_data/..." at startup. Logs whose run dir isn't found
    locally (still remote-only, archived, or non-training jobs like build_mptrj)
    stay in logs_dir. Idempotent: a re-fetched log is simply moved again,
    overwriting the identical copy in the run dir.
    """
    moved = skipped = 0
    for fname in sorted(os.listdir(logs_dir)):
        if not fname.endswith(".out"):
            continue
        out_path = os.path.join(logs_dir, fname)
        run_dir_rel = None
        try:
            with open(out_path, errors="replace") as fh:
                for i, line in enumerate(fh):
                    if line.startswith("Run output dir: "):
                        run_dir_rel = line[len("Run output dir: "):].strip()
                        break
                    if i > 300:           # marker prints near the top; don't scan GBs
                        break
        except OSError:
            continue
        if not run_dir_rel:
            skipped += 1
            continue
        # The printed path is relative to the (remote) project root, always
        # starting with the model_data component — remap onto our local root.
        parts = run_dir_rel.replace("\\", "/").split("/")
        if "model_data" in parts:
            parts = parts[parts.index("model_data") + 1:]
        target_dir = os.path.join(root, *parts)
        if not os.path.isdir(target_dir):
            skipped += 1                  # run not fetched locally (or archived)
            continue
        for ext in (".out", ".err"):
            src = os.path.join(logs_dir, fname[:-4] + ext)
            if os.path.exists(src):
                shutil.move(src, os.path.join(target_dir, os.path.basename(src)))
        moved += 1
    print(f">> logs: {moved} job log(s) moved into run dirs, {skipped} left in {logs_dir}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="path to the model_data/ tree")
    ap.add_argument("--apply", action="store_true", help="execute moves (default: dry run)")
    ap.add_argument("--index-only", action="store_true", help="skip reorg, just (re)write index.csv")
    ap.add_argument("--distribute-logs", metavar="LOGS_DIR", default=None,
                    help="move SLURM logs from LOGS_DIR into their model_data run dirs")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit(f"ERROR: root not found: {root}")

    if not args.index_only:
        do_reorg(root, apply=args.apply)
    if args.distribute_logs:
        distribute_logs(os.path.abspath(args.distribute_logs), root)
    write_index(root)


if __name__ == "__main__":
    main()
