#!/usr/bin/env python3
"""verify_smoke.py — sanity-check a (fetched) smoke-test training run.

Turns "did the smoke test work?" into an explicit checklist over the run dir's
metadata.json, *_epoch_log.csv, and artifacts, instead of eyeballing logs.
Written for the 2-epoch MPtrj pre-flight (configs/.../mptrj_smoke_2ep.json) but
works on any MPNN run directory.

Usage:
    python scripts/verify_smoke.py model_data/<date>/<run_tag>/<run_id>/

Exit code 0 = all checks passed (warnings allowed), 1 = at least one FAIL.
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

results = []   # (level, name, detail)


def check(name, ok, detail="", warn_only=False):
    level = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    results.append((level, name, detail))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="run directory (contains metadata.json)")
    ap.add_argument("--expect-material-split", action="store_true", default=None,
                    help="require split_by == material (default: required when the "
                         "run's index_path looks like an MPtrj pack/index)")
    args = ap.parse_args()
    rd = args.run_dir

    # ---- metadata ----------------------------------------------------------
    meta_path = os.path.join(rd, "metadata.json")
    check("metadata.json present", os.path.exists(meta_path))
    meta = {}
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        ds = meta.get("dataset", {})
        ss = ds.get("split_sizes", {})
        total = ds.get("total_indexed", 0)
        n_train, n_val, n_test = (ss.get("train_sc", 0) + ss.get("train_nonsc", 0),
                                  ss.get("val_sc", 0), ss.get("test_sc", 0))
        expect_material = args.expect_material_split
        if expect_material is None:
            expect_material = "mptrj" in str(ds.get("index_path", "")).lower()
        if expect_material:
            check("split_by == material", ds.get("split_by") == "material",
                  f"got {ds.get('split_by')!r}")
        covered = n_train + n_val + n_test
        check("splits cover the dataset", covered == total,
              f"train+val+test={covered:,} vs total_indexed={total:,} "
              "(material splits are whole-material, so exact coverage is expected)")
        check("val/test fractions sane",
              total > 0 and 0.01 < n_val / max(total, 1) < 0.15
              and 0.01 < n_test / max(total, 1) < 0.15,
              f"val {n_val/max(total,1):.3f}, test {n_test/max(total,1):.3f} of {total:,}",
              warn_only=True)

    # ---- epoch log + telemetry --------------------------------------------
    logs = glob.glob(os.path.join(rd, "*_epoch_log.csv"))
    check("epoch log present", bool(logs))
    if logs:
        rows = list(csv.DictReader(open(logs[0])))
        n_epochs = meta.get("training", {}).get("epochs")
        check("all epochs ran", n_epochs is None or len(rows) == n_epochs,
              f"{len(rows)} rows vs configured {n_epochs}")
        last = rows[-1] if rows else {}

        def fnum(row, k):
            try:
                return float(row.get(k, ""))
            except (TypeError, ValueError):
                return None

        maes = [fnum(r, "val_mae") for r in rows]
        shown = (f"{maes}" if len(maes) <= 6
                 else f"[{maes[0]:.4f}, {maes[1]:.4f}, ... {maes[-1]:.4f}]")
        check("val MAE finite", all(m is not None and math.isfinite(m) for m in maes),
              f"val_mae per epoch: {shown}")
        if len(maes) >= 2 and all(m is not None for m in maes):
            check("val MAE improving across epochs", maes[-1] < maes[0],
                  f"{maes[0]:.4f} -> {maes[-1]:.4f}", warn_only=True)

        gpu = fnum(last, "gpu_util_pct")
        check("GPU telemetry recorded", gpu is not None,
              "gpu_util_pct empty — run `deploy.sh setup-env` once to install "
              "psutil + nvidia-ml-py on the cluster", warn_only=True)
        if gpu is not None:
            check("GPU actually busy (util > 30%)", gpu > 30, f"{gpu}%",
                  warn_only=True)
        et, tt, dt = (fnum(last, "epoch_time_sec"), fnum(last, "train_time_sec"),
                      fnum(last, "data_time_sec"))
        if et and tt and dt is not None:
            frac = dt / tt if tt else 0
            check("input pipeline keeps up (data-wait < 30% of train loop)",
                  frac < 0.30, f"epoch {et:.0f}s, train {tt:.0f}s, "
                  f"data-wait {dt:.0f}s ({100*frac:.0f}%)", warn_only=True)
            print(f"  [info] big-run budget: ~{et:.0f}s/epoch -> "
                  f"{et*60/3600:.1f}h per 60 epochs (+ startup)")

    # ---- artifacts ----------------------------------------------------------
    for pat, name in [("*_model_best.pth.tar", "best checkpoint written"),
                      ("*_checkpoint.pth.tar", "last checkpoint written")]:
        check(name, bool(glob.glob(os.path.join(rd, pat))))
    pred = [p for p in glob.glob(os.path.join(rd, "*.csv"))
            if not p.endswith("_epoch_log.csv")]
    check("test predictions CSV written", bool(pred))
    if pred:
        n_pred = sum(1 for _ in open(pred[0]))
        exp = meta.get("dataset", {}).get("split_sizes", {}).get("test_sc")
        check("prediction rows == test split", exp is None or n_pred == exp,
              f"{n_pred:,} rows vs test_sc {exp:,}" if exp else f"{n_pred:,} rows")

    # ---- report -------------------------------------------------------------
    width = max(len(n) for _, n, _ in results)
    fails = 0
    for level, name, detail in results:
        mark = {"PASS": "✓", "WARN": "~", "FAIL": "✗"}[level]
        print(f"  {mark} [{level}] {name:<{width}}  {detail}")
        fails += (level == "FAIL")
    print(f"\n{'ALL CHECKS PASSED' if not fails else f'{fails} CHECK(S) FAILED'}"
          f" ({sum(1 for l, _, _ in results if l == 'WARN')} warnings)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
