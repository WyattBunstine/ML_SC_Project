#!/usr/bin/env python3
"""compare_runs.py — paired comparison of MPNN benchmark runs.

Standardizes the suite-results analysis: an overview table (params, val/test
MAE, timing/telemetry) plus PAIRED bootstrap significance tests of each run's
test MAE against a baseline run. Pairing works because suite runs share the
same split seed + dataset, hence the same test samples; the script verifies
that overlap before trusting it.

Usage:
    python scripts/compare_runs.py model_data/2026-06-12/mptrjB_*/mptrjB_*
    python scripts/compare_runs.py <run_dir>... [--baseline N] [--boot 3000]

(quote globs or let the shell expand them — each argument is one run dir).
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np


def load_run(run_dir):
    """Collect one run's metadata, epoch-log summary, and per-sample |error|."""
    r = {"dir": run_dir, "name": os.path.basename(run_dir.rstrip("/"))}
    meta = json.load(open(os.path.join(run_dir, "metadata.json")))
    arch = meta.get("architecture", {})
    r["tag"] = meta.get("model_type", r["name"])
    r["params"] = meta.get("model_size", {}).get("total_params")
    # Compact "what varies" descriptor from the architecture block.
    bits = [arch.get("edge_aggregation", "?")]
    bits.append("poly" if arch.get("use_poly_edges") else "no-poly")
    if arch.get("poly_fusion", "sum") != "sum":
        bits.append(f"fusion={arch['poly_fusion']}")
    if arch.get("use_coord_magnitude"):
        bits.append("coordmag")
    r["desc"] = ",".join(bits)
    r["split_by"] = meta.get("dataset", {}).get("split_by", "?")

    logs = glob.glob(os.path.join(run_dir, "*_epoch_log.csv"))
    if logs:
        rows = list(csv.DictReader(open(logs[0])))
        def f(row, k):
            try:
                return float(row.get(k, ""))
            except (TypeError, ValueError):
                return None
        vals = [(f(x, "val_mae"), i) for i, x in enumerate(rows)]
        vals = [(v, i) for v, i in vals if v is not None]
        r["best_val"], r["best_epoch"] = min(vals) if vals else (None, None)
        r["n_epochs"] = len(rows)
        last = rows[-1]
        r["s_per_epoch"] = f(last, "epoch_time_sec")
        r["gpu_pct"] = f(last, "gpu_util_pct")
        r["train_gap"] = (f(last, "val_mae") or 0) - (f(last, "train_mae") or 0)

    pred = [p for p in glob.glob(os.path.join(run_dir, "*.csv"))
            if not p.endswith("_epoch_log.csv")]
    r["errors"] = {}
    if pred:
        for row in csv.reader(open(pred[0])):
            try:
                r["errors"][row[0]] = abs(float(row[1]) - float(row[2]))
            except (ValueError, IndexError):
                continue   # header or malformed line
    r["test_mae"] = (sum(r["errors"].values()) / len(r["errors"])
                     if r["errors"] else None)
    return r


def paired_bootstrap(err_a, err_b, ids, n_boot, seed=0):
    """Mean of (a - b) per-sample |error| with a bootstrap 95% CI."""
    d = np.array([err_a[i] - err_b[i] for i in ids])
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, len(d)).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return d.mean(), lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--baseline", type=int, default=0,
                    help="index (0-based, in argument order) of the baseline run")
    ap.add_argument("--boot", type=int, default=3000, help="bootstrap resamples")
    args = ap.parse_args()

    runs = [load_run(rd) for rd in args.runs]
    base = runs[args.baseline]

    # ---- overview table ----
    print(f"{'run':<32} {'variant':<28} {'params':>8} {'best val':>9} "
          f"{'test MAE':>9} {'ep':>4} {'s/ep':>6} {'GPU%':>5}")
    for r in runs:
        print(f"{r['tag']:<32} {r['desc']:<28} {r['params'] or 0:>8,} "
              f"{r['best_val'] if r['best_val'] is not None else float('nan'):>9.5f} "
              f"{r['test_mae'] if r['test_mae'] is not None else float('nan'):>9.5f} "
              f"{r['n_epochs']:>4} {r['s_per_epoch'] or 0:>6.0f} "
              f"{r['gpu_pct'] if r['gpu_pct'] is not None else 0:>5.1f}")

    splits = {r["split_by"] for r in runs}
    if len(splits) > 1:
        print(f"\nWARNING: runs mix split_by values {splits} — comparisons may be invalid")

    # ---- paired significance vs baseline ----
    print(f"\npaired test-MAE difference vs baseline '{base['tag']}' "
          f"(negative = better than baseline; {args.boot} bootstrap resamples):")
    for r in runs:
        if r is base or not r["errors"] or not base["errors"]:
            continue
        ids = sorted(set(r["errors"]) & set(base["errors"]))
        union = len(set(r["errors"]) | set(base["errors"]))
        if union and len(ids) / union < 0.95:
            print(f"  {r['tag']:<32} SKIPPED — test sets overlap only "
                  f"{len(ids)}/{union} samples (different splits?)")
            continue
        mean, lo, hi = paired_bootstrap(r["errors"], base["errors"], ids, args.boot)
        sig = "SIGNIFICANT" if (lo > 0) == (hi > 0) else "not significant"
        better = "better" if mean < 0 else "worse"
        print(f"  {r['tag']:<32} Δ={mean:+.5f} [{lo:+.5f},{hi:+.5f}]  "
              f"{sig:<15} ({better}, n={len(ids):,})")

    # Overfit watch (end-of-training val - train MAE gap).
    gaps = [(r["tag"], r.get("train_gap")) for r in runs if r.get("train_gap") is not None]
    if gaps:
        print("\nend-of-run val-train MAE gap (large positive = overfitting):")
        for tag, g in gaps:
            print(f"  {tag:<32} {g:+.4f}")


if __name__ == "__main__":
    main()
