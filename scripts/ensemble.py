#!/usr/bin/env python3
"""Combine several models' predictions into an ensemble: mean prediction plus a
per-material uncertainty (std across members), with the ensemble vs. single-model
error so you can see what the ensemble bought.

Input is two or more headerless prediction CSVs in the `cif_id,target,pred`
format the trainers and scripts/eval_test.py emit. Members are joined on `cif_id`
(intersection), so to compare fairly each member should have been evaluated on the
SAME materials — e.g. run eval_test.py with a shared `--test-ids` file, then pass
the resulting CSVs here. The members must also be *diverse* to help: train them
with different `model_seed` (and/or different data) — identical members average to
themselves.

Output is a CSV `cif_id,target,mean_pred,std_pred,n_models`. `std_pred` is a cheap
epistemic-uncertainty estimate (members disagree most where they're least sure).

Usage:
    ensemble.py PRED1.csv PRED2.csv [PRED3.csv ...] [--out FILE]
"""
import argparse
import csv
import os
import statistics
import sys


def _read(path):
    """Read a headerless cif_id,target,pred CSV -> dict cif_id -> (target, pred)."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            cif_id = row[0]
            out[cif_id] = (float(row[1]), float(row[2]))
    if not out:
        raise SystemExit(f"ensemble: no usable rows in {path}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds", nargs="+", help="2+ prediction CSVs (cif_id,target,pred)")
    ap.add_argument("--out", help="output CSV (default: ensemble_<N>models.csv next to the first input)")
    a = ap.parse_args()

    if len(a.preds) < 2:
        raise SystemExit("ensemble: need at least 2 prediction files to ensemble.")

    members = [_read(p) for p in a.preds]
    # Join on the cif_ids present in every member.
    common = set(members[0])
    for m in members[1:]:
        common &= set(m)
    if not common:
        raise SystemExit("ensemble: no cif_ids common to all inputs — were they "
                         "evaluated on the same materials? (use eval_test --test-ids)")
    dropped = max(len(m) for m in members) - len(common)
    if dropped:
        print(f"ensemble: {len(common)} materials common to all {len(members)} members "
              f"({dropped} not shared by all were dropped)")

    rows = []
    per_member_abs = [[] for _ in members]   # |err| per member, over common ids
    ens_abs = []
    for cif_id in sorted(common):
        target = members[0][cif_id][0]
        preds = [m[cif_id][1] for m in members]
        mean_pred = statistics.fmean(preds)
        std_pred = statistics.pstdev(preds) if len(preds) > 1 else 0.0
        rows.append((cif_id, target, mean_pred, std_pred, len(preds)))
        ens_abs.append(abs(target - mean_pred))
        for i, p in enumerate(preds):
            per_member_abs[i].append(abs(target - p))

    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.preds[0])),
                                f"ensemble_{len(members)}models.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cif_id", "target", "mean_pred", "std_pred", "n_models"])
        w.writerows(rows)

    member_maes = [statistics.fmean(e) for e in per_member_abs]
    ens_mae = statistics.fmean(ens_abs)
    mean_member_mae = statistics.fmean(member_maes)
    mean_uncert = statistics.fmean(r[3] for r in rows)
    print(f"ensemble: wrote {len(rows)} rows -> {out}")
    print(f"  per-member MAE:    {[round(m, 4) for m in member_maes]}")
    print(f"  mean single MAE:   {mean_member_mae:.4f}")
    print(f"  ENSEMBLE MAE:      {ens_mae:.4f}  "
          f"({(mean_member_mae - ens_mae) / mean_member_mae * 100:+.1f}% vs mean single)")
    print(f"  mean uncertainty:  {mean_uncert:.4f} (std across members)")


if __name__ == "__main__":
    main()
