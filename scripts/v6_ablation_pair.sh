#!/bin/bash
# Two V6-protocol ablations on rung 20, sequential (one GPU), vs the A+B
# exclusion run (same-68 r=0.254 / onset 2.69K / peak 23.8K):
#   noforce — fixed holdout + A+B exclusions, NO train-force: isolates the
#             other-dopants-in-train design from NEMAD label quality
#   exclCE  — train-force kept, exclusions extended to ALL tiers (191 rows):
#             tests whether the review-tier C/E rows carry the residual noise
cd "$(dirname "$0")/.." || exit 1
STATUS=model_data/cf_calib/v6_ablation_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
: > "$STATUS"
note "v6 ablation pair started (noforce, exclCE)"
for TAG in noforce exclCE; do
  CFG="configs/head/gps_tc_v6v45_lsco_20_${TAG}.json"
  note "${TAG}: launching"
  if python main.py train-head "$CFG" > "model_data/cf_calib/v6dome20_${TAG}.log" 2>&1; then
    note "${TAG}: $(grep -oE '7-seed ensemble: .*' "model_data/cf_calib/v6dome20_${TAG}.log" | tail -1)"
  else
    note "${TAG}: FAILED (see v6dome20_${TAG}.log)"
  fi
done
note "V6 ABLATION PAIR COMPLETE"
