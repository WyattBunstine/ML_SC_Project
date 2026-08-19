#!/bin/bash
# Pretraining-target ablation x SC-class matrix: re-probe the June target-
# ablation checkpoints (rungs 01-08) under the FIXED post-split-bug protocol
# (3-seed / chemsys / msle / norms — the modern probe), then score per family.
# The old era's fine-tune comparisons ran on scrambled folds; this makes the
# target axis citable and adds the per-class breakdown that never existed.
cd "$(dirname "$0")/.." || exit 1
STATUS=model_data/cf_calib/target_probe_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
: > "$STATUS"
note "target-probe sweep started (7 checkpoints, fixed protocol)"
RUNS=()
for TAG in 01_energy 02_forces 03_magbg 04_dosfull 06_forcesw2 07_nodos 08_nomagmom; do
  if python main.py train-head "configs/head/gps_tc_tprobe_${TAG}.json" \
       > "model_data/cf_calib/tprobe_${TAG}.log" 2>&1; then
    note "${TAG}: $(grep -oE '3-seed ensemble: .*' "model_data/cf_calib/tprobe_${TAG}.log" | tail -1)"
    RUNS+=("$(ls -dt model_data/*/gps_tc_tprobe_${TAG}_2* 2>/dev/null | head -1)")
  else
    note "${TAG}: FAILED (see tprobe_${TAG}.log)"
  fi
done
note "per-family matrix:"
python scripts/family_stats.py "${RUNS[@]}" >> "$STATUS" 2>&1
note "TARGET-PROBE SWEEP COMPLETE"
