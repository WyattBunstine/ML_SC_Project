#!/bin/bash
# Dome loss sweep on the rung-18 encoder (current dome champion, l1 baseline
# r_all=0.817 / onset 6.0K / peak 19.6K / holdout MAE 5.62): byte-identical
# la_series protocol, ONLY the loss space swapped. Sequential on the local GPU
# (~20 min per 7-seed run); note-and-continue per loss; stats table at the end.
cd "$(dirname "$0")/.." || exit 1
STATUS=model_data/cf_calib/loss_sweep_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
: > "$STATUS"
note "dome loss sweep started (rung 18, losses: mse_k msle wl1_k huber_k)"

RUNS=()
for LOSS in mse_k msle wl1_k huber_k; do
  CFG="configs/head/gps_tc_v4_la_series_18_${LOSS}.json"
  note "${LOSS}: launching"
  if python main.py train-head "$CFG" > "model_data/cf_calib/dome18_${LOSS}.log" 2>&1; then
    note "${LOSS}: $(grep -oE '7-seed ensemble: .*' "model_data/cf_calib/dome18_${LOSS}.log" | tail -1)"
    RUN=$(ls -dt model_data/*/gps_tc_v4_la_series_18_${LOSS}_2* 2>/dev/null | head -1)
    note "${LOSS}: run dir $RUN"
    RUNS+=("$RUN")
  else
    note "${LOSS}: FAILED (see dome18_${LOSS}.log)"
  fi
done

note "computing dome stats (incl. l1 baseline)"
python scripts/dome_stats.py \
  model_data/2026-08-11/gps_tc_v4_la_series_18cfbvsnd_2026-08-11_12-33-26 \
  "${RUNS[@]}" > model_data/cf_calib/loss_sweep_stats.txt 2>&1
cat model_data/cf_calib/loss_sweep_stats.txt >> "$STATUS"
note "DOME LOSS SWEEP COMPLETE"
