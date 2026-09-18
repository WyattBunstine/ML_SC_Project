#!/bin/bash
# RESUME watcher for the target-ablation wave + rungs 34/35 — after the original
# autopilot watcher was killed. Does NOT resubmit anything: parses the status
# file for already-recorded run dirs, waits only on jobs still queued/running
# (32/33 pending on AssocGrpBillingMinutes may sit for hours), then fetches once
# and runs ALL the evals (probe + dome + family line per rung), which the dead
# watcher never reached.
cd "$(dirname "$0")/.." || exit 1
SSH="wbunsti1@login.rockfish.jhu.edu"
RP=/data/tmcquee2/wbunsti1/ML_SC_Proj
STATUS=model_data/cf_calib/target_ablation_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
note "RESUME watcher started (evals pending for all landed rungs)"

declare -A JID RDIR CFG
CFG[27]=27_t_energy; CFG[28]=28_t_forces_stress; CFG[29]=29_t_no_dos
CFG[30]=30_t_all_equal; CFG[31]=31_t_no_magmom; CFG[32]=32_t_no_bandgap
CFG[33]=33_t_no_forces; CFG[34]=34_t_eph; CFG[35]=35_t_a2f
for N in 27 28 29 30 31 32 33; do
  JID[$N]=$(grep -oE "${CFG[$N]} submitted: job [0-9]+" "$STATUS" | tail -1 | grep -oE '[0-9]+$')
  RDIR[$N]=$(grep -oE "${CFG[$N]} run dir: [^ ]+" "$STATUS" | tail -1 | sed 's/.*run dir: //')
done
JID[34]=30239520; JID[35]=30242097

wait_verify() {
  for i in $(seq 1 432); do
    state=$(ssh "$SSH" "squeue -j $1 -h -o '%T' 2>/dev/null" 2>/dev/null)
    rc=$?
    [ "$rc" -ne 255 ] && [ -z "$state" ] && break
    sleep 600
  done
  LOG="$RP/logs/$2"
  ssh "$SSH" "grep -q 'Multitask pretraining done' $LOG" || { note "$3: no completion marker"; return 1; }
  ssh "$SSH" "grep -qi 'Traceback' $LOG" && { note "$3: Traceback in log"; return 1; }
  ssh "$SSH" "grep -oE 'Best val energy MAE: [0-9.]+' $LOG | tail -1" | { read -r m; note "$3: $m"; }
  ssh "$SSH" "grep -oE 'Run output dir: [^ ]+' $LOG | head -1" | sed 's/Run output dir: //'
}
for N in 27 28 29 30 31 32 33 34 35; do
  A="${CFG[$N]}"
  if [ -n "${RDIR[$N]}" ] && [ "${RDIR[$N]}" != FAILED ]; then
    note "$A: already landed (${RDIR[$N]})"
    continue
  fi
  [ -z "${JID[$N]}" ] && { note "$A: no job id — skipped"; continue; }
  RDIR[$N]=$(wait_verify "${JID[$N]}" "gps_${A}_*-${JID[$N]}.out" "$A")
  note "$A run dir: ${RDIR[$N]:-FAILED}"
done
./scripts/deploy.sh fetch >> model_data/cf_calib/target_ablation_fetch.log 2>&1 || { note "HALT: fetch failed"; exit 1; }

run_eval() {
  [ -z "$2" ] || [ "$2" = FAILED ] && { note "$3: skipped (no run dir)"; return 0; }
  CKPT=$(ls "$2"/*model_best.pth.tar 2>/dev/null | head -1)
  [ -n "$CKPT" ] || { note "$3: no checkpoint"; return 0; }
  python3 - "$1" "$CKPT" <<'PYEOF' || { note "$3: patch failed"; return 0; }
import json, sys
cfg = json.load(open(sys.argv[1])); cfg["checkpoint"] = sys.argv[2]
json.dump(cfg, open(sys.argv[1], "w"), indent=2)
PYEOF
  python main.py train-head "$1" > "model_data/cf_calib/${3}.log" 2>&1 \
    || { note "$3: FAILED"; return 0; }
  note "$3: $(grep -oE '[0-9]+-seed ensemble: .*' "model_data/cf_calib/${3}.log" | tail -1)"
  NAME=$(python3 -c "import json;print(json.load(open('$1'))['name'])")
  RUN=$(ls -dt model_data/*/"${NAME}"_2* 2>/dev/null | head -1)
  [ -f "$RUN/predictions.csv" ] && python scripts/family_stats.py "$RUN" | tail -1 >> "$STATUS" 2>/dev/null
  if [ -n "$4" ] && [ -f "$RUN/predictions.csv" ]; then
    python scripts/plot_lsco_dome.py "$RUN/predictions.csv" \
      "docs/figures/tc_vs_cu_oxidation_la_series_dome_${4}.png" "$5" \
      >> "model_data/cf_calib/${3}.log" 2>&1 && note "$3: figure dome_${4}.png"
    python scripts/dome_stats.py "$RUN" >> "$STATUS" 2>/dev/null
  fi
}
for N in 27 28 29 30 31 32 33 34 35; do
  D="${RDIR[$N]}"
  run_eval "configs/head/gps_tc_probe_${N}tg.json"     "$D" "probe${N}tg" "" ""
  run_eval "configs/head/gps_tc_la_series_${N}tg.json" "$D" "dome${N}tg" "${N}tg" \
    "La\$_2\$CuO\$_4\$ family doping series — pretraining-target rung ${N}"
done
note "RESUME WATCHER COMPLETE — refs: rung 25 probe 4.58 (cup 22.4), dome 0.823/1.33K/22.9/5.30"
