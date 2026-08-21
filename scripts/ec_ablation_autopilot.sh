#!/bin/bash
# Electronic-configuration consolidation ablations, on the rung-21 (no-poly)
# base — one pre-approved chain:
#   25 = drop the valence subshell block (CF is the sole baked e-config block;
#        formal oxidation scalar kept — the doping dial + non-TM coverage)
#   26 = 25 + mask the oxidation column (orbital filling is the ONLY
#        electronic-configuration channel; also removes the n_f spectator-
#        lanthanide shortcut vehicle)
# References: rung 21 probe 4.48/5.74/0.844/15.99, dome 0.805/2.21K/21.7/5.62;
# rung 20 probe 4.94/6.52/0.845/18.91, dome 0.826/1.54K/17.4/5.32.
cd "$(dirname "$0")/.." || exit 1
SSH="wbunsti1@login.rockfish.jhu.edu"
RP=/data/tmcquee2/wbunsti1/ML_SC_Proj
STATUS=model_data/cf_calib/ec_ablation_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
die()  { note "HALT: $*"; exit 1; }
: > "$STATUS"
note "ec ablation autopilot started (rungs 25-26, icgpu04 excluded)"

declare -A JID
for ARM in 25_cf_only_valence_off 26_cf_pure_no_oxidation; do
  OUT=$(SLURM_EXCLUDE=icgpu04 ./scripts/deploy.sh run "configs/gps_mt_ablation_suite/${ARM}.json" 2>&1)
  J=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
  [ -n "$J" ] || { note "${ARM} submission FAILED"; continue; }
  JID[$ARM]=$J; note "${ARM} submitted: job $J"
done
[ ${#JID[@]} -gt 0 ] || die "no submissions succeeded"

wait_verify() {
  for i in $(seq 1 120); do
    state=$(ssh "$SSH" "squeue -j $1 -h -o '%T' 2>/dev/null" 2>/dev/null)
    [ -z "$state" ] && break
    sleep 600
  done
  LOG="$RP/logs/$2"
  ssh "$SSH" "grep -q 'Multitask pretraining done' $LOG" || { note "$3: no completion marker"; return 1; }
  ssh "$SSH" "grep -oE 'Best val energy MAE: [0-9.]+' $LOG | tail -1" | { read -r m; note "$3: $m"; }
  ssh "$SSH" "grep -E '^>> epoch 99:' $LOG | tail -1" | { read -r l; note "$3 final: $l"; }
  ssh "$SSH" "grep -oE 'Run output dir: [^ ]+' $LOG | head -1" | sed 's/Run output dir: //'
}
declare -A RDIR
for ARM in "${!JID[@]}"; do
  RDIR[$ARM]=$(wait_verify "${JID[$ARM]}" "gps_${ARM}_*-${JID[$ARM]}.out" "$ARM")
  note "${ARM} run dir: ${RDIR[$ARM]:-FAILED}"
done
./scripts/deploy.sh fetch >> model_data/cf_calib/ec_ablation_fetch.log 2>&1 || die "fetch failed"

run_eval() {
  [ -z "$2" ] && { note "$3: skipped"; return 0; }
  CKPT=$(ls "$2"/*model_best.pth.tar 2>/dev/null | head -1)
  [ -n "$CKPT" ] || { note "$3: no checkpoint"; return 0; }
  python3 - "$1" "$CKPT" <<'PYEOF' || { note "$3: patch failed"; return 0; }
import json, sys
cfg = json.load(open(sys.argv[1])); cfg["checkpoint"] = sys.argv[2]
json.dump(cfg, open(sys.argv[1], "w"), indent=2)
PYEOF
  python main.py train-head "$1" > "model_data/cf_calib/${3}.log" 2>&1 \
    || { note "$3: FAILED"; return 0; }
  note "$3: $(grep -oE '3-seed ensemble: .*|7-seed ensemble: .*' "model_data/cf_calib/${3}.log" | tail -1)"
  NAME=$(python3 -c "import json;print(json.load(open('$1'))['name'])")
  RUN=$(ls -dt model_data/*/"${NAME}"_2* 2>/dev/null | head -1)
  if [ -n "$4" ] && [ -f "$RUN/predictions.csv" ]; then
    python scripts/plot_lsco_dome.py "$RUN/predictions.csv" \
      "docs/figures/tc_vs_cu_oxidation_la_series_dome_${4}.png" "$5" \
      >> "model_data/cf_calib/${3}.log" 2>&1 && note "$3: figure dome_${4}.png"
    python scripts/dome_stats.py "$RUN" >> "$STATUS" 2>/dev/null
  fi
}
for ARM in 25 26; do
  KEY=$(ls configs/gps_mt_ablation_suite/ | grep "^${ARM}_cf" | sed 's/.json//')
  D="${RDIR[$KEY]}"
  run_eval "configs/head/gps_tc_probe_${ARM}ec.json"     "$D" "probe${ARM}ec" "" ""
  run_eval "configs/head/gps_tc_la_series_${ARM}ec.json" "$D" "dome${ARM}ec" "${ARM}ec" \
    "La\$_2\$CuO\$_4\$ family doping series — e-config consolidation rung ${ARM}"
done
note "EC ABLATION AUTOPILOT COMPLETE — vs rung 21 (4.48/5.74/0.844/15.99; 0.805/2.21K/21.7/5.62)"
