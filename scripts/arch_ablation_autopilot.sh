#!/bin/bash
# Architecture ablation ladder on the rung-20 recipe (CF+BVS fixed, no
# disorder, v45 packs) — one pre-approved unattended chain:
#   21 no-poly | 22 no-attention (poly off + mean shells, no angle) |
#   23 raw atoms (n_conv 0; forces/stress dropped BY CONSTRUCTION) |
#   24 no-angle-bias
# Submit all four (queue runs ~2 concurrent), wait, verify, fetch, then probe
# + la-series dome + figure per landed encoder. Reference: rung 20 probe
# 4.94/6.52/0.845/18.91; dome r_all 0.826 / onset 1.54K / peak 17.4K / 5.32.
cd "$(dirname "$0")/.." || exit 1
SSH="wbunsti1@login.rockfish.jhu.edu"
RP=/data/tmcquee2/wbunsti1/ML_SC_Proj
STATUS=model_data/cf_calib/arch_ablation_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
die()  { note "HALT: $*"; exit 1; }
: > "$STATUS"
note "arch ablation autopilot started (rungs 21-24)"

declare -A JID
for ARM in 21_ablate_no_poly 22_ablate_no_attention 23_ablate_raw_atoms 24_ablate_no_angle; do
  OUT=$(./scripts/deploy.sh run "configs/gps_mt_ablation_suite/${ARM}.json" 2>&1)
  J=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
  [ -n "$J" ] || { note "${ARM} submission FAILED: $(echo "$OUT" | tail -1)"; continue; }
  JID[$ARM]=$J
  note "${ARM} submitted: job $J"
done
[ ${#JID[@]} -gt 0 ] || die "no submissions succeeded"

wait_verify() {  # $1=jid $2=log-glob $3=tag -> echoes run dir, empty on failure
  for i in $(seq 1 260); do   # ~43h: 4 jobs, ~2 concurrent, ~9.5h each + queue
    state=$(ssh "$SSH" "squeue -j $1 -h -o '%T' 2>/dev/null" 2>/dev/null)
    [ -z "$state" ] && break
    sleep 600
  done
  LOG="$RP/logs/$2"
  ssh "$SSH" "grep -q 'Multitask pretraining done' $LOG" || { note "$3: no completion marker"; return 1; }
  ssh "$SSH" "grep -qi 'Traceback' $LOG" && { note "$3: Traceback in log"; return 1; }
  ssh "$SSH" "grep -oE 'Best val energy MAE: [0-9.]+' $LOG | tail -1" | { read -r m; note "$3: $m"; }
  ssh "$SSH" "grep -E '^>> epoch 99:' $LOG | tail -1" | { read -r l; note "$3 final: $l"; }
  ssh "$SSH" "grep -oE 'Run output dir: [^ ]+' $LOG | head -1" | sed 's/Run output dir: //'
}
declare -A RDIR
for ARM in "${!JID[@]}"; do
  RDIR[$ARM]=$(wait_verify "${JID[$ARM]}" "gps_${ARM}_*-${JID[$ARM]}.out" "$ARM")
  note "${ARM} run dir: ${RDIR[$ARM]:-FAILED}"
done

./scripts/deploy.sh fetch >> model_data/cf_calib/arch_ablation_fetch.log 2>&1 || die "fetch failed"
note "fetch OK"

run_eval() {  # $1=config $2=rundir $3=tag $4=dome-figure-suffix-or-empty $5=title
  [ -z "$2" ] && { note "$3: skipped (no checkpoint)"; return 0; }
  CKPT=$(ls "$2"/*model_best.pth.tar 2>/dev/null | head -1)
  [ -n "$CKPT" ] || { note "$3: no checkpoint under $2"; return 0; }
  python3 - "$1" "$CKPT" <<'PYEOF' || { note "$3: config patch failed"; return 0; }
import json, sys
cfg = json.load(open(sys.argv[1]))
cfg["checkpoint"] = sys.argv[2]
json.dump(cfg, open(sys.argv[1], "w"), indent=2)
PYEOF
  python main.py train-head "$1" > "model_data/cf_calib/${3}.log" 2>&1 \
    || { note "$3: FAILED (see ${3}.log)"; return 0; }
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
for ARM in 21 22 23 24; do
  KEY=$(ls configs/gps_mt_ablation_suite/ | grep "^${ARM}_ablate" | sed 's/.json//')
  D="${RDIR[$KEY]}"
  run_eval "configs/head/gps_tc_probe_${ARM}abl.json"     "$D" "probe${ARM}abl" "" ""
  run_eval "configs/head/gps_tc_la_series_${ARM}abl.json" "$D" "dome${ARM}abl" "${ARM}abl" \
    "La\$_2\$CuO\$_4\$ family doping series — arch ablation rung ${ARM}"
done
note "ARCH ABLATION AUTOPILOT COMPLETE — compare vs rung 20 (probe 4.94/6.52/0.845/18.91; dome 0.826/1.54K/17.4K/5.32)"
