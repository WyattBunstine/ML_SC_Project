#!/bin/bash
# v4.5 autopilot — ONE pre-approved unattended chain (overnight-autonomy
# pattern; deliberate halts only, status file, no agent hops):
#   1. local wave: rebuild SC/disorder graphs + re-augment DOS with the FIXED
#      builder, verify the fixes are present, pack _v45, sync (cf_v45_wave.sh)
#   2. cluster MPtrj re-augment (schema 4) -> packed_v45, verify header
#   3. submit rung 19 (CF+BVS-fixed + disorder) and rung 20 (no disorder),
#      confirm clean startup
#   4. wait for both, verify completion markers, fetch
#   5. probes + la-series domes + figures for both rungs
# Status: model_data/cf_calib/v45_status.txt
cd "$(dirname "$0")/.." || exit 1
SSH="wbunsti1@login.rockfish.jhu.edu"
RP=/data/tmcquee2/wbunsti1/ML_SC_Proj
STATUS=model_data/cf_calib/v45_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
die()  { note "HALT: $*"; exit 1; }
: > "$STATUS"
note "v45 autopilot started"

# 1. local wave (hours: 2 Voronoi rebuilds + 62k augment + 3 packs + sync)
./scripts/cf_v45_wave.sh || die "local wave failed (see v45_wave.log)"
note "local wave OK (graphs rebuilt, fixes verified, _v45 packs synced)"

# 2. cluster MPtrj re-augment -> packed_v45
OUT=$(AUGMENT_FORCE=1 ./scripts/deploy.sh augment-cf /scratch4/tmcquee2/wbunsti1/ML_SC_Proj/MPtrj/packed_v45 2>&1)
AJID=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
[ -n "$AJID" ] || die "MPtrj augment submission failed: $(echo "$OUT" | tail -2)"
note "MPtrj v45 augment submitted: job $AJID"
for i in $(seq 1 120); do   # up to 20h (queue + 12h walltime)
  state=$(ssh "$SSH" "squeue -j $AJID -h -o '%T' 2>/dev/null" 2>/dev/null)
  [ -z "$state" ] && break
  sleep 600
done
AUG_LOG="$RP/logs/augment_cf_*-${AJID}.out"
ssh "$SSH" "grep -qE 'DONE in .*, 0 failed' $AUG_LOG" || die "MPtrj augment reported failures (job $AJID)"
ssh "$SSH" "python3 -c \"
import json
h = json.load(open('/scratch4/tmcquee2/wbunsti1/ML_SC_Proj/MPtrj/packed_v45/pack_header.json'))
assert h['n_samples'] == 1578895 and h['has_cf'] and h['has_bvs'], h
print('v45 header OK')\"" || die "packed_v45 header verification failed"
note "packed_v45 verified (1,578,895 frames, has_cf+has_bvs)"

# 3. submit both pretrains (they ran concurrently at rungs 17/18)
submit_rung() {  # $1=config $2=tag -> echoes job id, empty on failure
  local out jid
  out=$(./scripts/deploy.sh run "$1" 2>&1)
  jid=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
  [ -n "$jid" ] || { note "$2 submission FAILED: $(echo "$out" | tail -2)"; return 1; }
  note "$2 submitted: job $jid"
  echo "$jid"
}
R19=$(submit_rung configs/gps_mt_ablation_suite/19_cf_bvs_fixed.json rung19)
R20=$(submit_rung configs/gps_mt_ablation_suite/20_cf_bvs_fixed_no_disorder.json rung20)
[ -z "$R19" ] && [ -z "$R20" ] && die "both pretrain submissions failed"

startup_check() {  # $1=jid $2=log-glob $3=tag
  [ -z "$1" ] && return 0
  for i in $(seq 1 96); do
    state=$(ssh "$SSH" "squeue -j $1 -h -o '%T' 2>/dev/null" 2>/dev/null)
    if [ "$state" = "RUNNING" ]; then
      sleep 300
      ssh "$SSH" "tail -n 40 $RP/logs/$2 2>/dev/null | grep -qiE 'Traceback|Error'" \
        && { note "$3 startup shows errors (job $1)"; return 1; }
      note "$3 RUNNING clean (job $1)"
      return 0
    fi
    [ -z "$state" ] && { note "$3 left the queue before starting (job $1)"; return 1; }
    sleep 600
  done
  note "$3 still queued after 16h (job $1) — continuing to wait in phase 4"
}
startup_check "$R19" "gps_19_cf_bvs_fixed_*-${R19}.out" rung19
startup_check "$R20" "gps_20_cf_bvs_fixed_no_disorder_*-${R20}.out" rung20

# 4. wait for completion + verify + fetch (rungs 17/18 trained ~9h15)
wait_verify() {  # $1=jid $2=log-glob $3=tag -> echoes run dir, empty on failure
  [ -z "$1" ] && return 1
  for i in $(seq 1 200); do   # up to ~33h
    state=$(ssh "$SSH" "squeue -j $1 -h -o '%T' 2>/dev/null" 2>/dev/null)
    [ -z "$state" ] && break
    sleep 600
  done
  LOG="$RP/logs/$2"
  ssh "$SSH" "grep -q 'Multitask pretraining done' $LOG" || { note "$3: no completion marker"; return 1; }
  ssh "$SSH" "grep -qi 'Traceback' $LOG" && { note "$3: Traceback in log"; return 1; }
  ssh "$SSH" "grep -oE 'Best val energy MAE: [0-9.]+' $LOG | tail -1" | \
    { read -r m; note "$3: $m"; }
  ssh "$SSH" "grep -oE 'Run output dir: [^ ]+' $LOG | head -1" | sed 's/Run output dir: //'
}
D19=$(wait_verify "$R19" "gps_19_cf_bvs_fixed_*-${R19}.out" rung19)
D20=$(wait_verify "$R20" "gps_20_cf_bvs_fixed_no_disorder_*-${R20}.out" rung20)
note "rung19 run dir: ${D19:-FAILED}"
note "rung20 run dir: ${D20:-FAILED}"
[ -z "$D19" ] && [ -z "$D20" ] && die "both pretrainings failed verification"

./scripts/deploy.sh fetch >> model_data/cf_calib/v45_fetch.log 2>&1 \
  || die "fetch failed (see v45_fetch.log)"
note "fetch OK"

# 5. evals (probe: 3-seed norms/chemsys/msle; dome: 7-seed parent_comp blocks:1)
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
  note "$3: launching with $CKPT"
  python main.py train-head "$1" > "model_data/cf_calib/${3}.log" 2>&1 \
    || { note "$3: FAILED (see ${3}.log)"; return 0; }
  note "$3: $(grep -oE '3-seed ensemble: .*|7-seed ensemble: .*|ensemble: .*' "model_data/cf_calib/${3}.log" | tail -1)"
  NAME=$(python3 -c "import json;print(json.load(open('$1'))['name'])")
  RUN=$(ls -dt model_data/*/"${NAME}"_2* 2>/dev/null | head -1)
  note "$3: run dir $RUN"
  if [ -n "$4" ] && [ -f "$RUN/predictions.csv" ]; then
    FIG="docs/figures/tc_vs_cu_oxidation_la_series_dome_${4}.png"
    python scripts/plot_lsco_dome.py "$RUN/predictions.csv" "$FIG" "$5" \
      >> "model_data/cf_calib/${3}.log" 2>&1 && note "$3: figure $FIG" \
      || note "$3: plot failed"
  fi
}
run_eval configs/head/gps_tc_probe_19cfbvsf.json           "$D19" probe19   "" ""
run_eval configs/head/gps_tc_probe_20cfbvsfnd.json         "$D20" probe20   "" ""
run_eval configs/head/gps_tc_v4_la_series_19cfbvsf.json    "$D19" dome19 19cfbvsf \
  'La$_2$CuO$_4$ family doping series — CF+BVS fixed, +disorder (rung 19)'
run_eval configs/head/gps_tc_v4_la_series_20cfbvsfnd.json  "$D20" dome20 20cfbvsfnd \
  'La$_2$CuO$_4$ family doping series — CF+BVS fixed, no disorder (rung 20)'
note "V45 AUTOPILOT COMPLETE — probes vs 16 (4.85/6.47/0.868/18.36) + champion 12 (4.26/5.33/0.847/15.23); domes vs 18 (r_all=0.817, onset 6.0K) + 16 (0.811/7.6K)"
