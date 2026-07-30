#!/bin/bash
# Eval autopilot for rungs 17 (CF, no disorder) / 18 (CF+BVS, no disorder):
# wait for pretraining jobs 29460152 / 29460158, verify + fetch, then run the
# four local evaluations (2 champion-recipe probes + 2 la_series domes) and
# render the dome figures. Single pre-approved script, deliberate halts only,
# per-rung failures note-and-continue (one bad rung can't block the other).
cd "$(dirname "$0")/.." || exit 1
SSH="wbunsti1@login.rockfish.jhu.edu"
RP=/data/tmcquee2/wbunsti1/ML_SC_Proj
STATUS=model_data/cf_calib/eval1718_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
die()  { note "HALT: $*"; exit 1; }
: > "$STATUS"
note "eval autopilot started (jobs 29460152=rung17, 29460158=rung18)"

wait_verify() {  # $1=jid $2=log-glob $3=tag -> echoes run dir, empty on failure
  for i in $(seq 1 90); do   # up to 15h
    state=$(ssh "$SSH" "squeue -j $1 -h -o '%T' 2>/dev/null" 2>/dev/null)
    [ -z "$state" ] && break
    sleep 600
  done
  LOG="$RP/logs/$2"
  ssh "$SSH" "grep -q 'Multitask pretraining done' $LOG" || { note "$3: no completion marker"; return 1; }
  ssh "$SSH" "grep -qi 'Traceback' $LOG" && { note "$3: Traceback in log"; return 1; }
  ssh "$SSH" "grep -oE 'Run output dir: [^ ]+' $LOG | head -1" | sed 's/Run output dir: //'
}

R17=$(wait_verify 29460152 "gps_17_cf_no_disorder_*-29460152.out" rung17)
R18=$(wait_verify 29460158 "gps_18_cf_bvs_no_disorder_*-29460158.out" rung18)
note "rung17 run dir: ${R17:-FAILED}"
note "rung18 run dir: ${R18:-FAILED}"
[ -z "$R17" ] && [ -z "$R18" ] && die "both pretrainings failed verification"

./scripts/deploy.sh fetch >> model_data/cf_calib/eval1718_fetch.log 2>&1 \
  || die "fetch failed (see eval1718_fetch.log)"

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

run_eval configs/head/gps_tc_probe_17cfnd.json          "$R17" probe17   "" ""
run_eval configs/head/gps_tc_probe_18cfbvsnd.json       "$R18" probe18   "" ""
run_eval configs/head/gps_tc_v4_la_series_17cfnd.json   "$R17" dome17 17cfnd \
  'La$_2$CuO$_4$ family doping series — CF, no disorder (rung 17)'
run_eval configs/head/gps_tc_v4_la_series_18cfbvsnd.json "$R18" dome18 18cfbvsnd \
  'La$_2$CuO$_4$ family doping series — CF+BVS, no disorder (rung 18)'
note "EVAL AUTOPILOT COMPLETE — compare probes vs champion 4.26/5.33/0.847/15.23; domes vs 09 onset (r=0.791) and 16 (r=0.811)"
