#!/bin/bash
# Rung-20 landing evals (resubmit job 29797654 @128G after the 96G OOM):
# wait for pretraining, verify + fetch, probe + la-series dome + figure.
# Same pre-approved single-script pattern as eval_autopilot_1718 / v45_autopilot.
cd "$(dirname "$0")/.." || exit 1
SSH="wbunsti1@login.rockfish.jhu.edu"
RP=/data/tmcquee2/wbunsti1/ML_SC_Proj
STATUS=model_data/cf_calib/eval20_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
die()  { note "HALT: $*"; exit 1; }
: > "$STATUS"
JID=29797654
GLOB="gps_20_cf_bvs_fixed_no_disorder_*-${JID}.out"
note "eval autopilot started (rung 20 resubmit, job $JID @128G)"

for i in $(seq 1 200); do   # queue + ~9.5h train
  state=$(ssh "$SSH" "squeue -j $JID -h -o '%T' 2>/dev/null" 2>/dev/null)
  [ -z "$state" ] && break
  sleep 600
done
LOG="$RP/logs/$GLOB"
ssh "$SSH" "sacct -j $JID --format=State,Elapsed -X -n" | { read -r s e; note "final state: $s ($e)"; }
ssh "$SSH" "grep -q 'Multitask pretraining done' $LOG" || die "rung20: no completion marker"
ssh "$SSH" "grep -qi 'Traceback' $LOG" && die "rung20: Traceback in log"
ssh "$SSH" "grep -oE 'Best val energy MAE: [0-9.]+' $LOG | tail -1" | { read -r m; note "rung20: $m"; }
D20=$(ssh "$SSH" "grep -oE 'Run output dir: [^ ]+' $LOG | head -1" | sed 's/Run output dir: //')
note "rung20 run dir: ${D20:-MISSING}"
[ -n "$D20" ] || die "no run dir in log"

./scripts/deploy.sh fetch >> model_data/cf_calib/eval20_fetch.log 2>&1 || die "fetch failed"
note "fetch OK"

run_eval() {  # $1=config $2=tag $3=dome-figure-suffix-or-empty $4=title
  CKPT=$(ls "$D20"/*model_best.pth.tar 2>/dev/null | head -1)
  [ -n "$CKPT" ] || { note "$2: no checkpoint under $D20"; return 0; }
  python3 - "$1" "$CKPT" <<'PYEOF' || { note "$2: config patch failed"; return 0; }
import json, sys
cfg = json.load(open(sys.argv[1]))
cfg["checkpoint"] = sys.argv[2]
json.dump(cfg, open(sys.argv[1], "w"), indent=2)
PYEOF
  note "$2: launching with $CKPT"
  python main.py train-head "$1" > "model_data/cf_calib/${2}.log" 2>&1 \
    || { note "$2: FAILED (see ${2}.log)"; return 0; }
  note "$2: $(grep -oE '3-seed ensemble: .*|7-seed ensemble: .*|ensemble: .*' "model_data/cf_calib/${2}.log" | tail -1)"
  NAME=$(python3 -c "import json;print(json.load(open('$1'))['name'])")
  RUN=$(ls -dt model_data/*/"${NAME}"_2* 2>/dev/null | head -1)
  note "$2: run dir $RUN"
  if [ -n "$3" ] && [ -f "$RUN/predictions.csv" ]; then
    FIG="docs/figures/tc_vs_cu_oxidation_la_series_dome_${3}.png"
    python scripts/plot_lsco_dome.py "$RUN/predictions.csv" "$FIG" "$4" \
      >> "model_data/cf_calib/${2}.log" 2>&1 && note "$2: figure $FIG" || note "$2: plot failed"
    python scripts/dome_stats.py "$RUN" >> "$STATUS" 2>/dev/null
  fi
}
run_eval configs/head/gps_tc_probe_20cfbvsfnd.json probe20 "" ""
run_eval configs/head/gps_tc_v4_la_series_20cfbvsfnd.json dome20 20cfbvsfnd \
  'La$_2$CuO$_4$ family doping series — CF+BVS fixed, no disorder (rung 20)'
note "RUNG-20 EVAL COMPLETE — compare probe vs 18 (4.98/6.72/0.860/19.32); dome vs 19 (r_all .813 / onset 4.14K / peak 22.0K / MAE 5.39) and 18 (.817/5.99K/19.6K/5.62)"
