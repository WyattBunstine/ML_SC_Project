#!/bin/bash
# Rung-21 third-attempt watcher (job 30103278, icgpu04 excluded): verify node,
# wait, verify completion, fetch, probe + dome + figure. Completes the arch
# ablation ladder.
cd "$(dirname "$0")/.." || exit 1
SSH="wbunsti1@login.rockfish.jhu.edu"
RP=/data/tmcquee2/wbunsti1/ML_SC_Proj
STATUS=model_data/cf_calib/arch_ablation_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
die()  { note "HALT: $*"; exit 1; }
J=30103278
note "rung 21 attempt 3: job $J (icgpu04 excluded)"

# node check once it starts
for i in $(seq 1 90); do
  state=$(ssh "$SSH" "squeue -j $J -h -o '%T %N' 2>/dev/null" 2>/dev/null)
  case "$state" in
    RUNNING*) note "rung21 running on: ${state#RUNNING }"; break ;;
    "") break ;;
  esac
  sleep 600
done
for i in $(seq 1 200); do
  state=$(ssh "$SSH" "squeue -j $J -h -o '%T' 2>/dev/null" 2>/dev/null)
  [ -z "$state" ] && break
  sleep 600
done
LOG="$RP/logs/gps_21_ablate_no_poly_*-${J}.out"
ssh "$SSH" "grep -q 'Multitask pretraining done' $LOG" || die "rung21 attempt 3: no completion marker"
ssh "$SSH" "grep -oE 'Best val energy MAE: [0-9.]+' $LOG | tail -1" | { read -r m; note "rung21: $m"; }
ssh "$SSH" "grep -E '^>> epoch 99:' $LOG | tail -1" | { read -r l; note "rung21 final: $l"; }
D21=$(ssh "$SSH" "grep -oE 'Run output dir: [^ ]+' $LOG | head -1" | sed 's/Run output dir: //')
./scripts/deploy.sh fetch >> model_data/cf_calib/arch_ablation_fetch.log 2>&1 || die "fetch failed"

run_eval() {
  CKPT=$(ls "$D21"/*model_best.pth.tar 2>/dev/null | head -1)
  [ -n "$CKPT" ] || { note "$2: no checkpoint"; return 0; }
  python3 - "$1" "$CKPT" <<'PYEOF' || { note "$2: patch failed"; return 0; }
import json, sys
cfg = json.load(open(sys.argv[1])); cfg["checkpoint"] = sys.argv[2]
json.dump(cfg, open(sys.argv[1], "w"), indent=2)
PYEOF
  python main.py train-head "$1" > "model_data/cf_calib/${2}.log" 2>&1 \
    || { note "$2: FAILED"; return 0; }
  note "$2: $(grep -oE '3-seed ensemble: .*|7-seed ensemble: .*' "model_data/cf_calib/${2}.log" | tail -1)"
  NAME=$(python3 -c "import json;print(json.load(open('$1'))['name'])")
  RUN=$(ls -dt model_data/*/"${NAME}"_2* 2>/dev/null | head -1)
  if [ "$2" = "dome21abl" ] && [ -f "$RUN/predictions.csv" ]; then
    python scripts/plot_lsco_dome.py "$RUN/predictions.csv" \
      "docs/figures/tc_vs_cu_oxidation_la_series_dome_21abl.png" \
      'La$_2$CuO$_4$ family doping series — arch ablation rung 21' \
      >> "model_data/cf_calib/${2}.log" 2>&1 && note "$2: figure written"
    python scripts/dome_stats.py "$RUN" >> "$STATUS" 2>/dev/null
  fi
}
run_eval configs/head/gps_tc_probe_21abl.json probe21abl
run_eval configs/head/gps_tc_la_series_21abl.json dome21abl
note "ARCH ABLATION LADDER COMPLETE (rung 21 attempt 3)"
