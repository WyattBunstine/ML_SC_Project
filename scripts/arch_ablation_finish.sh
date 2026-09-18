#!/bin/bash
# Arch-ablation finisher: rung 21 hit a bad node (icgpu04, ~9700s/epoch,
# data-wait ~5s => hardware, would blow walltime at epoch ~26) — cancel +
# resubmit it; fetch the three landed rungs (22/23/24) and run their evals
# NOW; then wait for the 21 resubmit and eval it on landing.
cd "$(dirname "$0")/.." || exit 1
SSH="wbunsti1@login.rockfish.jhu.edu"
RP=/data/tmcquee2/wbunsti1/ML_SC_Proj
STATUS=model_data/cf_calib/arch_ablation_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
die()  { note "HALT: $*"; exit 1; }
note "finisher: cancel stuck rung 21 (29974001) + resubmit"

ssh "$SSH" "scancel 29974001" 2>/dev/null
OUT=$(./scripts/deploy.sh run configs/gps_mt_ablation_suite/21_ablate_no_poly.json 2>&1)
R21=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
[ -n "$R21" ] || die "rung 21 resubmission failed"
note "rung 21 resubmitted: job $R21"

./scripts/deploy.sh fetch >> model_data/cf_calib/arch_ablation_fetch.log 2>&1 || die "fetch failed"
note "fetch OK (rungs 22/23/24 landed)"

run_eval() {  # $1=config $2=rundir $3=tag $4=dome-suffix $5=title
  [ -d "$2" ] || { note "$3: no run dir $2"; return 0; }
  CKPT=$(ls "$2"/*model_best.pth.tar 2>/dev/null | head -1)
  [ -n "$CKPT" ] || { note "$3: no checkpoint"; return 0; }
  python3 - "$1" "$CKPT" <<'PYEOF' || { note "$3: patch failed"; return 0; }
import json, sys
cfg = json.load(open(sys.argv[1])); cfg["checkpoint"] = sys.argv[2]
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
D22=$(ls -dt model_data/*/gps_mt_22_no_attention/gps_mt_22_no_attention_2* 2>/dev/null | head -1)
D23=$(ls -dt model_data/*/gps_mt_23_raw_atoms/gps_mt_23_raw_atoms_2* 2>/dev/null | head -1)
D24=$(ls -dt model_data/*/gps_mt_24_no_angle/gps_mt_24_no_angle_2* 2>/dev/null | head -1)
for A in "22:$D22" "23:$D23" "24:$D24"; do
  N="${A%%:*}"; D="${A#*:}"
  run_eval "configs/head/gps_tc_probe_${N}abl.json"     "$D" "probe${N}abl" "" ""
  run_eval "configs/head/gps_tc_la_series_${N}abl.json" "$D" "dome${N}abl" "${N}abl" \
    "La\$_2\$CuO\$_4\$ family doping series — arch ablation rung ${N}"
done
note "rungs 22/23/24 evals done — waiting for rung 21 (job $R21)"

for i in $(seq 1 200); do
  state=$(ssh "$SSH" "squeue -j $R21 -h -o '%T' 2>/dev/null" 2>/dev/null)
  [ -z "$state" ] && break
  sleep 600
done
LOG="$RP/logs/gps_21_ablate_no_poly_*-${R21}.out"
ssh "$SSH" "grep -q 'Multitask pretraining done' $LOG" || die "rung 21 resubmit: no completion marker"
ssh "$SSH" "grep -oE 'Best val energy MAE: [0-9.]+' $LOG | tail -1" | { read -r m; note "rung21: $m"; }
ssh "$SSH" "grep -E '^>> epoch 99:' $LOG | tail -1" | { read -r l; note "rung21 final: $l"; }
D21=$(ssh "$SSH" "grep -oE 'Run output dir: [^ ]+' $LOG | head -1" | sed 's/Run output dir: //')
./scripts/deploy.sh fetch >> model_data/cf_calib/arch_ablation_fetch.log 2>&1 || die "fetch2 failed"
run_eval "configs/head/gps_tc_probe_21abl.json"     "$D21" "probe21abl" "" ""
run_eval "configs/head/gps_tc_la_series_21abl.json" "$D21" "dome21abl" "21abl" \
  'La$_2$CuO$_4$ family doping series — arch ablation rung 21'
note "ARCH ABLATION COMPLETE (all four rungs evaluated)"
