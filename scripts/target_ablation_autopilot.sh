#!/bin/bash
# Pretraining-target ablation ladder on the MODERN stack (rung-25 recipe:
# no-poly, CF+BVS baked, valence block off, oxidation scalar on, v45 packs,
# no disorder corpus) — one pre-approved chain.
#
# Design (2026-08-24): the 08-19 target x SC-class matrix ran on June's old
# 14-dim checkpoints; this ladder re-asks the question on the dome-champion
# recipe with BOTH June-era confounds removed:
#   - EQUAL loss weights everywhere (rung-04 lesson: forces-x2 was mis-crowned)
#   - FIXED data: both packs (packed_v45 + dos_pack_ef1_v45) in every rung,
#     only the task list varies (June's ladder added the DOS pack with the DOS
#     task, conflating data and target)
# Build-up: 27 energy / 28 +forces,stress / 29 +magmom,bandgap / 30 all six.
# Leave-one-out off 30: 31 -magmom / 32 -bandgap / 33 -forces,stress
# (29 doubles as -dos). Rung 25's existing checkpoint = the forces-x2 all-six
# cell for free. Watch: magmom's cuprate value on a CF-bearing base (input
# already carries cf_unpaired, the AOM magmom).
# References — rung 25: probe 4.58/5.94/0.868/16.85, dome 0.823/1.33K/22.9/5.30,
# family cup 22.4 / fer 7.5 / other 2.4. Old-stack matrix best: rung 04
# (all six equal) 4.49 / cup 21.3.
cd "$(dirname "$0")/.." || exit 1
SSH="wbunsti1@login.rockfish.jhu.edu"
RP=/data/tmcquee2/wbunsti1/ML_SC_Proj
STATUS=model_data/cf_calib/target_ablation_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
die()  { note "HALT: $*"; exit 1; }
: > "$STATUS"
note "target ablation autopilot started (rungs 27-33 on rung-25 base, icgpu04 excluded)"

ARMS="27_t_energy 28_t_forces_stress 29_t_no_dos 30_t_all_equal 31_t_no_magmom 32_t_no_bandgap 33_t_no_forces"
declare -A JID
for ARM in $ARMS; do
  OUT=$(SLURM_EXCLUDE=icgpu04 ./scripts/deploy.sh run "configs/gps_mt_ablation_suite/${ARM}.json" 2>&1)
  J=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
  [ -n "$J" ] || { note "${ARM} submission FAILED"; continue; }
  JID[$ARM]=$J; note "${ARM} submitted: job $J"
done
[ ${#JID[@]} -gt 0 ] || die "no submissions succeeded"

wait_verify() {
  # 432 x 600s = 72h, matching the SLURM time limit (24h was too short for a
  # 7-job wave's queue wait). Break only when ssh itself succeeded (exit != 255,
  # ssh's transport-failure code) AND squeue printed nothing — a transient ssh
  # outage must not be mistaken for job completion (fetch-while-running).
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
  ssh "$SSH" "grep -E '^>> epoch 99:' $LOG | tail -1" | { read -r l; note "$3 final: $l"; }
  ssh "$SSH" "grep -oE 'Run output dir: [^ ]+' $LOG | head -1" | sed 's/Run output dir: //'
}
declare -A RDIR
for ARM in $ARMS; do
  [ -n "${JID[$ARM]}" ] || continue
  RDIR[$ARM]=$(wait_verify "${JID[$ARM]}" "gps_${ARM}_*-${JID[$ARM]}.out" "$ARM")
  note "${ARM} run dir: ${RDIR[$ARM]:-FAILED}"
done
./scripts/deploy.sh fetch >> model_data/cf_calib/target_ablation_fetch.log 2>&1 || die "fetch failed"

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
  # per-family matrix line — the deliverable of this ladder
  [ -f "$RUN/predictions.csv" ] && python scripts/family_stats.py "$RUN" | tail -1 >> "$STATUS" 2>/dev/null
  if [ -n "$4" ] && [ -f "$RUN/predictions.csv" ]; then
    python scripts/plot_lsco_dome.py "$RUN/predictions.csv" \
      "docs/figures/tc_vs_cu_oxidation_la_series_dome_${4}.png" "$5" \
      >> "model_data/cf_calib/${3}.log" 2>&1 && note "$3: figure dome_${4}.png"
    python scripts/dome_stats.py "$RUN" >> "$STATUS" 2>/dev/null
  fi
}
for N in 27 28 29 30 31 32 33; do
  KEY=$(ls configs/gps_mt_ablation_suite/ | grep "^${N}_t_" | sed 's/.json//')
  D="${RDIR[$KEY]}"
  run_eval "configs/head/gps_tc_probe_${N}tg.json"     "$D" "probe${N}tg" "" ""
  run_eval "configs/head/gps_tc_la_series_${N}tg.json" "$D" "dome${N}tg" "${N}tg" \
    "La\$_2\$CuO\$_4\$ family doping series — pretraining-target rung ${N}"
done
note "TARGET ABLATION AUTOPILOT COMPLETE — vs rung 25 (4.58/5.94/0.868/16.85; 0.823/1.33K/22.9/5.30)"
