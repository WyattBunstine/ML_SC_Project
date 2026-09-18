#!/bin/bash
# Loss-weight ablation wave (rungs 36-40) on the rung-35 base (all six tasks +
# eph_a2f, n_phonon 256, v45 packs, CF+BVS, valence off) — weights are the ONLY
# variable; rung 35 (all 1.0) is the equal-weight control, rung 25 (forces x2,
# no a2f) the no-phonon forces cell.
#   36 forces x2 | 37 a2f x2 | 38 magmom x2 | 39 dos x2 | 40 forces x2 + a2f x4
# Chain: submit 5 pretrains -> verify each -> patch its 3 head configs (probe
# msle, probe l1_k, la_series l1) with the new checkpoint -> submit as one
# run-head job per rung -> hand the 5 head jobs to head_batch_watcher.sh
# (fetch + family_stats/dome_stats/figures into head_batch_status.txt).
cd "$(dirname "$0")/.." || exit 1
SSH="wbunsti1@login.rockfish.jhu.edu"
RP=/data/tmcquee2/wbunsti1/ML_SC_Proj
STATUS=model_data/cf_calib/weight_ablation_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
die()  { note "HALT: $*"; exit 1; }
: > "$STATUS"
note "weight ablation autopilot started (rungs 36-40 on rung-35 base)"

ARMS="36_w_forces2 37_w_a2f2 38_w_magmom2 39_w_dos2 40_w_forces2_a2f4"
declare -A JID
for ARM in $ARMS; do
  OUT=$(SLURM_EXCLUDE=icgpu04 ./scripts/deploy.sh run "configs/gps_mt_ablation_suite/${ARM}.json" 2>&1)
  J=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
  [ -n "$J" ] || { note "${ARM} submission FAILED"; continue; }
  JID[$ARM]=$J; note "${ARM} submitted: job $J"
done
[ ${#JID[@]} -gt 0 ] || die "no pretrain submissions succeeded"

wait_verify() {
  # 432 x 600s = 72h. Break only when ssh succeeded (exit != 255) AND squeue
  # printed nothing — a transient ssh outage must not look like completion.
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

HEAD_JOBS=""
HEAD_NAMES=""
for ARM in $ARMS; do
  [ -n "${JID[$ARM]}" ] || continue
  N="${ARM%%_*}"
  RDIR=$(wait_verify "${JID[$ARM]}" "gps_${ARM}_*-${JID[$ARM]}.out" "$ARM")
  note "${ARM} run dir: ${RDIR:-FAILED}"
  [ -n "$RDIR" ] || continue
  CKPT="${RDIR}/result_t${N}_model_best.pth.tar"
  ssh "$SSH" "[ -f '${RP}/${CKPT}' ]" || { note "${ARM}: checkpoint missing remotely (${CKPT})"; continue; }
  CFGS=""
  for HC in "configs/head/gps_tc_probe_${N}wt.json" \
            "configs/head/gps_tc_probe_${N}wt_l1k.json" \
            "configs/head/gps_tc_la_series_${N}wt.json"; do
    python3 - "$HC" "$CKPT" <<'PYEOF' || { note "${ARM}: patch failed ($HC)"; continue 2; }
import json, sys
cfg = json.load(open(sys.argv[1])); cfg["checkpoint"] = sys.argv[2]
json.dump(cfg, open(sys.argv[1], "w"), indent=2)
PYEOF
    CFGS="${CFGS} ${HC}"
  done
  OUT=$(./scripts/deploy.sh run-head ${CFGS} 2>&1)
  HJ=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
  [ -n "$HJ" ] || { note "${ARM}: head batch submission FAILED"; continue; }
  note "${ARM}: head batch job $HJ (probe_${N}wt probe_${N}wt_l1k la_series_${N}wt)"
  HEAD_JOBS="${HEAD_JOBS} ${HJ}"
  HEAD_NAMES="${HEAD_NAMES} probe_${N}wt probe_${N}wt_l1k la_series_${N}wt"
done
[ -n "$HEAD_JOBS" ] || die "no head batches submitted"

note "handing off to head_batch_watcher:${HEAD_JOBS}"
HEAD_BATCH_NAMES="${HEAD_NAMES}" ./scripts/head_batch_watcher.sh ${HEAD_JOBS}
note "WEIGHT ABLATION AUTOPILOT COMPLETE — controls: 35 equal-w (probe 4.64/l1k 4.23/dome r_lsco .492), 25 forces-x2-no-a2f (l1k 4.36), 30 all-six (l1k 4.13)"
