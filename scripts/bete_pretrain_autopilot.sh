#!/bin/bash
# Rung-42 "BETE" overnight chain: MSE spectral-function loss (spectrum_loss mse,
# the Hennig-group/BETE-NET convention — L1's median-seeking was flattening the
# predicted a2f) + magmom x2 + eph_a2f x2 on the rung-35 base.
# Chain: submit pretrain -> verify -> patch the 6 eval configs (baseline trio +
# G/LG pf arms; the pf job builds the rung-42 pred cache remotely, sequential
# within one job so there is no build race) -> 2 run-head jobs -> watcher.
cd "$(dirname "$0")/.." || exit 1
SSH="wbunsti1@login.rockfish.jhu.edu"
RP=/data/tmcquee2/wbunsti1/ML_SC_Proj
STATUS=model_data/cf_calib/bete_pretrain_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
die()  { note "HALT: $*"; exit 1; }
: > "$STATUS"
note "bete pretrain autopilot started (rung 42: mse spectrum, magmom x2, a2f x2)"

ARM="42_bete_mag2_a2f2"
OUT=$(SLURM_EXCLUDE=icgpu04 ./scripts/deploy.sh run "configs/gps_mt_ablation_suite/${ARM}.json" 2>&1)
J=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
[ -n "$J" ] || die "pretrain submission failed"
note "${ARM} submitted: job $J"

for i in $(seq 1 432); do
  state=$(ssh "$SSH" "squeue -j $J -h -o '%T' 2>/dev/null" 2>/dev/null)
  rc=$?
  [ "$rc" -ne 255 ] && [ -z "$state" ] && break
  sleep 600
done
LOG="$RP/logs/gps_${ARM}_*-${J}.out"
ssh "$SSH" "grep -q 'Multitask pretraining done' $LOG" || die "no completion marker"
ssh "$SSH" "grep -qi 'Traceback' $LOG" && die "Traceback in pretrain log"
ssh "$SSH" "grep -E '^>> epoch 99:' $LOG | tail -1" | { read -r l; note "final: $l"; }
RDIR=$(ssh "$SSH" "grep -oE 'Run output dir: [^ ]+' $LOG | head -1" | sed 's/Run output dir: //')
note "run dir: ${RDIR:-MISSING}"
[ -n "$RDIR" ] || die "no run dir in log"
CKPT="${RDIR}/result_t42_model_best.pth.tar"
ssh "$SSH" "[ -f '${RP}/${CKPT}' ]" || die "checkpoint missing remotely (${CKPT})"

for HC in configs/head/gps_tc_probe_42bete.json configs/head/gps_tc_probe_42bete_l1k.json \
          configs/head/gps_tc_la_series_42bete.json configs/head/gps_tc_probe_42bete_g_l1k.json \
          configs/head/gps_tc_la_series_42bete_g.json configs/head/gps_tc_la_series_42bete_lg.json; do
  python3 - "$HC" "$CKPT" <<'PYEOF' || die "patch failed ($HC)"
import json, sys
cfg = json.load(open(sys.argv[1])); cfg["checkpoint"] = sys.argv[2]
json.dump(cfg, open(sys.argv[1], "w"), indent=2)
PYEOF
done
note "eval configs patched with ${CKPT}"

HJ=""
OUT=$(./scripts/deploy.sh run-head configs/head/gps_tc_probe_42bete.json \
      configs/head/gps_tc_probe_42bete_l1k.json configs/head/gps_tc_la_series_42bete.json 2>&1)
J1=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
[ -n "$J1" ] && { note "baseline head batch: job $J1"; HJ="$HJ $J1"; }
OUT=$(./scripts/deploy.sh run-head configs/head/gps_tc_probe_42bete_g_l1k.json \
      configs/head/gps_tc_la_series_42bete_g.json configs/head/gps_tc_la_series_42bete_lg.json 2>&1)
J2=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
[ -n "$J2" ] && { note "pf head batch: job $J2"; HJ="$HJ $J2"; }
[ -n "$HJ" ] || die "no head batches submitted"

note "handing off to head_batch_watcher:${HJ}"
HEAD_BATCH_NAMES="probe_42bete probe_42bete_l1k la_series_42bete probe_42bete_g_l1k la_series_42bete_g la_series_42bete_lg" \
  ./scripts/head_batch_watcher.sh ${HJ}
note "BETE PRETRAIN AUTOPILOT COMPLETE — references: 38 L (4.65/4.23/dome .515/1.36/27.1), 38pf G dome .728, 38pf lambda quartiles .064/.216/.668"
