#!/bin/bash
# Rung-51 chain: the 47-wide recipe + the global ph_dos spectrum task, fed by the
# 31,097-row multi-source phonon-DOS pack (dfpt > togo > pheasy; member weight 4
# ~ the a2f pack's epoch share) on top of the Cerqueira ph_dos already in the eph
# pack. Submit pretrain -> verify -> patch the 3 eval configs (msle probe, l1k
# probe, La-series dome; 47wide twins) -> run-head -> watcher.
#   systemd-run --user ... bash -c './scripts/phdos_51_autopilot.sh >> model_data/cf_calib/phdos_51_autopilot.out 2>&1' 
cd "$(dirname "$0")/.." || exit 1
SSH="wbunsti1@login.rockfish.jhu.edu"
RP=/data/tmcquee2/wbunsti1/ML_SC_Proj
STATUS=model_data/cf_calib/phdos_51_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
die()  { note "HALT: $*"; exit 1; }
: > "$STATUS"
note "phdos-51 autopilot started (rung 51: 47-wide recipe + global ph_dos task on the 31k multi-source phonon-DOS pack, member weight 4)"

ARM="51_phdos_wide"
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
CKPT="${RDIR}/result_t51_model_best.pth.tar"
ssh "$SSH" "[ -f '${RP}/${CKPT}' ]" || die "checkpoint missing remotely (${CKPT})"

for HC in configs/head/gps_tc_probe_51phdos.json configs/head/gps_tc_probe_51phdos_l1k.json \
          configs/head/gps_tc_la_series_51phdos.json; do
  python3 - "$HC" "$CKPT" <<'PYEOF' || die "patch failed ($HC)"
import json, sys
cfg = json.load(open(sys.argv[1])); cfg["checkpoint"] = sys.argv[2]
json.dump(cfg, open(sys.argv[1], "w"), indent=2)
PYEOF
done
note "eval configs patched with ${CKPT}"

HJ=""
OUT=$(./scripts/deploy.sh run-head configs/head/gps_tc_probe_51phdos.json \
      configs/head/gps_tc_probe_51phdos_l1k.json configs/head/gps_tc_la_series_51phdos.json 2>&1)
J1=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
[ -n "$J1" ] && { note "head batch: job $J1"; HJ="$HJ $J1"; }
[ -n "$HJ" ] || die "no head batches submitted"

note "handing off to head_batch_watcher:${HJ}"
HEAD_BATCH_NAMES="probe_51phdos probe_51phdos_l1k la_series_51phdos" \
  ./scripts/head_batch_watcher.sh ${HJ}
note "PHDOS-51 AUTOPILOT COMPLETE — references: 47wide probe 4.38 (l1k) / dome .549 0.87K 21.4K MAE 5.09; 49site probe 4.26 / dome .364 10.2K"
