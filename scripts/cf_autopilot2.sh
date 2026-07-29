#!/bin/bash
# Autopilot 2 (v4.4/rung-16 leg) — single pre-approved script, no agent tool
# calls on the unattended path (see overnight-autonomy memory):
#   A. wait for the schema-2 MPtrj augment (job 29405728) to finish so the CPU
#      queue slot frees and rung 15.2's launch (autopilot 1) is unaffected
#   B. submit the v4.4 MPtrj re-augment (--force, schema 3) -> packed_v44
#   C. wait + verify (0 failed, has_bvs header)
#   D. wait for the rung 15.2 GPU job to COMPLETE (it holds the gpu quota),
#      then submit rung 16 (16_cf_bvs.json) and confirm startup
# Status: model_data/cf_calib/autopilot2_status.txt. Deliberate halts only.
cd "$(dirname "$0")/.." || exit 1
SSH="wbunsti1@login.rockfish.jhu.edu"
RP=/data/tmcquee2/wbunsti1/ML_SC_Proj
STATUS=model_data/cf_calib/autopilot2_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
die()  { note "HALT: $*"; exit 1; }
: > "$STATUS"
note "autopilot2 started"

# A: schema-2 augment out of the queue
for i in $(seq 1 60); do
  state=$(ssh "$SSH" "squeue -j 29405728 -h -o '%T' 2>/dev/null" 2>/dev/null)
  [ -z "$state" ] && break
  sleep 600
done
note "schema-2 augment done"

# B: v4.4 MPtrj re-augment -> packed_v44
OUT=$(AUGMENT_FORCE=1 ./scripts/deploy.sh augment-cf /scratch4/tmcquee2/wbunsti1/ML_SC_Proj/MPtrj/packed_v44 2>&1)
AJID=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
[ -n "$AJID" ] || die "v4.4 augment submission failed: $(echo "$OUT" | tail -2)"
note "v4.4 MPtrj augment submitted: job $AJID"

# C: completion + verification
for i in $(seq 1 96); do
  state=$(ssh "$SSH" "squeue -j $AJID -h -o '%T' 2>/dev/null" 2>/dev/null)
  [ -z "$state" ] && break
  sleep 600
done
AUG_LOG="$RP/logs/augment_cf_*-${AJID}.out"
ssh "$SSH" "grep -qE 'DONE in .*, 0 failed' $AUG_LOG" || die "v4.4 augment reported failures (job $AJID)"
ssh "$SSH" "python3 -c \"
import json
h = json.load(open('/scratch4/tmcquee2/wbunsti1/ML_SC_Proj/MPtrj/packed_v44/pack_header.json'))
assert h['n_samples'] == 1578895 and h['has_cf'] and h['has_bvs'], h
print('v44 header OK')\"" || die "packed_v44 header verification failed"
note "packed_v44 verified"

# D: wait for rung 15.2 to finish (read its job id from autopilot 1's status),
# then launch rung 16
R15=$(grep -oE 'rung 15.2 submitted: job [0-9]+' model_data/cf_calib/autopilot_status.txt | grep -oE '[0-9]+' | tail -1)
if [ -n "$R15" ]; then
  for i in $(seq 1 200); do   # up to ~33h (queue + ~10h train)
    state=$(ssh "$SSH" "squeue -j $R15 -h -o '%T' 2>/dev/null" 2>/dev/null)
    [ -z "$state" ] && break
    sleep 600
  done
  note "rung 15.2 (job $R15) finished"
else
  note "WARN: rung 15.2 job id not found in autopilot status; launching rung 16 anyway"
fi
OUT=$(./scripts/deploy.sh run configs/gps_mt_ablation_suite/16_cf_bvs.json 2>&1)
R16=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
[ -n "$R16" ] || die "rung 16 submission failed: $(echo "$OUT" | tail -2)"
note "rung 16 submitted: job $R16"
for i in $(seq 1 96); do
  state=$(ssh "$SSH" "squeue -j $R16 -h -o '%T' 2>/dev/null" 2>/dev/null)
  if [ "$state" = "RUNNING" ]; then
    sleep 300
    ssh "$SSH" "tail -n 30 $RP/logs/gps_16_cf_bvs_*-${R16}.out 2>/dev/null | grep -E 'Model:|split_by|Error|Traceback'" >> "$STATUS" 2>/dev/null
    ssh "$SSH" "tail -n 40 $RP/logs/gps_16_cf_bvs_*-${R16}.out 2>/dev/null | grep -qiE 'Traceback|Error'" \
      && die "rung 16 startup shows errors (job $R16)"
    note "rung 16 RUNNING clean (job $R16) — AUTOPILOT2 COMPLETE"
    exit 0
  fi
  [ -z "$state" ] && die "rung 16 left the queue before starting (job $R16)"
  sleep 600
done
die "rung 16 still queued after 16h (job $R16) — submitted, just slow"
