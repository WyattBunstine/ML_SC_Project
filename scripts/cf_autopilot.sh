#!/bin/bash
# Self-contained overnight autopilot: carries the CF schema-2 pipeline from
# "local wave running" all the way to "rung 15.2 running on the cluster"
# WITHOUT requiring any further agent tool calls (each agent wake-up can hit a
# permission prompt; this script needs exactly one approval, now).
#
# Stages (all state written to model_data/cf_calib/autopilot_status.txt):
#   A. wait for the local wave2 chain to submit the MPtrj augment job
#   B. wait for the augment job to finish; verify 0 failures + pack header
#   C. launch rung 15.2 (deploy.sh run 15_cf_features.json)
#   D. wait for it to reach RUNNING; capture startup + param count
# Any verification failure stops the pipeline and records WHY in the status
# file (deliberate halt, never a hang; nothing interactive anywhere).
cd "$(dirname "$0")/.." || exit 1
SSH="wbunsti1@login.rockfish.jhu.edu"
RP=/data/tmcquee2/wbunsti1/ML_SC_Proj
STATUS=model_data/cf_calib/autopilot_status.txt
WAVE_LOG=model_data/cf_calib/schema2_wave2.log
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
die()  { note "HALT: $*"; exit 1; }
: > "$STATUS"
note "autopilot started"

# --- A: local wave -> augment job id --------------------------------------
for i in $(seq 1 240); do   # up to 4h
  grep -q "=== WAVE SUBMITTED OK ===" "$WAVE_LOG" 2>/dev/null && break
  if [ -s "$WAVE_LOG" ] && ! tail -1 "$WAVE_LOG" | grep -qE '.' ; then :; fi
  # chain died? (log stops growing AND last stage marker present w/o OK)
  sleep 60
done
grep -q "=== WAVE SUBMITTED OK ===" "$WAVE_LOG" || die "local wave never submitted the cluster job (see $WAVE_LOG)"
AUG_JID=$(grep -oE 'Submitted batch job [0-9]+' "$WAVE_LOG" | tail -1 | grep -oE '[0-9]+')
[ -n "$AUG_JID" ] || die "could not parse augment job id"
note "local wave OK; augment job $AUG_JID"

# --- B: augment job completes + verification ------------------------------
for i in $(seq 1 96); do    # up to 16h (queue + 2.5h run)
  state=$(ssh "$SSH" "squeue -j $AUG_JID -h -o '%T' 2>/dev/null" 2>/dev/null)
  [ -z "$state" ] && break
  sleep 600
done
AUG_LOG="$RP/logs/augment_cf_*-${AUG_JID}.out"
ssh "$SSH" "grep -qE 'DONE in .*, 0 failed' $AUG_LOG" || die "augment reported failures (job $AUG_JID)"
ssh "$SSH" "grep -qE 'Pack done: 1,?578,?895 samples \(0 failed\)' $AUG_LOG" || die "MPtrj repack incomplete"
ssh "$SSH" "python3 -c \"
import json
h = json.load(open('/scratch4/tmcquee2/wbunsti1/ML_SC_Proj/MPtrj/packed_v4_cf/pack_header.json'))
assert h['n_samples'] == 1578895 and h['has_cf'] and h['has_valence_baked'], h
print('header OK')\"" || die "packed_v4_cf header verification failed"
note "augment + repack verified (job $AUG_JID)"

# --- C: launch rung 15.2 ---------------------------------------------------
OUT=$(./scripts/deploy.sh run configs/gps_mt_ablation_suite/15_cf_features.json 2>&1)
R15_JID=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
[ -n "$R15_JID" ] || die "rung 15 submission failed: $(echo "$OUT" | tail -2)"
note "rung 15.2 submitted: job $R15_JID"

# --- D: confirm startup ----------------------------------------------------
for i in $(seq 1 96); do    # up to 16h queue
  state=$(ssh "$SSH" "squeue -j $R15_JID -h -o '%T' 2>/dev/null" 2>/dev/null)
  if [ "$state" = "RUNNING" ]; then
    sleep 300
    R15_LOG="$RP/logs/gps_15_cf_features_*-${R15_JID}.out"
    ssh "$SSH" "tail -n 30 $R15_LOG 2>/dev/null | grep -E 'Model:|split_by|Error|Traceback'" >> "$STATUS" 2>/dev/null
    if ssh "$SSH" "tail -n 40 $R15_LOG 2>/dev/null | grep -qiE 'Traceback|Error'"; then
      die "rung 15.2 startup shows errors (job $R15_JID)"
    fi
    note "rung 15.2 RUNNING clean (job $R15_JID) — AUTOPILOT COMPLETE"
    exit 0
  fi
  [ -z "$state" ] && die "rung 15.2 left the queue before starting (job $R15_JID)"
  sleep 600
done
die "rung 15.2 still queued after 16h (job $R15_JID) — still submitted, just slow"
