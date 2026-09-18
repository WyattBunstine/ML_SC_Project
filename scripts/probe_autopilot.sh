#!/bin/bash
# Probe autopilot: when the rung 15.2 (CF) and rung 16 (CF+BVS) pretrainings
# finish on Rockfish, fetch the checkpoints and run the local FineTune Tc
# probes, unattended.  Pattern per the overnight-autonomy lesson: ONE
# pre-approved self-contained script, no agent tool calls on the unattended
# path, explicit chaining (set -e is unreliable under the tool wrapper), and
# any verification failure is a deliberate HALT recorded in the status file.
#
# Sequence (all state -> model_data/cf_calib/probe_autopilot_status.txt):
#   A. wait for job 29416943 (rung 15.2) to leave the queue
#   B. verify the training log (epoch-100 marker, no Traceback), parse run dir
#   C. deploy.sh fetch -> local checkpoint; patch gps_tc_probe_15cf.json
#   D. python main.py train-head (3-seed FineTune probe, local GPU); record metrics
#   E-H. same for job 29421607 (rung 16) with gps_tc_probe_16cfbvs.json
# FineTune probes encode graphs live from the checkpoint (embed_dir unused),
# so no embedding-cache staleness is possible here.
cd "$(dirname "$0")/.." || exit 1
SSH="wbunsti1@login.rockfish.jhu.edu"
RP=/data/tmcquee2/wbunsti1/ML_SC_Proj
STATUS=model_data/cf_calib/probe_autopilot_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
die()  { note "HALT: $*"; exit 1; }
: > "$STATUS"
note "probe autopilot started (jobs 29416943 -> 29421607)"

wait_job() {  # $1=jid  $2=max 600s polls
  for i in $(seq 1 "$2"); do
    state=$(ssh "$SSH" "squeue -j $1 -h -o '%T' 2>/dev/null" 2>/dev/null)
    [ -z "$state" ] && return 0
    sleep 600
  done
  return 1
}

run_probe() {  # $1=jid  $2=remote log glob  $3=probe config  $4=tag
  wait_job "$1" 72 || die "$4: job $1 still in queue after 12h"
  LOG="$RP/logs/$2"
  ssh "$SSH" "grep -q 'Multitask pretraining done' $LOG" || die "$4: no completion marker in $2"
  ssh "$SSH" "grep -qi 'Traceback' $LOG" && die "$4: Traceback in training log $2"
  RUNDIR=$(ssh "$SSH" "grep -oE 'Run output dir: [^ ]+' $LOG | head -1" | sed 's/Run output dir: //')
  [ -n "$RUNDIR" ] || die "$4: could not parse run dir from $2"
  note "$4: training complete, run dir $RUNDIR"

  ./scripts/deploy.sh fetch >> model_data/cf_calib/probe_fetch.log 2>&1 \
    || die "$4: deploy.sh fetch failed (see probe_fetch.log)"
  CKPT=$(ls "$RUNDIR"/*model_best.pth.tar 2>/dev/null | head -1)
  [ -n "$CKPT" ] || die "$4: no *model_best.pth.tar under $RUNDIR after fetch"

  python3 - "$3" "$CKPT" <<'PYEOF' || die "$4: probe-config patch failed"
import json, sys
cfg = json.load(open(sys.argv[1]))
cfg["checkpoint"] = sys.argv[2]
json.dump(cfg, open(sys.argv[1], "w"), indent=2)
print("patched", sys.argv[1], "->", sys.argv[2])
PYEOF
  note "$4: launching FineTune probe with $CKPT"
  python main.py train-head "$3" > "model_data/cf_calib/${4}_probe.log" 2>&1 \
    || die "$4: probe run failed (see model_data/cf_calib/${4}_probe.log)"

  NAME=$(python3 -c "import json;print(json.load(open('$3'))['name'])")
  RUN=$(ls -dt model_data/*/"${NAME}"_2* 2>/dev/null | head -1)
  [ -n "$RUN" ] && [ -f "$RUN/metrics.json" ] || die "$4: probe finished but no metrics.json found"
  note "$4: PROBE DONE -> $RUN"
  note "$4: metrics: $(python3 -c "import json;print(json.dumps(json.load(open('$RUN/metrics.json'))))" | cut -c1-800)"
}

run_probe 29416943 "gps_15_cf_features_*-29416943.out" configs/head/gps_tc_probe_15cf.json    rung15cf
run_probe 29421607 "gps_16_cf_bvs_*-29421607.out"      configs/head/gps_tc_probe_16cfbvs.json rung16cfbvs
note "PROBE AUTOPILOT COMPLETE — compare vs champion 4.26 / 5.33 / 0.847 / 15.23"
