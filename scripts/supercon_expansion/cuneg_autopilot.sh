#!/bin/bash
# Cu-oxide negatives chain: pool -> match-dope -> graphs -> merge (V9).
cd "$(dirname "$0")/../.." || exit 1
B="python scripts/supercon_expansion/build_cuprate_negatives.py"
LOG=model_data/cf_calib/cuneg.out
STATUS=model_data/cf_calib/cuneg_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
die()  { note "HALT: $*"; exit 1; }
note "cuneg autopilot started (653 Cu-bearing magnetic compositions -> tc=0 negatives)"
$B pool;        tail -40 "$LOG" | grep -q "pool done" || die "pool"
note "pool: $(grep 'pool done' $LOG | tail -1)"
$B match-dope;  tail -60 "$LOG" | grep -q "match-dope DONE" || die "match-dope"
note "match-dope: $(grep 'match-dope DONE' $LOG | tail -1)"
$B graphs;      tail -40 "$LOG" | grep -q "graphs DONE" || die "graphs"
note "graphs: $(grep 'graphs DONE' $LOG | tail -1)"
$B merge >> "$LOG" 2>&1 || die "merge"
note "MERGE: $(grep '^V9:' $LOG | tail -1)"
note "$(grep 'Cu+O family' $LOG | tail -1)"
note "CUNEG AUTOPILOT COMPLETE"
