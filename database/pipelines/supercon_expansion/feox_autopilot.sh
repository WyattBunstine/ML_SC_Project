#!/bin/bash
cd "$(dirname "$0")/../../.." || exit 1
N="python database/pipelines/supercon_expansion/build_feox_negatives.py"; L=model_data/cf_calib/feox.out; S=model_data/cf_calib/feox_status.txt
note(){ echo "$(date '+%m-%d %H:%M') $*" >> $S; }
note "feox autopilot started"
$N pool;       tail -40 $L | grep -q "pool done" || { note "HALT pool"; exit 1; }; note "pool: $(grep 'pool done' $L | tail -1 | cut -c1-60)"
$N match-dope; tail -60 $L | grep -q "match-dope DONE" || { note "HALT match-dope"; exit 1; }; note "match-dope: $(grep 'match-dope DONE' $L | tail -1)"
$N graphs;     tail -40 $L | grep -q "graphs DONE" || { note "HALT graphs"; exit 1; }; note "graphs: $(grep 'graphs DONE' $L | tail -1 | cut -c1-60)"
$N merge >> $L 2>&1 || { note "HALT merge"; exit 1; }; note "$(grep '^V11:' $L | tail -1)"; note "$(grep 'Fe-oxide rows' $L | tail -1)"
note "FEOX AUTOPILOT COMPLETE"
