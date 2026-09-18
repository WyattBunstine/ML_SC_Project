#!/bin/bash
# Re-match the audited-out rows with the gated builder: expansion first, then the
# Cu negatives (same matcher, same rules), then graphs for both.
cd "$(dirname "$0")/../../.." || exit 1
B="python database/pipelines/supercon_expansion/build_supercon_v7.py"; N="python database/pipelines/supercon_expansion/build_cuprate_negatives.py"
S=model_data/cf_calib/rebuild_status.txt; L=model_data/cf_calib/rebuild.out
note(){ echo "$(date '+%m-%d %H:%M') $*" >> $S; }
note "rebuild-after-audit started"
$B match-dope;  tail -60 $L | grep -q "match-dope DONE" || { note "HALT v7 match-dope"; exit 1; }
note "v7: $(grep 'match-dope DONE' $L | tail -1)"
$N match-dope;  tail -60 $L | grep -q "match-dope DONE" || { note "HALT cuneg match-dope"; exit 1; }
note "cuneg: $(grep 'match-dope DONE' $L | tail -1)"
$B graphs;      tail -40 $L | grep -q "graphs DONE" || { note "HALT v7 graphs"; exit 1; }
note "v7 graphs: $(grep 'graphs DONE' $L | tail -1)"
$N graphs;      tail -40 $L | grep -q "graphs DONE" || { note "HALT cuneg graphs"; exit 1; }
note "cuneg graphs: $(grep 'graphs DONE' $L | tail -1)"
note "REBUILD COMPLETE"
