#!/bin/bash
# SuperCon expansion chain (user-directed 2026-09-10): pool -> match-dope ->
# graphs -> descriptors -> report. One systemd user unit per the overnight rule;
# every stage verified by its OWN completion line, never by exit code alone.
#   systemd-run --user ... bash -c './scripts/supercon_expansion/supercon_v7_autopilot.sh >> model_data/cf_calib/supercon_v7.out 2>&1'
cd "$(dirname "$0")/../.." || exit 1
B="python scripts/supercon_expansion/build_supercon_v7.py"
LOG=model_data/cf_calib/supercon_v7.out
STATUS=model_data/cf_calib/supercon_v7_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
die()  { note "HALT: $*"; exit 1; }
note "supercon-v7 autopilot started (9,810 un-covered SuperCon compositions)"

$B pool
tail -40 "$LOG" | grep -q "pool done" || die "pool exited without its summary"
note "pool: $(grep 'pool done' $LOG | tail -1)"

$B match-dope
tail -60 "$LOG" | grep -q "match-dope DONE" || die "match-dope exited without its summary"
note "match-dope: $(grep 'match-dope DONE' $LOG | tail -1)"

$B graphs
tail -40 "$LOG" | grep -q "graphs DONE" || die "graphs exited without its summary"
note "graphs: $(grep 'graphs DONE' $LOG | tail -1)"

python - <<'PY' || die "descriptors failed"
import sys; sys.path.insert(0, "models/head")
from descriptors import build_descriptor_table
build_descriptor_table("database/datafiles/MP/SC_MP_V7_supercon.pickle",
                       "database/datafiles/MP/descriptors_v7_supercon.pickle")
PY
note "descriptors built"

$B report >> "$LOG" 2>&1
note "SUPERCON-V7 AUTOPILOT COMPLETE — $(python -c "import pandas as pd; d=pd.read_pickle('database/datafiles/MP/SC_MP_V7_supercon.pickle'); print(len(d),'entries')")"
