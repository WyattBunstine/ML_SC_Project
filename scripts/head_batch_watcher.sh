#!/bin/bash
# Wait for a set of Rockfish head-batch jobs, fetch, then score everything
# locally (family_stats for probes/nickelates, dome_stats + figure for domes).
#   ./scripts/head_batch_watcher.sh <jobid> [jobid ...]
cd "$(dirname "$0")/.." || exit 1
SSH="wbunsti1@login.rockfish.jhu.edu"
STATUS=model_data/cf_calib/head_batch_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
note "head-batch watcher: waiting on $*"
for J in "$@"; do
  for i in $(seq 1 432); do
    state=$(ssh "$SSH" "squeue -j $J -h -o '%T' 2>/dev/null" 2>/dev/null)
    rc=$?
    [ "$rc" -ne 255 ] && [ -z "$state" ] && break
    sleep 300
  done
  note "job $J left the queue"
done
./scripts/deploy.sh fetch >> model_data/cf_calib/head_batch_fetch.log 2>&1 || { note "HALT: fetch failed"; exit 1; }
note "fetch done — scoring"
# Score list: override with HEAD_BATCH_NAMES="name1 name2 ..." (run-name suffixes
# after gps_tc_); defaults to the 2026-08-26 wave.
NAMES="${HEAD_BATCH_NAMES:-la_series_scratch_e nickelate_nopre_cpoly nickelate_scratch_e nopre_ka_comp \
            probe_28tg la_series_28tg probe_29tg la_series_29tg probe_30tg la_series_30tg \
            probe_31tg la_series_31tg probe_32tg la_series_32tg probe_33tg la_series_33tg \
            probe_34tg la_series_34tg probe_35tg la_series_35tg}"
for NAME in $NAMES; do
  RUN=$(ls -dt model_data/*/gps_tc_${NAME}_2* 2>/dev/null | head -1)
  [ -z "$RUN" ] || [ ! -f "$RUN/metrics.json" ] && { note "$NAME: MISSING"; continue; }
  note "$NAME: $(python3 -c "import json;m=json.load(open('$RUN/metrics.json'));print('MAE %.2f pos %.2f msle %.3f'%(m['head']['overall']['mae_K'],m['mae_tc_pos_K'],m['msle']))" 2>/dev/null)"
  case $NAME in
    la_series*)
      python scripts/dome_stats.py "$RUN" >> "$STATUS" 2>/dev/null
      python scripts/plot_lsco_dome.py "$RUN/predictions.csv" \
        "docs/figures/tc_vs_cu_oxidation_la_series_dome_${NAME#la_series_}.png" \
        "La\$_2\$CuO\$_4\$ doping series — ${NAME}" >/dev/null 2>&1 ;;
    *) python scripts/family_stats.py "$RUN" 2>/dev/null | tail -1 >> "$STATUS" ;;
  esac
done
note "HEAD-BATCH WATCHER COMPLETE"
