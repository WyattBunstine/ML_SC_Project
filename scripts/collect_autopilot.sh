#!/bin/bash
# Collector: gather the four CF/BVS evaluation runs now in flight —
#   local:  gps_tc_probe_15cf (DONE), gps_tc_probe_16cfbvs (running)
#   remote: la_series_15cf dome (job 29459015), la_series_16cfbvs dome (job 29459022)
# Waits for everything, fetches remote runs, records all metrics in one status
# file, and renders both dome figures. Overnight-autonomy pattern: single
# pre-approved script, deliberate halts only, nothing interactive.
cd "$(dirname "$0")/.." || exit 1
SSH="wbunsti1@login.rockfish.jhu.edu"
STATUS=model_data/cf_calib/collect_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
: > "$STATUS"
note "collector started (domes: 29459015, 29459022)"

# A: local rung-16 probe (metrics.json appears in its run dir)
for i in $(seq 1 36); do   # up to 3h
  RUN16=$(ls -dt model_data/*/gps_tc_probe_16cfbvs_2* 2>/dev/null | head -1)
  [ -n "$RUN16" ] && [ -f "$RUN16/metrics.json" ] && break
  sleep 300
done
if [ -n "$RUN16" ] && [ -f "$RUN16/metrics.json" ]; then
  note "probe16cfbvs: $(grep -oE '3-seed ensemble: .*' model_data/cf_calib/rung16cfbvs_probe.log 2>/dev/null | tail -1)"
  note "probe16cfbvs run dir: $RUN16"
else
  note "WARN: local rung-16 probe not finished after 3h — continuing to domes"
fi

# B: remote dome jobs
for jid in 29459015 29459022; do
  for i in $(seq 1 72); do   # up to 12h each (queue + run)
    state=$(ssh "$SSH" "squeue -j $jid -h -o '%T' 2>/dev/null" 2>/dev/null)
    [ -z "$state" ] && break
    sleep 600
  done
  note "dome job $jid left the queue"
done

./scripts/deploy.sh fetch >> model_data/cf_calib/collect_fetch.log 2>&1 \
  || { note "HALT: fetch failed (see collect_fetch.log)"; exit 1; }

# C: dome figures
for tag in 15cf 16cfbvs; do
  RUN=$(ls -dt model_data/*/gps_tc_v4_la_series_${tag}_2* 2>/dev/null | head -1)
  if [ -z "$RUN" ] || [ ! -f "$RUN/predictions.csv" ]; then
    note "WARN: ${tag} dome run/predictions.csv not found after fetch"; continue
  fi
  FIG="docs/figures/tc_vs_cu_oxidation_la_series_dome_${tag}.png"
  if python scripts/plot_lsco_dome.py "$RUN/predictions.csv" "$FIG" \
       >> "model_data/cf_calib/${tag}_domeplot.log" 2>&1; then
    note "${tag} dome: $RUN -> $FIG"
  else
    note "WARN: ${tag} dome plot failed (run at $RUN)"
  fi
  [ -f "$RUN/metrics.json" ] && \
    note "${tag} dome metrics: $(python3 -c "import json;print(json.dumps(json.load(open('$RUN/metrics.json'))))" | cut -c1-500)"
done
note "COLLECTOR COMPLETE"
