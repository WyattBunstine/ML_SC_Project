#!/bin/bash
# Dome autopilot: after the probe autopilot finishes, run the La2CuO4-family
# doping-series ("dome") FineTune evaluations for the rung 15.2 (CF) and
# rung 16 (CF+BVS) encoders, and regenerate the dome figures.
#
# Design notes (overnight-autonomy pattern):
#  - waits for scripts/probe_autopilot.sh to reach a terminal state (COMPLETE
#    or HALT) before touching the GPU — everything stays strictly serialized
#    on the single local GPU
#  - reuses the probe autopilot's fetched+verified checkpoints by reading them
#    out of the (patched) probe configs; a 2026-07-29 date guard means a HALT
#    before patching can never make us dome the stale 15.0 checkpoint
#  - dome protocol is gps_tc_v4_la_series_both.json byte-identical except
#    checkpoint/index (clean encoder attribution); 7 seeds, parent_comp split,
#    la_series_both holdout
#  - per-rung failures note-and-continue so one bad dome can't kill the other
cd "$(dirname "$0")/.." || exit 1
STATUS=model_data/cf_calib/dome_autopilot_status.txt
PSTAT=model_data/cf_calib/probe_autopilot_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
die()  { note "HALT: $*"; exit 1; }
: > "$STATUS"
note "dome autopilot started (waiting on probe autopilot)"

for i in $(seq 1 132); do   # up to 22h
  grep -qE "PROBE AUTOPILOT COMPLETE|HALT:" "$PSTAT" 2>/dev/null && break
  sleep 600
done
grep -qE "PROBE AUTOPILOT COMPLETE|HALT:" "$PSTAT" 2>/dev/null \
  || die "probe autopilot not terminal after 22h"
note "probe autopilot terminal: $(tail -1 "$PSTAT")"

run_dome() {  # $1=probe config  $2=dome config  $3=tag
  CKPT=$(python3 -c "import json;print(json.load(open('$1'))['checkpoint'])" 2>/dev/null)
  case "$CKPT" in
    *2026-07-29*) : ;;   # must be a freshly fetched rung 15.2 / 16 checkpoint
    *) note "$3: SKIP — probe config not patched with a 07-29 checkpoint ($CKPT)"; return 0 ;;
  esac
  [ -f "$CKPT" ] || { note "$3: SKIP — checkpoint file missing: $CKPT"; return 0; }

  python3 - "$2" "$CKPT" <<'PYEOF' || { note "$3: dome-config patch failed"; return 0; }
import json, sys
cfg = json.load(open(sys.argv[1]))
cfg["checkpoint"] = sys.argv[2]
json.dump(cfg, open(sys.argv[1], "w"), indent=2)
PYEOF
  note "$3: dome run launching with $CKPT"
  python main.py train-head "$2" > "model_data/cf_calib/${3}_dome.log" 2>&1 \
    || { note "$3: dome run FAILED (see ${3}_dome.log)"; return 0; }

  NAME=$(python3 -c "import json;print(json.load(open('$2'))['name'])")
  RUN=$(ls -dt model_data/*/"${NAME}"_2* 2>/dev/null | head -1)
  if [ -z "$RUN" ] || [ ! -f "$RUN/predictions.csv" ]; then
    note "$3: dome finished but no predictions.csv found"; return 0
  fi
  FIG="docs/figures/tc_vs_cu_oxidation_la_series_dome_${3}.png"
  if python scripts/plot_lsco_dome.py "$RUN/predictions.csv" "$FIG" \
       >> "model_data/cf_calib/${3}_dome.log" 2>&1; then
    note "$3: DOME DONE -> $RUN ; figure $FIG"
  else
    note "$3: dome ran ($RUN) but plot failed — regenerate manually"
  fi
}

run_dome configs/head/gps_tc_probe_15cf.json    configs/head/gps_tc_v4_la_series_15cf.json    15cf
run_dome configs/head/gps_tc_probe_16cfbvs.json configs/head/gps_tc_v4_la_series_16cfbvs.json 16cfbvs
note "DOME AUTOPILOT COMPLETE"
