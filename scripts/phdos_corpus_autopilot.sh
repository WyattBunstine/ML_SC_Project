#!/bin/bash
# Unattended chain for the multi-source phonon-DOS corpus (tasks #8/#9):
# wait for the togo DOS processing + MP structure fetch already running, then
# build graphs -> bake ph_dos -> pack. One pre-approved autopilot per the
# overnight protocol; never trust set -e under the tool wrapper — explicit
# checks instead.
#   nohup setsid ./scripts/phdos_corpus_autopilot.sh > <log> 2>&1 &
cd "$(dirname "$0")/.." || exit 1
STATUS=model_data/cf_calib/phdos_corpus_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }

note "phdos autopilot: waiting on togo processing + structure fetch"
for i in $(seq 1 720); do
  pgrep -f "process_togo_phonondb|build_phdos_corpus.py fetch-structures" > /dev/null || break
  sleep 60
done
N_TOGO=$(ls database/datafiles/TogoPhononDB/dos_raw | wc -l)
note "prereqs done: togo dos_raw $N_TOGO"

python scripts/build_phdos_corpus.py build
if [ $? -ne 0 ] || [ ! -f database/datafiles/MP_PhononDOS/PHDOS_index.pickle ]; then
  note "HALT: build failed"; exit 1
fi
note "build done: $(python -c "import pandas as pd; d=pd.read_pickle('database/datafiles/MP_PhononDOS/PHDOS_index.pickle'); print(len(d),'rows', d['phdos_source'].value_counts().to_dict())")"

python scripts/build_phdos_corpus.py bake
if [ $? -ne 0 ]; then note "HALT: bake failed"; exit 1; fi
note "bake done"

python main.py pack-dataset --index database/datafiles/MP_PhononDOS/PHDOS_index.pickle \
    --out database/datafiles/MP_PhononDOS/phdos_pack_v45
if [ $? -ne 0 ]; then note "HALT: pack failed"; exit 1; fi
note "PACK COMPLETE: $(du -sh database/datafiles/MP_PhononDOS/phdos_pack_v45 | cut -f1)"
