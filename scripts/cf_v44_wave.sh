#!/bin/bash
# v4.4 wave (CF_SCHEMA 3: + bond-valence block): rebuild the three local graph
# sets, pack under NEW _v44 names (rung 15.2 is TRAINING on the _cf/_cf2 packs
# — never touch a pack under a live reader), verify, sync. The cluster MPtrj
# --force re-augment (schema 3 -> packed_v44) is submitted at the end; it runs
# on the CPU partition concurrently with rung 15.2's GPU training.
cd "$(dirname "$0")/.." || exit 1
LOG=model_data/cf_calib/v44_wave.log
{
echo "=== 1. SC graphs (v4.4) ===" && \
rm -rf database/datafiles/MP/graphs_v4_doped_v44 database/datafiles/MP/SC_MP_V4_doped_v44.pickle database/datafiles/MP/SC_MP_V4_doped_v44.csv && \
python main.py build-db --kind cgv4 \
  --source database/datafiles/MP/id_prop.csv database/datafiles/MP/cifs/ \
  --output database/datafiles/MP/SC_MP_V4_doped_v44 \
  --graph-dir database/datafiles/MP/graphs_v4_doped_v44 \
  --oxidation-parent-csv database/datafiles/MP/3DSC_MP.csv && \
echo "=== 2. disorder graphs (v4.4) ===" && \
rm -rf database/datafiles/MP/disorder_corpus/graphs_v44 database/datafiles/MP/disorder_corpus/disorder_index_v44.pickle database/datafiles/MP/disorder_corpus/disorder_index_v44.csv && \
python main.py build-db --kind cgv4 --has-header \
  --source database/datafiles/MP/disorder_corpus/source.csv database/datafiles/MP/disorder_corpus/cifs/ \
  --output database/datafiles/MP/disorder_corpus/disorder_index_v44 \
  --graph-dir database/datafiles/MP/disorder_corpus/graphs_v44 && \
echo "=== 3. DOS graphs (v4.4: re-augment the schema-2 rebuild IN a copy) ===" && \
python scripts/augment_cf.py --graph-dir database/datafiles/MP/dos_rebuild/graphs_v4_cf --force && \
echo "=== 4. verify + pack (_v44 names) ===" && \
CUDA_VISIBLE_DEVICES="" python scripts/cf_v44_verify_pack.py && \
echo "=== 5. sync ===" && \
rsync -a database/datafiles/MP/SC_MP_V4_doped_v44.pickle database/datafiles/MP/SC_MP_V4_doped_v44.csv \
  wbunsti1@login.rockfish.jhu.edu:/data/tmcquee2/wbunsti1/ML_SC_Proj/database/datafiles/MP/ && \
rsync -a --delete database/datafiles/MP/graphs_v4_doped_v44/ \
  wbunsti1@login.rockfish.jhu.edu:/data/tmcquee2/wbunsti1/ML_SC_Proj/database/datafiles/MP/graphs_v4_doped_v44/ && \
rsync -a database/datafiles/MP/SC_pack_doped_v44/ \
  wbunsti1@login.rockfish.jhu.edu:/scratch4/tmcquee2/wbunsti1/ML_SC_Proj/MP/SC_pack_doped_v44/ && \
rsync -a database/datafiles/MP/disorder_pack_v44/ \
  wbunsti1@login.rockfish.jhu.edu:/scratch4/tmcquee2/wbunsti1/ML_SC_Proj/MP/disorder_pack_v44/ && \
rsync -a database/datafiles/MP/dos_pack_ef1_v44/ \
  wbunsti1@login.rockfish.jhu.edu:/scratch4/tmcquee2/wbunsti1/ML_SC_Proj/MP/dos_pack_ef1_v44/ && \
echo "=== V44 LOCAL WAVE OK ==="
} > "$LOG" 2>&1
tail -3 "$LOG"
