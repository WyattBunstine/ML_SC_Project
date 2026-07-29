#!/bin/bash
# Schema-2 wave, steps 2-6 (step 1 = DOS rebuild completed with 14 investigated
# + excluded failures: 12 always-masked dos-less rows, 2 no-bonding-edge
# structures — index dos_all_index_cf.pickle is uniform at 62,464 rows).
cd "$(dirname "$0")/.." || exit 1
LOG=model_data/cf_calib/schema2_wave2.log
{
echo "=== 2. SC graphs ===" && \
rm -rf database/datafiles/MP/graphs_v4_doped_cf database/datafiles/MP/SC_MP_V4_doped_cf.pickle database/datafiles/MP/SC_MP_V4_doped_cf.csv && \
python main.py build-db --kind cgv4 \
  --source database/datafiles/MP/id_prop.csv database/datafiles/MP/cifs/ \
  --output database/datafiles/MP/SC_MP_V4_doped_cf \
  --graph-dir database/datafiles/MP/graphs_v4_doped_cf \
  --oxidation-parent-csv database/datafiles/MP/3DSC_MP.csv && \
echo "=== 3. disorder graphs ===" && \
rm -rf database/datafiles/MP/disorder_corpus/graphs_v4_cf database/datafiles/MP/disorder_corpus/disorder_index_cf.pickle database/datafiles/MP/disorder_corpus/disorder_index_cf.csv && \
python main.py build-db --kind cgv4 --has-header \
  --source database/datafiles/MP/disorder_corpus/source.csv database/datafiles/MP/disorder_corpus/cifs/ \
  --output database/datafiles/MP/disorder_corpus/disorder_index_cf \
  --graph-dir database/datafiles/MP/disorder_corpus/graphs_v4_cf && \
echo "=== 4. verify + repack ===" && \
CUDA_VISIBLE_DEVICES="" python scripts/cf_schema2_verify_pack.py && \
echo "=== 5. sync ===" && \
rsync -a database/datafiles/MP/SC_MP_V4_doped_cf.pickle database/datafiles/MP/SC_MP_V4_doped_cf.csv \
  wbunsti1@login.rockfish.jhu.edu:/data/tmcquee2/wbunsti1/ML_SC_Proj/database/datafiles/MP/ && \
rsync -a --delete database/datafiles/MP/graphs_v4_doped_cf/ \
  wbunsti1@login.rockfish.jhu.edu:/data/tmcquee2/wbunsti1/ML_SC_Proj/database/datafiles/MP/graphs_v4_doped_cf/ && \
rsync -a --delete database/datafiles/MP/SC_pack_doped_cf/ \
  wbunsti1@login.rockfish.jhu.edu:/scratch4/tmcquee2/wbunsti1/ML_SC_Proj/MP/SC_pack_doped_cf/ && \
rsync -a --delete database/datafiles/MP/disorder_pack_cf/ \
  wbunsti1@login.rockfish.jhu.edu:/scratch4/tmcquee2/wbunsti1/ML_SC_Proj/MP/disorder_pack_cf/ && \
rsync -a database/datafiles/MP/dos_pack_ef1_cf2/ \
  wbunsti1@login.rockfish.jhu.edu:/scratch4/tmcquee2/wbunsti1/ML_SC_Proj/MP/dos_pack_ef1_cf2/ && \
ssh wbunsti1@login.rockfish.jhu.edu "rm -rf /scratch4/tmcquee2/wbunsti1/ML_SC_Proj/MP/dos_pack_ef1_cf && echo 'corrupt scratch DOS pack deleted'" && \
mkdir -p database/datafiles/.retired/2026-07-28/MP && \
mv database/datafiles/MP/dos_pack_ef1_cf database/datafiles/.retired/2026-07-28/MP/dos_pack_ef1_cf_CORRUPT && \
echo "=== LOCAL WAVE OK ===" && \
echo "=== 6. cluster MPtrj re-augment (--force, schema 2) ===" && \
AUGMENT_FORCE=1 ./scripts/deploy.sh augment-cf && \
echo "=== WAVE SUBMITTED OK ==="
} > "$LOG" 2>&1
tail -4 "$LOG"
