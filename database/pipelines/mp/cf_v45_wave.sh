#!/bin/bash
# v4.5 wave (CF_SCHEMA 4: symmetric-BA + role-gated bond valence): rebuild the
# three local graph sets with the FIXED builder, pack under NEW _v45 names
# (v44 packs stay untouched as the pre-fix reference), verify the fixes are
# actually present (cf_v45_verify_pack spot checks), sync to the cluster.
# The cluster MPtrj re-augment (schema 4 -> packed_v45) is submitted by
# v45_autopilot.sh after this wave lands.
cd "$(dirname "$0")/../../.." || exit 1
LOG=model_data/cf_calib/v45_wave.log
{
echo "=== 1. SC graphs (v4.5) ===" && \
rm -rf database/datafiles/MP/graphs_v4_doped_v45 database/datafiles/MP/SC_MP_V4_doped_v45.pickle database/datafiles/MP/SC_MP_V4_doped_v45.csv && \
python main.py build-db --kind cgv4 \
  --source database/datafiles/MP/id_prop.csv database/datafiles/MP/cifs/ \
  --output database/datafiles/MP/SC_MP_V4_doped_v45 \
  --graph-dir database/datafiles/MP/graphs_v4_doped_v45 \
  --oxidation-parent-csv database/datafiles/MP/3DSC_MP.csv && \
echo "=== 2. disorder graphs (v4.5) ===" && \
rm -rf database/datafiles/MP/disorder_corpus/graphs_v45 database/datafiles/MP/disorder_corpus/disorder_index_v45.pickle database/datafiles/MP/disorder_corpus/disorder_index_v45.csv && \
python main.py build-db --kind cgv4 --has-header \
  --source database/datafiles/MP/disorder_corpus/source.csv database/datafiles/MP/disorder_corpus/cifs/ \
  --output database/datafiles/MP/disorder_corpus/disorder_index_v45 \
  --graph-dir database/datafiles/MP/disorder_corpus/graphs_v45 && \
echo "=== 3. DOS graphs (re-augment in place: CF_SCHEMA 3 -> 4) ===" && \
python database/pipelines/mp/augment_cf.py --graph-dir database/datafiles/MP/dos_rebuild/graphs_v4_cf --force && \
echo "=== 4. verify fixes + pack (_v45 names) ===" && \
CUDA_VISIBLE_DEVICES="" python scripts/cf_v45_verify_pack.py && \
echo "=== 5. sync ===" && \
./scripts/deploy.sh sync-datafiles database/datafiles/MP/SC_MP_V4_doped_v45.pickle database/datafiles/MP/SC_MP_V4_doped_v45.csv && \
./scripts/deploy.sh sync-datafiles --delete database/datafiles/MP/graphs_v4_doped_v45 && \
./scripts/deploy.sh sync-datafiles database/datafiles/MP/SC_pack_doped_v45 database/datafiles/MP/disorder_pack_v45 database/datafiles/MP/dos_pack_ef1_v45 && \
echo "=== V45 LOCAL WAVE OK ==="
} > "$LOG" 2>&1
tail -3 "$LOG"
grep -q "V45 LOCAL WAVE OK" "$LOG"
