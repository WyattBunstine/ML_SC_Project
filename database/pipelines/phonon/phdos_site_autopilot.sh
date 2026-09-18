#!/bin/bash
# Site-projected phonon DOS pack (Togo + MP-pheasy), user-directed 2026-09-09.
# Runs as ONE systemd user unit (memory-capped) alongside a separate fetch-fc
# unit: site-togo (CPU) runs while fetch-fc (network) pulls pheasy force
# constants; then site-pheasy -> bake -> pack -> ship to cluster scratch via deploy.sh sync-datafiles.
# Every stage is checked by its own completion line (overnight-autonomy rule).
#   systemd-run --user ... bash -c './database/pipelines/phonon/phdos_site_autopilot.sh >> model_data/cf_calib/phdos_site_autopilot.out 2>&1'
cd "$(dirname "$0")/../../.." || exit 1
STATUS=model_data/cf_calib/phdos_site_status.txt
FCLOG=model_data/cf_calib/phdos_site_fetchfc.out
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
die()  { note "HALT: $*"; exit 1; }
note "phdos-site autopilot started (Togo + pheasy site-projected DOS -> phdos_pack_v45_site)"

python database/pipelines/phonon/build_phdos_site_corpus.py site-togo --workers 4
tail -60 model_data/cf_calib/phdos_site_autopilot.out | grep -q "site-togo done" || die "site-togo exited without its summary"
note "site-togo: $(ls database/datafiles/TogoPhononDB/site_raw | wc -l) site files"

# wait for the fetch-fc unit (its own summary line proves completion)
for i in $(seq 1 720); do
  grep -q "fetch-fc done" "$FCLOG" 2>/dev/null && break
  pgrep -f "build_phdos_site_corpus.py fetch-fc" > /dev/null || { sleep 60; grep -q "fetch-fc done" "$FCLOG" 2>/dev/null && break; die "fetch-fc process gone without its summary"; }
  sleep 60
done
grep -q "fetch-fc done" "$FCLOG" || die "fetch-fc never finished"
note "fetch-fc: $(grep 'fetch-fc done' "$FCLOG" | tail -1) — $(ls database/datafiles/MP_PhononDOS/fc_pheasy | wc -l) fc files"

python database/pipelines/phonon/build_phdos_site_corpus.py site-pheasy --workers 4
tail -60 model_data/cf_calib/phdos_site_autopilot.out | grep -q "site-pheasy done" || die "site-pheasy exited without its summary"
note "site-pheasy: $(ls database/datafiles/MP_PhononDOS/site_raw | wc -l) site files"

python database/pipelines/phonon/build_phdos_site_corpus.py bake || die "bake failed"
note "bake: $(grep '^bake:' model_data/cf_calib/phdos_site_autopilot.out | tail -1)"
python database/pipelines/phonon/build_phdos_site_corpus.py report >> model_data/cf_calib/phdos_site_autopilot.out 2>&1

OUT=database/datafiles/MP_PhononDOS/phdos_pack_v45_site
[ -d "$OUT" ] && mv "$OUT" "${OUT}_old_$(date +%m%d-%H%M)"
python main.py pack-dataset --index database/datafiles/MP_PhononDOS/PHDOS_index.pickle --out "$OUT" || die "pack failed"
grep -q '"has_phdos_site": *true' "$OUT/pack_header.json" || die "pack header has_phdos_site != true"
note "PACK COMPLETE: $(du -sh $OUT | cut -f1) — $(python -c "import json;h=json.load(open('$OUT/pack_header.json'));print({k:h[k] for k in ('n_samples','has_ph_dos','has_phdos_site')})")"

./scripts/deploy.sh sync-datafiles "$OUT" || die "sync failed"
note "SYNCED to cluster scratch (deploy.sh sync-datafiles) — PHDOS-SITE AUTOPILOT COMPLETE"
