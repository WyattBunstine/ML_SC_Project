#!/bin/bash
# Stream every Materials Cloud batch of the Cerqueira e-ph archive and keep
# a2F.dos6 (the published-value smearing, verified 2026-08-25) + McMillan.dat
# + qe.dyn* (dynamical matrices per q-point — the FULL harmonic phonon data,
# from which dispersions / phonon DOS interpolate; kept for a possible
# phonon-spectrum pretraining target). ~48 GiB streamed, ~10-40 GB kept.
# Resumable per batch via .done markers.
cd "$(dirname "$0")/../../.." || exit 1
OUT=database/datafiles/EPH_Cerqueira/a2f_raw
LOG=database/datafiles/EPH_Cerqueira/a2f_extract.log
mkdir -p "$OUT"
for b in a b c d e f g h i j k l m n o p q; do
  marker="$OUT/.batch-$b.done"
  [ -f "$marker" ] && continue
  echo "$(date '+%m-%d %H:%M') batch-$b start" >> "$LOG"
  ok=""
  for attempt in 1 2; do
    if curl -sL --retry 2 "https://archive.materialscloud.org/records/3kbt5-r3n56/files/batch-$b.tar.bz2?download=1" \
        | tar -xj -C "$OUT" --wildcards '*/a2F.dos6' '*/McMillan.dat' '*/qe.dyn*' 2>> "$LOG"; then
      ok=1; touch "$marker"; break
    fi
    echo "$(date '+%m-%d %H:%M') batch-$b attempt $attempt FAILED" >> "$LOG"
  done
  n=$(find "$OUT/batch-$b" -name a2F.dos6 2>/dev/null | wc -l)
  echo "$(date '+%m-%d %H:%M') batch-$b ${ok:+done} ($n spectra)" >> "$LOG"
done
total=$(find "$OUT" -name a2F.dos6 | wc -l)
echo "$(date '+%m-%d %H:%M') EXTRACTION COMPLETE: $total spectra across $(ls "$OUT" | grep -c batch)" >> "$LOG"
