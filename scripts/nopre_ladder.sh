#!/bin/bash
# NO-PRETRAIN ladder (2026-08-24): what does each layer of information/processing
# buy on Tc supervision ALONE — no pretraining anywhere; diff vs the pretrained
# rungs isolates the pretraining contribution per step. Local GPU, serial
# (one head job at a time), same Tc-head protocol as the probes (3-seed,
# chemsys split, msle, deepsets pooling, descriptors alongside).
#   a_comp    composition-only node features -> head (identity encoder, 14-dim,
#             geometry cols masked)
#   b_valence + valence subshells (18-dim, still structure-free)
#   c_geom    + geometry scalars: base geo cols + CF + BVS (32-dim; the
#             "structure as descriptors, no message passing" cell)
#   d_bonds   c + GPS layers FROM SCRATCH: bond attention, no angle, no poly
#   e_angle   d + angle bias    (= rung-21 arch, untrained)
#   f_poly    e + poly edges    (= rung-20 arch, untrained)
# From-scratch rungs train the whole encoder (ft_unfreeze all) at the
# pretraining LR 3e-4 — 1e-5 barely moves a random init.
# Pretrained probe references (positives-only cuprate in parens): 25 4.58
# (22.4), 21 4.48 (21.2), 20 4.94 (26.1), 23/n_conv0-full-features 4.88 (24.9).
cd "$(dirname "$0")/.." || exit 1
STATUS=model_data/cf_calib/nopre_ladder_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
: > "$STATUS"
note "no-pretrain ladder started (rungs a-f, local GPU, serial)"

for TAG in a_comp b_valence c_geom d_bonds e_angle f_poly; do
  CFG="configs/head/gps_tc_nopre_${TAG}.json"
  note "${TAG}: training"
  python main.py train-head "$CFG" > "model_data/cf_calib/nopre_${TAG}.log" 2>&1 \
    || { note "${TAG}: FAILED (see nopre_${TAG}.log)"; continue; }
  note "${TAG}: $(grep -oE '[0-9]+-seed ensemble: .*' "model_data/cf_calib/nopre_${TAG}.log" | tail -1)"
  RUN=$(ls -dt model_data/*/gps_tc_nopre_${TAG}_2* 2>/dev/null | head -1)
  [ -f "$RUN/predictions.csv" ] && python scripts/family_stats.py "$RUN" | tail -1 >> "$STATUS" 2>/dev/null
done
note "NO-PRETRAIN LADDER COMPLETE — compare per-rung lines above against the pretrained references in the header"
