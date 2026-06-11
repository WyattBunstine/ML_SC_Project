# eform_rebaseline_suite — fresh baselines on the fixed pipeline

The June-8 `formation_energy_options_suite` results predate three changes that make
them incomparable to any new run:

1. **Edge-orientation fix** — directional edge columns (Voronoi/ECoN weights) are now
   center-relative at load time (previously the aggregation weight was the *source's*
   ECoN for ~half of directed edges). Applies retroactively to all graphs (legacy
   graphs orient by the builder's `source <= target` canonicalization) — no rebuild.
2. **`coord_sphere` pruned** — edge feature dim 8 → 7 (it was determined by the two
   ECoN columns, |r| ≈ 0.90). Old checkpoints are shape-incompatible.
3. **New knobs** — `set_transformer` aggregation (idea A) and `poly_fusion: "gate"`.

This suite re-establishes baselines and isolates ONE factor per run, all on
MP_Energy / formation energy with the same backbone (64 atom dim, 64 edge hidden,
3 conv, 128 head, `ecn_weighted`, mean pooling, 500 epochs):

| config | varies | question it answers |
|---|---|---|
| `01_poly_off` | `use_poly_edges: false` | baseline without the polyhedral channel |
| `02_poly_on` | poly on (sum fusion) | clean poly-information gain vs 01 (note: dual conv ≈ +40k params — gain is poly info + capacity) |
| `03_set_transformer` | bond agg → local set-transformer w/ angle bias | does attention-fused atom+edge+angle (idea A) beat the physics-weighted sum at 49k samples? |
| `04_attention` | bond+poly agg → learned scalar attention | retests the June-8 "attention hurt" finding without its pooling/size confounds |
| `05_gate` | `poly_fusion: "gate"` | does gated bond/poly fusion beat the plain sum? (+~49k params for the gates) |

Launch each on cluster with:

    ./scripts/deploy.sh run configs/eform_rebaseline_suite/<config>.json

They are independent — submit all five and they queue in parallel (~2.5 h each on
an a100). Compare runs with `model_data/index.csv` after `./scripts/deploy.sh fetch`,
or `scripts/eval_test.py --test-ids` for a shared holdout.

Interpretation notes:
- 02 vs 01: poly value on the *corrected* features.
- 03 vs 02 and 04 vs 02: learned attention vs physical weighting — at this data
  scale attention may need MPtrj (~1.5M frames) to pay off; treat MP_Energy results
  as a lower bound on its value.
- 05 vs 02: fusion mechanism only. If the gate wins, the poly-token /
  hypergraph upgrades (claude.md roadmap #7) inherit it as the new default.
