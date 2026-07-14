# ⚠️ Archived: figures from scrambled-split runs (pre-2026-07-14)

Every figure here was generated from FineTune predictions produced BEFORE the
positional split fix (`48f099d`): CIFDataV4 seed-shuffles rows at load, but
loader positions were computed in index order, so actual train/test membership
was chance-level relative to the intended parent-grouped split. In particular,
the "held-out LSCO family" premise behind these dome plots did not hold — LSCO
variants were partly trained on, and the apparent dome amplitude is
leak-inflated.

Valid replacement: `../tc_vs_cu_oxidation_mp1077929_dome.png`, regenerated from
the true 191-row Cu1La2 family holdout (run gps_tc_v4_lsco_holdout_2026-07-14):
the dome SHAPE reproduces zero-shot (bin-mean r=0.875 vs formal Cu oxidation,
optimum at the correct doping) but with strongly compressed amplitude (~1 K
predicted swing vs ~25 K true) — the scale, not the shape, was the leaked part.
