# No-pretrain ladder — raw features through the Tc head (2026-08-25)

Probe protocol: 3-seed, chemsys split, msle loss, deepsets pooling, 41 descriptors alongside.
NOTE: reported MAE columns are ABSOLUTE KELVIN, but the training objective is msle = MSE in
log1p-standardized z-space (relative-error emphasis; log-space seed ensembling) — identical for
every row, so deltas are apples-to-apples; MSLE is the column closest to the optimized objective.
Family columns are positives-only MAE (K); cuprate n=107, Fe-based n=66, heavy-fermion n=36, conventional n=450.
Rungs A-C+poly: identity encoder (raw standardized node features ARE the per-atom latents, ZERO encoder params).
Rung D: 967k-param GPS (bonds, no angle/poly) trained from scratch on Tc only. References: pretrained fine-tune probes.

| rung / input | features | MAE (K) | SC-only | MSLE | r | cuprate | Fe-based | heavy-f | convent. |
|---|---|---|---|---|---|---|---|---|---|
| A  composition only | 14 (geo masked) | 5.55 | 7.42 | 0.982 | 0.673 | 28.4 | 8.1 | 1.3 | 3.2 |
| B  + valence subshells | 18 | 5.31 | 7.17 | 0.885 | 0.805 | 29.8 | 8.5 | 1.4 | 2.4 |
| C  + geometry / CF / BVS | 32 | 5.22 | 7.06 | 0.859 | 0.8 | 28.6 | 7.6 | 1.5 | 2.7 |
| D  C + GPS from scratch | encoder 967k | 5.08 | 6.99 | 0.856 | 0.843 | 28.8 | 7.9 | 1.3 | 2.5 |
| C+poly  + poly summaries | 40 | 4.98 | 6.51 | 0.894 | 0.787 | 25.8 | 7.4 | 1.6 | 2.5 |
| rung 20  full pretrained | pretrained | 4.94 | 6.52 | 0.845 | 0.829 | 26.1 | 7.7 | 1.2 | 2.4 |
| rung 25  dome champion | pretrained | 4.58 | 5.94 | 0.868 | 0.826 | 22.4 | 7.5 | 1.3 | 2.4 |
| rung 12  broad champion | pretrained+dis | 4.26 | 5.33 | 0.847 | 0.863 | 19.2 | 7.0 | 1.2 | 2.3 |
