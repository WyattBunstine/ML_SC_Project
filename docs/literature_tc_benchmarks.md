# T_c Prediction — Literature Benchmarks

Reference list of published machine-learning models for superconducting critical
temperature (T_c) regression, kept so we don't refetch. Captured 2026-06-30.

**Read the comparability notes before ranking anything against our numbers.** Different
datasets (3DSC vs SuperCon), splits (random vs grouped), and metrics (MAE vs RMSE vs MSLE)
make raw numbers misleading. MSLE is a log/relative metric dominated by *low*-T_c materials;
MAE in Kelvin is dominated by *high*-T_c cuprates. A model can win one and lose the other.

---

## Our benchmarks (for reference — 3DSC_MP, parent-grouped leak-free split, 1154-row test)

| model | MAE (K) | RMSE (K) | R² | MSLE | cuprate MAE (K) | notes |
|---|---|---|---|---|---|---|
| GPS encoder, **fine-tuned** (top block, 7-seed) | **4.44** | 10.41 | 0.706 | 0.848 | 14.73 | our best |
| GPS encoder, frozen + head | 5.68 | — | — | — | 23.4 | best frozen |
| ORIG CGCNN from scratch (leak-free) | 5.92 | 16.13 | 0.173 | 1.396 | 24.81 | single seed |
| ORIG CGCNN (random/leaky split) | 4.04 | — | — | — | ~24.8 | leakage-inflated, NOT comparable |

Split = parent-grouped (doped variants of one MP parent kept together), seed 123, 70/10/20.
Training loss = L1 in log1p(K) space; reported MAE is in Kelvin.

---

## Literature

### Most directly comparable (3DSC dataset — same as ours)

**3DSC — a dataset of superconductors including crystal structures**
- Authors / year: Sommer, Willa, Schmalian, Friederich — 2023
- Venue: *Scientific Data* (Nature)
- DOI: [10.1038/s41597-023-02721-y](https://doi.org/10.1038/s41597-023-02721-y) · arXiv: [2212.06071](https://arxiv.org/abs/2212.06071) · [PMC10663493](https://pmc.ncbi.nlm.nih.gov/articles/PMC10663493)
- Dataset: **3DSC_MP** (5,759 SuperCon entries ↔ 5,773 Materials Project structures — *this is our dataset*); also 3DSC_ICSD (9,150 ↔ 86,490 ICSD structures)
- Model: **XGBoost** on MAGPIE (composition) + **DSOAP** (disordered SOAP, structure) features
- Metric: **MSLE = 0.748 ± 0.010** (test, structure features) on 3DSC_MP; 0.776 ± 0.010 chemical-formula-only. They do NOT report MAE/RMSE. Illustrative translation: MSLE 0.748 ≈ abs error 1.16 / 6.37 / 58.47 K at T_c = 1 / 10 / 100 K.
- Split: **grouped by chemical system** (Meredig et al. style — all materials sharing a chemical system are entirely in train or entirely in test), 80:20, **100 repetitions**. This is leak-free and STRICTER than our parent-grouping.
- Relevance: the canonical 3DSC benchmark. On MSLE their XGBoost (0.748) is **better than our FT (0.848)** — on a harder split. But it's a log-error metric (rewards low-T_c relative accuracy) and a composition+SOAP gradient-boost, not a structure GNN. They report no MAE, so no absolute-K head-to-head.

**ALIGNN on 3DSC** — ⚠️ SOURCE UNIDENTIFIED (from a search summary; NOT the 3DSC paper itself, which used XGBoost. The HuggingFace `shreyaspullehf/supervision-alignn-tc-prediction` repo was checked and is UNRELATED. Exact source still not found.)
- Reported: MAE 5.34 K, RMSE 10.27 K, R² 0.7186 — fine-tuned ALIGNN on 3D crystal-structure graphs
- **Corroboration:** our FT encoder on 3DSC_MP gets RMSE 10.41, R² 0.706 — almost identical to this ALIGNN (10.27, 0.72). So structure GNNs on 3DSC cluster around RMSE ≈ 10.3, R² ≈ 0.71, which makes the 5.34/10.27/0.72 figure plausible as a real 3DSC benchmark. Our FT MAE 4.44 < their 5.34 (but verify their split before claiming the win).
- TODO: confirm the exact source/paper + split.

### Highest reported (but different dataset + unverifiable split — treat with caution)

**Crystal structure graph neural networks for high-performance superconducting critical temperature prediction**
- Authors / year: J. Zhang, X. Lin, K. Hu et al. (HIT Shenzhen) — 2024
- Venue: *Science China Materials* 67, 3253–3261
- DOI: [10.1007/s40843-024-3026-8](https://doi.org/10.1007/s40843-024-3026-8)
- Dataset: **their own SuperCon→ICSD matched set, ~5,713 entries** (NOT 3DSC — corroborated by the same first author's 2026 ACS Omega review). ICSD also used as a screening pool (found 76 candidates with T_c ≥ 77 K).
- Model: a "crystal structure graph neural network" — **specific architecture name not confirmed** (paywalled; not stated to be CGCNN/MEGNet/ALIGNN).
- Metrics: **R² = 0.962, RMSE = 6.192 K** (CONFIRMED from publisher summary); claims to "outperform all previously reported models." **MAE not reported.**
- Split: **UNKNOWN** — paywalled, no accessible source states random vs grouped, and no leakage discussion found.
- ⚠️ **Caution:** this is the best headline number found, but (a) different dataset (SuperCon→ICSD, not 3DSC), (b) split method unverifiable, (c) **R² 0.962 is a large outlier** vs the R² ≈ 0.71 cluster that every honest 3DSC structure-GNN (ALIGNN, our FT) lands in — strongly suggesting an easier dataset and/or a non-grouped (leaky) split. Not a fair comparison to our parent-grouped 3DSC numbers without the methods section. (The "92.9%/86.3%" accuracy figures seen in some snippets are search-engine confabulation — disregard.)

### Our project lineage (SuperCon — different, larger dataset)

**Identifying New Classes of High Temperature Superconductors With Convolutional Neural Networks**
- Authors / year: Quinn, McQueen — 2022 (the "MQCNN" / cnn_supercon work; McQueen = JHU)
- Venue: *Frontiers in Electronic Materials*
- DOI: [10.3389/femat.2022.893797](https://doi.org/10.3389/femat.2022.893797)
- Dataset: **SuperCon** (~33,000 entries), crystal structures assigned by correlation with Materials Project + structural DBs
- Model: CNN (image-like representation), both classification and regression
- Metrics: classification accuracy > 95%; **regression R² > 0.92, MAE ≈ 5.6 K**
- Split: not clearly specified (likely random) → possible leakage; not stated.
- Relevance: our project's lineage. Apples-to-oranges with our 3DSC number — different dataset (SuperCon, ~6× larger), composition-image CNN, different/unstated split.

### Other (pending precise numbers from the broad sweep — URLs captured)

- **Stanev et al. 2018**, *npj Computational Materials* — SuperCon, random forest on composition. Commonly cited RMSE ≈ 9.5 K (VERIFY). DOI: 10.1038/s41524-018-0085-8
- **Konno et al. 2021** — deep learning on periodic-table representation of composition (SuperCon). (VERIFY metric)
- **"Predicting the critical temperature of superconductors ... with a balanced dataset"**, *J. Appl. Phys.* 2026 — has an RMSE/MAE/R² cross-validation table. [pubs.aip.org/.../3377287](https://pubs.aip.org/aip/jap/article/139/2/023903/3377287)
- **"Predicting superconducting transition temperature through advanced ML and feature engineering"**, *Scientific Reports* 2024 — [10.1038/s41598-024-54440-y](https://doi.org/10.1038/s41598-024-54440-y) (likely SuperCon, composition)
- **"Accelerating superconductor discovery through tempered deep learning of the electron-phonon spectral function"**, arXiv [2401.16611](https://arxiv.org/abs/2401.16611) (2024) — predicts e-ph spectral function α²F → relevant to our Phase-3 λ/ω_log idea
- **Closed-loop superconducting materials discovery**, *npj Comp Mater* 2023 — [10.1038/s41524-023-01131-3](https://doi.org/10.1038/s41524-023-01131-3)
- **Data-Driven Superconductivity: a Review of ML Methods**, *J. Supercond. Nov. Magn.* 2026 — [10.1007/s10948-026-07175-y](https://doi.org/10.1007/s10948-026-07175-y) (survey — good for a numbers table)
- **"Learning Superconductivity" benchmark**, NeurIPS 2024 (datasets/benchmarks track) — a curated T_c benchmark with (reportedly) leakage-aware splits; notably does NOT cite Zhang 2024. Worth pulling for standardized splits + baseline numbers. (VERIFY exact title/metrics)
- **Zhang et al. review**, *ACS Omega* 2026, 11(22) 31853 — [10.1021/acsomega.6c01100](https://doi.org/10.1021/acsomega.6c01100) — "ML for Superconductor Discovery" survey; first-authored by Zhang (self-describes the 5,713 SuperCon→ICSD set above)

---

## Comparability cheatsheet

- **Metric**: MSLE (3DSC paper) ≠ MAE (us, Quinn) ≠ RMSE (most SuperCon work). RMSE ≈ 1.5–2× MAE for these skewed distributions. MSLE rewards low-T_c relative accuracy; MAE rewards high-T_c absolute accuracy (cuprates).
- **Dataset**: 3DSC_MP (5.8k, structures) is ours and the 3DSC paper's. SuperCon (~33k, composition ± assigned structures) is Quinn/Stanev/most others — larger, composition-driven, often easier on MAE.
- **Split**: random splits LEAK on these datasets (duplicate/doped near-siblings) — worth ≈ 1.9 K of false optimism for us (4.04 leaky → 5.92 leak-free, ORIG). The 3DSC paper's chemical-system grouping is the gold standard; our parent-grouping is leak-free but slightly less strict. Always check the split before trusting a low number.
- **To make the XGBoost comparison airtight**: re-evaluate our FT with chemical-system grouping + report MSLE (and ideally 100 split reps).
