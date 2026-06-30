# T_c Prediction — Literature Benchmarks

Reference list of published machine-learning models for superconducting critical
temperature (T_c) regression, kept so we don't refetch. Captured 2026-06-30.

**Read the comparability notes before ranking anything against our numbers.** The single
most important fact: headline numbers across this literature are **not comparable**, because
they differ in (a) dataset (composition-only SuperCon vs structure-augmented 3DSC), (b) metric
(MAE/RMSE in K vs MSLE in log-space vs R²), and (c) — most consequentially — **split protocol**:
a **random** split leaks doped variants of the same parent compound across train/test and
inflates scores; a **grouped** split (by chemical system / parent) is the honest, harder setting.
**Be skeptical of any R² ≳ 0.95 or RMSE ≲ 6 K unless the split is explicitly grouped** — those
are almost always random-split numbers inflated by doped-variant leakage. The split is flagged
for every entry below.

---

## Our benchmarks (3DSC_MP, parent-grouped leak-free split, 1154-row test)

| model | MAE (K) | RMSE (K) | R² | MSLE | cuprate MAE (K) | notes |
|---|---|---|---|---|---|---|
| GPS encoder, **fine-tuned** (top block, 7-seed) | **4.44** | 10.41 | 0.706 | 0.848 | 14.73 | our best (parent-grouped, L1 loss) |
| GPS encoder, FT — **3DSC protocol** (chemsys split + MSLE loss) | 5.47 | 11.67 | — | **0.779** | 20.32 | for direct XGBoost comparison ↓ |
| GPS encoder, frozen + head | 5.68 | — | — | — | 23.4 | best frozen |
| ORIG CGCNN from scratch (leak-free) | 5.92 | 16.13 | 0.173 | 1.396 | 24.81 | single seed |
| ORIG CGCNN (random/leaky split) | 4.04 | — | — | — | ~24.8 | leakage-inflated, NOT comparable |

Split = parent-grouped (doped variants of one MP parent kept together), seed 123, 70/10/20.
Loss = L1 in log1p(K); reported MAE in Kelvin. Our split is leak-free but slightly *less* strict
than the 3DSC paper's chemical-system grouping.

---

## A. Structure-based, on 3DSC (our dataset — the directly comparable group)

**3DSC — a dataset of superconductors including crystal structures** — *the reference protocol*
- Sommer, Willa, Schmalian, Friederich — 2023, *Scientific Data* 10, 816
- DOI [10.1038/s41597-023-02721-y](https://doi.org/10.1038/s41597-023-02721-y) · arXiv [2212.06071](https://arxiv.org/abs/2212.06071) · [PMC10663493](https://pmc.ncbi.nlm.nih.gov/articles/PMC10663493/)
- Dataset: **3DSC_MP** (5,759 SuperCon ↔ 5,773 MP structures — *our dataset*); 3DSC_ICSD (9,150 ↔ 86,490)
- Model: **XGBoost** on MAGPIE + disordered-SOAP (DSOAP)
- Metric: **MSLE only** (no MAE/RMSE/R²). Test MSLE **0.748 ± 0.010** (3DSC_MP, structure) vs 0.776 (composition); 1.085 ± 0.073 (3DSC_ICSD). Illustratively MSLE 0.748 ≈ abs error ~1.2 / 6.4 / 58 K at T_c = 1 / 10 / 100 K. Structure helps mainly cuprates; gains within error for most families.
- Split: ✅ **GROUPED by chemical system** (Meredig-style, all-train or all-test), 80:20, **100 reps**. The gold-standard honest protocol; STRICTER than our parent-grouping.
- vs us: **DIRECT TEST DONE (2026-06-30).** Re-ran our FT under their exact protocol — chemical-system grouped split + MSLE loss, 7-seed: **MSLE 0.779 vs their 0.748** (MAE 5.47, RMSE 11.67, cuprate 20.32). So the XGBoost still edges us by ~4% on the log metric on its own turf. Caveats: ours is a SINGLE chem-system split vs their 100 reps (split variance ~±0.02–0.04, so possibly within noise); and the MSLE objective traded away our absolute-K/cuprate strength (MAE 4.44→5.47, cuprate 14.73→20.32). Net: roughly comparable; XGBoost wins on the low-T_c-dominated log metric, our GNN wins on absolute-K/cuprate MAE (which they don't report). Strong composition+SOAP gradient boost = recurring "composition is a hard baseline" lesson.

**SuperVision-ALIGNN** (this IS the source of the "ALIGNN on 3DSC" figure)
- HuggingFace `shreyaspullehf/supervision-alignn-tc-prediction`
- Dataset: 3DSC_MP (5,773); Model: ALIGNN
- Metrics: **MAE 5.34 K, RMSE 10.27 K, R² 0.719**
- Split: ⚠️ **random 70/15/15** (leaky)
- vs us: our FT RMSE 10.41 / R² 0.706 ≈ ALIGNN's 10.27 / 0.719 — but **ALIGNN is on a random (leaky) split and ours is leak-free**, so matching it on a harder task means we're effectively stronger; and our MAE 4.44 < their 5.34.

**Electronegativity-informed CGCNN (mCGCNN-EΔEN)**
- ACS *Inorg. Chem.*, DOI [10.1021/acs.inorgchem.6c01169](https://doi.org/10.1021/acs.inorgchem.6c01169)
- Dataset: 3DSC; Metrics: **RMSE 8.02 K, R² 0.824**
- Split: ⚠️ **not confirmed** (likely random — better RMSE/R² than ALIGNN/ours; verify before trusting)

**SOAP-descriptor model** — *JPCC* 2022, DOI [10.1021/acs.jpcc.2c01904](https://doi.org/10.1021/acs.jpcc.2c01904)
- Dataset: 5,713 structure-matched compounds; Metric: **R² 0.929** (SOAP) vs 0.863 (no structure); R² only
- Split: ⚠️ cross-validated, **effectively NOT grouped-by-system** (so leaky-ish)

## B. Composition-only (SuperCon — different, usually larger dataset)

**Stanev et al. 2018** — *the canonical baseline*
- npj *Comput. Mater.* 4, 29; DOI [10.1038/s41524-018-0085-8](https://doi.org/10.1038/s41524-018-0085-8) · arXiv [1709.02727](https://arxiv.org/abs/1709.02727)
- Dataset: ~16,400 SuperCon (~12,400 finite T_c; ~5,700 cuprates); Model: Random Forest on MAGPIE+AFLOW
- Metric: predicts **ln(T_c)** for T_c > 10 K. **R² ≈ 0.88** overall (low-T_c 0.85, cuprate <0.8, Fe-based 0.74). ⚠️ **Reports NO RMSE/MAE in Kelvin** — any "Stanev K-error" is not from the paper.
- Split: ⚠️ random 85/15 (+ RF out-of-bag)

**Roter & Dordevic 2020**
- *Physica C* 575, 1353689; arXiv [2002.07266](https://arxiv.org/abs/2002.07266)
- Dataset: ~30,000 SuperCon (+~3,000 non-SC); Model: SVD/PCA element vectors + Bagged Tree
- Metrics: **R² ≈ 0.93, RMSE ≈ 8.91 K** (all ~30k); Split: ⚠️ random
- 📌 Notable: estimates **~20% of SuperCon entries are mislabeled** — a field-wide data-quality ceiling and an argument for cleaning.

**Quinn & McQueen 2022** — *our project lineage (MQCNN / cnn_supercon; McQueen = JHU)*
- *Front. Electron. Mater.*; DOI [10.3389/femat.2022.893797](https://doi.org/10.3389/femat.2022.893797)
- Dataset: >10,000 SuperCon superconductors (regression); Model: CNN
- Metrics: combined **R² ≈ 0.92, MAE = 5.6 K** (per-class MAE 1.6–8.0, R² 0.82–0.89)
- Split: ⚠️ random 75/10/15

**Konno et al. 2021** — PRB 103, 014509 ("reading the periodic table" CNN). Primarily a **classification** model (above/below a T_c threshold); not a clean K-scale regression benchmark.

## C. Highest reported — but different dataset + unverifiable split (treat with caution)

**Crystal-structure GNN — Zhang et al. 2024**
- *Science China Materials* 67, 3253–3261; DOI [10.1007/s40843-024-3026-8](https://doi.org/10.1007/s40843-024-3026-8)
- Dataset: their **own SuperCon→ICSD matched set, ~5,713** (NOT 3DSC; corroborated by Zhang's own 2026 ACS Omega review)
- Model: "crystal structure graph neural network" — specific architecture **not named** (an "ALIGNN" attribution in search snippets is unverified)
- Metrics: **R² = 0.962, RMSE = 6.192 K** (verified, publisher summary); claims SOTA. **MAE not reported.**
- Split: ⚠️ **UNKNOWN** — fully paywalled (Unpaywall: no OA), methods unreachable; no leakage discussion findable.
- ⚠️ The best headline number found, but: different dataset, unverifiable split, and **R² 0.962 is a large outlier** vs the R² ≈ 0.71–0.82 cluster of honest 3DSC structure models — consistent with (not proof of) a random/leaky split. Do NOT use as a grouped-split target. (The "92.9%/86.3%" accuracy and "LightGBM MAE 2.93 K" figures seen in some snippets are confabulations / from other studies — disregard.)

## Other references
- **"Learning Superconductivity" benchmark**, NeurIPS 2024 (datasets/benchmarks) — curated T_c benchmark, reportedly leakage-aware splits; does NOT cite Zhang 2024. Worth pulling for standardized splits. (VERIFY title/metrics)
- **ML for Superconductor Discovery** review (Zhang, first author), *ACS Omega* 2026 11(22) 31853 — [10.1021/acsomega.6c01100](https://doi.org/10.1021/acsomega.6c01100)
- **Closed-loop superconducting materials discovery**, npj *Comput. Mater.* 2023 — [10.1038/s41524-023-01131-3](https://doi.org/10.1038/s41524-023-01131-3)
- **Tempered deep learning of the electron-phonon spectral function** (predicts α²F → relevant to our Phase-3 λ/ω_log idea), arXiv [2401.16611](https://arxiv.org/abs/2401.16611) (2024)

---

## Comparability cheatsheet

- **Metric**: MSLE (3DSC paper) ≠ MAE (us, Quinn) ≠ RMSE (Roter, ALIGNN) ≠ R²-on-ln T_c (Stanev). RMSE ≈ 1.5–2× MAE on these skewed distributions. MSLE rewards low-T_c *relative* accuracy; MAE/RMSE in K reward high-T_c *absolute* accuracy (cuprates).
- **Split is the decider**: random splits LEAK on these datasets (duplicate/doped near-siblings) — worth ≈ 1.9 K of false optimism for us (ORIG 4.04 leaky → 5.92 leak-free). 3DSC's chemical-system grouping is the gold standard; our parent-grouping is leak-free, slightly less strict. **Honest, grouped-split structure results are scarce — basically just the 3DSC XGBoost (MSLE 0.748).** ALIGNN/mCGCNN/JPCC/Zhang are all random or unverified.
- **Data ceiling**: ~20% of SuperCon entries estimated mislabeled (Roter & Dordevic) — caps achievable accuracy.
- **Bottom line vs us**: On 3DSC, every flashier number (R² ≥ 0.92, RMSE ≤ 8 K) rides on a random/unverified split. The only fair competitor on an honest split is the **3DSC XGBoost**, and the direct same-protocol test (chemsys split + MSLE, 2026-06-30) gives **XGBoost 0.748 vs our FT 0.779 MSLE** — XGBoost edges us ~4% on the log metric (within ~split-noise of being a tie). But our GNN leads on absolute-K MAE / cuprates (4.44 / 14.7 parent-grouped), which they don't report. Roughly comparable models with opposite strengths; a strong composition+SOAP gradient boost remains hard to beat on MSLE. Remaining rigor step: a few chem-system split seeds to pin our MSLE spread (single split so far).
