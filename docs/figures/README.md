# Figures

| folder | contents |
|---|---|
| `domes/` | every T_c-vs-doping dome: La-series (`tc_vs_cu_oxidation_la_series_dome_<rung>.png`), family/structure holdouts (`tc_vs_cu_oxidation_fam_*`), index comparisons (`dome_v*_ladome_51`, `dome_common60_*`), dome evolution panels |
| `matthias/` | valence-electron domes for the conventional alloys (holdout arms + training coverage) |
| `parity/` | predicted-vs-experimental T_c panels per encoder, index comparisons, formation-energy parity, and their generators |
| `eph/` | electron-phonon screening figures |
| `training/` | pretraining learning curves and formation-energy ladder figures |
| `dataset/` | dataset composition: T_c histograms by family, no-pretrain ladder table |
| `latex/` | TikZ/LaTeX schematics (graph construction, GPS encoder, pretraining-transfer) with their PDFs and generator scripts |
| `scrambled_split_archive/` | pre-2026-07-14 figures produced on the scrambled folds (kept for the record, not citable) |

One-off figures stay at this level. Generators write to these folders by default
(`plot_lsco_dome.py`, `head_batch_watcher.sh`, `plot_parity.py`, `plot_tc_histogram.py`, ...).
