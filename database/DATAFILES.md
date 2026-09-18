# database/datafiles — provenance and regeneration map

`database/datafiles/` is git-ignored as a whole (`.gitignore`: `/database/datafiles/`).
The only tracked files under it are small, hand-curated **experiment definitions**
(holdout / exclusion id lists, the banned-typo keys, the ICSD request list) — 31
files, all under 50 KB. Raw downloads and every derived artifact (graphs, packs,
indices, descriptors, predictions) are regenerated from the scripts below.

Legend — **raw**: an external download with no in-repo fetch step (URL given);
**fetch**: downloaded by a script; **derived**: built by a script from the row above.

## Raw inputs (download by hand unless a fetch script is listed)

| file | status | source | consumer |
|---|---|---|---|
| `MP/3DSC_MP.csv`, `MP/cifs/`, `MP/id_prop.csv` | raw | 3DSC repository (Sommer et al. 2023), `superconductors_3D` release; 3DSC_MP.csv is their metadata table, cifs/ their MP-matched CIFs | `main.py build-db --kind cgv4` |
| `MP/SuperCon_Stanev2018.csv` | raw | SuperCon export from Stanev et al. 2018 (npj Comput. Mater. 4, 29) supplementary | `database/pipelines/supercon_expansion/build_supercon_v7.py` |
| `NE_SCDB/magnetic_materials.csv`, `superconductor_materials_full.csv` | raw | NEMAD (Itani et al. 2024) CSV exports | `database/pipelines/nemad_expansion/`, `database/pipelines/supercon_expansion/build_{cuprate,feox}_negatives.py` |
| `MP/ICSD_Parent_Cifs/EntryWithCollCode*.cif` | raw, licensed | ICSD (FIZ Karlsruhe), hand-selected per `docs/data_curation/icsd_parents_wanted.md` | the SuperCon builder (tier-0 parents) |
| `MPtrj/MPtrj_2022.9_full.json` | raw | MPtrj (Deng et al. 2023, CHGNet), figshare | `main.py build-mptrj` → `database/Extract_MPtrj.py` |
| `EPH_Cerqueira/DS-A.pk.bz2`, `DS-B.pk.bz2` | raw | Cerqueira / Sanna / Marques 2023 e-ph dataset (Materials Cloud) | `database/pipelines/eph/build_eph_corpus.py` |
| `EPH_Cerqueira/a2f_raw/` | fetch | `database/pipelines/eph/extract_a2f_corpus.sh` streams the Materials Cloud batches (a2F.dos6, McMillan.dat, qe.dyn*) | `database/pipelines/eph/build_phonon_targets.py` |
| `TogoPhononDB/zips/`, `registry.csv` | fetch | `database/pipelines/phonon/crawl_togo_phonondb.py` (MDR@NIMS, 10,034 zips) | `database/pipelines/phonon/process_togo_phonondb.py` |
| `MP_PhononDOS/raw/`, `fc_pheasy/`, `structures.jsonl.gz` | fetch | `database/pipelines/phonon/fetch_mp_phonon_dos.py` (DFPT), `database/pipelines/phonon/build_phdos_corpus.py {dfpt,togo,pheasy,fetch-structures}`, `database/pipelines/phonon/build_phdos_site_corpus.py fetch-fc` (MP API, needs `MP_API_KEY`) | phonon corpus build |
| `MP_Energy/mp_energy.csv`, `MP_Energy/cifs/` | fetch | `main.py download-energy` → `database/Download_MP_energy.py` (MP API) | MP-energy graphs |
| `Non_SC_DB_MP/Non_SC.csv`, `cifs/` | fetch | `main.py download-nonsc` → `database/Download_MP_data.py` (MP API) | non-SC pool |
| `WBM/wbm-cse.jsonl.gz`, `wbm-summary.csv.gz` | raw | Matbench Discovery data release (Riebesell et al.), figshare | `database/pipelines/benchmarks/wbm_ingest.py` |
| `Matbench/matbench_mp_e_form.json.gz` | raw | matbench hosted json.gz | `database/pipelines/benchmarks/matbench_ingest.py` |
| `MP/atom_init.json` | derived | `main.py atom-init` | all graph loaders |

## Derived artifacts by folder

### MP — superconductor indices, graphs, packs
| artifact | generator |
|---|---|
| `graphs_v4_doped_v45/`, `SC_MP_V4_doped_v45.{pickle,csv}` | `database/pipelines/mp/cf_v45_wave.sh` step 1 = `main.py build-db --kind cgv4 --source id_prop.csv cifs/ --oxidation-parent-csv 3DSC_MP.csv` |
| `descriptors_doped.pickle` | `models/head/descriptors.build_descriptor_table(index)` (run inside the head pipelines) |
| `SC_pack_doped_v45/`, every `*_pack_*` | `main.py pack-dataset --index <index.pickle> --out <pack>` |
| `disorder_corpus/`, `disorder_pack_v45/` | `database/pipelines/mp/build_disorder_corpus.py` then `cf_v45_wave.sh` step 2 |
| `dos_rebuild/`, `dos_pack_ef1_v45/` | `database/pipelines/dos_rebuild/` (README there), `database/pipelines/mp/rebuild_dos_graphs_cf.py`, `database/pipelines/mp/augment_cf.py`, `main.py fetch-dos` (`database/Download_MP_dos.py`) |
| `graphs_v4/` (89k, the full-MP screen set), `screen_mp_index.pickle`, `descriptors_screen_mp.pickle`, `screen_mp_metadata.pickle`, `pred_cache*` | `database/pipelines/mp/build_mp_screen.py`, `scripts/screen_mp.py`, `scripts/screen_eph.py` |
| `cifs_v5_nemad/`, `graphs_v5_nemad_v45/`, `SC_MP_V5_nemad*`, `cifs_v5_icsd/`, `SC_MP_V5_icsd*`, `SC_MP_V6_*`, `magnetic_rebuild/`, `SC_MP_V4M*`, `SC_MP_V6M*`, `descriptors_v4m*`, `descriptors_v6m` | `database/pipelines/nemad_expansion/` (fetch_pool → match_and_dope / build_icsd_set / match_magnetic → build_magnetic_index; README there) — the 2026-07/08 expansions, superseded by the SuperCon expansion below |
| `cifs_v7_supercon/`, `graphs_v4_v7_supercon/`, `SC_MP_V7_supercon*`, `descriptors_v7_supercon` | `database/pipelines/supercon_expansion/supercon_v7_autopilot.sh` = `build_supercon_v7.py candidates → pool → match-dope → graphs` (+ `rebuild_after_audit.sh` after the audit) |
| `cifs_v9_cuneg/`, `graphs_v4_v9_cuneg/`, `SC_MP_V9_cuneg*` | `database/pipelines/supercon_expansion/cuneg_autopilot.sh` (`build_cuprate_negatives.py`) |
| `cifs_v11_feox/`, `graphs_v4_v11_feox/`, `SC_MP_V11_feox*` | `database/pipelines/supercon_expansion/feox_autopilot.sh` (`build_feox_negatives.py`) |
| `SC_MP_V8_supercon*`, `descriptors_v8_supercon`, `3DSC_MP_v8.csv`, `holdout_matthias_*` | `database/pipelines/supercon_expansion/merge_and_holdouts.py` |
| `SC_MP_V9_cuneg.pickle` (merged), `descriptors_v9_cuneg`, `3DSC_MP_v9.csv`, `la_series_both_holdout_ids_v9`, `nickelate_holdout_ids_v45_v9` | `build_cuprate_negatives.py merge` |
| `SC_MP_V10*`, `descriptors_v10`, `3DSC_MP_v10.csv`, `*_v10.csv` holdouts | `database/pipelines/supercon_expansion/finalize_v10.py` (applies `docs/data_curation/v9_audit_exclude_ids.csv`, `v10_conflict_minority_ids.csv`, `v10_conflict_pairs_drop_ids.csv`) |
| `SC_MP_V11*`, `descriptors_v11*`, `3DSC_MP_v11.csv`, `*_v11.csv` holdouts | `build_feox_negatives.py merge`; V11c = the stratified 600-row cap (built inline 2026-09-14, ids in `SC_MP_V11c.pickle`) |
| `holdout_struct_*`, `exclude_struct_*` | `scripts/structure_holdout.py` |
| `holdout_fam_*`, `la_series_both_holdout_ids.csv`, `lsco_*`, `nickelate_holdout_ids*.csv` | `scripts/family_dome.py`, `scripts/nickelate_holdout.py`, the July holdout builders (tracked as experiment definitions) |
| `embeddings/` | `main.py embed-gps` / FineTune caches (rebuilt on demand) |
| `descriptors_v*_*.pickle` (all) | `models/head/descriptors.build_descriptor_table` on the matching index; re-assembled by `database/pipelines/supercon_expansion/rebuild_graphs_histfix.sh` |

### Phonon / e-ph corpora
| artifact | generator |
|---|---|
| `MP_PhononDOS/graphs_v45_phdos/`, `PHDOS_index*`, `phdos_pack_v45/` | `database/pipelines/phonon/phdos_corpus_autopilot.sh` (`build_phdos_corpus.py build → bake`, `main.py pack-dataset`) |
| `MP_PhononDOS/site_raw/`, `phdos_pack_v45_site/` | `database/pipelines/phonon/phdos_site_autopilot.sh` (`build_phdos_site_corpus.py fetch-fc → site-togo → site-pheasy → bake`, pack) |
| `TogoPhononDB/dos_raw/`, `site_raw/` | `database/pipelines/phonon/process_togo_phonondb.py`, `build_phdos_site_corpus.py site-togo` |
| `EPH_Cerqueira/graphs_v45_eph/`, `EPH_index*` | `database/pipelines/eph/build_eph_corpus.py` |
| `EPH_Cerqueira/eph_pack_v45/`, `eph_pack_v45_site/` | `database/pipelines/eph/build_phonon_targets.py a2f → phdos → phdos-site`, then `main.py pack-dataset` (chain: `model_data/cf_calib/histfix_eph.sh`) |

### Other corpora
| artifact | generator |
|---|---|
| `MP_Energy/graphs_v4/`, `MP_Energy_V4*` | `main.py build-db --kind cgv4` on the MP-energy CIFs |
| `MPtrj/` packs (cluster: `/scratch/.../MPtrj/packed_v45`) | `main.py build-mptrj` → `augment-cf` (`scripts/v45_autopilot.sh`) |
| `Matbench/graphs_v45_eform/` | `database/pipelines/benchmarks/matbench_ingest.py` |
| `WBM/graphs_v45_wbm/`, `WBM_eform_index*`, `wbm_test_index.pickle`, `wbm_pack_v45/`, `wbm_predictions.csv` | `database/pipelines/benchmarks/wbm_ingest.py` → `main.py pack-dataset` → `scripts/wbm_predict.py` (chain: `model_data/cf_calib/histfix_wbm.sh`) |
| `NE_SCDB/*.csv|pickle` (candidates, matches, pools), `SC_EXPAND/*` | `database/pipelines/nemad_expansion/`, `database/pipelines/supercon_expansion/build_*.py candidates|pool` |

## Superseded / backup directories (safe to delete; ~15 GB)
`*_prefix_0917-*` (pre-fix graphs/packs/indices kept when the 2026-09-17 rebuild
ran), `MP_PhononDOS/phdos_pack_v45_togo6511`, `MP/graphs_v4_doped`, `graphs_v4_doped_oxifix`,
`graphs_v4_doped_cf`, `graphs_v4_doped_v44`, `SC_pack_doped`, `SC_pack_doped_cf`,
`SC_pack_doped_v44`, `disorder_pack`, `disorder_pack_cf`, `disorder_pack_v44`,
`dos_pack_ef1`, `dos_pack_ef1_cf2`, `dos_pack_ef1_v44`, `graphs_v5_icsd`, `graphs_v5_nemad`
(pre-v45 schema eras).

## Known gaps
- No in-repo fetch step for the 3DSC, SuperCon, NEMAD, MPtrj, Cerqueira DS-A/B, WBM and
  Matbench raw files — download by hand from the sources above into the listed paths.
- The ICSD CIFs are licensed and cannot be redistributed or scripted.
- V11c was built inline, not by a script (its ids are the index itself).
- The corpus pipelines live under `database/pipelines/` (`supercon_expansion/`, `nemad_expansion/`,
  `dos_rebuild/`, `phonon/`, `eph/`, `mp/`, `benchmarks/`) beside the graph builder
  (`database_main.py`), the MP downloaders and the MPtrj streamer.
