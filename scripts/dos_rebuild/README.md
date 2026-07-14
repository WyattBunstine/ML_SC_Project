# DOS re-fetch — narrow ±1 eV window

Rebuilds the multitask **DOS** target on a narrow **±1 eV / 128-bin** E_F-aligned grid
(`database/Download_MP_dos.py`; was −10..+5 eV / 256 bins) for the full ~62,972 MP materials
with DOS — concentrating the loss on the SC-relevant near-E_F states and nearly doubling the
effective sample count. Full record in the `valence-features-nickelate` memory (DOS REBUILD).

**Requirements:** run from the repo root; `MP_API_KEY` (env). **Outputs** under
`database/datafiles/MP/` (gitignored).

Pipeline:
1. *(inline)* query MP `has_props=["dos"]` → `dos_rebuild/dos_material_ids.json` (all 62,972)
   and `dos_missing_graphs.json` (those lacking an undoped graph).
2. `fetch_dos_structures.py` — fetch structures for the ~23k materials missing graphs →
   `dos_rebuild/cifs/`; then `main.py build-db --kind cgv4` on them → `dos_rebuild/graphs_v4/`.
3. `dos_build_index.py` — unified index over the **undoped** (mp-id-named) DOS graphs →
   `dos_rebuild/dos_all_index.pickle` (doped SC graphs excluded — DOS is an undoped property).
4. `main.py fetch-dos --index dos_all_index.pickle --workers 12` — attach the ±1 eV DOS onto
   each graph (resampled + Gaussian-broadened).
5. `main.py pack-dataset --index dos_all_index.pickle --out dos_pack_ef1` — the DOS pack.
6. `smoke_dos_pack.py` — end-to-end smoke of the packed DOS exactly as pretraining consumes it.

Pairs with the per-atom DOS target (`/n_atoms`) + `segment_sum→mean` head reconditioning
(`models/common/data.py`, `models/GPSTransformer/model.py`), and feeds the valence re-pretrain
config `configs/gps_mt_ablation_suite/09_forces_w2_valence.json` (`n_energy=128`, `dos_pack_ef1`).

Both pieces are config-gated so they ablate independently (a 2×2 with the historic 06 run):
`dos_per_atom` (false = legacy extensive total-DOS target + segment-SUM head) and `n_energy`
(must match the pack's stored grid; the reader raises on mismatch). Cells:
06 = neither · `10_forces_w2_valence_only` (legacy `dos_pack`, 256, extensive) = valence only ·
`11_forces_w2_dos_ef1_only` = new DOS only · 09 = both.
Smoke any cell: `python scripts/dos_rebuild/smoke_dos_pack.py <config> <pack_dir>`.
