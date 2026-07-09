# NEMAD → MP/ICSD dataset expansion

Expands the 3DSC T_c transfer set with **NEMAD** consensus labels matched to Materials
Project + ICSD structures (`3DSC 5,773 → SC_MP_V6_expanded 7,852`). Full record in the
`nemad-icsd-expansion` memory.

**Requirements:** run from the repo root; `MP_API_KEY` (env) for MP fetches; `THREEDSC_REPO`
(env, default `~/Downloads/old_files/3DSC-main`) for the synthetic-doping internals.
**Outputs** land under `database/datafiles/` (gitignored — re-run to regenerate).

Pipeline:
1. *(inline)* NEMAD consensus per fractional composition → `NE_SCDB/nemad_candidates.csv`;
   the unique chemical systems → `need_systems.txt`.
2. `fetch_pool.py` — MP crystal pool (lightweight fields) for the needed systems →
   `mp_crystal_pool.pickle`.
3. `match_nemad2.py` — faithful re-impl of 3DSC's formula matcher (`totreldiff`, tier-first
   selection) → `nemad_matches.csv`.
4. `synth_dope.py` — self-contained wrapper around 3DSC `_4_synthetic_doping` (imported by 5/6).
5. `match_and_dope.py` — top-K match + synth-dope (retries lower-ranked parents) →
   `SC_MP_V5_nemad_source.csv` + `cifs_v5_nemad/`.
6. `build_icsd_set.py` — the oxygen-interstitial cuprates that fail MP doping: assign
   oxygen-rich **ICSD** parents (`ICSD_Parent_Cifs/`) + the combined cation+O doper in
   `database/icsd_doping.py` (RE-123 by substitution into one ortho template).

Diagnostics: `scan_icsd.py` / `coverage.py` (ICSD parent inventory + target coverage),
`analyze_fails.py` (synth-doping failure breakdown — which are O-interstitial, grouped by
parent, to pick ICSD fetches).

Then `main.py build-db --kind cgv4 … --oxidation-parent-csv` builds the graphs, and the rows
are merged into `SC_MP_V6_expanded.{pickle,csv}`.
