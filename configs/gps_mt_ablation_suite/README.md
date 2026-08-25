# GPS multi-task physics-signal ablation (→ T_c transfer)

Phase 6 of the conservative multi-task pretraining plan. Each rung pretrains the **same**
`GPSCrystalNet` encoder (rung-06 local-transformer architecture, identical hyperparameters)
and differs *only* in which physical signals co-train. The encoder is then **frozen** and
probed at superconductor T_c. The metric that matters is **T_c transfer**, not pretraining MAE.

| rung | tasks | packs |
|------|-------|-------|
| `01_energy_only`    | energy                                  | `packed_v4` |
| `02_forces_stress`  | + forces, stress (conservative autograd)| `packed_v4` |
| `03_magmom_bandgap` | + magmom, bandgap                       | `packed_v4` |
| `04_dos_full` *(REVIVED)* | + DOS                             | `packed_v4` ∪ `dos_pack` |

`forces`/`stress` are `−∂E/∂cart` / `∂E/∂strain` (conservative autograd, not direct heads).
`bandgap` is the per-structure electronic-structure signal that co-trains in rung 03.

> **`04_dos_full` is REVIVED (2026-06-24).** DOS is a per-structure total spectrum (per-atom
> Softplus head → `_segment_sum`) trained over the masked union of `packed_v4` and a relaxed-MP
> DOS pack. The earlier 278-coverage "park" was a self-inflicted bug — `_dos_object` had been
> switched to a material-id-ONLY S3 key, which 404'd every canonical material. With the stock
> `get_dos_by_material_id` (task-id) route restored as the primary path (commits `084017b`,
> `02a2ed4`), a re-fetch now covers **31,403 / 49,280** relaxed-MP materials — ample to train a
> DOS head. The DOS pack (`database/datafiles/MP/dos_pack`, `has_dos=true`, `has_positions=true`,
> exact `has_to_jimage=true`) is built locally and shipped with `deploy.sh sync-dos-pack`.
> Build it: `python main.py fetch-dos --index database/datafiles/MP_Energy/MP_Energy_V4.pickle`
> then `python main.py pack-dataset --index database/datafiles/MP_Energy/MP_Energy_V4.pickle
> --out database/datafiles/MP/dos_pack --derive-mp-id`. The DOS pack carries no
> forces/magmom/stress (relaxed structures) — those are NaN-masked; energy + DOS co-train on it.
>
> **`--derive-mp-id` is REQUIRED for this pack.** The union splits by material
> (`split_by="material"`), which needs every member to carry an `mp_id`. The relaxed-MP index has
> no `mp_id` column (its `id` *is* the material id, one structure each), so the flag synthesizes
> `mp_id = id` (minus `.cif`). Without it the pack's `groups` is `None`, collapses the union's
> groups to `None`, and the run dies in `get_sc_nonsc_loaders` ("no usable 'mp_id' column"). The
> flag is opt-in precisely so it does NOT touch `MP_Energy_V4.pickle` itself — the `eform`
> benchmark suite trains solo on that index with a frame split, which must stay unchanged.

## Run protocol (per rung)

```bash
# 1. pretrain the multi-task encoder (cluster; double-backward path, amp=false, fp32 forces)
scripts/deploy.sh run configs/gps_mt_ablation_suite/01_energy_only.json

# 2. export frozen per-structure embeddings for the T_c transfer set
python main.py embed-gps --checkpoint <run_dir>/result_model_best.pth.tar \
    --index database/datafiles/MP/SC_MP_V4.pickle --out embeddings/gps_mt_01_energy_only

# 3. train the frozen-encoder T_c head (reuses models/head/, MACE-compatible layout)
python main.py train-head --embeddings embeddings/gps_mt_01_energy_only ...
```

Compare the rungs' T_c heads with `compare_runs.py` (paired, same split/seed) against the
frozen-MACE and set-transformer baselines. The hypothesis: each added physics signal yields a
more transferable encoder; the ladder isolates each signal's marginal transfer value.

> `04_dos_full` and `gps/gps_multitask.json` are the same full-union config; `gps_multitask.json`
> is the canonical "everything on" entry point, the suite rungs isolate each signal.

## Rungs 27–33: pretraining-target ladder on the modern stack (2026-08-24)

The 08-19 target × SC-class matrix re-probed **June's old 14-dim checkpoints**; these rungs
re-ask the question on the **rung-25 dome-champion recipe** (no-poly, CF+BVS baked, valence
block off, oxidation scalar on, v45 packs, no disorder corpus), with both June-era confounds
removed: **equal loss weights everywhere** (the rung-04 lesson — forces×2 was mis-crowned on
scrambled folds) and **fixed data** — every rung trains on the same two packs
(`packed_v45` ∪ `dos_pack_ef1_v45`), only the task list varies (June's ladder added the DOS
pack together with the DOS task, conflating data and target).

| rung | tasks | reading |
|------|-------|---------|
| `27_t_energy`        | energy                                    | build-up 1 |
| `28_t_forces_stress` | + forces, stress                          | build-up 2 |
| `29_t_no_dos`        | + magmom, bandgap                         | build-up 3 ≡ leave-one-out −DOS |
| `30_t_all_equal`     | all six, equal weights                    | reference cell |
| `31_t_no_magmom`     | all six − magmom                          | LOO — magmom was cuprate-specific on the old stack; the CF input already carries `cf_unpaired` (an AOM magmom), so this tests whether the task is now redundant |
| `32_t_no_bandgap`    | all six − bandgap                         | LOO |
| `33_t_no_forces`     | all six − forces, stress                  | LOO |

Rung **25**'s existing checkpoint is the *forces×2 all-six* cell for free (probe
4.58/5.94/0.868/16.85; dome 0.823/1.33K/22.9/5.30; family cup 22.4 / fer 7.5 / other 2.4).
Old-stack matrix best: rung 04 (all six, equal) — ALL 4.49 / cuprate 21.3.

Autopilot (submit all 7, watch, fetch, probe + LSCO-dome eval + per-family
`family_stats.py` line per rung): `scripts/target_ablation_autopilot.sh`, status in
`model_data/cf_calib/target_ablation_status.txt`. Eval head configs:
`configs/head/gps_tc_{probe,la_series}_{27..33}tg.json` (checkpoint patched at eval time).

## No-pretrain ladder (rungs a–f, 2026-08-24) — companion experiment, local GPU

Not a pretraining suite: **no pretraining anywhere**. Measures what each layer of
information/processing buys on **Tc supervision alone**; the diff against the pretrained
rungs isolates the pretraining contribution per step. Same Tc-head protocol as the probes
(3-seed, chemsys, msle, deepsets, descriptors alongside); only the head's per-atom input
changes width.

| rung | encoder input | mechanism |
|------|---------------|-----------|
| `a_comp`    | composition-only node features (14-dim, geometry cols masked) | identity encoder — raw features ARE the per-atom latents |
| `b_valence` | + valence subshells (18-dim, still structure-free) | identity |
| `c_geom`    | + geometry scalars: base geo cols + CF + BVS (32-dim) | identity — "structure as descriptors, no message passing" |
| `d_bonds`   | c + GPS layers **from scratch**: bond attention, no angle, no poly | `ft_unfreeze: all`, encoder LR 3e-4 |
| `e_angle`   | d + angle bias (= rung-21 arch, untrained) | " |
| `f_poly`    | e + poly edges (= rung-20 arch, untrained) | " |

Mechanics: `encoder_args` in a head config (FineTune) replaces the checkpoint — type
`identity` returns standardized raw node features as the per-atom representation (phase B
skipped); type `scratch` builds a seeded random GPSCrystalNet trained end-to-end on Tc.
Feature normalizers fit on the transfer index. `mask_geometry_features` (data.py
`GEO_FEATURE_COLS`) zeroes the 7 geometry-derived base columns for the structure-free rungs.
Configs `configs/head/gps_tc_nopre_{a..f}*.json`; runner `scripts/nopre_ladder.sh` (serial —
the local GPU takes one head job at a time); status `model_data/cf_calib/nopre_ladder_status.txt`.
