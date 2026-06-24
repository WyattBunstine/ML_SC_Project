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
> --out database/datafiles/MP/dos_pack`. The DOS pack carries no forces/magmom/stress (relaxed
> structures) — those are NaN-masked; energy + DOS co-train on it.

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
