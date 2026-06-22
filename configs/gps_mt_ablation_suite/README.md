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
| `04_dos_full` *(PARKED)* | + DOS                              | `packed_v4` ∪ `dos_pack` |

`forces`/`stress` are `−∂E/∂cart` / `∂E/∂strain` (conservative autograd, not direct heads).
`bandgap` is the per-structure electronic-structure signal that co-trains in rung 03.

> **`04_dos_full` is PARKED (2026-06-22).** DOS is a per-structure total spectrum (per-atom
> Softplus head → `_segment_sum`) trained over the masked union of `packed_v4` and a relaxed-MP
> DOS pack. But MP's open-data DOS objects (`s3://materialsproject-parsed/dos/<mid>.json.gz`)
> are only mirrored for **278** materials of this `theoretical=False` (experimental/ICSD) set —
> too few to train a DOS head. The code path (DOS pack, `ConcatMTDataset` union, dos head) is
> built and tested; revive `04` only with a DOS-rich material set (e.g. theoretical materials).
> The live ladder is **01 → 02 → 03**; `bandgap` carries the electronic leg in the interim.

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
