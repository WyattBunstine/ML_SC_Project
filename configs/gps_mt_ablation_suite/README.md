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
| `04_dos_full`       | + DOS                                   | `packed_v4` ∪ `dos_pack` |

`forces`/`stress` are `−∂E/∂cart` / `∂E/∂strain` (conservative autograd, not direct heads);
`dos` is a per-structure total spectrum (per-atom Softplus head → `_segment_sum`, 256-bin
E_F-aligned grid). The DOS rung trains over the masked **union** of the MPtrj pack (E/F/stress/
magmom/bandgap) and the relaxed-MP DOS pack (dos only) — each pack supplies its targets, the
others NaN-masked.

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
