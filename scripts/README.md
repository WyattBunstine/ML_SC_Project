# `scripts/` — local tools

Everything here runs on a workstation against `database/datafiles/` and `model_data/`.
Cluster deployment (the ssh/rsync/sbatch wrapper and the run autopilots that drive it)
lives in `scripts/remote/`, which is git-ignored because it carries our site's host,
account and storage paths — see the README's *Running on a SLURM cluster* section for
the three conventions any replacement wrapper must follow.

| Group | Scripts |
|---|---|
| T_c head training / inference | `run_head.py` (train a head from a config), `predict_tc.py`, `probe_encoders.py` (frozen-encoder probes across rungs), `head_hpo_sweep.py`, `a2f_head_retrain.py`, `ensemble.py`, `eval_test.py` |
| Holdout analyses | `family_dome.py`, `family_stats.py`, `dome_stats.py`, `plot_lsco_dome.py` (cuprate doping domes), `matthias_dome.py`, `plot_matthias_combined.py` (valence-electron domes), `nickelate_holdout.py`, `structure_holdout.py` (structure-type holdout lists), `fe_parity.py`, `plot_parity.py`, `plot_v8v9_parity.py`, `compare_runs.py` |
| Discovery screens / benchmarks | `screen_mp.py`, `screen_eph.py`, `screen_report.py`, `wbm_predict.py` (Matbench Discovery) |
| Figures | `plot_tc_histogram.py`, `plot_fe_training_curves.py`, `plot_phonon_dispersion.py` |
| Data QA | `audit_doping_labels.py`, `audit_ferrite_labels.py`, `calibrate_cf_magmom.py`, `oxidation_doping_prototype.py`, `cf_schema2_verify_pack.py`, `cf_v44_verify_pack.py`, `cf_v45_verify_pack.py`, `smoke_dataset.py` |
| Checks | `validate_config.py` (run before launching), `verify_smoke.py`, `verify_autograd_forces.py`, `verify_tf32_forces.py`, `verify_ddp_multitask.py`, `verify_multitask_train.py`, `verify_union_masking.py`, `verify_dos_fetch.py` |
| Run bookkeeping | `reorg_runs.py` (nest flat `model_data/` runs by date) |
| Local sweeps (bash, no cluster) | `dome_autopilot.sh`, `loss_sweep_dome.sh`, `nopre_ladder.sh`, `target_probe_sweep.sh`, `v6_ablation_pair.sh`, `v6_v45_dome.sh` |

## `eval_test.py` — score a saved model on a test set

Run a saved checkpoint over a test set and write per-sample predictions
(`cif_id,target,pred`, the same headerless format the trainers emit, so `plot.py`
reads it directly). Two uses:

1. **Recover an interrupted run.** The trainer only writes the test CSV after
   training *finishes*, so a cancelled run leaves a good `*_model_best.pth.tar`
   but no predictions. Regenerate them:
   ```bash
   python scripts/eval_test.py --run model_data/<run_dir>/
   # -> model_data/<run_dir>/<ckpt>_eval.csv  (+ prints MAE)
   python main.py plot --results model_data/<run_dir>/<ckpt>_eval.csv
   ```
2. **Apples-to-apples model comparison.** Evaluate different models on the SAME
   materials. Capture one run's test IDs, then score the others on them:
   ```bash
   python scripts/eval_test.py --run model_data/MPNN_run/  --export-ids holdout.txt
   python scripts/eval_test.py --run model_data/Orig_run/  --test-ids  holdout.txt
   ```

The model family is auto-detected from `config.json` (`index_path` → MPNN,
`dataset` → original CGCNN); it's pluggable — adding a model is one `ModelAdapter`
subclass. Options: `--checkpoint best|last`, `--split test|val|train|all`,
`--out FILE`.

> ⚠ A *fair* comparison needs the shared IDs to have been held out of every
> compared model's training. The MPNN and original CGCNN shuffle different
> pickles independently, so their own test splits don't share materials even at
> the same seed — designate a common holdout (or train both on a shared split)
> and pass it with `--test-ids`. The tool scores whatever IDs you give it but
> can't verify they were held out.

## `ensemble.py` — combine models into a mean prediction + uncertainty

Average several models' predictions into an ensemble (lower MAE) with a
per-material uncertainty (std across members). Inputs are the `cif_id,target,pred`
CSVs the trainers / `eval_test.py` write; members are joined on `cif_id`.

```bash
# 1. Train diverse members — same config, different model_seed (and/or data):
#    (e.g. configs with "model_seed": 1, 2, 3, ...)
# 2. Score each on the SAME materials:
python scripts/eval_test.py --run model_data/run_seed1/ --export-ids holdout.txt --out p1.csv
python scripts/eval_test.py --run model_data/run_seed2/ --test-ids  holdout.txt --out p2.csv
python scripts/eval_test.py --run model_data/run_seed3/ --test-ids  holdout.txt --out p3.csv
# 3. Ensemble:
python scripts/ensemble.py p1.csv p2.csv p3.csv --out ensemble.csv
```

Output is `cif_id,target,mean_pred,std_pred,n_models`; it also prints per-member
MAE, the ensemble MAE, and the % improvement. Members must be **diverse** to help
(vary `model_seed`) and evaluated on the **same** materials (use `--test-ids`) —
identical members average to themselves. `std_pred` is a cheap epistemic
uncertainty: members disagree most where they're least confident.

> Tip: to evaluate the SWA-averaged weights from a run, use
> `eval_test.py --checkpoint swa`.

