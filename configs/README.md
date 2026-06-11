# MPNN training config reference

JSON config files for the crystal_graph_v4 MPNN model (`CNN/MPNN/MPNNMain.py`).

Run a config with:

```bash
python main.py train-mpnn configs/<your_config>.json
```

(The baseline CGCNN trainer, `python main.py train ...`, uses a **different**
schema — see `CNN/CGCNNMain.py`. This document is only for `train-mpnn`.)

A config is a flat JSON object of key/value pairs. Unknown keys are ignored.
Every key except the few listed under **Required** has a default, so a minimal
config is short.

---

## Required keys

These have no default; the run fails (or behaves undefined) without them.

| Key | Type | Description |
|-----|------|-------------|
| `index_path` | string | Path to the index pickle produced by `build-db --kind cgv4` (e.g. `database/datafiles/MP/SC_MP_V4.pickle`), OR a **packed dataset directory** produced by `main.py pack-dataset` (auto-detected; ~30x faster sample reads, recommended for large datasets like MPtrj). |
| `out_file` | string | Output path **prefix**. All artifacts are written as `<out_file>...` (see [Outputs](#outputs)). |
| `epochs` | int | Number of training epochs. |
| `batch_size` | int | Crystals per batch. |
| `learning_rate` | float | Initial optimizer learning rate. |
| `val_ratio` | float | Fraction of each class held out for validation (0–1). |
| `test_ratio` | float | Fraction of each class held out for test (0–1). |

Minimal valid config (everything else defaulted):

```json
{
    "index_path":    "database/datafiles/MP/SC_MP_V4.pickle",
    "out_file":      "CNN/MPNN/mpnn_result",
    "epochs":        1000,
    "batch_size":    64,
    "learning_rate": 0.01,
    "val_ratio":     0.1,
    "test_ratio":    0.1
}
```

---

## Optional keys (with defaults)

### Task

| Key | Type | Default | Valid values | Description |
|-----|------|---------|--------------|-------------|
| `task` | string | `"regression"` | `"regression"`, `"classification"` | Regression predicts T_c (L1 loss, MAE); classification predicts SC vs non-SC (NLL loss, AUC/F1, model selected on realistic-val AUC). |

### Regression target (regression only)

| Key | Type | Default | Valid values | Description |
|-----|------|---------|--------------|-------------|
| `target_transform` | string | `"none"` | `"none"`, `"log1p"` | Transform applied to T_c before training (inverted for reporting). `log1p` = train in `log(1+T_c)` space so the loss isn't dominated by high-T_c materials and the densely-packed low-T_c region gets relative weight. T_c=0 maps to 0. MAE is still reported in real T_c units. |

### Classification recall control (classification only)

These bias the SC/non-SC head toward catching superconductors (recall) at the
cost of more false positives — appropriate when missing a real SC is worse than
a false alarm.

| Key | Type | Default | Valid values | Description |
|-----|------|---------|--------------|-------------|
| `sc_class_weight` | float | `1.0` | `>= 0` | Loss weight on the SC (positive) class. `> 1` penalizes false negatives more (training-time recall bias). E.g. `3.0`. |
| `sc_decision_threshold` | float | `0.5` | `0–1` | Predict SC when `p_sc >= threshold`. Lowering it (e.g. `0.3`) raises recall at the cost of precision (inference-time, no retrain). |
| `sc_fbeta` | float | `1.0` | `> 0` | F-beta weighting for the reported/selection F-score. `beta>1` favors recall (`2.0` = F2). |
| `selection_metric` | string | `"auc"` | `"auc"`, `"fbeta"`, `"f1"`, `"recall"`, `"precision"`, `"accuracy"` | Validation metric used to pick the best checkpoint. For a recall-priority model use `"fbeta"` with `sc_fbeta: 2`. Falls back to F1 when the chosen metric is nan. |

### Dataset / data loading

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `target_column` | string | `null` | Which index column to regress on, for multi-target indexes (e.g. the MP energy dataset: `"formation_energy_per_atom"` or `"e_above_hull"`). When unset, falls back to the legacy `value`/`tc` column, so existing SC indexes work unchanged; a tc-less index (e.g. the energy dataset) has no usable default and **raises** asking you to set this key, rather than silently picking a target. Rows whose chosen target is missing (NaN) are dropped. |
| `max_num_nbr` | int | `14` | Bonding-edge neighbors per atom (padded/truncated to this length). |
| `max_num_poly_nbr` | int | `16` | Polyhedral-edge neighbors per atom (padded/truncated). |
| `graph_cache_size` | int | `4096` | Max number of **fully-built samples** kept in each worker's in-memory LRU cache. The cache stores the built tensors (not the parsed JSON), so a cached item skips the read *and* all the per-atom neighbor sorting/padding/stacking — the dominant CPU cost — on every epoch after the first. `0` = unbounded. Each worker holds its own cache (with `num_workers > 0`), so cache RAM ≈ `num_workers × graph_cache_size × (built-sample size)`. A built sample is ~60 KB at `max_num_poly_nbr=16`, ~145 KB at `64`. Keep that product under roughly half of the job's RAM. Because built samples are smaller than the old JSON cache, you can afford a much larger value here — ideally large enough to hold each worker's working set so later epochs are nearly free on CPU. |
| `num_workers` | int | `0` | DataLoader worker processes. `> 0` overlaps loading with compute; workers are persistent so their caches survive across epochs. Ignored when `prebuild_dataset` is on (forced to 0 — nothing left to build, and GPU-resident tensors can't cross a worker). |
| `prebuild_dataset` | bool | `false` | Build **every** sample once up front and keep them all resident, so `__getitem__` becomes a list index (no per-epoch rebuild). Best when the built dataset fits in memory; the build is a one-time ~3–4 min for ~49k graphs. Forces `num_workers=0`. For datasets too large to reside, leave this `false` and use the streaming/LRU path (`num_workers>0` + `graph_cache_size`). |
| `prebuild_device` | string | `"cpu"` | Where a prebuilt dataset lives: `"cpu"` (RAM; per-batch pinned `non_blocking` copy to GPU — recommended), `"cuda"` (VRAM; removes the per-batch copy but slower startup and spends VRAM), or `"auto"` (cuda if it fits with model headroom, else cpu). Training throughput is ~identical between cpu/cuda. |
| `split_seed` | int | `123` | Seed for the stratified train/val/test split and the per-epoch non-SC sampling. |
| `split_by` | string | auto | `"material"` splits whole materials (all frames of one `mp_id` land in the same split — REQUIRED for trajectory datasets like MPtrj, where a frame-level split leaks near-duplicate frames); `"frame"` is the historical row-level split. Default: `material` when the index carries `mp_id`, else `frame`. The resolved value is recorded in `metadata.json`. |

> **Scaling past memory:** for large datasets, pack once with
> `python main.py pack-dataset` and point `index_path` at the pack directory
> (~30x faster reads, bitwise-identical samples). Details and the other
> residence modes: [`CNN/MPNN/DATALOADING.md`](../CNN/MPNN/DATALOADING.md).

### Model architecture

| Key | Type | Default | Valid values | Description |
|-----|------|---------|--------------|-------------|
| `atom_feat_len` | int | `64` | — | Hidden atom-embedding dimension. |
| `edge_hidden_dim` | int | `128` | — | Hidden dimension inside each EdgeNet. |
| `n_conv` | int | `3` | — | Number of message-passing layers. |
| `h_feat_len` | int | `128` | — | MLP hidden dimension after pooling. |
| `n_hidden` | int | `1` | — | Number of post-pool MLP layers. |
| `use_poly_edges` | bool | `true` | — | Message-pass over polyhedral (corner/edge/face-sharing) edges in addition to bonding edges. When `false`, poly inputs are ignored. |
| `edge_aggregation` | string | `"ecn_weighted"` | `"ecn_weighted"`, `"attention"`, `"set_transformer"` | How edge messages are aggregated onto each atom. `ecn_weighted` uses the physical ECoN weight (bonds) / shared_count (poly); `attention` learns a per-edge score; `set_transformer` runs multi-head self-attention over the neighbor tokens with the inter-neighbor bond angle as a per-head attention bias (bond channel only; poly stays `ecn_weighted`). |
| `set_transformer_heads` | int | `4` | — | Attention heads for `edge_aggregation: "set_transformer"`; `atom_feat_len` must be divisible by it. Ignored otherwise. |
| `poly_fusion` | string | `"sum"` | `"sum"`, `"gate"` | How the bond and poly messages combine in the dual conv: plain addition, or learned per-channel sigmoid gates over both messages (initialized near the additive baseline). |
| `use_coord_magnitude` | bool | `false` | — | Re-inject each atom's total raw coordination weight (log1p of the summed ECoN / shared_count) through a learned projection — the intensive weighted aggregation otherwise discards coordination magnitude. |
| `atom_pooling` | string | `"mean"` | `"mean"`, `"mean_max"`, `"attention"`, `"set2set"` | How atom embeddings are read out into one crystal vector. `mean` = global mean; `mean_max` = concat(mean, max) (2× width, surfaces the most active atom); `attention` = learned per-atom softmax weighting (single step); `set2set` = LSTM-driven multi-step attention readout (Vinyals 2015), 2× width. |
| `set2set_steps` | int | `3` | — | Number of Set2Set processing steps. Only used when `atom_pooling` is `"set2set"`. |
| `normalize_features` | bool | `true` | — | Standardize node/edge/poly input features (per-feature z-score) using stats computed on the training split. Strongly recommended. |
| `feature_stat_graphs` | int | `4000` | — | Max training graphs sampled to compute the normalization stats (only used when `normalize_features` is true). |

### SC / non-SC sampling

| Key | Type | Default | Valid values | Description |
|-----|------|---------|--------------|-------------|
| `SC_to_non_SC_ratio` | float or string | `"inf"` (none) | `> 0`, or `"inf"`/`"infinity"`/`"none"`/`null` | SC count ÷ non-SC count. **Governs the composition of every split — train, val, and test.** `1.0` = one non-SC per SC; `2.0` = half as many non-SC as SC; `"inf"`/omitted = no non-SC in **any** split (so non-SC never leak into the eval sets). Train re-samples its non-SC fresh each epoch; val/test get a fixed ratio-sized non-SC subset. Requires the index to actually contain non-SC rows (build with `--nonsc-source`). |

### Optimizer / schedule

| Key | Type | Default | Valid values | Description |
|-----|------|---------|--------------|-------------|
| `optim` | string | `"SGD"` | `"SGD"`, `"Adam"`, `"AdamW"` | Optimizer. Unknown values raise an error. `AdamW` uses decoupled weight decay (better regularization than `Adam`'s coupled L2 when `weight_decay > 0`; identical when it's 0). |
| `momentum` | float | `0.9` | — | SGD momentum (ignored by Adam/AdamW). |
| `weight_decay` | float | `0` | — | Weight decay (coupled L2 for SGD/Adam; decoupled for AdamW). A light value like `1e-4` is a cheap regularizer for an over-capacity model. |
| `lr_milestones` | list[int] | `[100]` | — | Epochs at which the LR is multiplied by 0.1 (`MultiStepLR`, gamma fixed at 0.1). Ignored during the SWA phase (see below). |

### Regularization & generalization

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `dropout` | float | `0.0` (regression), `0.5` (classification) | Dropout probability on the pooled crystal vector before the readout MLP. Off by default for regression; set e.g. `0.2`–`0.5` to regularize a model with a large train/val gap. |
| `swa` | bool | `false` | Enable **Stochastic Weight Averaging**: after `swa_start`, average the weights visited under a low constant LR and use that average for the final test eval (saved as `<out>_swa.pth.tar`). The model uses LayerNorm (no BatchNorm), so no `update_bn` pass is needed. Often a few % MAE in the post-plateau regime. |
| `swa_start` | int | `0.75 × epochs` | Epoch to begin averaging. Should be after the LR has decayed / the curve has plateaued. |
| `swa_lr` | float | `0.05 × learning_rate` | Constant LR (`SWALR`) held during the SWA phase. |
| `model_seed` | int | `null` | Seeds torch before weight init. Vary it across runs to build a **diverse ensemble** (without it, every run inits identically). Combine the runs' predictions with `scripts/ensemble.py`. |

### Logging / output

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `print_split` | int | `10` | Print a progress line every N batches. |
| `model_data_dir` | string | `"model_data"` | Parent directory under which each run gets its own subdirectory. |
| `run_tag` | string | `"MPNN"` | Label prefixed to the run directory name (use `"Orig"` for baseline CGCNN). |
| `out_file` | string | (required) | Now interpreted as a **basename**: its directory part is ignored and all artifacts are written inside the run directory using its basename as the file prefix. |

### Cluster resources (`slurm`) — used only by `scripts/deploy.sh`

A config may carry an optional top-level **`slurm`** object to request different
cluster resources per experiment. It is read by `scripts/deploy.sh run` when
submitting the SLURM job; `MPNNMain.py` ignores it (unknown key), so the same
file drives both training and the resource request. Any key present **overrides**
the matching default in `deploy.sh`'s `EDIT THIS BLOCK`; missing keys keep that
default.

| Key | Maps to | Example |
|-----|---------|---------|
| `partition` | `--partition` | `"a100"` |
| `account` | `--account` | `"tmcquee2-paradim_gpu"` |
| `time` | `--time` | `"24:00:00"` |
| `gpus` | `--gpus` | `1` |
| `cpus` | `--cpus-per-task` | `24` |
| `mem` | `--mem` | `"96G"` |
| `mail_user` | `--mail-user` | `"you@jh.edu"` |

```json
{
    "num_workers": 12,
    "slurm": { "cpus": 24, "mem": "96G", "time": "24:00:00" }
}
```

Keep `cpus` ≥ `num_workers` + 1, and size `mem` for the per-worker sample cache
(see `graph_cache_size` above). See `configs/mpnn_basic_rockfish.json` for a
worked example and [`scripts/README.md`](../scripts/README.md) for the deploy
workflow.

---

## Legacy / removed keys

- **`aggregation`** — renamed to `edge_aggregation`. The old key is still read as
  a fallback, so existing configs keep working, but prefer `edge_aggregation`.
- **`n_nonsc`** — removed. Non-SC sampling is now controlled by
  `SC_to_non_SC_ratio` (a ratio, applied to both tasks), not an absolute count.

---

## Outputs

Each run creates a self-contained directory nested by date and run tag:
`<model_data_dir>/<YYYY-MM-DD>/<run_tag>/<run_tag>_<YYYY-MM-DD_HH-MM-SS>/` (e.g.
`model_data/2026-06-03/MPNN/MPNN_2026-06-03_14-30-12/`). The nesting keeps
`model_data/` navigable as runs accumulate; the leaf keeps the full run id so
it's self-describing in isolation. Inside, artifacts use the `out_file`
basename as their prefix (shown below as `<base>`):

| File | When | Contents |
|------|------|----------|
| `config.json` | always | Copy of the exact config used for this run. |
| `metadata.json` | always | Resolved hyperparameters, feature dims, split sizes, and model size — total/trainable params, effective train samples/epoch, and **params per train sample**. |
| `<base>_epoch_log.csv` | always | Per-epoch metrics (loss, MAE/acc, val metrics, LR, time, is_best) **plus resource telemetry**: `train_time_sec` / `data_time_sec` (time in the train loop vs. waiting on the dataloader), `gpu_util_pct` (mean), `gpu_mem_gb` (peak), `cpu_pct` (mean, process tree incl. workers; 100 = one core), `rss_gb` (peak). Resource columns need `psutil` + `nvidia-ml-py` (blank otherwise). |
| `<base>_checkpoint.pth.tar` | always | Latest checkpoint (model + optimizer + normalizer + args). |
| `<base>_model_best.pth.tar` | always | Best checkpoint (regression: lowest realistic-val MAE; classification: best realistic-val `selection_metric`). |
| `<base>_losstrain.csv.npy`, `<base>_lossval.csv.npy` | regression | Train/val loss curves. |
| `<base>.csv` | regression test | `(cif_id, true, pred)` on the realistic test set. |
| `<base>_test_balanced.csv` | regression/classification test | Test predictions on the balanced set (only written when the eval split contains non-SC). |
| `<base>_test_realistic.csv` | classification test | `(cif_id, true_label, p_sc)` on the realistic test set. |

The train/val/test split is **per-class stratified**. val and test are reported
in a **realistic** form (the composition set by `SC_to_non_SC_ratio`) and a
**balanced** form (1:1, drawn from the same ratio-limited non-SC pool). The two
coincide when the ratio is ≥ 1, and both are SC-only when the ratio is `inf`
(non-SC are then excluded from every split, not just train).

### Managing accumulated runs

`scripts/deploy.sh fetch` regenerates `model_data/index.csv` — one row per run
(date, tag, target, params, epochs trained vs target, completeness, and the
best-epoch val metrics) so runs are comparable without opening folders.

- `./scripts/deploy.sh reorg` migrates any older **flat** runs into the
  `<date>/<run_tag>/` layout on both the cluster and locally (idempotent; new
  runs are already born nested). Add `--dry-run` to preview the move plan.
- `./scripts/deploy.sh archive <rel_path> [...]` retires a run (use the
  `rel_path` from `index.csv`) into `model_data/.archive/` on both sides; `fetch`
  skips `.archive`, so retired runs stop being pulled back and the clutter is
  gone from the active tree for good (still recoverable under `.archive/`).

---

## Example configs

### Regression (T_c), SC-only — current baseline

```json
{
    "index_path":         "database/datafiles/MP/SC_MP_V4.pickle",
    "out_file":           "CNN/MPNN/mpnn_result",
    "task":               "regression",

    "max_num_nbr":        14,
    "max_num_poly_nbr":   16,
    "graph_cache_size":   4096,
    "num_workers":        4,

    "atom_feat_len":      64,
    "edge_hidden_dim":    128,
    "n_conv":             3,
    "h_feat_len":         128,
    "n_hidden":           1,
    "use_poly_edges":     true,
    "edge_aggregation":   "ecn_weighted",
    "atom_pooling":       "mean",
    "normalize_features": true,

    "SC_to_non_SC_ratio": "inf",

    "optim":              "AdamW",
    "learning_rate":      0.001,
    "weight_decay":       0.00001,
    "lr_milestones":      [500, 800],
    "epochs":             1000,

    "batch_size":         64,
    "val_ratio":          0.1,
    "test_ratio":         0.1,
    "print_split":        10
}
```

### Regression on the MP energy dataset (multi-target index)

Trains on the experimental-MP benchmark, selecting one of its stored targets via
`target_column`. The index is built from `mp_energy.csv` (which carries both
`e_above_hull` and `formation_energy_per_atom` columns); switch targets by
changing `target_column` alone — no rebuild needed.

```json
{
    "index_path":         "database/datafiles/MP_Energy/MP_Energy_V4.pickle",
    "out_file":           "CNN/MPNN/mpnn_eform",
    "task":               "regression",
    "target_column":      "formation_energy_per_atom",

    "use_poly_edges":     true,
    "atom_pooling":       "set2set",
    "normalize_features": true,

    "SC_to_non_SC_ratio": "inf",

    "optim":              "AdamW",
    "learning_rate":      0.001,
    "weight_decay":       0.00001,
    "epochs":             1000,

    "batch_size":         64,
    "val_ratio":          0.1,
    "test_ratio":         0.1
}
```

### Classification (SC vs non-SC), 1:1 balanced

Requires an index built with non-SC rows
(`build-db --kind cgv4 --nonsc-source ...`).

```json
{
    "index_path":         "database/datafiles/MP/SC_MP_V4.pickle",
    "out_file":           "CNN/MPNN/mpnn_clf",
    "task":               "classification",

    "use_poly_edges":     true,
    "atom_pooling":       "attention",
    "normalize_features": true,

    "SC_to_non_SC_ratio": 1.0,

    "optim":              "Adam",
    "learning_rate":      0.001,
    "epochs":             300,

    "batch_size":         64,
    "val_ratio":          0.1,
    "test_ratio":         0.1
}
```
