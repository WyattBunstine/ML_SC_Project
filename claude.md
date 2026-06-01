# Chemically and Geometrically Motivated Crystal Graph Neural Networks for Superconductor Screening

This project predicts and screens superconductors from crystal structure. It has two
goals, realized as a **two-stage hurdle**:

1. **Stage 1 — classify** whether a material is a superconductor candidate at all
   (SC vs. non-SC), trained on 3DSC_MP superconductors (~5,773) plus large-band-gap
   non-superconductors pulled from the Materials Project (~55k).
2. **Stage 2 — regress** the critical temperature T_c (in Kelvin) for superconductors.

Two graph-neural-network families are trained and compared:
- **Baseline CGCNN** (`CNN/OriginalCGCNN/`, after [txie-93/cgcnn](https://github.com/txie-93/cgcnn)),
  driven by `CNN/CGCNNMain.py`.
- **MPNN** (`CNN/MPNN/`) — a message-passing network with learned edge features that
  consumes the richer `crystal_graph_v4` graphs.

> The earlier coordination-environment variant (`CGCNNCoordEnv`) has been **retired**
> and superseded by the MPNN. It is gone from the codebase (still in git history).

## Working Style
When working on this project, iterate with the user on ideas before making implementations. When the user asks for something to be done, think through the possible impacts of the changes and specifically how this might change or reduce the accuracy of the scripts. After thinking through, ask the user about their preferred way of implementing the changes. Highlight different possible implementations and the benefits and drawbacks of each approach. Do not make changes large until explicitly told to by the user. Once a change is made, make sure that the script still runs properly and make additional changes if there are runtime errors or similar coding errors. If the result changes to something undesirable, highlight this for the user and suggest additional changes to address it, but do not make additional changes until told to do so by the user.

---

## Project Structure

```
ML_SC_Project/
├── main.py                           # CLI: build-db / download-nonsc / train / train-mpnn / plot
├── plot.py                           # Scatter-plot predictions vs targets
├── configs/
│   ├── basic.json                    # Baseline CGCNN, regression (T_c)
│   ├── classify_basic.json           # Baseline CGCNN, classification (SC/non-SC)
│   └── mpnn_basic.json               # MPNN (crystal_graph_v4), regression
├── database/
│   ├── database_main.py              # DB generation: generate_atom_init(),
│   │                                 #   generate_Basic_DB(), generate_CGv4_DB()
│   ├── crystal_graph_v4_import.py    # build_crystal_graph_from_cif() — rich graph builder for cgv4
│   ├── atom_init.json                # Per-element feature vectors (84 elements, 8 features each)
│   ├── MP/                           # superconductor (3DSC_MP) data
│   │   ├── id_prop.csv               # Headerless: filename.cif, T_c (5,773 rows)
│   │   ├── 3DSC_MP.csv               # Full 3DSC dataset with metadata
│   │   ├── id_prop_basic.pickle      # SC-only basic dataset: [id, value, struc_dict, label]
│   │   ├── id_prop_basic_combined.*  # SC + non-SC combined (label 1/0) — classifier input
│   │   ├── id_prop_v4.pickle/.csv    # cgv4 index: [id, value, graph_path, label]
│   │   ├── graphs_v4/                # cgv4 per-material JSON graphs (+ failed.txt log)
│   │   └── cifs/                     # CIF structure files
│   └── Non_SC_DB_MP/                 # non-superconductor (large-band-gap) negatives
│       ├── Download_MP_data.py       # gen_dataset(): MP API download (needs MP_API_KEY)
│       ├── Non_SC.csv                # headerless filename.cif,0.0 (T_c placeholder)
│       └── cifs/                     # downloaded non-SC CIFs
└── CNN/
    ├── CGCNNMain.py                  # Baseline CGCNN trainer (main.py train) — regression + classification
    ├── classify_result*, test_result*  # outputs (predictions, checkpoints, epoch logs)
    ├── OriginalCGCNN/                # baseline CGCNN
    │   ├── CGCNNMainOrig.py          # standalone argparse trainer (not wired into main.py)
    │   ├── CGCNNOrig.py              # model (classification-capable)
    │   └── data.py                   # dataset, collate, loaders, BalancedEpochSampler
    └── MPNN/                         # message-passing net (main.py train-mpnn)
        ├── MPNNMain.py               # MPNN trainer — regression + classification
        ├── MPNNModel.py              # CrystalMPNN model (classification-capable)
        └── MPNNData.py               # graph loader, collate, loaders, BalancedEpochSampler
```

---

## Data Pipeline

### Step 0: Download non-SC negatives (`main.py download-nonsc`)
`Non_SC_DB_MP/Download_MP_data.py::gen_dataset()` queries the Materials Project for
large-band-gap (default ≥ 1.0 eV) materials and writes one CIF each plus a headerless
`Non_SC.csv` (`<material_id>.cif,0.0` — the T_c is a placeholder; these rows are marked
non-SC by the database `label`, **not** by this value). Requires the **`MP_API_KEY`**
environment variable. `--limit N` caps the count for class balance.

### Step 1: Build Database (`main.py build-db --kind <kind>`)
Sources are `(CSV, CIF_DIR)` pairs. `--source` rows are labeled **SC (label 1)**;
`--nonsc-source` rows are labeled **non-SC (label 0)**. The label is written to the
DB's `label` column and is what the Stage-1 classifier trains on — it is **not** derived
from T_c (≈31% of the SC dataset has T_c = 0.0, so a value-derived label would be wrong).
Three kinds:

- **`atom-init`** → `generate_atom_init()`: per-element feature file `atom_init.json`
  (not tied to a dataset; regenerating won't byte-match the committed file because
  pymatgen's electron-affinity data changed).
- **`basic`** → `generate_Basic_DB()`: parse CIF → pymatgen Structure → `struc_dict`,
  emitting `[id, value, struc_dict, label]`. Rows accumulate in a list and the DataFrame
  is built once (O(n)); progress prints every 500 structures. `--parallel` threads it.
- **`cgv4`** → `generate_CGv4_DB()`: builds a rich `crystal_graph_v4` graph per material
  (one JSON each in `graphs_v4/`) plus an index `[id, value, graph_path, label]`. Fully
  resumable (skips existing JSONs; failures → `graphs_v4/failed.txt`). Consumed by the MPNN.

Common flags: `--has-header`, `--limit N` (first N rows/source), `--output` (path/prefix),
`--nonsc-source CSV CIF_DIR` (repeatable).

### Step 2: Training (`main.py train <config>` / `main.py train-mpnn <config>`)
Both trainers are JSON-config driven and support two tasks via the `"task"` key:

- **`"regression"`** (default): predict T_c. L1Loss (MAE), train/val/test split, T_c
  normalized by training mean/std. Trained on superconductors only.
- **`"classification"`**: predict SC vs non-SC. 2-class LogSoftmax head + NLLLoss, with:
  - **Stratified split** by label into train/val/test (no leakage).
  - **`BalancedEpochSampler`**: each epoch trains on *all* train-SC + a fresh random
    `n_nonsc` (default 5000) train-non-SC, re-drawn per epoch — balanced batches while
    eventually covering the large non-SC pool.
  - **Two eval views per split**: `realistic` (true imbalance) and `balanced` (non-SC
    subsampled to the SC count). Metrics: accuracy, precision, recall, F1, AUC.
  - **Model selection (`is_best`) by AUC on the realistic val split** (falls back to F1
    if a split is single-class).

The baseline CGCNN trains via `CNN/CGCNNMain.py` on the `struc_dict` pickle; the MPNN
trains via `CNN/MPNN/MPNNMain.py` on the `graphs_v4` index. Both share the same crystal
graph construction idea (up to ~12–14 neighbors, Gaussian-expanded distances for the CGCNN;
learned edge features for the MPNN).

### Step 3: Output
- **Regression**: `<out_file>.csv` (`cif_id, target_tc, predicted_tc`),
  `<out_file>_model_best.pth.tar`, `<out_file>_losstrain.csv.npy`, `<out_file>_lossval.csv.npy`.
- **Classification**: `<out_file>_test_realistic.csv` and `<out_file>_test_balanced.csv`
  (`cif_id, true_label, p_sc`), plus the best checkpoint.
- **Per-epoch telemetry** (`<out_file>_epoch_log.csv`), flushed every epoch by all trainers:
  - regression: `epoch, train_loss, train_mae, val_loss, val_mae, lr, epoch_time_sec, is_best`
  - classification: `epoch, train_loss, train_acc, val_loss, val_acc, val_precision,
    val_recall, val_f1, val_auc, val_bal_acc, lr, epoch_time_sec, is_best`
- `plot.py`: scatter plot of predictions vs targets (regression), reports MSE.

---

## Key Configuration

| Key | Example | Meaning |
|-----|---------|---------|
| `task` | `"regression"` / `"classification"` | Training objective (default regression) |
| `dataset_rd` | `"database/MP"` | Directory holding the pickle + `atom_init.json` (CGCNN) |
| `dataset` | `"id_prop_basic_combined.pickle"` | Pickle name (CGCNN) |
| `index_path` | `"database/MP/id_prop_v4.pickle"` | cgv4 index (MPNN) |
| `atom_init` | `"atom_init.json"` | Per-element feature file (CGCNN) |
| `n_nonsc` | 5000 | [classification] non-SC sampled per epoch |
| `split_seed` | 123 | [classification] stratified-split seed |
| `atom_feat_len` / `n_conv` / `h_feat_len` / `n_hidden` | 64 / 3 / 128 / 1 | Model dims |
| `aggregation` | `"ecn_weighted"` / `"attention"` | [MPNN] edge aggregation |
| `batch_size` / `epochs` | 128 / 300–1000 | Training loop |
| `optim` / `learning_rate` / `momentum` / `weight_decay` / `lr_milestones` | `"SGD"` / 0.01 / 0.9 / 0 / [100] | Optimizer |
| `out_file` | `"CNN/classify_result"` | Output prefix |

Provided configs: `configs/basic.json` (baseline regression), `configs/classify_basic.json`
(baseline classification on the combined DB), `configs/mpnn_basic.json` (MPNN regression).

---

## Atom Features (`database/atom_init.json`)
Each element (Z = 1–84) has 8 features: `[Z, block (0=s,1=p,2=d,3=f), valence,
atomic_radius, electron_affinity, ionization_energy, electronegativity, electron_affinity]`.
The baseline CGCNN sums these per site weighted by occupancy → (N, 8), embedded to 64-d.
The MPNN instead uses the 12 node + 8 edge features defined in `MPNN/MPNNData.py`.

---

## Key Concepts

- **Two-stage hurdle**: a Stage-1 SC/non-SC classifier gates a Stage-2 T_c regressor.
  Stage 2 is the existing regression model (trained on SC only, T_c=0 rows kept). The
  composition command that gates one with the other is the remaining work.
- **Label**: 1 = superconductor, 0 = non-superconductor. Set per data source at build
  time (`--source` vs `--nonsc-source`), stored in the `label` column. Loaders default it
  to 1 when absent, so old pickles still work for regression.
- **Class imbalance** (~9.5:1 non-SC:SC): handled in training by the per-epoch
  `BalancedEpochSampler`, not by subsampling the database. Evaluated on both realistic and
  balanced splits.
- **Baseline vs MPNN**: baseline CGCNN (`OriginalCGCNN`) is the published-architecture
  control; the MPNN with `crystal_graph_v4` features is the project's contribution.
- **Target variable**: T_c in Kelvin (regression, L1Loss) or SC/non-SC class (NLLLoss).

---

## Running the Project

```bash
# Per-element feature file
python main.py build-db --kind atom-init

# SC-only basic dataset
python main.py build-db --kind basic

# Download non-SC negatives (needs MP_API_KEY); --limit caps the count
python main.py download-nonsc --limit 5000

# Combined SC + non-SC dataset (labels 1 / 0) for the classifier
python main.py build-db --kind basic \
  --source database/MP/id_prop.csv database/MP/cifs/ \
  --nonsc-source database/Non_SC_DB_MP/Non_SC.csv database/Non_SC_DB_MP/cifs/ \
  --output database/MP/id_prop_basic_combined

# Build crystal_graph_v4 graphs for the MPNN (add --nonsc-source for classification)
python main.py build-db --kind cgv4

# Train baseline CGCNN — regression (T_c) or classification (SC/non-SC)
python main.py train configs/basic.json
python main.py train configs/classify_basic.json

# Train MPNN (requires cgv4 graphs)
python main.py train-mpnn configs/mpnn_basic.json

# Plot regression results
python main.py plot --results CNN/test_result.csv
```

---

## Known Notes
- The MP API key is read from the **`MP_API_KEY`** environment variable (no longer hardcoded).
- `basic` DB build is ~linear in #structures (~20 ms each); the ~61k combined build takes
  roughly 20 min. The old `DataFrame.loc[...] = row` append was O(n²) and was replaced.
- ~31% of the SC dataset has T_c = 0.0; these are treated as SC (label 1) for the classifier
  and kept (as T_c=0) in the regressor.
- CIF files outnumber `id_prop` entries — not all CIFs are used.
- Requires `torch`, `pymatgen`, `mp-api`, `scikit-learn` (classification metrics), `pandas`, `numpy`.
