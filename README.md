# ML_SC_Project — Chemically and Geometrically Motivated Crystal Graph Neural Networks

This project predicts and screens superconductors from crystal structure. It frames the
problem as a **two-stage hurdle**: first classify whether a material is a superconductor
candidate (SC vs. non-SC), then regress the critical temperature T_c for superconductors.

Two graph-neural-network families are trained and compared:

- a **baseline CGCNN** ([txie-93/cgcnn](https://github.com/txie-93/cgcnn)), and
- an **MPNN** with learned edge features that consumes richer `crystal_graph_v4` graphs
  (this project's contribution).

The earlier coordination-environment variant (`CGCNNCoordEnv`) has been retired and
superseded by the MPNN.

## Background

Computational prediction of materials properties has a long, successful history,
but some phenomena — superconductivity in particular — remain poorly described by
analytical or first-principles methods. Because the materials properties that give
rise to superconductivity interact in subtle ways, and because there are only on the
order of 30,000 known superconductors (limited training data), the **representation**
of a material is critical. This project represents composition as per-site feature
vectors (element, occupancy, oxidation/valence, local geometry) and structure via the
crystal graph, then learns superconductivity (SC/non-SC) and T_c from them.

## Project Structure

```
ML_SC_Project/
├── main.py                 # entry point: build-db / download-nonsc / download-energy / train / train-mpnn / plot
├── plot.py                 # plots CNN predictions vs. targets
├── requirements.txt        # pinned Python dependencies (Python 3.11)
│
├── scripts/
│   └── deploy.sh           # push code/data to a SLURM cluster (cluster) and submit jobs
│
├── configs/
│   ├── basic.json          # baseline CGCNN, regression (T_c)
│   ├── classify_basic.json # baseline CGCNN, classification (SC/non-SC)
│   └── mpnn_basic.json     # MPNN (crystal_graph_v4), regression
│
├── database/               # dataset construction
│   ├── database_main.py    # core data-prep library (generators + helpers)
│   ├── crystal_graph_v4_import.py  # rich graph builder used by the cgv4 kind
│   ├── atom_init.json      # per-element feature vectors used by the CGCNN
│   ├── MP/                 # superconductor (3DSC_MP) data
│   │   ├── 3DSC_MP.csv     # raw 3DSC dataset (headered: cif paths + tc)
│   │   ├── id_prop.csv     # headerless "<cif_filename>,<tc>" rows
│   │   ├── id_prop_basic_combined.pickle  # SC + non-SC, labeled (classifier input)
│   │   └── cifs/           # CIF structure files
│   ├── Non_SC_DB_MP/
│   │   ├── Download_MP_data.py  # downloads non-superconductors from the MP API
│   │   ├── Non_SC.csv      # headerless "<material_id>.cif,0.0"
│   │   └── cifs/           # downloaded non-SC CIFs
│   └── MP_Energy/          # MP energy-target dataset (formation energy / e_above_hull)
│       ├── Download_MP_energy.py  # downloads experimental MP materials + energy targets
│       ├── mp_energy.csv   # header "cif,material_id,e_above_hull,formation_energy_per_atom"
│       └── cifs/           # downloaded CIFs
│
└── CNN/                    # the models (adapted from txie-93/cgcnn)
    ├── CGCNNMain.py        # baseline CGCNN trainer (JSON-config; regression + classification)
    ├── OriginalCGCNN/      # baseline CGCNN
    │   ├── CGCNNMainOrig.py  # standalone argparse trainer (not wired into main.py)
    │   ├── CGCNNOrig.py    # model
    │   └── data.py         # dataset, collate, loaders, BalancedEpochSampler
    └── MPNN/               # message-passing network
        ├── MPNNMain.py     # MPNN trainer (regression + classification)
        ├── MPNNModel.py    # CrystalMPNN model
        └── MPNNData.py     # graph loader, collate, loaders, BalancedEpochSampler
```

### Script reference

| Script | Role | Entry point |
| --- | --- | --- |
| `main.py` | Project CLI for the whole workflow: build datasets, download negatives, train + evaluate, plot. | `python main.py {build-db,download-nonsc,train,train-mpnn,plot} ...` |
| `plot.py` | `plot_results()` — reads a results CSV, computes MSE, scatter-plots predicted vs. target T_c. | `python main.py plot` |
| `database/database_main.py` | Core data-prep library: `generate_atom_init`, `generate_Basic_DB`, `generate_CGv4_DB`. | imported (called by `main.py`) |
| `database/Non_SC_DB_MP/Download_MP_data.py` | `gen_dataset` — pulls non-superconductors (by band gap) from the MP API into CIFs + a prop CSV. Needs `MP_API_KEY`. | `python main.py download-nonsc` |
| `database/MP_Energy/Download_MP_energy.py` | `gen_dataset` — pulls experimentally-observed MP materials into CIFs + a CSV with `e_above_hull` and `formation_energy_per_atom` target columns (training-time target chosen via the config's `target_column`). Needs `MP_API_KEY`. | `python main.py download-energy` |
| `CNN/CGCNNMain.py` | Baseline CGCNN training/validation/test loop with checkpointing; regression or classification per config. | `python main.py train <config.json>` |
| `CNN/MPNN/MPNNMain.py` | MPNN trainer over `crystal_graph_v4` graphs; regression or classification per config. | `python main.py train-mpnn <config.json>` |
| `CNN/OriginalCGCNN/*` | Baseline CGCNN (model, data loader + sampler, standalone `argparse` trainer). | imported (+ optional standalone) |
| `CNN/MPNN/MPNNModel.py`, `MPNNData.py` | MPNN model and graph dataset/loaders. | imported |

### Files read and generated per command

Paths are relative to the project root (the working directory you run from).

| Command / function | Reads | Generates |
| --- | --- | --- |
| `main.py build-db --kind atom-init` | pymatgen element data | `database/atom_init.json` |
| `main.py build-db --kind basic` | id→property CSV(s) + CIF dir(s) | `<output>.{pickle,csv}` with columns `id, value, struc_dict, label` |
| `main.py build-db --kind cgv4` | id→property CSV(s) + CIF dir(s) | per-material JSONs in `graphs_v4/` + index `<output>.{pickle,csv}` (`id, value, graph_path, label` + every recognized target column present: `tc`, `e_above_hull`, `formation_energy_per_atom`) |
| `main.py download-nonsc` | MP API (needs `MP_API_KEY`) | `Non_SC.csv` + one CIF per material |
| `main.py download-energy` | MP API (needs `MP_API_KEY`) | `mp_energy.csv` (`cif, material_id, e_above_hull, formation_energy_per_atom`) + one CIF per material |
| `main.py train <config>` | config JSON; `<dataset_rd>/<dataset>` pickle + `<dataset_rd>/<atom_init>` | checkpoints, predictions, `<out_file>_epoch_log.csv` (see below) |
| `main.py train-mpnn <config>` | config JSON; `index_path` (cgv4 index) + the referenced graph JSONs | same output families as `train` |
| `plot.py` | a results CSV (default `CNN/test_result.csv`) | a matplotlib plot |

Outputs by task: **regression** → `<out_file>.csv` (`cif_id, target_tc, predicted_tc`),
`<out_file>_model_best.pth.tar`, loss `.npy` dumps; **classification** →
`<out_file>_test_realistic.csv` and `<out_file>_test_balanced.csv` (`cif_id, true_label, p_sc`)
plus the best checkpoint. All trainers also write `<out_file>_epoch_log.csv` (per-epoch metrics).

> The dataset pickle embeds each structure as `struc_dict` (`Structure.as_dict()`), so the
> baseline CGCNN does **not** read CIFs at train time — CIFs are only consumed when the
> pickle (or cgv4 graphs) are built. The pickle and `atom_init.json` must share `dataset_rd`.

## Usage

> Run all commands from the project root. Paths like `database/MP/cifs/` are resolved
> relative to it.

### 1. Build the feature file and datasets

```bash
# Per-element feature vectors -> database/atom_init.json
python main.py build-db --kind atom-init

# SC-only basic dataset (id, value, struc_dict, label) -> database/MP/id_prop_basic.{pickle,csv}
python main.py build-db --kind basic

# Quick smoke test on the first 50 rows
python main.py build-db --kind basic --limit 50
```

**Database file kinds**

| `--kind` | Output (default) | Columns | Used by |
| --- | --- | --- | --- |
| `atom-init` | `database/atom_init.json` | `{Z: [Z, block, valence, atomic_radius, electron_affinity, ionization_energy, electronegativity, electron_affinity]}` | CGCNN |
| `basic` | `database/MP/id_prop_basic.{pickle,csv}` | `id, value, struc_dict, label` | CGCNN |
| `cgv4` | `database/MP/id_prop_v4.{pickle,csv}` (+ `graphs_v4/`) | `id, value, graph_path, label` (+ any target columns: `tc`, `e_above_hull`, `formation_energy_per_atom`) | MPNN |

**Useful flags** (`python main.py build-db -h` for the full list):

- `--source CSV CIF_DIR` — a superconductor id→property CSV and its cif dir (label 1); repeatable.
- `--nonsc-source CSV CIF_DIR` — a non-superconductor source (label 0); repeatable.
- `--output PATH` — output path/prefix.
- `--has-header` — source CSV has a `cif`/`tc` header (e.g. `3DSC_MP.csv`); default is headerless.
- `--limit N` — first `N` rows per source (quick testing).
- `--parallel` / `--batch-size N` — `basic` only: thread the CIF parsing.

### 2. Download non-SC negatives and build the combined classifier dataset

```bash
# Requires the MP_API_KEY environment variable; --limit caps the count for class balance
python main.py download-nonsc --limit 5000

# Combine SC (label 1) and non-SC (label 0) into one labeled dataset
python main.py build-db --kind basic \
  --source database/MP/id_prop.csv database/MP/cifs/ \
  --nonsc-source database/Non_SC_DB_MP/Non_SC.csv database/Non_SC_DB_MP/cifs/ \
  --output database/MP/id_prop_basic_combined
```

The `label` column (1 = SC, 0 = non-SC) is set by which flag the source came from — it is
**not** derived from T_c, because ~31% of the SC dataset has T_c = 0.0.

### 2b. (Optional) Build an MP energy-target dataset

A downloader pulls experimentally-observed materials from the Materials Project to use as a
general benchmark (a sanity check on how the models learn, independent of the superconductor
task). It records two thermodynamic targets per material:

```bash
# Requires MP_API_KEY. theoretical=False (experimental only).
python main.py download-energy

# Quick smoke test
python main.py download-energy --limit 50
```

This writes one CIF per material into `database/MP_Energy/cifs/` and `mp_energy.csv` with
columns `cif, material_id, e_above_hull, formation_energy_per_atom`. `e_above_hull` is the
energy above the convex hull (eV/atom, ≥ 0; 0 = stable); `formation_energy_per_atom` is the
formation enthalpy (eV/atom, typically negative). **Both targets are kept as columns** — no
target is baked in at download time. Flags: `--include-theoretical`, `--limit N`,
`--chunk-size N`. Build graphs from it (`--has-header` reads the column row):

```bash
python main.py build-db --kind cgv4 --has-header \
  --source database/MP_Energy/mp_energy.csv database/MP_Energy/cifs/ \
  --output database/MP_Energy/id_prop_v4_energy

# then train on it (ready-made config; formation energy as the target):
python main.py train-mpnn configs/mpnn_eform.json
```

The cgv4 index preserves **every recognized target column** present in the source
(`tc`, `e_above_hull`, `formation_energy_per_atom`). You then choose which to regress on at
train time with the config's `target_column` key — e.g. `"target_column": "formation_energy_per_atom"`
or `"e_above_hull"` — so switching targets needs no rebuild. The bundled `configs/mpnn_eform.json`
sets this (and `target_transform: none`, since formation energies are negative and `log1p` would
be invalid). When `target_column` is unset the trainer falls back to the legacy `value`/`tc`
column, so existing SC configs are unaffected; an index with no `tc` (like the energy dataset)
has no usable default and errors out asking you to set `target_column`, rather than silently
regressing on the wrong target. Rows whose chosen target is missing (NaN) are dropped automatically.

### 3. Train a model (`main.py train` / `main.py train-mpnn`)

Training is driven by a JSON config. The `"task"` key selects the objective:

```bash
# Baseline CGCNN, T_c regression (superconductors only)
python main.py train configs/basic.json

# Baseline CGCNN, SC/non-SC classification (combined dataset)
python main.py train configs/classify_basic.json

# MPNN over crystal_graph_v4 graphs
python main.py train-mpnn configs/mpnn_basic.json
```

For **classification**, the trainer does a stratified split, trains with a
`BalancedEpochSampler` (all SC + a fresh random `n_nonsc` non-SC each epoch), and reports
accuracy / precision / recall / F1 / AUC on both a *realistic* (true-imbalance) and a
*balanced* validation/test split. Model selection uses realistic-split AUC.

### 4. Plot predictions (`main.py plot`)

```bash
python main.py plot                              # reads CNN/test_result.csv
python main.py plot --results path/to/other.csv  # or a custom results file
```

### End-to-end (regression baseline)

```bash
python main.py build-db --kind atom-init   # element feature file (if not present)
python main.py build-db --kind basic       # dataset pickle
python main.py train configs/basic.json    # train + evaluate
python main.py plot                        # visualize results
```

## Running on a SLURM cluster (cluster)

The `crystal_graph_v4` MPNN trains best on a GPU. `scripts/deploy.sh` automates
shipping the project to the JHU **cluster** cluster and submitting SLURM jobs.
It separates the **one-time** setup (build a conda env, push the 14 GB graph
database) from the **per-run** step (push the tiny code+config, submit a job), so
each training run only transfers kilobytes.

```
  LOCAL (your machine)                         CLUSTER (<cluster-login-host>)
  ─────────────────────                        ──────────────────────────────────
  build-db --kind cgv4  ──┐
   → graphs_v4/ + index   │  deploy.sh sync-data (once)
   pickles                ├───────────────────▶ /data/.../ML_SC_Proj/database/MP/
                          │  deploy.sh setup-env (once)
  requirements.txt ───────┴───────────────────▶ conda env `ml_sc`
                             deploy.sh run <cfg>
  configs/*.json, CNN/MPNN/*.py ──────────────▶ sbatch job → python MPNNMain.py
                                                   → model_data/<run>/ + logs/
  model_data/, logs/  ◀───────────────────────  deploy.sh fetch
   → main.py plot
```

The SLURM job runs `python CNN/MPNN/MPNNMain.py <config>` from the project root —
exactly what `main.py train-mpnn <config>` does locally — so results are
identical to a local run, just on a GPU node.

**One-time setup** (edit the `EDIT THIS BLOCK` at the top of `deploy.sh` first —
host/user/path, SLURM partition + account, conda env name):

```bash
./scripts/deploy.sh setup-env     # create conda env `ml_sc` from requirements.txt
./scripts/deploy.sh sync-data     # rsync graphs_v4/ (14 GB, ~89k files) + pickles
```

**Per training run:**

```bash
./scripts/deploy.sh run configs/mpnn_basic.json   # push code+config, submit job
./scripts/deploy.sh status                        # squeue for your jobs
./scripts/deploy.sh logs <jobid>                  # tail the live SLURM log
./scripts/deploy.sh fetch                         # pull model_data/ + logs/ back
python main.py plot --results model_data/<run>/<base>.csv   # visualize locally
```

See [`scripts/README.md`](scripts/README.md) for the full subcommand reference,
the config block, what each step transfers, and troubleshooting.

## References

[1] Pogue, Elizabeth & New, Alexander & McElroy, Kyle & Le, Nam & Pekala, Michael & McCue, Ian & Gienger, Eddie & Domenico, Janna & Hedrick, Elizabeth & McQueen, Tyrel & Wilfong, Brandon & Piatko, Christine & Ratto, Christopher & Lennon, Andrew & Chung, Christine & Montalbano, Timothy & Bassen, Gregory & Stiles, Christopher. (2022). Closed-loop machine learning for discovery of novel superconductors. 10.48550/arXiv.2212.11855.

[2] Quinn Margaret R., McQueen Tyrel M. (2022) Identifying New Classes of High Temperature Superconductors With Convolutional Neural Networks, Frontiers in Electronic Materials https://www.frontiersin.org/articles/10.3389/femat.2022.893797

[3] Goodall, R.E.A., Lee, A.A. Predicting materials properties without crystal structure: deep representation learning from stoichiometry. Nat Commun 11, 6280 (2020). https://doi.org/10.1038/s41467-020-19964-7

[4] Cheng, J., Zhang, C. & Dong, L. A geometric-information-enhanced crystal graph network for predicting properties of materials. Commun Mater 2, 92 (2021). https://doi.org/10.1038/s43246-021-00194-3
