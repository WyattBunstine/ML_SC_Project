# ML_SC_Project — Chemically and Geometrically Motivated Crystal Graph Neural Network

This project predicts superconducting critical temperature (T_c) from crystal
structure using a Crystal Graph Convolutional Neural Network (CGCNN). It extends
the baseline CGCNN ([txie-93/cgcnn](https://github.com/txie-93/cgcnn)) with a more
chemically and geometrically motivated graph construction — most notably encoding
each site's **coordination environment** as an additional atom feature. The
original CGCNN is kept alongside as a baseline for comparison.

## Background

Computational prediction of materials properties has a long, successful history,
but some phenomena — superconductivity in particular — remain poorly described by
analytical or first-principles methods. Because the materials properties that give
rise to superconductivity interact in subtle ways, and because there are only on the
order of 30,000 known superconductors (limited training data), the **representation**
of a material is critical. This project represents composition as per-site feature
vectors (element, occupancy, oxidation/valence, coordination environment) and
geometry via the crystal graph, then learns T_c from them.

## Project Structure

```
ML_SC_Project/
├── main.py                 # << entry point: build-db / train / plot
├── plot.py                 # plots CNN predictions vs. targets
│
├── database/               # dataset construction
│   ├── database_main.py    # core data-prep library (generators + helpers)
│   ├── atom_init.json      # per-element feature vectors used by the CNN
│   ├── MP/                 # Materials Project / 3DSC data
│   │   ├── 3DSC_MP.csv     # raw 3DSC dataset (headered: cif paths + tc)
│   │   ├── id_prop.csv     # headerless "<cif_filename>,<tc>" rows
│   │   ├── id_prop.pickle  # parsed basic dataset (id, value, struc_dict)
│   │   └── cifs/           # ~10.9k CIF structure files
│   └── Non_SC_DB_MP/
│       └── Download_MP_data.py  # downloads non-superconductors from Materials Project
│
└── CNN/                    # the models (adapted from txie-93/cgcnn)
    ├── CGCNNMain.py        # << CNN training/eval entry point (JSON-config driven)
    ├── CGCNNCoordEnv/      # this project's modified CGCNN
    │   ├── CGCNNCE.py      # ConvLayer + CrystalGraphConvNet
    │   └── CEdata.py       # CIFData dataset; appends coordination env to atom features
    └── OriginalCGCNN/      # unmodified baseline for comparison
        ├── CGCNNMainOrig.py
        ├── CGCNNOrig.py
        └── data.py
```

### Script reference

| Script | Role | Entry point |
| --- | --- | --- |
| `main.py` | Project CLI / entry point for the whole workflow: build datasets, train + evaluate, plot. | `python main.py {build-db,train,plot} ...` |
| `plot.py` | `plot_results()` — reads a results CSV, computes MSE, and scatter-plots predicted vs. target T_c. | `python main.py plot` |
| `database/database_main.py` | Core data-prep library: `generate_atom_init`, `generate_Basic_DB`, `generate_CE_DB`. | imported (called by `main.py`) |
| `database/Non_SC_DB_MP/Download_MP_data.py` | `gen_dataset` — pulls non-superconductors (by band gap) from the Materials Project API into CIFs + a prop CSV. | imported |
| `CNN/CGCNNMain.py` | Main training/validation/test loop with checkpointing; selects the CE or original model from the config. | `python main.py train <config.json>` (wraps it) |
| `CNN/CGCNNCoordEnv/CGCNNCE.py` | The modified crystal graph conv-net model. | imported |
| `CNN/CGCNNCoordEnv/CEdata.py` | `CIFData` dataset reading the parsed pickle; builds atom features and appends the coordination-environment value. | imported |
| `CNN/OriginalCGCNN/*` | Unmodified baseline CGCNN (model, data loader, standalone `argparse` trainer). | `python CNN/OriginalCGCNN/CGCNNMainOrig.py <args>` |

### Files read and generated per script

Paths are relative to the project root (the working directory you run from).

| Script / function | Reads | Generates |
| --- | --- | --- |
| `main.py build-db --kind atom-init` | pymatgen's built-in element data (no project files) | `database/atom_init.json` |
| `main.py build-db --kind basic` | an id→property CSV (default `database/MP/id_prop.csv`) + its CIF files (default `database/MP/cifs/`) | `<output>.pickle` + `<output>.csv` (default `database/MP/id_prop_basic.{pickle,csv}`) |
| `main.py build-db --kind ce` | same as `basic` | `<output>.pickle` + `<output>.csv` (default `database/MP/id_prop_ce.{pickle,csv}`) |
| `plot.py` | `CNN/test_result.csv` | none — opens a matplotlib plot window |
| `database_main.generate_atom_init()` | pymatgen element data | `atom_init.json` (default `database/atom_init.json`) |
| `database_main.generate_Basic_DB()` | id→property CSV + CIF files | `<output>.pickle` + `<output>.csv` |
| `database_main.generate_CE_DB()` | id→property CSV + CIF files | `<output>.pickle` + `<output>.csv` (adds a `ce` column) |
| `Non_SC_DB_MP/Download_MP_data.gen_dataset()` | Materials Project API (network + API key) | a prop CSV (e.g. `Non_SC.csv`) + one CIF per material in the cif dir |
| `CNN/CGCNNMain.py` | config JSON (arg 1); `<dataset_rd>/<dataset>` pickle (`id, value, struc_dict[, ce]`) + `<dataset_rd>/<atom_init>` | `<out_file>_checkpoint.pth.tar`, `<out_file>_model_best.pth.tar`, `<out_file>.csv` (test predictions), `<out_file>_losstrain.csv.npy`, `<out_file>_lossval.csv.npy` |
| `CNN/OriginalCGCNN/CGCNNMainOrig.py` | argparse args; pickle + `atom_init.json` in the dataset root dir | same outputs as `CGCNNMain.py` |
| `CEdata.py` / `OriginalCGCNN/data.py` | the dataset pickle + `atom_init.json` | none — provide tensors to the data loader |
| `CGCNNCE.py` / `CGCNNOrig.py` | — | none — model definitions only |

> The dataset pickle embeds each structure as `struc_dict` (`Structure.as_dict()`), so
> the CNN does **not** read CIF files at train time — CIFs are only consumed when the
> pickle is built. The pickle and `atom_init.json` must live in the same `dataset_rd`.

## Usage

> Run all commands from the project root. The database code resolves relative paths
> such as `database/MP/cifs/`.

### Generating the database files (`main.py`)

`main.py` exposes a `build-db` subcommand that generates or regenerates the three
database files the CNN consumes. (If a target already exists it prints
`Regenerating`, otherwise `Generating`.)

```bash
# Per-element feature vectors -> database/atom_init.json
python main.py build-db --kind atom-init

# Parsed basic dataset (id, value, struc_dict) -> database/MP/id_prop_basic.{pickle,csv}
python main.py build-db --kind basic --parallel

# Basic dataset + per-site coordination environments (adds a 'ce' column)
python main.py build-db --kind ce

# Quick smoke test on just the first 50 rows
python main.py build-db --kind ce --limit 50

# Custom source(s): an id->property CSV and its cif directory (repeat --source for more)
python main.py build-db --kind basic --source database/MP/id_prop.csv database/MP/cifs/
```

**Database file kinds**

| `--kind` | Output (default) | Columns | Used by |
| --- | --- | --- | --- |
| `atom-init` | `database/atom_init.json` | `{Z: [Z, block, valence, atomic_radius, electron_affinity, ionization_energy, electronegativity, electron_affinity]}` | both models |
| `basic` | `database/MP/id_prop_basic.{pickle,csv}` | `id, value, struc_dict` | `OriginalCGCNN` |
| `ce` | `database/MP/id_prop_ce.{pickle,csv}` | `id, value, struc_dict, ce` | `CGCNNCoordEnv` |

**Useful flags** (see `python main.py -h` for the full list):

- `--source CSV CIF_DIR` — an id→property CSV and its cif directory; repeatable.
  Default: `database/MP/id_prop.csv database/MP/cifs/`.
- `--output PATH` — output path/prefix (kind-specific default; `basic`/`ce` append `.pickle`/`.csv`).
- `--has-header` — the source CSV has a header row with `cif`/`tc` columns (e.g. `3DSC_MP.csv`).
  Default assumes a headerless `<cif_filename>,<tc>` file like `id_prop.csv`.
- `--limit N` — process only the first `N` rows of each source (quick testing).
- `--parallel` / `--batch-size N` — `basic` only: parse CIFs across threads.
- `--timing` — `basic` only: print construction timing.
- `--max-z N` — `atom-init` only: generate features for `Z = 1 .. N-1` (default 85).

> **Note:** regenerating `atom-init` reproduces the original *logic* but will not
> byte-match the committed `atom_init.json` — pymatgen's electron-affinity reference
> data has changed since that file was created (only the electron-affinity columns differ).

### Training a model (`main.py train`)

The CNN is driven by a JSON config that points at the parsed pickle, the
`atom_init.json` feature file, and model/training hyperparameters. A ready-to-run
config for the basic (baseline) model is provided at `configs/basic.json`:

```bash
python main.py train configs/basic.json
```

This wraps `CNN/CGCNNMain.py` (running it as a subprocess so its package imports
resolve). The config selects the model (`"models"` containing `"ORIG"` uses the
baseline `OriginalCGCNN`, otherwise the `CGCNNCoordEnv` variant) and provides
`dataset_rd` (root dir holding both the pickle and `atom_init.json`), `atom_init`,
`dataset` (pickle name), `batch_size`, `epochs`, `learning_rate`, `optim`, etc.
Training also evaluates on the held-out test split and writes predictions to
`<out_file>.csv` (e.g. `CNN/test_result.csv`).

### Plotting predictions (`main.py plot`)

Scatter-plot the test predictions against their targets:

```bash
python main.py plot                              # reads CNN/test_result.csv
python main.py plot --results path/to/other.csv  # or a custom results file
```

### End-to-end

```bash
python main.py build-db --kind atom-init   # element feature file (if not present)
python main.py build-db --kind basic       # dataset pickle
python main.py train configs/basic.json    # train + evaluate
python main.py plot                        # visualize results
```

## References

[1] Pogue, Elizabeth & New, Alexander & McElroy, Kyle & Le, Nam & Pekala, Michael & McCue, Ian & Gienger, Eddie & Domenico, Janna & Hedrick, Elizabeth & McQueen, Tyrel & Wilfong, Brandon & Piatko, Christine & Ratto, Christopher & Lennon, Andrew & Chung, Christine & Montalbano, Timothy & Bassen, Gregory & Stiles, Christopher. (2022). Closed-loop machine learning for discovery of novel superconductors. 10.48550/arXiv.2212.11855.

[2] Quinn Margaret R., McQueen Tyrel M. (2022) Identifying New Classes of High Temperature Superconductors With Convolutional Neural Networks, Frontiers in Electronic Materials https://www.frontiersin.org/articles/10.3389/femat.2022.893797

[3] Goodall, R.E.A., Lee, A.A. Predicting materials properties without crystal structure: deep representation learning from stoichiometry. Nat Commun 11, 6280 (2020). https://doi.org/10.1038/s41467-020-19964-7

[4] Cheng, J., Zhang, C. & Dong, L. A geometric-information-enhanced crystal graph network for predicting properties of materials. Commun Mater 2, 92 (2021). https://doi.org/10.1038/s43246-021-00194-3
