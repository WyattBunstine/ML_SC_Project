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
├── main.py                           # CLI: build-db / download-nonsc / download-energy / train / train-mpnn / plot
├── plot.py                           # Scatter-plot predictions vs targets
├── configs/
│   ├── basic.json                    # Baseline CGCNN, regression (T_c)
│   ├── classify_basic.json           # Baseline CGCNN, classification (SC/non-SC)
│   └── mpnn_basic.json               # MPNN (crystal_graph_v4), regression
├── database/                        # SCRIPTS only — all data is under datafiles/ (gitignored)
│   ├── database_main.py             # DB generation: generate_atom_init(),
│   │                                #   generate_Basic_DB(), generate_CGv4_DB()
│   ├── crystal_graph_v4_import.py   # build_crystal_graph_from_cif() — rich graph builder for cgv4
│   ├── Download_MP_data.py          # gen_dataset(): non-SC MP API download (main.py download-nonsc)
│   ├── Download_MP_energy.py        # gen_dataset(): MP energy-target download (main.py download-energy)
│   └── datafiles/                   # ALL data files — GITIGNORED (CIFs, graphs, pickles, CSVs)
│       ├── atom_init.json           # Per-element feature vectors (84 elements, 8 features each)
│       ├── MP/                      # superconductor (3DSC_MP) data
│       │   ├── id_prop.csv          # Headerless: filename.cif, T_c (5,773 rows)
│       │   ├── 3DSC_MP.csv          # Full 3DSC dataset with metadata
│       │   ├── id_prop_basic.pickle # SC-only basic dataset: [id, value, struc_dict, label]
│       │   ├── id_prop_v4.pickle/.csv # cgv4 index: [id, value, graph_path, label] (+ target cols)
│       │   ├── graphs_v4/           # cgv4 per-material JSON graphs (+ failed.txt log)
│       │   └── cifs/                # CIF structure files
│       ├── Non_SC_DB_MP/            # non-SC negatives: Non_SC.csv + cifs/
│       └── MP_Energy/               # MP energy benchmark: mp_energy.csv + cifs/
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
`database/Download_MP_data.py::gen_dataset()` queries the Materials Project for
large-band-gap (default ≥ 4.0 eV) materials and writes one CIF each plus a headerless
`Non_SC.csv` (`<material_id>.cif,0.0` — the T_c is a placeholder; these rows are marked
non-SC by the database `label`, **not** by this value). Requires the **`MP_API_KEY`**
environment variable. `--limit N` caps the count for class balance. The cutoff is high on
purpose: superconductors are metallic, and MP's DFT (PBE) band gaps underestimate the true
gap, so a low cutoff (e.g. 1 eV) can admit metallic/SC materials — even known SCs from the
positive set — into the negatives. 4 eV keeps only unambiguous insulators; tune with
`--min-band-gap`.

### Step 0b (optional): MP energy-target benchmark (`main.py download-energy`)
`main.py download-energy` → `database/Download_MP_energy.py::gen_dataset()` pulls
experimentally-observed MP materials (`theoretical=False`, with a usable structure) and
writes one CIF each plus `mp_energy.csv`
(`cif, material_id, e_above_hull, formation_energy_per_atom`). This is a general
property-prediction benchmark — a sanity check on how the models learn, independent of the
superconductor task. Two thermodynamic targets are recorded per material as **separate
columns**: `e_above_hull` (eV/atom, ≥ 0; 0 = on-hull/stable) and `formation_energy_per_atom`
(eV/atom, usually negative). No target is baked in here — the cgv4 index keeps both columns
and the MPNN selects one at train time via the config `target_column` (see Step 1 / Key
Configuration). Rows missing both energies (or a structure) are skipped. Flags:
`--include-theoretical`, `--limit N`, `--chunk-size N`. Needs **`MP_API_KEY`**.

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
  (one JSON each in `graphs_v4/`) plus an index `[id, value, graph_path, label]` **+ every
  recognized target column** present in the source (`tc`, `e_above_hull`,
  `formation_energy_per_atom`; missing → NaN per row). `value` mirrors `tc` (or the first
  target) for legacy single-target consumers. The MPNN picks which column to regress on via
  the config `target_column`. Fully resumable (skips existing JSONs; failures →
  `graphs_v4/failed.txt`). Consumed by the MPNN.

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
| `dataset_rd` | `"database/datafiles/MP"` | Directory holding the pickle + `atom_init.json` (CGCNN) |
| `dataset` | `"id_prop_basic_combined.pickle"` | Pickle name (CGCNN) |
| `index_path` | `"database/datafiles/MP/id_prop_v4.pickle"` | cgv4 index (MPNN) |
| `target_column` | `"formation_energy_per_atom"` | [MPNN] which index target column to regress on; unset → legacy `value`/`tc` |
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

## Atom Features (`database/datafiles/atom_init.json`)
Each element (Z = 1–84) has 8 features: `[Z, block (0=s,1=p,2=d,3=f), valence,
atomic_radius, electron_affinity, ionization_energy, electronegativity, electron_affinity]`.
The baseline CGCNN sums these per site weighted by occupancy → (N, 8), embedded to 64-d.

The MPNN instead uses the **14 node + 8 edge + 7 poly-edge** features defined in
`MPNN/MPNNData.py` (`NODE_FEA_LEN`, `NBR_FEA_LEN`, `POLY_FEA_LEN`). The 14 node features are:
`Z, oxidation_state, ion_role, chi_pauling, chi_allen, ecn_value, shannon_radius, cn_core,
hist_{corner,edge,face,other}, ionization_energy, electron_affinity`. The last two are pure
per-element lookups (carried over from the original CGCNN `atom_init` vector); they are read
from the stored graph when present and otherwise computed from `Z` via pymatgen, so graphs
built before they were added need **no** rebuild. Feature dims are inferred at runtime from
the first sample and recorded in each run's `metadata.json` (`feature_dims`).

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

# (Optional) MP energy-target benchmark dataset (needs MP_API_KEY)
python main.py download-energy                             # both energy columns
# Build its graphs, then pick the target in the config via target_column
python main.py build-db --kind cgv4 --has-header \
  --source database/datafiles/MP_Energy/mp_energy.csv database/datafiles/MP_Energy/cifs/ \
  --output database/datafiles/MP_Energy/id_prop_v4_energy
python main.py train-mpnn configs/mpnn_eform.json         # regress formation energy

# Combined SC + non-SC dataset (labels 1 / 0) for the classifier
python main.py build-db --kind basic \
  --source database/datafiles/MP/id_prop.csv database/datafiles/MP/cifs/ \
  --nonsc-source database/datafiles/Non_SC_DB_MP/Non_SC.csv database/datafiles/Non_SC_DB_MP/cifs/ \
  --output database/datafiles/MP/id_prop_basic_combined

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

---

## Future Directions / Research Roadmap

**Premise.** The end goal is predicting superconducting **T_c**; formation energy is a
"more solved" benchmark used to refine the model. T_c is physically governed by (a)
**phonons / lattice dynamics** (electron-phonon coupling λ, ω_log — the BCS channel),
(b) **electronic structure** (DOS at E_F), and (c) **magnetism / spin fluctuations**
(dominant in unconventional SCs, but phonons likely still contribute there too — the
cuprate isotope effect is nonzero and grows toward the dome edges, and phonon–spin
coupling is real). The current model encodes **chemistry + static geometry** but **no
dynamics and no magnetism** — that gap is where the roadmap aims. The recurring
constraint is that T_c is **data-starved and label-noisy**, so *transfer learning is the
through-line* of most of these ideas.

Ordered roughly by effort; each notes *why it should help*, the *honest caveat*, and a
*first step*.

1. **Three-body bond angles (line graph, ALIGNN-style) — near-term, in progress.**
   Bonding edges are currently purely two-body (distance/weights/Δχ); the only angles are
   the coarse polyhedral mean/std on poly-edges. Add the full set of bond angles at each
   atom via a line graph (nodes = bonds, line-graph edges = bond pairs at a shared atom),
   the angle expanded in an **RBF(cos θ)** basis. *Why:* the complete rotation-invariant
   3-body descriptor disambiguates local environments (square-planar vs octahedral, etc.).
   *Caveat:* 3-body is provably incomplete (Pozdnyakov–Ceriotti 2020) but captures most
   signal for scalar targets. *First step:* extend the external `crystal_graph_v4` builder
   to emit bond-angle triplets (and **store dihedrals in the same rebuild** — see #2), add a
   line-graph channel to `CrystalMPNN`, rebuild graphs once.

2. **Dihedral / 4-body terms (GemNet-style) — a falsifiable T_c hypothesis.**
   *Why:* dihedrals close the 3-body completeness gap and help GemNet most on *forces /
   dynamics* — which, if phonons matter for pairing, may transfer to T_c. *Caveat / the
   honest gap:* T_c is a rotation-invariant **scalar**, mechanistically more like formation
   energy (where 4-body gains are modest) than like the **vector** force targets where
   dihedrals shine. *Clean experiment:* store dihedrals during the #1 rebuild but gate them
   behind a flag, then ablate on **both** formation energy and T_c — the hypothesis predicts
   *Δ(T_c) ≫ Δ(formation energy)*. Confirm or refute with one controlled test, no second
   rebuild.

3. **Magnetism — cheapest real gap.** Encode **MP per-site DFT magnetic moments** (and
   total magnetization) as node features. *Why:* magnetism is central to unconventional SC
   (AFM cuprate parents, spin-fluctuation pairing in Fe-based) and is entirely absent now.
   *Caveat:* DFT magmoms are calculation-dependent and least reliable for exactly the
   strongly-correlated systems that matter; and a *static* moment is a coarse proxy for the
   *dynamic* spin fluctuations that mediate pairing (same static-vs-dynamic gap as phonons).
   *First step:* pull `magmom` per site from MP into the cgv4 node features.

4. **Phonon/dynamics features via a pretrained universal MLIP — high-value shortcut.**
   Instead of training a phonon model on the scarce explicit-phonon data (~1.5k materials in
   the MP/Petretto DB), run a **pretrained foundation MLIP** (MACE-MP-0, CHGNet, M3GNet,
   MatterSim, ORB — trained on millions of energy/force/stress points) over the T_c dataset
   and harvest **site-resolved phonon descriptors** (on-site force constants, atomic MSD /
   Debye–Waller factors, site-projected phonon-DOS moments) plus global ω_log as node/global
   features. *Why:* injects the electron-phonon-relevant dynamics the static graph lacks,
   without the phonon-data bottleneck. *Note:* "per-atom phonon modes" is ill-defined (modes
   are delocalized) — use site-resolved descriptors. *First step:* prototype with one MLIP
   (e.g. MACE-MP-0) computing force constants on a few hundred structures.

5. **Pretrain on dynamics, fine-tune on T_c — highest ceiling.** Pretrain the encoder on
   **energies + forces** from large datasets (MPtrj, Alexandria, OMat24, GNoME), then
   fine-tune on T_c. *Why:* an encoder trained to predict forces has learned the
   gradient/curvature of the energy surface — the dynamical response — and forces data is far
   more abundant than phonon data. The current formation-energy work is the warm-up for this.
   *Caveat:* biggest infrastructure lift (data plumbing, architecture/objective changes).

6. **Data scale.** For final runs, train on the **million-sample** datasets modern models
   use (Alexandria/OMat24/GNoME for pretraining); for the T_c target itself, prioritize more
   superconductor data and label-quality handling (experimental T_c is heterogeneous/noisy),
   since that — not representation cleverness — likely dominates the error budget.

**Generalization toolkit already in place** (use these to evaluate every change above):
config-controlled `dropout`, decoupled `weight_decay`, **SWA** (`swa`), seed-varied
**ensembling** (`model_seed` + `scripts/ensemble.py`), checkpoint re-evaluation
(`scripts/eval_test.py`, incl. shared `--test-ids` for apples-to-apples model comparison),
and epoch-log diagnostics (`main.py plot --epoch-log`).
