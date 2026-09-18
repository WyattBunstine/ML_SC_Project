# Chemically and Geometrically Motivated Crystal Graph Neural Networks for Superconductor Screening

This project predicts and screens superconductors from crystal structure. It has two
goals, realized as a **two-stage hurdle**:

1. **Stage 1 — classify** whether a material is a superconductor candidate at all
   (SC vs. non-SC), trained on 3DSC_MP superconductors (~5,773) plus large-band-gap
   non-superconductors pulled from the Materials Project (~55k).
2. **Stage 2 — regress** the critical temperature T_c (in Kelvin) for superconductors.

Two graph-neural-network families are trained and compared:
- **Baseline CGCNN** (`models/OriginalCGCNN/`, after [txie-93/cgcnn](https://github.com/txie-93/cgcnn)),
  driven by `models/CGCNNMain.py`.
- **MPNN** (`models/MPNN/`) — a message-passing network with learned edge features that
  consumes the richer `crystal_graph_v4` graphs.

> The earlier coordination-environment variant (`CGCNNCoordEnv`) has been **retired**
> and superseded by the MPNN. It is gone from the codebase (still in git history).

## Working Style
When working on this project, iterate with the user on ideas before making implementations. When the user asks for something to be done, think through the possible impacts of the changes and specifically how this might change or reduce the accuracy of the scripts. After thinking through, ask the user about their preferred way of implementing the changes. Highlight different possible implementations and the benefits and drawbacks of each approach. Do not make changes large until explicitly told to by the user. Once a change is made, make sure that the script still runs properly and make additional changes if there are runtime errors or similar coding errors. If the result changes to something undesirable, highlight this for the user and suggest additional changes to address it, but do not make additional changes until told to do so by the user.

---

## Project Structure

```
ML_SC_Project/
├── main.py                           # CLI: build-db / build-mptrj / augment-positions / augment-physics /
│                                     #   pack-dataset / fetch-dos / download-nonsc / download-energy /
│                                     #   train / train-mpnn / train-gps / train-head / embed-mace / embed-gps / plot
├── plot.py                           # Scatter-plot predictions vs targets
├── configs/
│   ├── orig_basic.json / orig_classify_basic.json / mpnn_basic.json   # legacy CGCNN/MPNN entries
│   ├── gps/ + gps_ablation_suite/    # GPS encoder single-task + architecture ablations
│   ├── gps_mt_ablation_suite/        # multitask pretraining rungs (01..15; 15 = +crystal-field features)
│   └── head/                         # T_c transfer-head configs (frozen probes, fine-tune, HPO, holdouts)
├── scripts/
│   ├── remote/ (git-ignored)         # deploy.sh = SLURM cluster wrapper: setup-env/sync-*/run/run-head/sweep-head/build-mptrj/
│   │                                 #   pack-mptrj/augment-cf/status/fetch/... (header lists all)
│   ├── augment_cf.py                 # backfill baked v4.3 valence+cf onto existing compact graphs (ordered sets)
│   ├── calibrate_cf_magmom.py        # AOM crystal-field knob calibration vs MPtrj DFT magmoms (held-out)
│   ├── build_disorder_corpus.py      # mean-field disordered-structure pretraining corpus from ordered MP entries
│   ├── head_hpo_sweep.py / run_head.py / probe_encoders.py   # head HPO stage A, multi-config head runner, encoder zoo
│   └── (eval/compare/ensemble utilities)
├── database/                        # SCRIPTS only — all data is under datafiles/ (gitignored)
│   ├── database_main.py             # DB generation: generate_atom_init(), generate_Basic_DB(), generate_CGv4_DB()
│   ├── crystal_graph_v4_import.py   # imports the cgv4 builder from ../RPToleranceFactor (RP_TOLERANCE_FACTOR_PATH)
│   ├── Extract_MPtrj.py             # stream the MPtrj release JSON (frames + physics targets)
│   ├── Download_MP_data.py / Download_MP_energy.py / Download_MP_dos.py   # MP API fetches
│   ├── icsd_doping.py / oxidation_doping.py   # doped-structure synthesis + doping-aware oxidation states
│   └── datafiles/                   # ALL data files — GITIGNORED (CIFs, graphs, pickles, packs)
│       │                            # Retirement convention: superseded/era-closed datasets move
│       │                            #   to datafiles/.retired/<date>/ (README manifest: why + how
│       │                            #   to restore/rebuild) instead of deletion — check there
│       │                            #   before rebuilding something that "disappeared".
│       ├── atom_init.json
│       ├── MP/                      # superconductor data + relaxed-MP electronic structure
│       │   ├── id_prop.csv / 3DSC_MP.csv          # 5,773-row T_c source + metadata
│       │   ├── SC_MP_V4_doped*.pickle             # cgv4 index lineage: _doped (+_oxifix), V4M (magnetic
│       │   │                                      #   negatives), V6 (NEMAD/ICSD expansion), _doped_cf (v4.3
│       │   │                                      #   baked valence + crystal-field features)
│       │   ├── graphs_v4*/                        # per-material JSON graphs per index variant
│       │   ├── SC_pack_doped*/ dos_pack*/ disorder_pack*/   # packed columnar stores (_cf = v4.3 features)
│       │   ├── disorder_corpus/                   # mean-field doped-structure corpus (cifs + graphs + index)
│       │   ├── dos_rebuild/                       # DOS fetch + ±1eV-grid index (dos_pack_ef1 source)
│       │   └── cifs*/                             # CIF structure files (incl. v5 ICSD/NEMAD sets)
│       ├── Non_SC_DB_MP/            # non-SC negatives: Non_SC.csv + cifs/
│       ├── MP_Energy/               # MP energy benchmark (experimental-only by default!)
│       └── MPtrj/                   # MPtrj release JSON (graphs+packs live on cluster scratch)
└── models/
    ├── common/                       # SHARED infra: data.py (graph/pack loaders, features, splits),
    │                                 #   pack.py (columnar store), train.py (regression + multitask loops)
    ├── GPSTransformer/               # GPS crystal encoder (main.py train-gps): model.py + gps_main.py
    ├── head/                         # T_c transfer: HeadMain/HeadModel/HeadData/FineTune/embed_gps/
    │                                 #   embed_cache (+ MACE embedding path)
    ├── CGCNNMain.py + OriginalCGCNN/ # baseline CGCNN (main.py train)
    └── MPNN/                         # message-passing net (main.py train-mpnn): MPNNMain.py + MPNNModel.py
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

The baseline CGCNN trains via `models/CGCNNMain.py` on the `struc_dict` pickle; the MPNN
trains via `models/MPNN/MPNNMain.py` on the `graphs_v4` index. Both share the same crystal
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
| `dataset` | `"SC_MP_basic_combined.pickle"` | Pickle name (CGCNN) |
| `index_path` | `"database/datafiles/MP/SC_MP_V4.pickle"` | cgv4 index (MPNN) |
| `target_column` | `"formation_energy_per_atom"` | [MPNN] which index target column to regress on; unset → legacy `value`/`tc` |
| `atom_init` | `"atom_init.json"` | Per-element feature file (CGCNN) |
| `n_nonsc` | 5000 | [classification] non-SC sampled per epoch |
| `split_seed` | 123 | [classification] stratified-split seed |
| `atom_feat_len` / `n_conv` / `h_feat_len` / `n_hidden` | 64 / 3 / 128 / 1 | Model dims |
| `aggregation` | `"ecn_weighted"` / `"attention"` | [MPNN] edge aggregation |
| `batch_size` / `epochs` | 128 / 300–1000 | Training loop |
| `optim` / `learning_rate` / `momentum` / `weight_decay` / `lr_milestones` | `"SGD"` / 0.01 / 0.9 / 0 / [100] | Optimizer |
| `out_file` | `"models/classify_result"` | Output prefix |

Provided configs: `configs/orig_basic.json` (baseline regression), `configs/orig_classify_basic.json`
(baseline classification on the combined DB), `configs/mpnn_basic.json` (MPNN regression).

---

## Atom Features (`database/datafiles/atom_init.json`)
Each element (Z = 1–84) has 8 features: `[Z, block (0=s,1=p,2=d,3=f), valence,
atomic_radius, electron_affinity, ionization_energy, electronegativity, electron_affinity]`.
The baseline CGCNN sums these per site weighted by occupancy → (N, 8), embedded to 64-d.

The MPNN/GPS models instead use the **14 node + 7 edge + 7 poly-edge** features defined in
`models/common/data.py` (`NODE_FEA_LEN`, `NBR_FEA_LEN`, `POLY_FEA_LEN`). Optional node blocks
extend this (concat order `[base | rich | valence | cf | dihedral]`, config-flag-gated):
`use_rich_node_features` (+4), `use_valence_features` (+4, [n_s,n_p,n_d,n_f] subshells —
baked per-species in builder-v4.3 graphs, load-time fallback for legacy), `use_cf_features`
(+12, the AOM crystal-field block: sorted d-levels, occupancies, frontier gap, unpaired —
baked-only, v4.3 packs), `use_dihedrals` (+12). The 14 base node features are:
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
  --output database/datafiles/MP_Energy/MP_Energy_V4 \
  --graph-dir database/datafiles/MP_Energy/graphs_v4
python main.py train-mpnn configs/mpnn_eform.json         # regress formation energy

# Combined SC + non-SC dataset (labels 1 / 0) for the classifier
python main.py build-db --kind basic \
  --source database/datafiles/MP/id_prop.csv database/datafiles/MP/cifs/ \
  --nonsc-source database/datafiles/Non_SC_DB_MP/Non_SC.csv database/datafiles/Non_SC_DB_MP/cifs/ \
  --output database/datafiles/MP/SC_MP_basic_combined

# Build crystal_graph_v4 graphs for the MPNN (add --nonsc-source for classification)
python main.py build-db --kind cgv4

# Train baseline CGCNN — regression (T_c) or classification (SC/non-SC)
python main.py train configs/orig_basic.json
python main.py train configs/orig_classify_basic.json

# Train MPNN (requires cgv4 graphs)
python main.py train-mpnn configs/mpnn_basic.json

# MPtrj energy-pretraining dataset (~1.6M trajectory frames; runs on the cluster):
./scripts/deploy.sh build-mptrj    # CPU job: stream the 12 GB JSON -> cgv4 graphs on scratch
./scripts/deploy.sh pack-mptrj     # CPU job: pack graphs into the fast columnar store
./scripts/deploy.sh run configs/formation_energy_options_suite/mptrj_eform_minimal.json
# (local equivalents: python main.py build-mptrj / pack-dataset --index ... --out ...)

# Pack ANY cgv4 dataset for ~30x faster training reads (point index_path at the dir)
python main.py pack-dataset --index database/datafiles/MP_Energy/MP_Energy_V4.pickle \
  --out database/datafiles/MP_Energy/packed_v1

# Plot regression results
python main.py plot --results models/test_result.csv
```

Cluster workflow (deploy/fetch/reorg/archive, /data-vs-scratch storage):
see `scripts/README.md`. Per-epoch resource telemetry (GPU/CPU/data-wait columns
in every run's `*_epoch_log.csv`) needs `psutil` + `nvidia-ml-py` (in
requirements; blank columns if absent). Trajectory datasets MUST split by
material (`split_by`, auto when the index has `mp_id`) — frame-level splits leak
near-duplicate frames.

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

## Research Roadmap (rewritten 2026-06-12; phased)

> **⚠️ STATUS NOTE (2026-07-28).** The phase write-ups below are a point-in-time
> snapshot (2026-06-24) and predate two major developments:
> 1. **The FineTune positional-split bug** (fixed `48f099d`, 2026-07-14): loader
>    positions were joined to split labels by index order while the dataset
>    seed-shuffles at load, scrambling every fine-tune fold from 2026-06-29 to
>    2026-07-14. All transfer numbers from that window (incl. the 4.18 K
>    "champion") are INVALID; docs/literature_tc_benchmarks.md carries the
>    re-validated table.
> 2. **The pretraining-representation program** (rungs 09-15): valence subshell
>    features, ±1 eV DOS reconditioning, the mean-field disorder corpus (rung 12
>    = current champion encoder: MAE 4.26 K / SC-only 5.33 / MSLE 0.847 /
>    cuprate 15.23 under chemsys + norms-unfreeze), whole-crystal attention
>    (rung 13: best MSLE 0.836, worse Kelvin — two-encoder portfolio), and the
>    AOM crystal-field features (builder v4.3, rung 15 in flight).
> Session-to-session state lives in the auto-memory (MEMORY.md), which is more
> current than this section.

**Premise.** The end target is superconducting **T_c**: data-starved (~6k labels) and
label-noisy. Physically, T_c is governed by (a) **phonons / lattice dynamics**
(electron-phonon coupling λ, ω_log — the BCS channel), (b) **electronic structure**
(DOS at E_F), and (c) **magnetism / spin fluctuations** (dominant in unconventional
SC). The central strategy is now **transfer learning**: borrow a force-pretrained
universal MLIP backbone (MACE-MP-0 / CHGNet — both trained on MPtrj), attach our
chemically-informed model as the T_c head, and fine-tune through a staged curriculum.
The scientific deliverable beyond screening: the **conventional-vs-unconventional
generalization gap** of a phonon-pretrained model — measured *differentially* against
matched no-pretraining baselines — as an interpretable probe of what physics beyond
electron-phonon is missing. Literature check (2026-06-12) found no prior study doing
this; closest precedent is Stanev 2018's cross-family non-transfer. Key physics
caveat: forces carry the phonon *denominator* of λ = N(E_F)⟨I²⟩/M⟨ω²⟩ only — the
electronic numerator needs auxiliary supervision (Phase 3).

**Status ledger (what the original roadmap items became).** Done: 3-body angles
(old #1) — angle triplets stored and consumed two ways (`use_bond_angles` mean-RBF;
`set_transformer` angle-biased local attention); the MPtrj 6-arm benchmark
(2026-06-11) made set_transformer the best arm (val MAE 0.0397 vs 0.0429 poly_on,
~−7%), and the gain is not explained by parameter count (the gate arm has more
params, zero gain). Dihedrals stored but unconsumed (old #2 — consumption lands in
Phase 4). MPtrj scale infra (old #6): 1.58M-frame packed store, material splits,
telemetry, paired-stats tooling (`scripts/compare_runs.py`), vectorized readout.
Benchmarked and rejected: `coord_magnitude` (worse than baseline at MPtrj scale —
off by default). Old #4 (MLIP-derived features) folds into Phase 1; old #5
(pretrain→fine-tune) is the spine of Phases 1–4; old #3 (magnetism) lands in
Phase 3; old #7 (hypergraph) is Phase 5 stretch.

### Phase 0 — close out the MPtrj benchmark round
Resubmit poly_off (job 25518913 hit SLURM launch-failure requeue hold; exclude the
flaky node), walltimes 12h (60 epochs needs 9–11h). Add two control arms to settle
*why* set_transformer won: a **width-matched poly_on** (~126k params — kills the
capacity explanation for good) and **`use_bond_angles` + ecn_weighted** (angle
*information* vs attention *mechanism*). Fix `scripts/eval_test.py` to honor
`build_angle_bias` from the checkpoint config before producing any test-set numbers.

### Phase 1 — pluggable encoder contract (representation-level plug)
**STATUS: BUILT + first results (2026-06-12).** `models/head/` — embed_mace.py
(`main.py embed-mace`; per-structure alignment assertions, 5,773/5,773 SC
embedded clean), descriptors.py (41-dim bypass vector, all 60,937 rows),
HeadModel.py (PCA-whiten + standardizer buffers; 7,189 fresh params at
defaults), HeadMain.py (`main.py train-head configs/head/...`: ridge probe →
optional class-pretrain → 5-seed T_c ensemble; family/group-resolved metrics
in K and log1p-K). First numbers (frozen MACE-MP-0 medium + bypass, no class
pretraining, test split): probe 6.49 K / head **4.68 K** overall MAE (trivial
median-predictor: 9.75 K); log-space conventional 0.453 vs cuprates 0.924 —
the family gap is already visible at the linear-probe level. Probe predictions
are clamped to the train target range (unclamped ridge + expm1 inverse
exploded to 12,544 K). Non-SC embed pass + classification-pretrain arm pending
(embeddings were still building).

Contract (decided 2026-06-12): an **encoder** is anything mapping a structure to
**contextualized per-atom embeddings `(N, D_enc)`**, aligned to the graph builder's
atom order. The plug point is the encoder OUTPUT — after all local + global context
— NOT the model's input features, so encoders are interchangeable beneath one fixed,
small T_c head (controlled comparison: same head + protocol, swap encoder,
family-resolved paired stats). First encoder: **frozen MACE-MP-0** (post-interaction
invariant l=0 node features — what its own readout consumes); later: our pretrained
GPS model behind the same contract; also `[MACE ‖ ours]` fusion to test
complementarity. Plumbing: one-time embed pass → packed per-atom column / per-graph
array + atom-order alignment test. The head additionally takes a
**physical-descriptor bypass**: pooled hand-crafted invariants (bridge-angle stats,
sharing-mode histogram, ECoN distribution, composition scalars) — inputs, not
weights, so zero fresh parameters; in the MACE-first config this is the only path
our descriptors reach the head. The same lane later carries MLIP-harvested
site-resolved phonon descriptors (on-site force constants, MSD, phonon-DOS moments
— old #4). Caveats: MACE-MP-0 was trained on MPtrj (contaminated there; honest
reads at SuperCon or via an OMat/MPA-trained variant); 3DSC embeddings inherit the
idealized-structure/doping problem. **Deep fusion** (MACE at the node-feature level
of a curriculum-pretrained trainable model) is demoted to a later experiment, only
if shallow `[MACE ‖ ours]` fusion shows the representations are complementary.

### Phase 2 — small T_c head + SuperCon transfer pipeline + our encoder
**Parameter budget rule:** ~5.8k noisy T_c labels → freshly-initialized parameters
(the scarce resource) capped around ~10k. Pretrained params being fine-tuned count
much more gently (low LR + early stopping ⇒ effective capacity ≪ count); frozen
params are free. Protocol per encoder, in order of increasing risk, validation
decides where to stop: (1) **linear probe** on the pooled frozen embedding (~300
params — the standard representation-quality metric, always reported first), (2)
small MLP head (~5–15k) + dropout/weight-decay/seed ensembles, (3) head + last-block
unfreezing (LR-grouped or LoRA-style adapters), (4) full fine-tune. Budget
stretchers: head-trunk pretraining on the **~61k-label Stage-1 SC/non-SC
classification task** before T_c regression; ensembles (BETE-NET precedent).
**Our encoder** = the GPS-style sibling model (validated angle-biased local shell
attention interleaved per block with within-crystal global attention; global token
may condition the local center query; poly channel as second local relation;
register-token readout) — pretrained on MPtrj energies (+ Phase 3 curriculum), then
plugged behind the Phase 1 contract. It is NOT trained on SuperCon from scratch.
**Status (2026-06-17): BUILT and pretraining (branch `gps-tier2`).** Full
pre-LN-transformer local channel, per-atom energy head, a PBC long-range distance
bias on the global attention, and an 8-rung complexity ablation ladder
(`configs/gps_ablation_suite/`, raw→…→+distance-bias) currently running on MPtrj
formation energy. The positions data the distance bias needs — also the Phase-4
forces prerequisite — is regenerated without a Voronoi rebuild via
`deploy.sh augment-positions` → `packed_v2`. Current build state: ARCHITECTURE.md §10.
**Update (2026-06-18→24): the multi-task physics pretraining is BUILT and running,
pulling Phase 3's electronic/magnetic heads and Phase 4's autograd forces forward into
one combined pretrain (branch `gps-tier2`).** `GPSCrystalNet` now co-trains
conservative-autograd forces (`−∂E/∂cart`) + stress + per-atom magmom + per-structure
bandgap + per-structure total DOS (256-bin spectrum) over a masked union of `packed_v4`
(MPtrj) and a relaxed-MP DOS pack (`run_multitask` in `common/train.py`;
`model.encode()`/`main.py embed-gps` for the frozen transfer export). The **signal**
ablation ladder `configs/gps_mt_ablation_suite/01–04` (energy → +forces/stress →
+magmom/bandgap → +DOS) is the multitask one running now — distinct from the
architecture ladder above; rung 04's DOS was revived 2026-06-24 (MP coverage
31,403/49,280). See ARCHITECTURE.md §9–§10.
SuperCon side: 3DSC family labels (cuprates/Fe-based/heavy-fermion/…) for
family-resolved evaluation; ordered-compound subsets to control the
doping-representation problem. Deliverable: the conventional/unconventional
**differential** experiment, designed against its three confounds — OOD
distribution shift (differential vs same architecture without phonon-pretrained
encoder), doping noise (cuprate T_c is doping-domed and structure-matching destroys
doping), and family-correlated DFT quality (PBE worst for correlated oxides).

### Phase 3 — curriculum middle stage + electronic/magnetic gaps
Intermediate fine-tune on *computed* electron-phonon datasets before the empirical
DB: Marques-group high-throughput λ/ω_log (~7k), JARVIS-EPC (~1k), BETE-NET α²F
(~800). A backbone+head that predicts λ and ω_log well should nail conventional
SuperCon entries via Allen-Dynes — a direct, clean test of hypothesis 1. The
**electronic + magnetic auxiliaries are no longer future here:** per-structure
**band gap** and **per-structure total DOS** (the full 256-bin spectrum, not just
DOS@E_F) and per-atom **magmom** are BUILT multi-task heads co-training now
(2026-06-18; rungs 03–04 of the signal ladder) — magmom as a per-atom head *target*,
not a node feature (old #3). Caveat unchanged: DFT moments least reliable exactly for
correlated systems; static moment is a coarse proxy for dynamic spin fluctuations. The
*computed* e-ph curriculum (λ/ω_log from Marques/JARVIS-EPC/BETE-NET) is the
genuinely-future Phase-3 step.

### Phase 4 — own the backbone (Tier-1 differentiable rework)
**STATUS (2026-06-18): the core is BUILT.** Conservative autograd forces `−∂E/∂cart` +
stress `∂E/∂strain` are implemented in `GPSCrystalNet` via a differentiable
column-replacement geometry (recompute bond-length/ratio/angle-cos from an in-forward
Cartesian leaf using exact per-edge `to_jimage`; topology/Voronoi/chemistry held fixed),
and the model is force-training in the multi-task pretrain. Dihedral consumption and the
Voronoi→fixed-topology demotion below remain; **Tier-2** (e3nn) stays out of scope.
Key fact: forces need **position-differentiability, not internal equivariance** —
the autograd gradient of an invariant energy is automatically equivariant
(SchNet/ALIGNN-style). Move featurization in-model: positions + PBC image vectors
in the batch; recompute bond lengths/ratios, **ECoN weights (smooth closed form —
the signature prior survives force training)**, bond angles, and dihedrals (finally
consuming old #2; its falsifiable prediction — dihedrals help T_c more than E_form
— rides along) inside the forward. Voronoi demotes to fixed topology /
attention-logit constants (same epistemic status as a neighbor list); smooth cutoff
envelopes handle topology changes between frames. Then: force-train our model and
swap it into the same `atom_context` contract in place of MACE; A/B. **Tier-2**
(irreps/e3nn equivariant internal features) is explicitly out of scope unless
force-accuracy SOTA becomes a goal — though note attention logits must be invariant
even in equivariant transformers, so the angle/ECoN/sharing-mode bias machinery
would survive that rewrite intact.

### Phase 5 — stretch
Coordination-environment hypergraph (old #7): polyhedra as hyperedges via
factor-nodes, reusing the set-attention primitive one scale up (AllSetTransformer);
needs a builder rebuild (polyhedron member-sets, poly-pair torsions); A/B against
the existing poly channel rather than assuming a gain. Generative/screening
integration once the T_c head is trustworthy.

**Generalization toolkit already in place** (use to evaluate every change above):
config-controlled `dropout`, decoupled `weight_decay`, **SWA** (`swa`), seed-varied
**ensembling** (`model_seed` + `scripts/ensemble.py`), checkpoint re-evaluation
(`scripts/eval_test.py`, incl. shared `--test-ids` for apples-to-apples model
comparison), and epoch-log diagnostics (`main.py plot --epoch-log`).

---

## Potential Future Directions (non-binding idea bank)

Recorded from the 2026-06-15/16 design discussions — **exploratory, not committed**.
Kept so the reasoning isn't lost; revisit if/when the relevant stage is reached.
The full GPS encoder spec lives in `models/GPSTransformer/ARCHITECTURE.md`.

1. **Own the encoder via MACE distillation.** Train our GPS encoder to reproduce
   MACE's energies/forces (teacher = MACE, labels free + dense + noise-free — run
   MACE over many structures, no DFT). Makes "match MACE's forces" an achievable
   training target instead of a from-scratch fight; the force-matching *residual*
   diagnoses where our explicit-physics inductive bias is insufficient (metals,
   off-equilibrium frames). Then compare distilled-ours vs. MACE under one head.
   Faithful distillation of an equivariant teacher may want l=1 (PaiNN-style)
   vector channels — equivariance in service of distillation fidelity, not a SOTA
   chase.

2. **Off-equilibrium DOS acquisition** (to supervise the electronic *coupling*
   ∂DOS/∂x, which MP's equilibrium-only DOS can't). Options, with the catch on each:
   - *ML-DOS teacher (Mat2Spec/DOSnet): TRAP* — they're equilibrium-trained, so
     they share our blind spot and just propagate the equilibrium prior.
   - *Hamiltonian-learning models (DeepH/DeepH-E3): the real shortcut IF a
     broadly-pretrained, universal-chemistry one exists* — they predict H(R), so
     DOS generalizes across configurations. Coverage was the gap as of early 2026;
     worth re-checking.
   - *DFTB: chemistry-limited* — ~1000x faster, but Slater-Koster params are patchy
     exactly for the TM/f-element/intermetallic SC chemistries.
   - *Targeted DFT single-points: the only path to NEW trustworthy off-eq DOS* —
     bounded if curated (equilibrium + a few small displacements per material,
     stratified to SC chemistries → ~10-50k single-points, not 1.4M). Workflow +
     projected-DOS parsing + E_F alignment is the real lift, not core-hours. Hold
     behind a value check (does the electronic channel even help T_c?).
   - *Free stopgap:* MPtrj's per-frame `bandgap` (off-eq electronic scalar) — but
     uninformative among metals (gap≡0), so weak for SC families specifically.

3. **AFLOW as a supplementary source** (NOT a base — it's equilibrium-only, no
   trajectory forces, so it can't replace MPtrj for the force leg). Use for: bigger
   equilibrium DOS coverage if MP is thin; its phonon subset (APL/AAPL) as a direct
   curvature/λ signal for a Phase-3 curriculum. Caveat: AFLOW's DFT protocol differs
   from MP/MPtrj → cross-dataset label inconsistency when mixed. The latent fork:
   force-centric (MPtrj base) vs. broad-equilibrium-property pretraining (AFLOW
   base, drop forces) — the latter only if we relax the force thesis.

4. **Hessian / phonon fine-tuning (PFT).** E+F training can still get curvature
   (2nd derivatives) slightly wrong; directly supervising force constants / phonon
   properties (e.g. from AFLOW-APL or a computed λ/ω_log set) sharpens the
   phonon-relevant content — a Phase-3 curriculum step on top of force pretraining.

5. **Physics ensemble / multi-teacher distillation.** Several specialist physics
   models each provide one label channel (MACE→forces, a DOS model→DOS, …) and our
   encoder learns from all. Coherent for *equilibrium* multi-property labels;
   inherits the Option-2 trap for anything off-equilibrium (a teacher can't label
   outside its training distribution).
