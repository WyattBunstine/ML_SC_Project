# mptrj_benchmark_suite — the architecture matrix at MPtrj scale

The same one-factor-per-run matrix as `eform_rebaseline_suite`, moved to the
**MPtrj packed dataset** (~1.58M trajectory frames / 146k materials, formation
energy per frame). This is where the attention-family verdicts actually get
decided: on 49k-sample MP_Energy the set-transformer overfit (train 0.013 / val
0.058) and learned attention only tied the physics-weighted sum — 1.4M training
frames is the regime those results were provisionally blamed on.

Shared setup (every run): packed reads from scratch, **material-level split**
(seed 123 → identical splits across runs, so paired per-sample comparisons are
valid), batch **512** with LR **0.004** (linear scaling of the batch-128 LR
0.001 — 4x fewer optimizer steps per epoch, so the step size scales up with the
batch), 60 epochs, milestones [30, 50], val/test 5%, backbone 64/64/3conv/128.

| config | varies | question |
|---|---|---|
| `01_poly_off` | poly channel off | baseline |
| `02_poly_on` | poly on (sum fusion) | does the poly gain (~6% on MP_Energy) replicate at scale? |
| `03_set_transformer` | bond agg → set-transformer w/ angle bias | does idea A win once data is no longer the constraint? |
| `04_attention` | agg → learned scalar attention | attention vs physics weighting at scale |
| `05_gate` | `poly_fusion: "gate"` | does learned fusion separate from the sum at scale? |
| `06_coord_magnitude` | `use_coord_magnitude: true` | first benchmark of the coordination-strength channel (never tested) |

Launch (each ~3.5–6 h on an a100; queue all six in parallel):

    for f in configs/mptrj_benchmark_suite/0*.json; do ./scripts/deploy.sh run "$f"; done

After fetch: `model_data/index.csv` for the overview;
`python scripts/verify_smoke.py <run_dir>` per run for the health checklist;
paired per-sample comparison on the shared test CSVs for significance.

Interpretation caveats (same as the MP_Energy round): 02/05 add parameters as
well as structure (poly conv ≈ +40k, gates ≈ +49k); 03's util/epoch-time will
naturally run higher than the others (more compute per atom). Epoch-time budget
measured from the 2-epoch smoke at batch 128: ~360 s/epoch → expect ~180–200 s
at 512 for the ecn arms.
