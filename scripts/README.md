# `scripts/` — cluster deployment

`deploy.sh` pushes this project to a SLURM cluster (configured for JHU
**Rockfish**) and runs MPNN training jobs on a GPU node. It is a thin wrapper
around `ssh`, `rsync`, and `sbatch` — no extra dependencies on either side.

## Why it's split into subcommands

The `crystal_graph_v4` graph database is **~14 GB across ~89k small JSON files**.
Copying that on every run would be dominated by per-file SSH overhead, so the
bulk transfer is a **one-time** `sync-data`. Likewise the conda environment is
built once with `setup-env`. Each training run then ships only the kilobyte-sized
code + config and submits a job. Nothing destructive ever runs on the remote:
code/config syncs use no `--delete`, so remote `model_data/` and results are
never touched.

## Subcommands

| Command | Frequency | What it does |
|---|---|---|
| `setup-env` | once | rsyncs `requirements.txt`, creates conda env `ml_sc` (Python 3.11) on the cluster, `pip install -r requirements.txt`. Idempotent — safe to re-run after editing deps. |
| `sync-data` | once | rsyncs `database/MP/graphs_v4/` (14 GB) + `database/MP/*.pickle`. Resumable (`--partial`); re-running only sends new/changed files. |
| `run <config>` | per run | rsyncs `CNN/MPNN/*.py` + `configs/`, generates an `sbatch` script under remote `jobs/`, and submits it. |
| `sync-code` | (auto) | pushes just code + configs. Called automatically by `run`; rarely needed directly. |
| `status` | as needed | `squeue` for your jobs. |
| `logs <jobid>` | as needed | `tail -f` the live SLURM stdout (`logs/<jobname>-<jobid>.out`). |
| `fetch` | after a run | rsyncs remote `model_data/` + `logs/` back to your machine. |

## First-time configuration

Edit the `EDIT THIS BLOCK` section at the top of `deploy.sh`:

| Variable | Meaning |
|---|---|
| `REMOTE_HOST` / `REMOTE_USER` | cluster login host and your username (ssh keys assumed set up for passwordless login). |
| `REMOTE_PATH` | remote project root (use group/scratch storage, e.g. `/data/<pi>/<user>/ML_SC_Proj`). |
| `SLURM_PARTITION` | GPU partition (Rockfish: `a100`). |
| `SLURM_ACCOUNT` | billing account. **Verify the exact name** with `sacctmgr show assoc user=<user> format=account,partition`. A wrong account fails fast at `sbatch` time. |
| `SLURM_TIME` / `SLURM_GPUS` / `SLURM_CPUS` / `SLURM_MEM` | walltime, GPU count, CPUs (≥ config `num_workers` + 1), and RAM (the graph LRU `graph_cache_size` lives in memory). |
| `SLURM_MAIL_USER` | email for `BEGIN,END,FAIL` notifications. Leave `""` to disable email. |
| `CONDA_ENV` / `PYTHON_VERSION` | conda env name and Python version built by `setup-env`. |
| `TORCH_CUDA_CHANNEL` | PyTorch CUDA build to install (see [PyTorch / CUDA on Rockfish](#pytorch--cuda-on-rockfish)). |
| `ENV_SETUP` | commands run at the top of each job to make `python`/torch/pymatgen importable (loads the anaconda module and activates `CONDA_ENV`). |

### Per-config resource overrides

The variables above are the **defaults**. Any config can override them for its
own run by carrying a top-level `"slurm"` object — handy when different
experiments need different resources (more workers → more cores, a bigger model
→ more time/RAM). `run` reads it at submit time and prints the resolved request:

```jsonc
// configs/mpnn_basic_rockfish.json
{
    "num_workers": 12,
    "slurm": { "cpus": 24 }     // this run requests 24 cores; everything else
                                // (partition, gpus, mem, time, account) keeps the default
}
```

Recognized keys: `partition`, `account`, `time`, `gpus`, `cpus`, `mem`,
`mail_user`. `MPNNMain.py` ignores the `slurm` key, so one file configures both
training and the SLURM request. Full reference: [config README](../configs/README.md#cluster-resources-slurm--used-only-by-scriptsdeploysh).

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

## PyTorch / CUDA on Rockfish

**Use `TORCH_CUDA_CHANNEL="cu128"`** — confirmed working as of 2026-06-04
(torch `2.11.0+cu128`, `torch.cuda.is_available()` → `True` on an `a100` node).

The gotcha: a plain `pip install torch` pulls the **CUDA 13** wheel from PyPI,
but Rockfish's GPU driver only supports up to **CUDA 12.9** (`nvidia-smi`,
top-right). The CUDA-13 wheel can't initialize CUDA on that driver, so torch
prints a "driver too old" `UserWarning` and **silently falls back to CPU** —
training still runs, just ~10–50× slower. So `setup-env` installs torch from
PyTorch's `cu128` channel *before* the rest of `requirements.txt`, and `torch`
is left unpinned in `requirements.txt` so the requirements step can't drag the
CUDA-13 wheel back in.

If the driver ever changes (check `nvidia-smi` on a GPU node), set
`TORCH_CUDA_CHANNEL` to a build at or below its CUDA version (e.g. `cu124`,
`cu121`). A channel with no matching wheel makes `setup-env` **fail loudly**
rather than fall back to CPU. Note `setup-env` runs the install on the **login
node**, where the verification line correctly prints `cuda? False` (no GPU
there) — only the value inside a GPU job matters.

## Typical workflow

```bash
# --- one time ---
./scripts/deploy.sh setup-env
./scripts/deploy.sh sync-data

# --- each experiment ---
./scripts/deploy.sh run configs/mpnn_basic.json
./scripts/deploy.sh status
./scripts/deploy.sh logs 1234567        # job id from `status`
./scripts/deploy.sh fetch

# --- analyze locally ---
python main.py plot --results model_data/<run_dir>/<base>.csv
```

Each `run` writes a timestamped job script to remote `jobs/`, names the job
`mpnn_<config>_<timestamp>`, and streams output to `logs/<jobname>-<jobid>.out`.
Training artifacts land in the remote `model_data/<run_dir>/` (config, metadata,
checkpoints, epoch log, predictions) — see the [config README](../configs/README.md#outputs)
for the per-run file layout — and `fetch` mirrors them back.

## What gets transferred

| Step | Local → Remote | Remote → Local |
|---|---|---|
| `setup-env` | `requirements.txt` | — |
| `sync-data` | `database/MP/graphs_v4/`, `database/MP/*.pickle` | — |
| `run` / `sync-code` | `CNN/MPNN/*.py`, `configs/`, generated `jobs/*.slurm` | — |
| `fetch` | — | `model_data/`, `logs/` |

`run` does **not** re-send the graph database — it relies on `sync-data` having
been run. If you regenerate or extend the graphs, re-run `sync-data` (it only
sends the delta).

## Troubleshooting

- **`Invalid account or account/partition combination`** at submit — `SLURM_ACCOUNT`
  or `SLURM_PARTITION` is wrong. Run `sacctmgr show assoc user=<user>` and fix the block.
- **`conda: command not found` / activation fails in the job** — the `ENV_SETUP`
  module name is wrong for the cluster, or the env wasn't created. Re-run `setup-env`;
  adjust the `module load anaconda` line if your cluster names it differently.
- **Job can't find graphs / pickles** — `sync-data` hasn't completed, or `REMOTE_PATH`
  differs from where data was synced. The training entrypoint resolves graph paths
  relative to the project root, so the job's cwd must be `REMOTE_PATH`.
- **`CUDA available? False` in `setup-env` output** — expected: the login node has no
  GPU. CUDA is available inside the GPU job. Confirm with `nvidia-smi` in the job log.
- **Out of memory** — raise `SLURM_MEM`, or lower `graph_cache_size` / `num_workers`
  in the config.
