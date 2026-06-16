#!/usr/bin/env bash
#
# deploy.sh — push the MPNN training code/config to a SLURM cluster and submit a job.
#
# Why it's split into subcommands: the graph database is ~14 GB across ~89k small
# JSON files. Copying that every run would be painfully slow (per-file SSH
# overhead), so it's a ONE-TIME `sync-data`. Each training cycle only ships the
# tiny code+config and submits an sbatch job.
#
#   ./scripts/deploy.sh setup-env                 # one-time: create the conda env on the cluster
#   ./scripts/deploy.sh sync-data                 # one-time: rsync the 14 GB DB + pickles
#   ./scripts/deploy.sh run configs/mpnn_basic.json   # push code+config, submit SLURM job
#   ./scripts/deploy.sh build-mptrj               # build MPtrj cgv4 graphs on the cluster (CPU job)
#   ./scripts/deploy.sh pack-mptrj                # pack graphs into the fast columnar training format
#   ./scripts/deploy.sh status                    # squeue for your jobs
#   ./scripts/deploy.sh logs <jobid>              # tail a running job's log
#   ./scripts/deploy.sh fetch                     # rsync model_data/ + logs back here
#   ./scripts/deploy.sh reorg                     # tidy runs into model_data/<date>/<tag>/ + index.csv (both sides)
#   ./scripts/deploy.sh archive <run> [run...]    # retire run(s) to .archive/ so fetch stops pulling them
#
# First run: edit the CONFIG block below (host/user/path + SLURM resources + env).
# ---------------------------------------------------------------------------

set -euo pipefail

# ========================= EDIT THIS BLOCK =================================
# --- Connection (Rockfish @ JHU; ssh keys already set up for passwordless login) ---
REMOTE_HOST="login.rockfish.jhu.edu"     # supercomputer hostname (or an ~/.ssh/config alias)
REMOTE_USER="wbunsti1"                   # your cluster username
REMOTE_PATH="/data/tmcquee2/wbunsti1/ML_SC_Proj"   # remote project root (PI group data dir)
# Bulk regenerable data (the ~1.5M MPtrj graph JSONs, ~308 GB) lives on scratch:
# /data hit its group quota at ~120k graphs. Scratch is PURGED PERIODICALLY — only
# re-buildable files go here (graphs re-build resumably from the MPtrj JSON, which
# stays on /data, as do the index pickles, code, and logs). If a purge removes
# graphs, re-run `deploy.sh build-mptrj` to regenerate, then re-run training.
SCRATCH_PATH="/scratch4/tmcquee2/wbunsti1"
SCRATCH_MPTRJ_GRAPHS="${SCRATCH_PATH}/ML_SC_Proj/MPtrj/graphs_v4"
# Packed columnar training store (MPNNPack): built from the graphs by
# `deploy.sh pack-mptrj`, read directly by training (configs point index_path
# here). Regenerable from graphs+index in a few hours, so scratch is fine.
SCRATCH_MPTRJ_PACK="${SCRATCH_PATH}/ML_SC_Proj/MPtrj/packed_v1"

# --- SLURM resource request (Rockfish-specific — verify against your allocation) ---
SLURM_PARTITION="a100"                   # Rockfish GPU partition (a100 nodes)
SLURM_ACCOUNT="tmcquee2-paradim_gpu"     # billing account; VERIFY exact name (check `sacctmgr show assoc user=wbunsti1`)
SLURM_TIME="12:00:00"                    # walltime HH:MM:SS
SLURM_GPUS="1"                           # training auto-uses CUDA if a GPU is present
SLURM_CPUS="8"                           # >= num_workers+1 in the config (config uses 4)
SLURM_MEM="48G"                         # Rockfish min node RAM; the graph LRU cache (graph_cache_size) lives in RAM
SLURM_MAIL_USER="wbunsti1@jh.edu"        # email for END/FAIL notifications; VERIFY address (leave "" to disable)

# --- CPU build job (graph generation: `build-mptrj`) ---------------------------
# Graph building is CPU-bound (Voronoi tessellation per structure) and uses NO
# GPU, so it runs on a standard compute partition, not the a100 nodes. These are
# used only by the `build-mptrj` subcommand.
SLURM_CPU_PARTITION="parallel"           # Rockfish CPU partition (full-node, ~48 cores)
SLURM_CPU_ACCOUNT="tmcquee2-paradim"     # CPU billing account (non-GPU allocation)
SLURM_CPU_TIME="24:00:00"                # walltime for the full ~1.5M-frame build (resumable if it runs over)
SLURM_CPU_CPUS="48"                       # worker processes = this; one full parallel node
SLURM_CPU_MEM="0"                         # "0" = request all memory on the node (graphs stream; modest per-worker RAM)

# --- Remote environment (a conda env created once via `./deploy.sh setup-env`) ---
CONDA_ENV="ml_sc"                        # conda env name on Rockfish (created by setup-env)
PYTHON_VERSION="3.11"                    # python version for the env
TORCH_CUDA_CHANNEL="cu128"               # PyTorch CUDA build matched to Rockfish's GPU driver (CUDA 12.9).
                                         #   The default PyPI wheel is CUDA 13 and won't run on that driver
                                         #   (torch falls back to CPU). Use cu124/cu121 if the driver is older
                                         #   — check `nvidia-smi` (top-right "CUDA Version") on a GPU node.

# This block runs at the top of every SLURM job to make python+torch+pymatgen
# importable. It activates the conda env that `setup-env` builds (below).
ENV_SETUP=$(cat <<ENVEOF
module load anaconda
# Make 'conda activate' work in a non-interactive batch shell.
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}
ENVEOF
)
# ==========================================================================

SSH="${REMOTE_USER}@${REMOTE_HOST}"

# Resolve repo root so the script works from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
    # Print the contiguous comment header (everything after the shebang up to the
    # first non-comment line), stripping the leading "# ".
    awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"
    exit 1
}

# ---------------------------------------------------------------------------
# One-time: create the conda env on Rockfish with all training deps.
# Run this ONCE before your first job (and again only if deps change).
# Idempotent: `conda create` is skipped if the env already exists; the pip
# install then just ensures/updates the packages.
# ---------------------------------------------------------------------------
setup_env() {
    [ -f requirements.txt ] || { echo "ERROR: requirements.txt not found at repo root"; exit 1; }

    echo ">> Pushing requirements.txt..."
    ssh "${SSH}" "mkdir -p '${REMOTE_PATH}'"
    rsync -a requirements.txt "${SSH}:${REMOTE_PATH}/"

    echo ">> Creating/updating conda env '${CONDA_ENV}' (python ${PYTHON_VERSION}) on ${REMOTE_HOST}..."
    # Heredoc runs remotely. Unquoted EOF so local vars (${CONDA_ENV}, paths)
    # expand now; \$(...) and \$1 are escaped to run on the remote instead.
    ssh "${SSH}" bash -s <<EOF
set -euo pipefail
module load anaconda
source "\$(conda info --base)/etc/profile.d/conda.sh"
# Create the env only if it doesn't already exist.
if ! conda env list | awk '{print \$1}' | grep -qx '${CONDA_ENV}'; then
    conda create -y -n '${CONDA_ENV}' python='${PYTHON_VERSION}'
fi
conda activate '${CONDA_ENV}'
pip install --upgrade pip
# Install PyTorch matched to the cluster GPU driver FIRST. The default PyPI wheel
# is the CUDA 13 build, which Rockfish's ~CUDA 12.9 driver can't run — torch then
# silently falls back to CPU (very slow training). Installing the cuXXX build from
# PyTorch's channel up front means the requirements step below sees torch already
# satisfied and never pulls the CUDA-13 wheel.
echo ">> Installing PyTorch (${TORCH_CUDA_CHANNEL}) matched to the cluster GPU driver..."
pip install --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_CHANNEL}" torch
echo ">> Installing remaining deps from requirements.txt..."
pip install -r '${REMOTE_PATH}/requirements.txt'
echo ">> Env ready. python: \$(which python)"
python -c "import torch, pymatgen, sklearn, pandas, numpy; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
EOF
    echo ">> setup-env complete. (CUDA shows False on the login node — that's expected; it'll be True inside the GPU job.)"
}

# ---------------------------------------------------------------------------
# One-time bulk transfer of the static dataset (14 GB, ~89k files).
# rsync is resumable and skips unchanged files, so re-running it after a
# partial transfer (or after adding new graphs) only sends the delta.
# ---------------------------------------------------------------------------
sync_data() {
    echo ">> Creating remote dirs..."
    ssh "${SSH}" "mkdir -p '${REMOTE_PATH}/database/datafiles/MP' '${REMOTE_PATH}/database/datafiles/MP_Energy' '${REMOTE_PATH}/database/datafiles/MPtrj'"

    # Element feature file for the baseline CGCNN (ORIG configs resolve it as
    # database/datafiles/atom_init.json; CIFData hard-asserts on it at startup —
    # without this line, every remote ORIG run died at job start).
    echo ">> Syncing atom_init.json..."
    rsync -a database/datafiles/atom_init.json "${SSH}:${REMOTE_PATH}/database/datafiles/"

    # Each dataset keeps its own graphs_v4/ + index pickles (SC under MP/, the
    # energy benchmark under MP_Energy/). Sync each so either can be trained on
    # the cluster. rsync globs/dirs that don't exist locally are skipped harmlessly.
    for ds in MP MP_Energy; do
        echo ">> Syncing ${ds} graphs + pickles (first run is slow; later runs send only the delta)..."
        rsync -a --info=progress2 --partial \
            "database/datafiles/${ds}/graphs_v4" \
            "${SSH}:${REMOTE_PATH}/database/datafiles/${ds}/" 2>/dev/null || true
        rsync -a --info=progress2 \
            database/datafiles/${ds}/*.pickle \
            "${SSH}:${REMOTE_PATH}/database/datafiles/${ds}/" 2>/dev/null || true
    done

    # MPtrj is CLUSTER-BUILT (`deploy.sh build-mptrj`): its graphs live on scratch
    # (${SCRATCH_MPTRJ_GRAPHS}) and its index is written remotely with scratch
    # paths — never rsync local MPtrj graphs/pickles into /data (a local index
    # would carry local paths and clobber the cluster one anyway).

    echo ">> Dataset sync complete."
}

# ---------------------------------------------------------------------------
# Push just the training code + all configs. Tiny, runs every job submission.
# No --delete: never touches remote model_data/ or results.
# ---------------------------------------------------------------------------
sync_code() {
    ssh "${SSH}" "mkdir -p '${REMOTE_PATH}/models/common' '${REMOTE_PATH}/models/MPNN' '${REMOTE_PATH}/models/GPSTransformer' '${REMOTE_PATH}/models/OriginalCGCNN' '${REMOTE_PATH}/configs' '${REMOTE_PATH}/logs' '${REMOTE_PATH}/jobs'"
    # Ship the shared infra (models/common: data layer, packer, resmon, trainer) +
    # each model package (MPNN, GPSTransformer) + the baseline CGCNN
    # (models/CGCNNMain.py + the OriginalCGCNN package it imports), so `run` can
    # dispatch any model. *.py only — never the local checkpoints / test_result
    # artifacts / __pycache__ that also live under models/.
    rsync -a models/*.py              "${SSH}:${REMOTE_PATH}/models/"
    rsync -a models/common/*.py       "${SSH}:${REMOTE_PATH}/models/common/"
    rsync -a models/MPNN/*.py         "${SSH}:${REMOTE_PATH}/models/MPNN/"
    rsync -a models/GPSTransformer/*.py "${SSH}:${REMOTE_PATH}/models/GPSTransformer/"
    rsync -a models/OriginalCGCNN/*.py "${SSH}:${REMOTE_PATH}/models/OriginalCGCNN/"
    rsync -a configs/              "${SSH}:${REMOTE_PATH}/configs/"
}

# ---------------------------------------------------------------------------
# Push code+config and submit a SLURM job for the given config.
# The entrypoint is chosen from the config (MPNN vs baseline CGCNN — see the
# dispatch in run()), mirroring main.py's train / train-mpnn split. Either trainer
# resolves data paths RELATIVE to the project root (pickles store paths like
# database/datafiles/MP/graphs_v4/...) and imports its siblings from its own dir —
# so the job must `cd ${REMOTE_PATH}` and run `python <entrypoint> <config>`
# exactly as we do locally.
# ---------------------------------------------------------------------------
run() {
    local config="${1:-}"
    [ -n "${config}" ] || { echo "ERROR: pass a config, e.g. run configs/mpnn_basic.json"; exit 1; }
    [ -f "${config}" ] || { echo "ERROR: config not found locally: ${config}"; exit 1; }

    # Config path must be relative to the repo root (that's the remote cwd too).
    local config_rel="${config#./}"
    local cfg_base; cfg_base="$(basename "${config_rel}" .json)"
    local stamp; stamp="$(date +%Y%m%d-%H%M%S)"
    local job_name="mpnn_${cfg_base}_${stamp}"
    local job_file="jobs/${job_name}.slurm"

    # --- Per-config SLURM overrides ----------------------------------------
    # A config may carry an optional top-level "slurm" object to request
    # different resources per experiment, e.g.:
    #     "slurm": { "cpus": 24, "mem": "96G", "time": "24:00:00" }
    # Any key present overrides the matching default from the EDIT THIS BLOCK
    # above; missing keys keep the default. MPNNMain.py ignores unknown keys,
    # so the same file drives both training and resource requests.
    # Recognized keys: partition, account, time, gpus, cpus, mem, mail_user.
    local s_part="${SLURM_PARTITION}" s_acct="${SLURM_ACCOUNT}" s_time="${SLURM_TIME}"
    local s_gpus="${SLURM_GPUS}" s_cpus="${SLURM_CPUS}" s_mem="${SLURM_MEM}"
    local s_mail="${SLURM_MAIL_USER}"
    # Read overrides via command substitution (NOT process substitution) so the
    # parser's exit code is visible: a bad config must fail loudly here, not
    # silently fall back to the defaults and submit with the wrong resources. The
    # parser exits 0 with no output when there's no "slurm" block (defaults are
    # then correct), and nonzero on a malformed block / unknown key / bad value.
    local slurm_kv
    if ! slurm_kv="$(python3 - "${config}" <<'PY'
import json, sys

VALID = ("partition", "account", "time", "gpus", "cpus", "mem", "mail_user")
try:
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
except Exception as e:
    sys.stderr.write(f"  invalid config JSON: {e}\n"); sys.exit(1)

slurm = cfg.get("slurm")
if slurm is None:
    sys.exit(0)                       # no override block -> use deploy.sh defaults
if not isinstance(slurm, dict):
    sys.stderr.write("  'slurm' must be a JSON object\n"); sys.exit(1)

for k, v in slurm.items():
    if k not in VALID:                # catches typos like "cpu"/"memory" that
        sys.stderr.write(            # would otherwise be silently ignored
            f"  unknown slurm key '{k}' (valid: {', '.join(VALID)})\n"); sys.exit(1)
    if v is None:
        continue
    # A bare number for mem means MEGABYTES to SLURM (e.g. 96 -> 96M, instant
    # OOM). Require an explicit unit suffix.
    if k == "mem" and str(v)[-1:].upper() not in ("K", "M", "G", "T"):
        sys.stderr.write(
            f"  slurm.mem='{v}' needs a unit suffix, e.g. \"96G\" "
            f"(a bare number means MB to SLURM)\n"); sys.exit(1)
    # cpus/gpus must be positive integers.
    if k in ("cpus", "gpus"):
        try:
            if int(v) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            sys.stderr.write(f"  slurm.{k}='{v}' must be a positive integer\n")
            sys.exit(1)
    print(f"{k}\t{v}")
PY
)"; then
        echo "ERROR: bad SLURM override in ${config} (see above)." >&2
        exit 1
    fi

    local key val
    while IFS=$'\t' read -r key val; do
        [ -n "${key}" ] || continue   # skip the empty line from an empty result
        case "${key}" in
            partition) s_part="${val}" ;;
            account)   s_acct="${val}" ;;
            time)      s_time="${val}" ;;
            gpus)      s_gpus="${val}" ;;
            cpus)      s_cpus="${val}" ;;
            mem)       s_mem="${val}" ;;
            mail_user) s_mail="${val}" ;;
        esac
    done <<< "${slurm_kv}"
    echo ">> SLURM request: partition=${s_part} cpus=${s_cpus} gpus=${s_gpus} mem=${s_mem} time=${s_time}"

    # --- Validate the config before pushing/submitting ---------------------
    # Fail fast LOCALLY (referenced files exist, enum/range sanity, target
    # resolvable, log1p not used on a negative target) so a typo never costs a
    # queue wait + a wasted allocation. Cross-checks num_workers against the
    # resolved cpus. Run from the repo root so the config's relative paths resolve.
    echo ">> Validating config..."
    if ! python3 scripts/validate_config.py "${config}" --cpus "${s_cpus}"; then
        echo "ERROR: config validation failed for ${config}; not submitting." >&2
        exit 1
    fi

    # --- Pick the trainer entrypoint from the config -----------------------
    # Mirror main.py's train / train-mpnn split (which locally chooses the script
    # by subcommand) and validate_config.py's classification: an MPNN config
    # carries "index_path" (pre-computed graphs); the baseline CGCNN carries
    # "dataset" (a struc_dict pickle) and/or "models":"ORIG". Routing on the data
    # key is the reliable signal — nothing in the code reads "models". Without
    # this, every job ran models/MPNN/MPNNMain.py, so an ORIG config silently trained
    # the wrong model.
    local entrypoint
    if ! entrypoint="$(python3 - "${config}" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
arch = str(cfg.get("architecture", "")).strip().lower()
models = str(cfg.get("models", "")).strip().upper()
has_index, has_dataset = "index_path" in cfg, "dataset" in cfg
if arch == "gps":
    # GPS is a v4-graph model (carries index_path); route it to its own trainer.
    # Without this branch it would fall through to MPNNMain and silently train a
    # CrystalMPNN on the GPS config.
    if not has_index:
        sys.stderr.write("  architecture=gps but config has no 'index_path'\n")
        sys.exit(1)
    print("models/GPSTransformer/gps_main.py")
elif models == "ORIG" or (has_dataset and not has_index):
    if has_index and not has_dataset:        # contradictory: ORIG model, MPNN data
        sys.stderr.write("  models=ORIG but config has 'index_path' (MPNN data), not 'dataset'\n")
        sys.exit(1)
    print("models/CGCNNMain.py")
elif has_index:
    print("models/MPNN/MPNNMain.py")
else:
    sys.stderr.write("  cannot determine model: config has neither 'index_path' (MPNN) nor 'dataset' (ORIG)\n")
    sys.exit(1)
PY
)"; then
        echo "ERROR: could not determine trainer entrypoint for ${config} (see above)." >&2
        exit 1
    fi
    echo ">> Model entrypoint: ${entrypoint}"

    echo ">> Pushing code + configs..."
    sync_code

    # Build the sbatch script. Unquoted heredoc so local vars expand now;
    # %x/%j are SLURM runtime tokens (job name / job id), left literal.
    local account_line=""
    [ -n "${s_acct}" ] && account_line="#SBATCH --account=${s_acct}"
    # Email on completion/failure only if an address is configured. Both #SBATCH
    # lines go on one bash line joined by a literal \n so they expand correctly.
    local mail_lines=""
    [ -n "${s_mail}" ] && mail_lines=$'#SBATCH --mail-type=BEGIN,END,FAIL\n'"#SBATCH --mail-user=${s_mail}"

    echo ">> Writing remote job script: ${job_file}"
    ssh "${SSH}" "cat > '${REMOTE_PATH}/${job_file}'" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${s_part}
${account_line}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=${s_time}
#SBATCH --gpus=${s_gpus}
#SBATCH --cpus-per-task=${s_cpus}
#SBATCH --mem=${s_mem}
${mail_lines}
#SBATCH --output=${REMOTE_PATH}/logs/%x-%j.out
#SBATCH --error=${REMOTE_PATH}/logs/%x-%j.err

set -euo pipefail
${ENV_SETUP}

export PYTHONUNBUFFERED=1          # stream training output to the log live
cd "${REMOTE_PATH}"
echo "Host: \$(hostname)   GPU(s):"; nvidia-smi -L || true
echo "Running: python ${entrypoint} ${config_rel}"
python ${entrypoint} "${config_rel}"
EOF

    echo ">> Submitting..."
    ssh "${SSH}" "cd '${REMOTE_PATH}' && sbatch '${job_file}'"
    echo ">> Submitted. Track with: ./scripts/deploy.sh status"
}

# ---------------------------------------------------------------------------
# Build the MPtrj cgv4 graphs ON THE CLUSTER (CPU job) instead of locally.
# The graphs land where training runs, so they never need to be fetched back.
#
# One command does the whole setup:
#   1. ensures ijson + tess (Voro++) in the conda env on the login node
#      (network + compilers are available there, not on compute nodes);
#   2. ships the graph builder (RPToleranceFactor/crystal_graph_v4.py),
#      main.py + database/*.py, and the 12 GB MPtrj JSON (one-time, resumable);
#   3. writes + submits a CPU SLURM job that runs `python main.py build-mptrj`.
#
# The job REQUIRES tess: it imports it up front and aborts if missing, so the
# build uses Voro++ (not the scipy fallback). Resumable — re-submitting skips
# frames whose graph already exists.
# ---------------------------------------------------------------------------
build_mptrj() {
    local mptrj_json="database/datafiles/MPtrj/MPtrj_2022.9_full.json"
    [ -f "${mptrj_json}" ] || { echo "ERROR: ${mptrj_json} not found locally"; exit 1; }
    [ -f "../RPToleranceFactor/crystal_graph_v4.py" ] || {
        echo "ERROR: ../RPToleranceFactor/crystal_graph_v4.py not found (the cgv4 builder)"; exit 1; }

    # Remote sibling path for the builder, mirroring the local layout
    # (crystal_graph_v4_import.py resolves RP_TOLERANCE_FACTOR_PATH, set in the job).
    local remote_rp="${REMOTE_PATH%/*}/RPToleranceFactor"

    echo ">> Ensuring ijson + tess (Voro++) in env '${CONDA_ENV}' on the login node..."
    # gcc is needed to compile tess's Voro++ extension; module name may vary by
    # cluster — adjust if 'module load gcc' fails.
    ssh "${SSH}" bash -s <<EOF
set -euo pipefail
module load anaconda
module load gcc 2>/dev/null || true
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}
python -c "import ijson" 2>/dev/null || pip install ijson
# tess (Voro++) needs a from-isolation rebuild: its shipped Cython-generated
# tess/_voro.cpp #includes "longintrepr.h", a CPython header removed in 3.12+,
# so the default build fails. setup.py only re-cythonizes if Cython is importable
# at build time — pip's build isolation hides it. So install a modern Cython into
# the env and build with --no-build-isolation so it regenerates _voro.cpp for
# this Python. (setuptools+wheel must be present too, since isolation is off.)
if ! python -c "import tess" 2>/dev/null; then
    pip install "cython>=3.0" setuptools wheel
    pip install tess --no-build-isolation
fi
python -c "import tess; print('  tess OK — Voro++ available')"
EOF

    echo ">> Syncing builder + code + the 12 GB MPtrj JSON (one-time; resumable)..."
    ssh "${SSH}" "mkdir -p '${remote_rp}' '${REMOTE_PATH}/database/datafiles/MPtrj' '${REMOTE_PATH}/logs' '${REMOTE_PATH}/jobs' '${SCRATCH_MPTRJ_GRAPHS}'"
    rsync -a ../RPToleranceFactor/crystal_graph_v4.py "${SSH}:${remote_rp}/"
    rsync -a main.py            "${SSH}:${REMOTE_PATH}/"
    rsync -a database/*.py      "${SSH}:${REMOTE_PATH}/database/"
    rsync -a --info=progress2 --partial \
        "${mptrj_json}" "${SSH}:${REMOTE_PATH}/database/datafiles/MPtrj/"

    # One-time migration: graphs built by earlier runs into the OLD /data location
    # move to scratch so (a) the blown /data group quota is freed and (b) the
    # resumable build skips them instead of rebuilding. rsync --remove-source-files
    # is itself resumable; ~120k files take a while on GPFS — run only if needed.
    local old_graphs="${REMOTE_PATH}/database/datafiles/MPtrj/graphs_v4"
    if ssh "${SSH}" "[ -d '${old_graphs}' ] && [ -n \"\$(ls -f '${old_graphs}' 2>/dev/null | head -3 | tail -n +3)\" ]"; then
        echo ">> Migrating existing graphs from /data to scratch (one-time; frees the /data quota;"
        echo ">>   ~120k files can take tens of minutes — safe to interrupt and re-run)..."
        ssh "${SSH}" "
            rsync -a --remove-source-files '${old_graphs}/' '${SCRATCH_MPTRJ_GRAPHS}/' &&
            find '${old_graphs}' -type d -empty -delete
            echo '   migrated; /data graph dir removed.'
        "
    fi

    # Payload runs after the shared CPU-job preamble (env + cd, see submit_cpu_job).
    # Force Voro++ (tess): abort if it isn't importable so the build never
    # silently uses the scipy fallback. Graphs -> scratch (bulk, regenerable;
    # /data quota can't hold 1.5M files); index pickle -> /data (small, NOT
    # purged) with absolute scratch graph paths.
    submit_cpu_job "build_mptrj" "${SLURM_CPU_TIME}" "$(cat <<EOF
module load gcc 2>/dev/null || true
export RP_TOLERANCE_FACTOR_PATH="${remote_rp}"
python -c "import tess; print('Using tess/Voro++ for Voronoi tessellation')"
echo "Host: \$(hostname)   CPUs: ${SLURM_CPU_CPUS}"
python main.py build-mptrj --workers ${SLURM_CPU_CPUS} \\
    --graph-dir "${SCRATCH_MPTRJ_GRAPHS}" \\
    --output database/datafiles/MPtrj/MPtrj_V4
EOF
)"
    echo ">> Submitted. Track with: ./scripts/deploy.sh status   (logs: ./scripts/deploy.sh logs <jobid>)"
    echo ">> When it finishes: graphs on scratch (${SCRATCH_MPTRJ_GRAPHS}),"
    echo ">>   index at ${REMOTE_PATH}/database/datafiles/MPtrj/MPtrj_V4.pickle — train directly, no fetch."
    echo ">> NOTE: scratch is purged periodically; if graphs vanish, re-run build-mptrj to regenerate."
}

# ---------------------------------------------------------------------------
# Pack the MPtrj graphs into the columnar training format (CPU job).
# One-time after build-mptrj finishes (and after any graph rebuild): parses every
# graph JSON once and writes memmap-able arrays + offsets to scratch. Training
# configs point index_path at ${SCRATCH_MPTRJ_PACK}; sample tensors are
# bitwise-identical to the lazy loader but ~30x faster to read. Resumable is NOT
# needed (a re-run overwrites; ~30-60 min on a parallel node).
# ---------------------------------------------------------------------------
pack_mptrj() {
    echo ">> Pushing code..."
    sync_code
    rsync -a main.py "${SSH}:${REMOTE_PATH}/"

    submit_cpu_job "pack_mptrj" "06:00:00" "$(cat <<EOF
python main.py pack-dataset \\
    --index database/datafiles/MPtrj/MPtrj_V4.pickle \\
    --out "${SCRATCH_MPTRJ_PACK}" \\
    --workers ${SLURM_CPU_CPUS}
EOF
)"
    echo ">> Submitted. When done, train with configs whose index_path = ${SCRATCH_MPTRJ_PACK}"
}

# ---------------------------------------------------------------------------
# Shared CPU SLURM job submitter: $1 = job-name prefix, $2 = walltime,
# $3 = script body (runs after env activation + cd into the project root).
# One copy of the #SBATCH preamble / mail / log plumbing so the CPU jobs
# (build-mptrj, pack-mptrj) can't drift apart.
# ---------------------------------------------------------------------------
submit_cpu_job() {
    local name_prefix="$1" walltime="$2" payload="$3"
    local stamp; stamp="$(date +%Y%m%d-%H%M%S)"
    local job_name="${name_prefix}_${stamp}"
    local job_file="jobs/${job_name}.slurm"
    local account_line=""
    [ -n "${SLURM_CPU_ACCOUNT}" ] && account_line="#SBATCH --account=${SLURM_CPU_ACCOUNT}"
    local mail_lines=""
    [ -n "${SLURM_MAIL_USER}" ] && mail_lines=$'#SBATCH --mail-type=END,FAIL\n'"#SBATCH --mail-user=${SLURM_MAIL_USER}"

    echo ">> SLURM request: partition=${SLURM_CPU_PARTITION} cpus=${SLURM_CPU_CPUS} mem=${SLURM_CPU_MEM} time=${walltime}"
    echo ">> Writing remote job script: ${job_file}"
    ssh "${SSH}" "cat > '${REMOTE_PATH}/${job_file}'" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${SLURM_CPU_PARTITION}
${account_line}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=${walltime}
#SBATCH --cpus-per-task=${SLURM_CPU_CPUS}
#SBATCH --mem=${SLURM_CPU_MEM}
${mail_lines}
#SBATCH --output=${REMOTE_PATH}/logs/%x-%j.out
#SBATCH --error=${REMOTE_PATH}/logs/%x-%j.err

set -euo pipefail
${ENV_SETUP}
export PYTHONUNBUFFERED=1
cd "${REMOTE_PATH}"
${payload}
EOF

    echo ">> Submitting..."
    ssh "${SSH}" "cd '${REMOTE_PATH}' && sbatch '${job_file}'"
}

status() {
    ssh "${SSH}" "squeue -u '${REMOTE_USER}' -o '%.10i %.30j %.8T %.10M %.10l %.6D %R'"
}

logs() {
    local jobid="${1:-}"
    [ -n "${jobid}" ] || { echo "ERROR: pass a job id (see 'status')"; exit 1; }
    # Logs are named <jobname>-<jobid>.out; match on the id.
    ssh "${SSH}" "tail -f ${REMOTE_PATH}/logs/*-${jobid}.out"
}

# ---------------------------------------------------------------------------
# Pull finished run artifacts back. model_data/<run>/ holds the checkpoint,
# metadata.json, epoch log, and predictions; logs/ holds the SLURM stdout.
# ---------------------------------------------------------------------------
fetch() {
    mkdir -p model_data logs
    echo ">> Fetching model_data/ ..."
    # --exclude '.archive': runs you've retired (see `archive`) stay on the
    # remote but are never pulled back, so deleting clutter actually sticks.
    rsync -a --info=progress2 --exclude='.archive' \
        "${SSH}:${REMOTE_PATH}/model_data/" model_data/ || true
    echo ">> Fetching SLURM logs/ ..."
    rsync -a "${SSH}:${REMOTE_PATH}/logs/" logs/ || true
    # Refresh the at-a-glance comparison index (best val metric, params, epochs,
    # completeness per run) and move each fetched SLURM log into the run dir it
    # belongs to (mapping read from the log's "Run output dir:" line), so a run's
    # stdout lives next to its checkpoints. Non-training logs stay in logs/.
    # Reorg of older flat runs is a separate explicit step.
    echo ">> Refreshing model_data/index.csv + distributing logs into run dirs ..."
    python3 scripts/reorg_runs.py --root model_data --index-only --distribute-logs logs || true
    echo ">> Done."
}

# ---------------------------------------------------------------------------
# Tidy run dirs into model_data/<date>/<run_tag>/<run_id>/ and (re)write
# index.csv. New runs are already born nested (the trainers write there); this
# migrates older FLAT runs. It moves on BOTH the remote and locally with the
# SAME leaf names, so the next `fetch` (a plain rsync mirror) re-pairs them with
# no re-download. Idempotent: already-nested runs are skipped. Dry-run first
# with `reorg --dry-run` to preview the move plan.
# ---------------------------------------------------------------------------
reorg() {
    local apply="--apply"
    [ "${1:-}" = "--dry-run" ] && apply=""

    echo ">> Pushing reorg helper to remote..."
    ssh "${SSH}" "mkdir -p '${REMOTE_PATH}/scripts'"
    rsync -a scripts/reorg_runs.py "${SSH}:${REMOTE_PATH}/scripts/"

    echo ">> Reorganizing remote model_data/ ..."
    ssh "${SSH}" "python3 '${REMOTE_PATH}/scripts/reorg_runs.py' --root '${REMOTE_PATH}/model_data' ${apply}"

    echo ">> Reorganizing local model_data/ ..."
    python3 scripts/reorg_runs.py --root model_data ${apply}

    # Explicit if (not `[ ... ] && echo`): a bare test as the function's last
    # command would return 1 in apply mode, and `set -e` would fail the whole
    # script right after a fully successful reorg.
    if [ -z "${apply}" ]; then
        echo ">> (dry run — re-run 'reorg' without --dry-run to apply)"
    fi
}

# ---------------------------------------------------------------------------
# Retire one or more runs: move them into model_data/.archive/ on BOTH sides.
# `fetch` excludes .archive, so an archived run stops being pulled back and the
# clutter is gone from the active tree for good (the data is still recoverable
# under .archive/ on either machine). Pass run paths RELATIVE to model_data/,
# e.g. the rel_path column from index.csv:
#     ./scripts/deploy.sh archive 2026-06-08/eform_minimal/eform_minimal_2026-06-08_12-02-08
# ---------------------------------------------------------------------------
archive() {
    [ "$#" -ge 1 ] || { echo "ERROR: pass one or more run paths relative to model_data/ (see index.csv rel_path)"; exit 1; }
    for rel in "$@"; do
        rel="${rel#model_data/}"; rel="${rel%/}"     # tolerate a model_data/ prefix or trailing slash
        [ -n "${rel}" ] || continue
        echo ">> Archiving '${rel}' (remote + local)..."
        # Remote: only move if it exists there.
        ssh "${SSH}" "
            set -e
            src='${REMOTE_PATH}/model_data/${rel}'
            dst='${REMOTE_PATH}/model_data/.archive/${rel}'
            if [ -e \"\$src\" ]; then mkdir -p \"\$(dirname \"\$dst\")\"; mv \"\$src\" \"\$dst\"; echo '   remote: archived'; else echo '   remote: not present, skipped'; fi
        "
        # Local mirror.
        if [ -e "model_data/${rel}" ]; then
            mkdir -p "model_data/.archive/$(dirname "${rel}")"
            mv "model_data/${rel}" "model_data/.archive/${rel}"
            echo "   local: archived"
        else
            echo "   local: not present, skipped"
        fi
    done
    echo ">> Refreshing model_data/index.csv ..."
    python3 scripts/reorg_runs.py --root model_data --index-only || true
    echo ">> Done. Archived runs live under model_data/.archive/ and won't be re-fetched."
}

cmd="${1:-}"
shift || true
case "${cmd}" in
    setup-env) setup_env ;;
    sync-data) sync_data ;;
    sync-code) sync_code ;;
    run)         run "$@" ;;
    build-mptrj) build_mptrj ;;
    pack-mptrj)  pack_mptrj ;;
    status)    status ;;
    logs)      logs "$@" ;;
    fetch)     fetch ;;
    reorg)     reorg "$@" ;;
    archive)   archive "$@" ;;
    *)         usage ;;
esac
