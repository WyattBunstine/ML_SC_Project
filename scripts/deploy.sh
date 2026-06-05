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
#   ./scripts/deploy.sh status                    # squeue for your jobs
#   ./scripts/deploy.sh logs <jobid>              # tail a running job's log
#   ./scripts/deploy.sh fetch                     # rsync model_data/ + logs back here
#
# First run: edit the CONFIG block below (host/user/path + SLURM resources + env).
# ---------------------------------------------------------------------------

set -euo pipefail

# ========================= EDIT THIS BLOCK =================================
# --- Connection (Rockfish @ JHU; ssh keys already set up for passwordless login) ---
REMOTE_HOST="login.rockfish.jhu.edu"     # supercomputer hostname (or an ~/.ssh/config alias)
REMOTE_USER="wbunsti1"                   # your cluster username
REMOTE_PATH="/data/tmcquee2/wbunsti1/ML_SC_Proj"   # remote project root (PI group data dir)

# --- SLURM resource request (Rockfish-specific — verify against your allocation) ---
SLURM_PARTITION="a100"                   # Rockfish GPU partition (a100 nodes)
SLURM_ACCOUNT="tmcquee2-paradim_gpu"     # billing account; VERIFY exact name (check `sacctmgr show assoc user=wbunsti1`)
SLURM_TIME="12:00:00"                    # walltime HH:MM:SS
SLURM_GPUS="1"                           # training auto-uses CUDA if a GPU is present
SLURM_CPUS="8"                           # >= num_workers+1 in the config (config uses 4)
SLURM_MEM="48G"                         # Rockfish min node RAM; the graph LRU cache (graph_cache_size) lives in RAM
SLURM_MAIL_USER="wbunsti1@jh.edu"        # email for END/FAIL notifications; VERIFY address (leave "" to disable)

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
    ssh "${SSH}" "mkdir -p '${REMOTE_PATH}/database/datafiles/MP'"

    echo ">> Syncing graph database (14 GB, ~89k files) — first run is slow, later runs are fast..."
    rsync -a --info=progress2 --partial \
        database/datafiles/MP/graphs_v4 \
        "${SSH}:${REMOTE_PATH}/database/datafiles/MP/"

    echo ">> Syncing index pickles..."
    rsync -a --info=progress2 \
        database/datafiles/MP/*.pickle \
        "${SSH}:${REMOTE_PATH}/database/datafiles/MP/"

    echo ">> Dataset sync complete."
}

# ---------------------------------------------------------------------------
# Push just the training code + all configs. Tiny, runs every job submission.
# No --delete: never touches remote model_data/ or results.
# ---------------------------------------------------------------------------
sync_code() {
    ssh "${SSH}" "mkdir -p '${REMOTE_PATH}/CNN/MPNN' '${REMOTE_PATH}/configs' '${REMOTE_PATH}/logs' '${REMOTE_PATH}/jobs'"
    rsync -a CNN/MPNN/*.py "${SSH}:${REMOTE_PATH}/CNN/MPNN/"
    rsync -a configs/      "${SSH}:${REMOTE_PATH}/configs/"
}

# ---------------------------------------------------------------------------
# Push code+config and submit a SLURM job for the given config.
# The training entrypoint resolves graph paths RELATIVE to the project root
# (the pickle stores paths like database/datafiles/MP/graphs_v4/...), and imports its
# siblings from CNN/MPNN — so the job must `cd ${REMOTE_PATH}` and run
# `python CNN/MPNN/MPNNMain.py <config>` exactly as we do locally.
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
echo "Running: python CNN/MPNN/MPNNMain.py ${config_rel}"
python CNN/MPNN/MPNNMain.py "${config_rel}"
EOF

    echo ">> Submitting..."
    ssh "${SSH}" "cd '${REMOTE_PATH}' && sbatch '${job_file}'"
    echo ">> Submitted. Track with: ./scripts/deploy.sh status"
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
    rsync -a --info=progress2 "${SSH}:${REMOTE_PATH}/model_data/" model_data/ || true
    echo ">> Fetching SLURM logs/ ..."
    rsync -a "${SSH}:${REMOTE_PATH}/logs/" logs/ || true
    echo ">> Done."
}

cmd="${1:-}"
shift || true
case "${cmd}" in
    setup-env) setup_env ;;
    sync-data) sync_data ;;
    sync-code) sync_code ;;
    run)       run "$@" ;;
    status)    status ;;
    logs)      logs "$@" ;;
    fetch)     fetch ;;
    *)         usage ;;
esac
