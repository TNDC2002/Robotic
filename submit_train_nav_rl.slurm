#!/bin/bash
#SBATCH --job-name=nav_rl_train
#SBATCH --partition=mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0
# Paths below are overridden by submit_train_nav_rl.sh (absolute paths).
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out

# Always produce debug output even when the job fails early.
set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

mkdir -p logs runs

# Mirror everything to a repo log (backup if SLURM output path fails).
JOB_LOG="${REPO_ROOT}/logs/job_${SLURM_JOB_ID:-local}.log"
exec > >(tee -a "${JOB_LOG}") 2>&1

log() { echo "[$(date -Is)] $*"; }

on_exit() {
  local ec=$?
  log "EXIT trap: exit_code=${ec} job_id=${SLURM_JOB_ID:-n/a}"
  if [[ ${ec} -ne 0 ]]; then
    log "Job failed. Check SLURM logs and ${JOB_LOG}"
  fi
  exit "${ec}"
}
trap on_exit EXIT

on_err() {
  log "ERR trap: command failed at line ${1} (exit ${2:-?})"
}
trap 'on_err ${LINENO} $?' ERR

log "========== job start =========="
log "REPO_ROOT=${REPO_ROOT}"
log "SLURM_JOB_ID=${SLURM_JOB_ID:-n/a}"
log "SLURM_JOB_NAME=${SLURM_JOB_NAME:-n/a}"
log "SLURM_PARTITION=${SLURM_PARTITION:-n/a}"
log "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-n/a}"
log "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-n/a}"
log "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-n/a}"
log "SLURM_SUBMIT_HOST=${SLURM_SUBMIT_HOST:-n/a}"
log "host=$(hostname)"
log "pwd=$(pwd)"
log "user=${USER:-n/a}"
log "bash=${BASH_VERSION}"

if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  log "ERROR: missing ${REPO_ROOT}/.venv/bin/python"
  exit 1
fi

export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export VIRTUAL_ENV="${REPO_ROOT}/.venv"
export PYTHONUNBUFFERED=1
unset PYTHONHOME

log "python=$(command -v python)"
python -V
python -c "import sys; print('executable', sys.executable)"

TRAIN_CMD=(
  python train_nav_rl.py
  --timesteps 4000000
  --n-envs 8
  --n-steps 1024
  --batch-size 512
)
log "cmd=${TRAIN_CMD[*]}"

"${TRAIN_CMD[@]}"
train_ec=$?
log "python exit_code=${train_ec}"

if [[ ${train_ec} -ne 0 ]]; then
  log "ERROR: training command failed"
  exit "${train_ec}"
fi

log "========== job end (success) =========="
