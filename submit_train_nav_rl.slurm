#!/bin/bash
#SBATCH --job-name=nav_rl_train
#SBATCH --partition=mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0
# Overridden to absolute paths by submit_train_nav_rl.sh
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out

# --- Bootstrap (no set -e): always append to a file on disk ---
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
BOOT_LOG="${LOG_DIR}/bootstrap.log"
JOB_ID="${SLURM_JOB_ID:-unknown}"
JOB_LOG="${LOG_DIR}/job_${JOB_ID}.log"

mkdir -p "${LOG_DIR}" "${REPO_ROOT}/runs" 2>/dev/null || true

_boot() {
  # Write directly to file (works even if SLURM stdout is broken).
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "${BOOT_LOG}"
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "${JOB_LOG}"
}

_boot "=== job bootstrap job_id=${JOB_ID} ==="
_boot "repo=${REPO_ROOT}"
_boot "host=$(hostname 2>&1 || echo unknown)"
_boot "user=${USER:-unknown} pwd=$(pwd 2>&1)"
_boot "bash=${BASH_VERSION:-unknown} shell=${SHELL:-unknown}"
_boot "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-n/a}"

cd "${REPO_ROOT}" || {
  _boot "FATAL: cannot cd to ${REPO_ROOT}"
  exit 1
}

set -eo pipefail
set -x

log() {
  local line="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "${line}"
  echo "${line}" >> "${JOB_LOG}"
}

on_exit() {
  local ec=$?
  # Avoid calling log() if it might fail; use _boot for reliability.
  _boot "EXIT trap exit_code=${ec}"
  exit "${ec}"
}
trap on_exit EXIT

log "========== job start =========="
log "REPO_ROOT=${REPO_ROOT}"
log "JOB_LOG=${JOB_LOG}"
log "BOOT_LOG=${BOOT_LOG}"
log "SLURM_JOB_ID=${SLURM_JOB_ID:-n/a}"
log "SLURM_PARTITION=${SLURM_PARTITION:-n/a}"
log "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-n/a}"

if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  log "ERROR: missing ${REPO_ROOT}/.venv/bin/python"
  _boot "ERROR: missing venv python"
  exit 1
fi

export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export VIRTUAL_ENV="${REPO_ROOT}/.venv"
export PYTHONUNBUFFERED=1
unset PYTHONHOME

log "python=$(command -v python)"
python -V || { log "ERROR: python -V failed"; exit 1; }

set +e
python -c "import sys; print('executable', sys.executable)"
py_check=$?
set -e
if [[ ${py_check} -ne 0 ]]; then
  log "ERROR: python sanity check failed (exit ${py_check})"
  _boot "ERROR: python sanity check failed exit=${py_check}"
  exit "${py_check}"
fi

log "cmd=python train_nav_rl.py --timesteps 4000000 --n-envs 8 --n-steps 1024 --batch-size 512"

set +e
python train_nav_rl.py \
  --timesteps 4000000000000 \
  --n-envs 8 \
  --n-steps 1024 \
  --batch-size 512
train_ec=$?
set -e

log "python exit_code=${train_ec}"
_boot "training finished exit_code=${train_ec}"

if [[ ${train_ec} -ne 0 ]]; then
  log "ERROR: training command failed"
  exit "${train_ec}"
fi

log "========== job end (success) =========="
