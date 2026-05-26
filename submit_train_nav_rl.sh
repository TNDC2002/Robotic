#!/usr/bin/env bash
# Submit nav RL training on SLURM (mig: 16 CPUs, 128 GiB RAM).
#
# Usage:
#   ./submit_train_nav_rl.sh
#   ./submit_train_nav_rl.sh --time 72:00:00
#   sbatch --dependency=afterok:12345 submit_train_nav_rl.slurm

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  echo "error: ${REPO_ROOT}/.venv/bin/python not found or not executable" >&2
  exit 1
fi

exec sbatch "$@" "${REPO_ROOT}/submit_train_nav_rl.slurm"
