#!/bin/bash
#SBATCH --job-name=nav_rl_train
#SBATCH --partition=mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

# Repo root = directory containing this .slurm file ($0 is the script path under sbatch).
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

mkdir -p logs runs

if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  echo "error: ${REPO_ROOT}/.venv/bin/python not found" >&2
  exit 1
fi

# Use project venv (no system Python).
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export VIRTUAL_ENV="${REPO_ROOT}/.venv"
unset PYTHONHOME

echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "python=$(command -v python)"
python -V
echo "start=$(date -Is)"
echo "cmd=python train_nav_rl.py --timesteps 4000000 --n-envs 16"

python train_nav_rl.py --timesteps 4000000 --n-envs 16

echo "end=$(date -Is)"
