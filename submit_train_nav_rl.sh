#!/usr/bin/env bash
# Submit nav RL training on SLURM with guaranteed log paths.
#
# Usage:
#   ./submit_train_nav_rl.sh
#   ./submit_train_nav_rl.sh --time=0          # unlimited (this cluster)
#   ./submit_train_nav_rl.sh --time=72:00:00     # cap walltime if needed
#   ./submit_train_nav_rl.sh --partition=mig --cpus-per-task=16

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  echo "error: ${REPO_ROOT}/.venv/bin/python not found or not executable" >&2
  exit 1
fi

# SLURM opens --output/--error when the job starts; directory must exist at submit time.
mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/runs"

SLURM_OUT="${REPO_ROOT}/logs/slurm-%j.out"
SLURM_ERR="${REPO_ROOT}/logs/slurm-%j.out"

echo "Submitting from: ${REPO_ROOT}"
echo "SLURM stdout/stderr: ${SLURM_OUT}"
echo "Backup job log:     ${REPO_ROOT}/logs/job_<jobid>.log"
echo "Bootstrap log:      ${REPO_ROOT}/logs/bootstrap.log (always appended)"

job_id="$(
  sbatch \
    --chdir="${REPO_ROOT}" \
    --output="${SLURM_OUT}" \
    --error="${SLURM_ERR}" \
    "$@" \
    "${REPO_ROOT}/submit_train_nav_rl.slurm"
)"

echo "${job_id}"
job_num="${job_id##* }"
echo ""
echo "Monitor:"
echo "  squeue -j ${job_num}"
echo "  sacct -j ${job_num} --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem"
echo "  tail -f ${REPO_ROOT}/logs/slurm-${job_num}.out"
echo "  tail -f ${REPO_ROOT}/logs/job_${job_num}.log"
echo "  tail -f ${REPO_ROOT}/logs/bootstrap.log"
