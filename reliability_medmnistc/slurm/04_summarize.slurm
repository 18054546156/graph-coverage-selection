#!/bin/bash
# Consolidate only completed run metrics; does not train or evaluate.
#SBATCH --job-name=medmnistc-summary
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/summary_%j.out
#SBATCH --error=logs/summary_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
PYTHON="${PYTHON:-python}"
cd "$REPO_ROOT"
export RELIABILITY_OUT="${RELIABILITY_OUT:-$REPO_ROOT/runs/train_eval}"
exec "$PYTHON" reliability_medmnistc/scripts/summarize.py \
  --config reliability_medmnistc/configs/full_5datasets_8methods.yaml
