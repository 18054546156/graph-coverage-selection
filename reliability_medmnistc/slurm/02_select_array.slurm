#!/bin/bash
# Create selected-index artifacts. Set DATASET for a per-dataset array task.
#SBATCH --job-name=medmnistc-select
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/select_%A_%a.out
#SBATCH --error=logs/select_%A_%a.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-python}"
cd "$REPO_ROOT"
export MEDMNIST_ROOT="${MEDMNIST_ROOT:-$REPO_ROOT/data/medmnist}"
export MEDMNISTC_ROOT="${MEDMNISTC_ROOT:-$REPO_ROOT/data/medmnistc}"
export RELIABILITY_OUT="${RELIABILITY_OUT:-$REPO_ROOT/runs/select}"
export SELECTION_OUT="${SELECTION_OUT:-$REPO_ROOT/artifacts/selections}"
export GRAPH_CACHE="${GRAPH_CACHE:-$REPO_ROOT/cache}"
export DATASETS="${DATASET:-${DATASETS:-organsmnist,organamnist,pathmnist,tissuemnist,bloodmnist}}"
exec "$PYTHON" reliability_medmnistc/scripts/select.py \
  --config reliability_medmnistc/configs/full_5datasets_8methods.yaml
