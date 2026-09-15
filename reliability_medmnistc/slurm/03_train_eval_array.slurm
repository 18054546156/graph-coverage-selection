#!/bin/bash
# Train selected subsets, save final.pt, then run clean and corruption tests.
#SBATCH --job-name=medmnistc-train-eval
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-9%4
#SBATCH --output=logs/train_eval_%A_%a.out
#SBATCH --error=logs/train_eval_%A_%a.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
PYTHON="${PYTHON:-python}"
cd "$REPO_ROOT"
DATASET_NAMES=(organsmnist organamnist pathmnist tissuemnist bloodmnist)
RATIO_VALUES=(0.02 0.05)
TASK_ID="${SLURM_ARRAY_TASK_ID:?submit this file as its declared array}"
export DATASET="${DATASET_NAMES[$((TASK_ID / 2))]}"
export RATIOS="${RATIO_VALUES[$((TASK_ID % 2))]}"
export MEDMNIST_ROOT="${MEDMNIST_ROOT:-$REPO_ROOT/data/medmnist}"
export MEDMNISTC_ROOT="${MEDMNISTC_ROOT:-$REPO_ROOT/data/medmnistc}"
export RELIABILITY_OUT="${RELIABILITY_OUT:-$REPO_ROOT/runs/train_eval}"
export SELECTION_OUT="${SELECTION_OUT:-$REPO_ROOT/artifacts/selections}"
export GRAPH_CACHE="${GRAPH_CACHE:-$REPO_ROOT/cache}"
export DATASETS="${DATASET:-${DATASETS:-organsmnist,organamnist,pathmnist,tissuemnist,bloodmnist}}"
export RUN_FULL_TRAIN=0
export AUTO_CONSOLIDATE=0
"$PYTHON" reliability_medmnistc/scripts/train.py \
  --config reliability_medmnistc/configs/full_5datasets_8methods.yaml
"$PYTHON" reliability_medmnistc/scripts/evaluate.py \
  --config reliability_medmnistc/configs/full_5datasets_8methods.yaml
