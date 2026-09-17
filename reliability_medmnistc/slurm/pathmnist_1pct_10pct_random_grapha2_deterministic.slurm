#!/bin/bash
# PathMNIST 1% and 10%: fixed seed-42 selections x five training seeds.
# Task map: ratio x method x training seed = 2 x 2 x 5 = 20 tasks.
# One task uses one GPU, 8 CPUs, and 48G; at most five run concurrently.
# Selection artifacts are the existing benchmark artifacts, not regenerated here.
#SBATCH --job-name=path-r01-r10-det
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-19%5
#SBATCH --output=logs/path_r01_r10_det_%A_%a.out
#SBATCH --error=logs/path_r01_r10_det_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?set REPO_ROOT or submit from the repository}}"
OUT_ROOT="${OUT_ROOT:?set OUT_ROOT}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
CORR_ROOT="${CORR_ROOT:?set CORR_ROOT}"
SELECTION_BASE="${SELECTION_BASE:?set SELECTION_BASE}"

cd "$REPO_ROOT"
export PYTHON="${PYTHON:-/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/envs/medmnistc-py311/bin/python}"
source "$REPO_ROOT/reliability_medmnistc/slurm/common_env.sh"
configure_imagemagick

RATIOS=(0.01 0.1)
RATIO_TAGS=(001 010)
METHODS_LIST=(random graph_a2)
SEEDS=(42 43 44 45 46)
TASK_ID="${SLURM_ARRAY_TASK_ID:?submit this file as its declared array}"
RATIO_INDEX=$((TASK_ID / 10))
METHOD_INDEX=$(((TASK_ID % 10) / 5))
SEED_INDEX=$((TASK_ID % 5))
RATIO="${RATIOS[$RATIO_INDEX]}"
RATIO_TAG="${RATIO_TAGS[$RATIO_INDEX]}"
METHOD="${METHODS_LIST[$METHOD_INDEX]}"
TRAINING_SEED="${SEEDS[$SEED_INDEX]}"

export METHODS="$METHOD"
export DATASETS=pathmnist
export RATIOS="$RATIO"
export SELECTION_SEED=42
export TRAINING_SEEDS="$TRAINING_SEED"
export DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export RUN_FULL_TRAIN=0
export AUTO_CONSOLIDATE=0
export RELIABILITY_OUT="$OUT_ROOT"
export SELECTION_OUT="$SELECTION_BASE/benchmark_pathmnist_${METHOD}_ratio${RATIO_TAG}_sel42_train42"
export MEDMNIST_ROOT="$DATA_ROOT"
export MEDMNISTC_ROOT="$CORR_ROOT"
export GRAPH_ROOT="${GRAPH_ROOT:-$REPO_ROOT}"
export MEDC_ROOT="${MEDC_ROOT:-$REPO_ROOT/third_party/medmnistc}"
export GRAPH_CACHE="$OUT_ROOT/cache"

echo "method=$METHOD ratio=$RATIO selection_seed=42 training_seed=$TRAINING_SEED deterministic=1"
"$PYTHON" reliability_medmnistc/scripts/train.py \
  --config reliability_medmnistc/configs/pathmnist_1pct_10pct_random_grapha2_train5_deterministic.yaml
"$PYTHON" reliability_medmnistc/scripts/evaluate.py \
  --config reliability_medmnistc/configs/pathmnist_1pct_10pct_random_grapha2_train5_deterministic.yaml
