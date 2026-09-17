#!/bin/bash
# PathMNIST 2%: fixed seed-42 selections x five training seeds.
# Task map: 0-4 Random, train seeds 42-46; 5-9 Graph-A2, train seeds 42-46.
# One task uses one GPU, 8 CPUs, and 48G; at most five run concurrently.
#SBATCH --job-name=path-r02-det
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
# gpu-a100 currently enforces a shorter partition walltime than the project QoS.
#SBATCH --time=24:00:00
#SBATCH --array=0-9%5
#SBATCH --output=logs/path_r02_det_%A_%a.out
#SBATCH --error=logs/path_r02_det_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?set REPO_ROOT or submit from the repository}}"
FORMAL_ROOT="${FORMAL_ROOT:?set FORMAL_ROOT}"
OUT_ROOT="${OUT_ROOT:?set OUT_ROOT}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
CORR_ROOT="${CORR_ROOT:?set CORR_ROOT}"

cd "$REPO_ROOT"
export PYTHON="${PYTHON:-/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/envs/medmnistc-py311/bin/python}"
source "$REPO_ROOT/reliability_medmnistc/slurm/common_env.sh"
configure_imagemagick

METHODS_LIST=(random graph_a2)
SEEDS=(42 43 44 45 46)
TASK_ID="${SLURM_ARRAY_TASK_ID:?submit this file as its declared array}"
METHOD="${METHODS_LIST[$((TASK_ID / 5))]}"
TRAINING_SEED="${SEEDS[$((TASK_ID % 5))]}"

export METHODS="$METHOD"
export DATASETS=pathmnist
export RATIOS=0.02
export SELECTION_SEED=42
export TRAINING_SEEDS="$TRAINING_SEED"
export DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export RUN_FULL_TRAIN=0
export AUTO_CONSOLIDATE=0
export RELIABILITY_OUT="$OUT_ROOT"
export SELECTION_OUT="$FORMAL_ROOT/selections"
export MEDMNIST_ROOT="$DATA_ROOT"
export MEDMNISTC_ROOT="$CORR_ROOT"
export GRAPH_ROOT="${GRAPH_ROOT:-$REPO_ROOT}"
export MEDC_ROOT="${MEDC_ROOT:-$REPO_ROOT/third_party/medmnistc}"
export GRAPH_CACHE="$OUT_ROOT/cache"

echo "method=$METHOD ratio=0.02 selection_seed=42 training_seed=$TRAINING_SEED deterministic=1"
"$PYTHON" reliability_medmnistc/scripts/train.py \
  --config reliability_medmnistc/configs/pathmnist_2pct_random_grapha2_train5_deterministic.yaml
"$PYTHON" reliability_medmnistc/scripts/evaluate.py \
  --config reliability_medmnistc/configs/pathmnist_2pct_random_grapha2_train5_deterministic.yaml
