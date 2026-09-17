#!/bin/bash
# PathMNIST cross-factor pilot: Random/Graph-A2 x 2 ratios x 2 selection seeds.
# Submit in two batches of eight tasks because the account MaxSubmit is 15.
# BATCH_OFFSET=0 runs training_seed=42; BATCH_OFFSET=8 runs training_seed=43.
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-7%5
#SBATCH --output=logs/cross_path_%A_%a.out
#SBATCH --error=logs/cross_path_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?set REPO_ROOT or submit from the repository}}"
FORMAL_ROOT="${FORMAL_ROOT:?set FORMAL_ROOT}"
VARIANCE_ROOT="${VARIANCE_ROOT:?set VARIANCE_ROOT}"
OUT_ROOT="${OUT_ROOT:?set OUT_ROOT}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
CORR_ROOT="${CORR_ROOT:?set CORR_ROOT}"
BATCH_OFFSET="${BATCH_OFFSET:-0}"

cd "$REPO_ROOT"
export PYTHON="${PYTHON:-/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/envs/medmnistc-py311/bin/python}"
source "$REPO_ROOT/reliability_medmnistc/slurm/common_env.sh"
configure_imagemagick

TASK_ID=$((SLURM_ARRAY_TASK_ID + BATCH_OFFSET))
METHOD_LIST=(random graph_a2)
RATIO_LIST=(0.02 0.05)
SELECTION_SEED_LIST=(42 43)
TRAINING_SEED_LIST=(42 43)

METHOD="${METHOD_LIST[$((TASK_ID % 2))]}"
RATIO="${RATIO_LIST[$(((TASK_ID / 2) % 2))]}"
SELECTION_SEED="${SELECTION_SEED_LIST[$(((TASK_ID / 4) % 2))]}"
TRAINING_SEED="${TRAINING_SEED_LIST[$((TASK_ID / 8))]}"

if [[ "$SELECTION_SEED" == "42" ]]; then
  SELECTION_ROOT="$FORMAL_ROOT/selections"
else
  SELECTION_ROOT="$VARIANCE_ROOT/selections"
fi

export METHODS="$METHOD"
export DATASETS=pathmnist
export RATIOS="$RATIO"
export SELECTION_SEED
export TRAINING_SEEDS="$TRAINING_SEED"
export RUN_FULL_TRAIN=0
export AUTO_CONSOLIDATE=0
export RELIABILITY_OUT="$OUT_ROOT"
export SELECTION_OUT="$SELECTION_ROOT"
export MEDMNIST_ROOT="$DATA_ROOT"
export MEDMNISTC_ROOT="$CORR_ROOT"
export GRAPH_CACHE="$OUT_ROOT/cache"

echo "task=$TASK_ID method=$METHOD ratio=$RATIO selection_seed=$SELECTION_SEED training_seed=$TRAINING_SEED"
"$PYTHON" reliability_medmnistc/scripts/train.py \
  --config reliability_medmnistc/configs/full_5datasets_8methods.yaml
"$PYTHON" reliability_medmnistc/scripts/evaluate.py \
  --config reliability_medmnistc/configs/full_5datasets_8methods.yaml
