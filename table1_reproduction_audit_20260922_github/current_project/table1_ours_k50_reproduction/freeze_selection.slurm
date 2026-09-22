#!/usr/bin/env bash
# Freeze selections only; no downstream training and no validation/test access.

#SBATCH --job-name=table1_ours_select
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --qos=YOUR_SLURM_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --array=0-4%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/project/PROJECT_ROOT/USER/table1_ours_k50_reproduction/logs/select_%A_%a.out
#SBATCH --error=/project/PROJECT_ROOT/USER/table1_ours_k50_reproduction/logs/select_%A_%a.err

set -euo pipefail
ROOT=/project/PROJECT_ROOT/USER/table1_ours_k50_reproduction
PYTHON=/project/PROJECT_ROOT/USER/graph_select/envs/graphcov-py311/bin/python
DATASETS=(organsmnist organamnist pathmnist tissuemnist bloodmnist)
DATASET=${DATASETS[${SLURM_ARRAY_TASK_ID}]}
export PYTHONNOUSERSITE=1
export GRAPHCOV_USE_FAISS_GPU=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cd "${ROOT}"
test -x "${PYTHON}"
"${PYTHON}" freeze_selection.py \
  --protocol "${ROOT}/protocol.json" \
  --dataset "${DATASET}" \
  --output "${ROOT}/selection" \
  --cache /project/PROJECT_ROOT/USER/graph_a2_path_seed42_validation/cache
