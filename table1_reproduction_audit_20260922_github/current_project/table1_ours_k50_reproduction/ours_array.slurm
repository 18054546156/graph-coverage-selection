#!/usr/bin/env bash
# One array task per dataset. Each task runs both ratios and five trials.
# The vendor runner reports final BA as the primary field and retains best-test
# values only as an audit field.

#SBATCH --job-name=table1_ours_k50
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --qos=YOUR_SLURM_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --array=0-4%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/project/PROJECT_ROOT/USER/table1_ours_k50_reproduction/logs/ours_%A_%a.out
#SBATCH --error=/project/PROJECT_ROOT/USER/table1_ours_k50_reproduction/logs/ours_%A_%a.err

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
test -s "${HOME}/.medmnist/${DATASET}_224.npz"
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'

echo "dataset=${DATASET} job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} host=$(hostname) start=$(date --iso-8601=seconds)"
"${PYTHON}" run_ours.py \
  --protocol "${ROOT}/protocol.json" \
  --dataset "${DATASET}" \
  --output "${ROOT}/results/raw/${DATASET}" \
  --cache /project/PROJECT_ROOT/USER/graph_a2_path_seed42_validation/cache
echo "dataset=${DATASET} end=$(date --iso-8601=seconds)"
