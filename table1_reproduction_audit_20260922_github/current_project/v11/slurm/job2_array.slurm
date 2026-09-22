#!/bin/bash
# Generic v11 Job 2 worker. Override V11_CONFIG for seed-42 or frozen test runs.

#SBATCH --job-name=v11_train
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --qos=YOUR_SLURM_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --array=0-9%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=/project/PROJECT_ROOT/USER/graph_bench/v11/logs/job2_%A_%a.out
#SBATCH --error=/project/PROJECT_ROOT/USER/graph_bench/v11/logs/job2_%A_%a.err

set -euo pipefail

PROJECT=${GRAPHCOV_PROJECT_ROOT:-/project/PROJECT_ROOT/USER/graph_bench}
PYTHON=${GRAPHCOV_PYTHON:-${PROJECT}/envs/graphcov-py311/bin/python}
CONFIG=${V11_CONFIG:-${PROJECT}/v11/configs/job2_table1_validation_3seeds.json}
NUM_SHARDS=10

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "${PROJECT}"
test -x "${PYTHON}"
test -s "${CONFIG}"

echo "job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} shard=${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}"
echo "config=${CONFIG} host=$(hostname) start=$(date --iso-8601=seconds)"
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'

"${PYTHON}" v11/experiments/job2_downstream.py \
  --config "${CONFIG}" \
  --project-root "${PROJECT}" \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --num-shards "${NUM_SHARDS}"

echo "end=$(date --iso-8601=seconds)"
