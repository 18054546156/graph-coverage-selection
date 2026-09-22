#!/bin/bash
# Table 1 Job 2: 400 downstream runs split across ten resumable workers.
# Slurm runs at most five workers concurrently.

#SBATCH --job-name=table1_train
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --qos=YOUR_SLURM_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --array=0-9%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=/project/PROJECT_ROOT/USER/graph_bench/table1_reproduction/logs/job2_%A_%a.out
#SBATCH --error=/project/PROJECT_ROOT/USER/graph_bench/table1_reproduction/logs/job2_%A_%a.err

set -euo pipefail

PROJECT=${GRAPHCOV_PROJECT_ROOT:-/project/PROJECT_ROOT/USER/graph_bench}
PYTHON=${GRAPHCOV_PYTHON:-${PROJECT}/envs/graphcov-py311/bin/python}
CONFIG=${PROJECT}/table1_reproduction/configs/job2_table1.json
NUM_SHARDS=10

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "${PROJECT}"
test -x "${PYTHON}"

echo "job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} shard=${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}"
echo "host=$(hostname) start=$(date --iso-8601=seconds)"
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'

"${PYTHON}" table1_reproduction/experiments/job2_downstream.py \
  --config "${CONFIG}" \
  --project-root "${PROJECT}" \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --num-shards "${NUM_SHARDS}"

echo "end=$(date --iso-8601=seconds)"
