#!/bin/bash
# Graph-A2 seed-42 downstream test: one dataset and two ratios per array task.

#SBATCH --job-name=a2_s42_test
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-rtx4090
#SBATCH --array=0-4%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=/project/prj-sis01/xuxiaoyu/graph_bench/table1_reproduction/logs/a2_s42_test_%A_%a.out
#SBATCH --error=/project/prj-sis01/xuxiaoyu/graph_bench/table1_reproduction/logs/a2_s42_test_%A_%a.err

set -euo pipefail

PROJECT=${GRAPHCOV_PROJECT_ROOT:-/project/prj-sis01/xuxiaoyu/graph_bench}
PYTHON=${GRAPHCOV_PYTHON:-/project/prj-sis01/xuxiaoyu/graph_select/envs/graphcov-py311/bin/python}
CONFIG=${PROJECT}/table1_reproduction/configs/job2_graph_a2_seed42.json
DATASETS=(organsmnist organamnist pathmnist tissuemnist bloodmnist)
DATASET=${DATASETS[${SLURM_ARRAY_TASK_ID}]}

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "${PROJECT}"
test -x "${PYTHON}"
test -s "${CONFIG}"

echo "job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} dataset=${DATASET} method=graph_a2 seed=42"
echo "host=$(hostname) start=$(date --iso-8601=seconds)"
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'

"${PYTHON}" table1_reproduction/experiments/job2_downstream.py \
  --config "${CONFIG}" \
  --project-root "${PROJECT}" \
  --only-dataset "${DATASET}"

echo "end=$(date --iso-8601=seconds)"
