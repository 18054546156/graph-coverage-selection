#!/bin/bash
# v11 Job 1: one dataset per array task, train-only selection and diagnostics.

#SBATCH --job-name=v11_select
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --qos=YOUR_SLURM_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --array=0-4%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=/project/PROJECT_ROOT/USER/graph_bench/v11/logs/job1_%A_%a.out
#SBATCH --error=/project/PROJECT_ROOT/USER/graph_bench/v11/logs/job1_%A_%a.err

set -euo pipefail

PROJECT=${GRAPHCOV_PROJECT_ROOT:-/project/PROJECT_ROOT/USER/graph_bench}
PYTHON=${GRAPHCOV_PYTHON:-${PROJECT}/envs/graphcov-py311/bin/python}
CONFIG=${PROJECT}/v11/configs/job1_table1_calibration.json
DATASETS=(organsmnist organamnist pathmnist tissuemnist bloodmnist)
DATASET=${DATASETS[${SLURM_ARRAY_TASK_ID}]}

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "${PROJECT}"
test -x "${PYTHON}"

echo "job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} dataset=${DATASET}"
echo "host=$(hostname) start=$(date --iso-8601=seconds)"
"${PYTHON}" -c 'import faiss, torch; assert torch.cuda.is_available(); assert faiss.get_num_gpus() == 1; print(torch.__version__, torch.version.cuda, faiss.__version__, torch.cuda.get_device_name(0))'

"${PYTHON}" v11/experiments/job1_select.py \
  --config "${CONFIG}" \
  --project-root "${PROJECT}" \
  --only-dataset "${DATASET}"

echo "end=$(date --iso-8601=seconds)"
