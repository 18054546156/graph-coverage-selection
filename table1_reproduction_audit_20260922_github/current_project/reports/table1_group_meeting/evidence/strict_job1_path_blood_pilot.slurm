#!/bin/bash
# Job 1 (Path+Blood pilot): one dataset per array task, all eight methods, both ratios, seed42 only.

#SBATCH --job-name=pilot_select_pb
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --qos=YOUR_SLURM_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/project/PROJECT_ROOT/USER/table1_strict_repro/table1_reproduction/logs/job1_pilot_%A_%a.out
#SBATCH --error=/project/PROJECT_ROOT/USER/table1_strict_repro/table1_reproduction/logs/job1_pilot_%A_%a.err

set -euo pipefail

PROJECT=${GRAPHCOV_PROJECT_ROOT:-/project/PROJECT_ROOT/USER/table1_strict_repro}
PYTHON=${GRAPHCOV_PYTHON:-${PROJECT}/envs/graphcov-py311/bin/python}
CONFIG=${PROJECT}/table1_reproduction/configs/job1_path_blood_pilot.json
DATASETS=(pathmnist bloodmnist)
DATASET=${DATASETS[${SLURM_ARRAY_TASK_ID}]}

export PYTHONNOUSERSITE=1
export GRAPHCOV_USE_FAISS_GPU=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "${PROJECT}"
test -x "${PYTHON}"
test -s "${HOME}/.medmnist/${DATASET}_224.npz"

echo "job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} dataset=${DATASET}"
echo "host=$(hostname) start=$(date --iso-8601=seconds)"
"${PYTHON}" -c "import faiss, torch; assert torch.cuda.is_available(); assert faiss.get_num_gpus() == 1; print(torch.__version__, torch.version.cuda, faiss.__version__, torch.cuda.get_device_name(0))"

"${PYTHON}" table1_reproduction/experiments/job1_select.py \
  --config "${CONFIG}" \
  --project-root "${PROJECT}" \
  --only-dataset "${DATASET}"

echo "end=$(date --iso-8601=seconds)"
