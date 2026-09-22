#!/bin/bash
# Complete one method/seed pair of the twelve missing Blood public-k10 runs.
# The submit command supplies METHOD and TRAINING_SEED as positional arguments.

#SBATCH --job-name=table1_k10_gap
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --qos=YOUR_SLURM_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=/project/PROJECT_ROOT/USER/_archive_fork_20260830/graph_bench/table1_reproduction/logs/missing_k10_%j.out
#SBATCH --error=/project/PROJECT_ROOT/USER/_archive_fork_20260830/graph_bench/table1_reproduction/logs/missing_k10_%j.err

set -euo pipefail

PROJECT=/project/PROJECT_ROOT/USER/_archive_fork_20260830/graph_bench
PYTHON=${GRAPHCOV_PYTHON:-${PROJECT}/envs/graphcov-py311/bin/python}
CONFIG=${PROJECT}/table1_reproduction/missing_k10_completion/public_k10_job2_config.json
METHOD=${1:?METHOD must be supplied as the first sbatch argument}
TRAINING_SEED=${2:?TRAINING_SEED must be supplied as the second sbatch argument}

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "${PROJECT}"
test -x "${PYTHON}"
test -s "${HOME}/.medmnist/bloodmnist_224.npz"
echo "job=${SLURM_JOB_ID} method=${METHOD} seed=${TRAINING_SEED}"
echo "host=$(hostname) start=$(date --iso-8601=seconds)"
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
"${PYTHON}" table1_reproduction/missing_k10_completion/run_missing.py \
  --config "${CONFIG}" \
  --project-root "${PROJECT}" \
  --method "${METHOD}" \
  --training-seed "${TRAINING_SEED}"
echo "end=$(date --iso-8601=seconds)"
