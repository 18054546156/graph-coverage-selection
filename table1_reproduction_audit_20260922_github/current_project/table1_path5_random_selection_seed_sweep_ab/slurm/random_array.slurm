#!/usr/bin/env bash
#SBATCH --job-name=path5_rand_sweep
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --qos=YOUR_SLURM_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --array=0-4%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/project/PROJECT_ROOT/USER/table1_path5_random_selection_seed_sweep_ab/logs/random_%A_%a.out
#SBATCH --error=/project/PROJECT_ROOT/USER/table1_path5_random_selection_seed_sweep_ab/logs/random_%A_%a.err
set -euo pipefail
ROOT=/project/PROJECT_ROOT/USER/table1_path5_random_selection_seed_sweep_ab
PYTHON=/project/PROJECT_ROOT/USER/graph_select/envs/graphcov-py311/bin/python
SEEDS=(42 43 44 45 46)
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID}]}
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cd "${ROOT}"
test -x "${PYTHON}"
test -s "${HOME}/.medmnist/pathmnist_224.npz"
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
echo "selection_seed=${SEED} training_seed=42 job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} host=$(hostname) start=$(date --iso-8601=seconds)"
"${PYTHON}" train_random.py --protocol "${ROOT}/protocol.json" --artifacts "${ROOT}/artifacts" --selection-seed "${SEED}" --output "${ROOT}/results"
echo "selection_seed=${SEED} end=$(date --iso-8601=seconds)"
