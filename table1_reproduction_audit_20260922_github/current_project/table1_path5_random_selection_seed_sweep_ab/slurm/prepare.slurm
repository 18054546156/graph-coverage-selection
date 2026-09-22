#!/usr/bin/env bash
#SBATCH --job-name=path5_rand_prep
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --qos=qos-normal
#SBATCH --partition=cpu-amd9754
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/project/PROJECT_ROOT/USER/table1_path5_random_selection_seed_sweep_ab/logs/prepare_%j.out
#SBATCH --error=/project/PROJECT_ROOT/USER/table1_path5_random_selection_seed_sweep_ab/logs/prepare_%j.err
set -euo pipefail
ROOT=/project/PROJECT_ROOT/USER/table1_path5_random_selection_seed_sweep_ab
PYTHON=/project/PROJECT_ROOT/USER/graph_select/envs/graphcov-py311/bin/python
export PYTHONNOUSERSITE=1
cd "${ROOT}"
test -x "${PYTHON}"
test -s "${HOME}/.medmnist/pathmnist_224.npz"
"${PYTHON}" prepare_random.py --protocol "${ROOT}/protocol.json" --artifacts "${ROOT}/artifacts"
