#!/usr/bin/env bash
#SBATCH --job-name=path5_fixed_sum
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --qos=qos-normal
#SBATCH --partition=cpu-amd9754
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/project/PROJECT_ROOT/USER/table1_path5_fixed_random_paired_ab/logs/summary_%j.out
#SBATCH --error=/project/PROJECT_ROOT/USER/table1_path5_fixed_random_paired_ab/logs/summary_%j.err

set -euo pipefail

ROOT=/project/PROJECT_ROOT/USER/table1_path5_fixed_random_paired_ab
PYTHON=/project/PROJECT_ROOT/USER/graph_select/envs/graphcov-py311/bin/python

cd "${ROOT}"
"${PYTHON}" aggregate.py \
  --protocol "${ROOT}/protocol.json" \
  --results "${ROOT}/results" \
  --output "${ROOT}/reports"
