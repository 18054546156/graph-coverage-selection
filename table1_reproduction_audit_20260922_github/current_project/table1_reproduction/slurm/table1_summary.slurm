#!/usr/bin/env bash
#SBATCH --job-name=table1_summary
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --qos=qos-normal
#SBATCH --partition=cpu-amd9754
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/project/PROJECT_ROOT/USER/graph_bench/table1_reproduction/logs/summary_%j.out
#SBATCH --error=/project/PROJECT_ROOT/USER/graph_bench/table1_reproduction/logs/summary_%j.err

set -euo pipefail

PROJECT=/project/PROJECT_ROOT/USER/graph_bench
PYTHON=/project/PROJECT_ROOT/USER/graph_select/envs/graphcov-py311/bin/python
INPUT_ROOT=${1:-${PROJECT}/table1_reproduction/outputs/job2_table1_k50_global_valckpt}
OUTPUT_DIR=${2:-${INPUT_ROOT}/summary}

cd "${PROJECT}"
test -x "${PYTHON}"
"${PYTHON}" table1_reproduction/experiments/summarize.py \
  --input-root "${INPUT_ROOT}" \
  --output-dir "${OUTPUT_DIR}"
