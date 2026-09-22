#!/usr/bin/env bash
#SBATCH --job-name=table1_ours_finalize
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --qos=YOUR_SLURM_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/project/PROJECT_ROOT/USER/table1_ours_k50_reproduction/logs/finalize_%j.out
#SBATCH --error=/project/PROJECT_ROOT/USER/table1_ours_k50_reproduction/logs/finalize_%j.err

set -euo pipefail
ROOT=/project/PROJECT_ROOT/USER/table1_ours_k50_reproduction
PYTHON=/project/PROJECT_ROOT/USER/graph_select/envs/graphcov-py311/bin/python
cd "${ROOT}"
"${PYTHON}" summarize.py --root "${ROOT}"
"${PYTHON}" build_notebook.py
