#!/bin/bash
# Freeze validation winners. Runs only after all validation workers succeed.

#SBATCH --job-name=v11_freeze
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --qos=YOUR_SLURM_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/project/PROJECT_ROOT/USER/graph_bench/v11/logs/freeze_%j.out
#SBATCH --error=/project/PROJECT_ROOT/USER/graph_bench/v11/logs/freeze_%j.err

set -euo pipefail

PROJECT=${GRAPHCOV_PROJECT_ROOT:-/project/PROJECT_ROOT/USER/graph_bench}
PYTHON=${GRAPHCOV_PYTHON:-${PROJECT}/envs/graphcov-py311/bin/python}
PROTOCOL_DIR=${PROJECT}/v11/outputs/frozen_protocol

cd "${PROJECT}"
mkdir -p "${PROTOCOL_DIR}"

"${PYTHON}" v11/experiments/freeze_validation_winners.py \
  --validation-root "${PROJECT}/v11/outputs/job2_table1_validation_k50baseline_valckpt" \
  --validation-config "${PROJECT}/v11/configs/job2_table1_validation_3seeds.json" \
  --selection-root "${PROJECT}/v11/outputs/job1_table1_calibration_k50baseline" \
  --output-config "${PROTOCOL_DIR}/job2_table1_test.json" \
  --test-output-root "${PROJECT}/v11/outputs/job2_table1_test_valckpt" \
  --required-calibration-seeds 3 \
  --minimum-ba-gain 0.005 \
  --worst-recall-tolerance 0.02

"${PYTHON}" v11/experiments/summarize_downstream.py \
  --input-root "${PROJECT}/v11/outputs/job2_table1_validation_k50baseline_valckpt" \
  --output-dir "${PROTOCOL_DIR}/validation_summary"

echo "frozen_config=${PROTOCOL_DIR}/job2_table1_test.json"
echo "end=$(date --iso-8601=seconds)"
