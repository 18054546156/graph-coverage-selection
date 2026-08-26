#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${GRAPHCOV_PYTHON:-/root/miniconda3/envs/graphcov_pathmnist/bin/python}"
CONFIG="${1:-v11/configs/job2_derma_atomic_seed42.json}"
NAME="$(basename "${CONFIG}" .json)"
LOG_DIR="${PROJECT_ROOT}/v11/logs"
mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -u v11/experiments/job2_downstream.py \
  --config "${CONFIG}" --num-shards 2 --shard-index 0 \
  2>&1 | tee "${LOG_DIR}/${NAME}_gpu0.log"
