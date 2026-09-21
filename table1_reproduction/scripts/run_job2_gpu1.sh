#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${GRAPHCOV_PYTHON:-/project/prj-sis01/xuxiaoyu/graph_select/envs/graphcov-py311/bin/python}"
CONFIG="${1:-table1_reproduction/configs/job2_table1.json}"
NAME="$(basename "${CONFIG}" .json)"
LOG_DIR="${PROJECT_ROOT}/table1_reproduction/logs"
mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}"
CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" -u table1_reproduction/experiments/job2_downstream.py \
  --config "${CONFIG}" --num-shards 2 --shard-index 1 \
  2>&1 | tee "${LOG_DIR}/${NAME}_gpu1.log"
