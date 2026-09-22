#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${GRAPHCOV_PYTHON:-/root/miniconda3/envs/graphcov_pathmnist/bin/python}"
CONFIG="${1:-v11/configs/job1_derma_atomic.json}"
NAME="$(basename "${CONFIG}" .json)"
LOG_DIR="${PROJECT_ROOT}/v11/logs"
mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -u v11/experiments/job1_select.py \
  --config "${CONFIG}" \
  2>&1 | tee "${LOG_DIR}/${NAME}.log"
