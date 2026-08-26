#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${GRAPHCOV_PYTHON:-/root/miniconda3/envs/graphcov_pathmnist/bin/python}"
CONFIG="${1:-table1_reproduction/configs/job1_table1.json}"
NAME="$(basename "${CONFIG}" .json)"
LOG_DIR="${PROJECT_ROOT}/table1_reproduction/logs"
mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" table1_reproduction/experiments/preflight.py \
  --config "${CONFIG}"
"${PYTHON_BIN}" -u table1_reproduction/experiments/job1_select.py \
  --config "${CONFIG}" 2>&1 | tee "${LOG_DIR}/${NAME}.log"
