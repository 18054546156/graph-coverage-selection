#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${GRAPHCOV_PYTHON:-/root/miniconda3/envs/graphcov_pathmnist/bin/python}"
INPUT_ROOT="${1:?usage: summarize_job2.sh INPUT_ROOT [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-${INPUT_ROOT}}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" v11/experiments/summarize_downstream.py \
  --input-root "${INPUT_ROOT}" --output-dir "${OUTPUT_DIR}"
