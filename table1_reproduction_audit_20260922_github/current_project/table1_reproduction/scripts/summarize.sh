#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${GRAPHCOV_PYTHON:-/project/PROJECT_ROOT/USER/graph_select/envs/graphcov-py311/bin/python}"
INPUT_ROOT="${1:-table1_reproduction/outputs/job2_table1_k50_global_valckpt}"
OUTPUT_DIR="${2:-${INPUT_ROOT}/summary}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" table1_reproduction/experiments/summarize.py \
  --input-root "${INPUT_ROOT}" --output-dir "${OUTPUT_DIR}"
