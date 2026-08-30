#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 GPU_ID GROUP_NAME METHOD [METHOD ...]" >&2
  exit 2
fi

GPU_ID="$1"
GROUP_NAME="$2"
shift 2
METHODS=("$@")

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${GRAPHCOV_PYTHON:-/root/miniconda3/envs/graphcov_pathmnist/bin/python3.11}"
OUTPUT_ROOT="${RACE_OUTPUT_ROOT:-$ROOT/results}"
LOG_ROOT="$OUTPUT_ROOT/logs"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[race] group=$GROUP_NAME gpu=$GPU_ID methods=${METHODS[*]}"
echo "[race] code=$ROOT commit=$(git rev-parse HEAD)"
echo "[race] output=$OUTPUT_ROOT python=$PYTHON"

for METHOD in "${METHODS[@]}"; do
  METHOD_OUTPUT="$OUTPUT_ROOT/$METHOD"
  METHOD_LOG="$LOG_ROOT/${GROUP_NAME}_${METHOD}.log"
  echo "[race] starting method=$METHOD at $(date --iso-8601=seconds)"

  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" -u -m graphcov.run \
    --datasets tissuemnist \
    --methods "$METHOD" \
    --embeddings uni \
    --ratios 0.02 \
    --seed 42 \
    --trials 2 \
    --global \
    --k-neighbors 50 \
    --k-hops 2 \
    --sparse-cpu \
    --training-paradigm iteration \
    --iterations 1000 \
    --test-interval 1000 \
    --batch-size 256 \
    --lr 0.1 \
    --momentum 0.9 \
    --weight-decay 0.0005 \
    --size 224 \
    --num-workers 4 \
    --output "$METHOD_OUTPUT" \
    2>&1 | tee "$METHOD_LOG"

  "$PYTHON" scripts/validate_method_run.py \
    --output "$METHOD_OUTPUT" \
    --method "$METHOD" \
    --expected-seeds 42 43
  echo "[race] completed method=$METHOD at $(date --iso-8601=seconds)"
done

echo "[race] group=$GROUP_NAME complete"
