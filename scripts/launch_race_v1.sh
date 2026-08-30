#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${RACE_OUTPUT_ROOT:-/root/graphcov_pathmnist_sota/results/sota_race_v1}"
GPU0_SESSION="sota_race_v1_gpu0"
GPU1_SESSION="sota_race_v1_gpu1"

for SESSION in "$GPU0_SESSION" "$GPU1_SESSION"; do
  if screen -list | grep -q "[.]$SESSION"; then
    echo "screen session already exists: $SESSION" >&2
    exit 1
  fi
done

mapfile -t GPU_MEMORY < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [[ ${#GPU_MEMORY[@]} -lt 2 ]]; then
  echo "two GPUs are required" >&2
  exit 1
fi
for INDEX in 0 1; do
  if (( GPU_MEMORY[INDEX] >= 500 )); then
    echo "GPU $INDEX is busy: ${GPU_MEMORY[INDEX]} MiB" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT/logs"

screen -dmS "$GPU0_SESSION" bash -lc \
  "cd '$ROOT' && RACE_OUTPUT_ROOT='$OUTPUT_ROOT' bash scripts/run_race_group.sh 0 gpu0 graph_a2 graph_a2_dec graph_a2_damp50 graph_a2_marginal"

screen -dmS "$GPU1_SESSION" bash -lc \
  "cd '$ROOT' && RACE_OUTPUT_ROOT='$OUTPUT_ROOT' bash scripts/run_race_group.sh 1 gpu1 graph_a2_sqrt graph_a2_dec_sqrt graph_a2_dec_marginal graph_a2_damp25 graph_a2_damp75"

echo "launched $GPU0_SESSION and $GPU1_SESSION"
echo "results: $OUTPUT_ROOT"
