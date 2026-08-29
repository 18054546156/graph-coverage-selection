#!/usr/bin/env bash
set -euo pipefail

group="${1:?usage: wait_for_gpu_and_run.sh geometry|baselines GPU_INDEX}"
gpu_index="${2:?usage: wait_for_gpu_and_run.sh geometry|baselines GPU_INDEX}"
project_dir="/root/graphcov_pathmnist_sota/reproduction-official-code"
result_root="/root/graphcov_pathmnist_sota/results/table1_single_seed_artifacts"

mkdir -p "$result_root/logs"
queue_log="$result_root/logs/${group}_queue.log"

while true; do
  used_memory="$({ nvidia-smi --id="$gpu_index" \
    --query-compute-apps=used_memory --format=csv,noheader,nounits || true; } \
    | awk '{sum += $1} END {print sum + 0}')"
  if (( used_memory < 500 )); then
    break
  fi
  printf '%s waiting: GPU %s uses %s MiB\n' "$(date -Is)" "$gpu_index" "$used_memory" \
    | tee -a "$queue_log"
  sleep 60
done

printf '%s starting %s on GPU %s\n' "$(date -Is)" "$group" "$gpu_index" | tee -a "$queue_log"
cd "$project_dir"
export CUDA_VISIBLE_DEVICES="$gpu_index"
exec bash scripts/run_table1_single_seed.sh "$group"
