#!/usr/bin/env bash
set -euo pipefail

group="${1:?usage: run_table1_single_seed.sh geometry|baselines}"
python_bin="/root/miniconda3/envs/graphcov_pathmnist/bin/python3.11"
project_dir="/root/graphcov_pathmnist_sota/reproduction-official-code"
cache_dir="/root/graphcov_pathmnist_sota/cache/embeddings"
result_root="/root/graphcov_pathmnist_sota/results/table1_single_seed_artifacts"

cd "$project_dir"
mkdir -p "$result_root/logs"

common_args=(
  --datasets pathmnist bloodmnist
  --embeddings uni
  --ratios 0.02 0.05
  --trials 1 --seed 42
  --k-neighbors 50
  --training-paradigm epoch --epochs 1000
  --test-every-n-epochs 10
  --batch-size 256 --lr 0.1 --momentum 0.9
  --weight-decay 0.0005 --size 224 --num-workers 4
  --cache "$cache_dir"
  -v
)

case "$group" in
  geometry)
    "$python_bin" -u -m graphcov.run \
      --methods random fps herding graph_a2 \
      --global --sparse-cpu \
      --output "$result_root/geometry" \
      "${common_args[@]}" 2>&1 | tee "$result_root/logs/geometry.log"
    ;;
  baselines)
    "$python_bin" -u -m graphcov.run \
      --methods facility el2n_top forgetting eva \
      --eva-epochs 200 --eva-window-size 10 --eva-size 28 \
      --output "$result_root/baselines" \
      "${common_args[@]}" 2>&1 | tee "$result_root/logs/baselines.log"
    ;;
  *)
    echo "unknown group: $group" >&2
    exit 2
    ;;
esac
