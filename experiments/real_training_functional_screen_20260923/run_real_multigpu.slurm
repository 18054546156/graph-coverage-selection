#!/bin/bash
# N independent single-GPU workers inside ONE slurm job.
#
# Why: qos-high-gpu caps a user at 5 running JOBS but 12 GPUs. One dataset per
# job at --gres=gpu:1 therefore used 5 of the 12 GPUs we are entitled to and left
# 7 idle. Packing 3 workers per job turns the same 4-5 job slots into 12 GPUs.
#
# Each worker is a plain single-GPU run of real_training_functional_screen.py,
# pinned with CUDA_VISIBLE_DEVICES and given its own --output directory (the
# driver opens "<output>/<dataset>.jsonl" with mode "w", so sharing an output
# directory between workers would truncate).
#
# Submit with:
#   sbatch --gres=gpu:3 --cpus-per-task=18 --mem=60G \
#     --partition=gpu-a100 --job-name=rfs-mg-a \
#     run_real_multigpu.slurm bloodmnist,0,2 pathmnist,0,3 tissuemnist,0,3
#
# Work items are POSITIONAL ARGUMENTS, one "dataset,shard_index,shard_count" per
# allocated GPU. They must NOT be passed through --export: sbatch splits an
# --export value on commas, so --export=ALL,WORK="bloodmnist,0,2 pathmnist,0,3"
# silently sets WORK=bloodmnist and invents variables named "0" and "2 pathmnist".
#
# Shards are STRIDED (index % count == shard), not sliced, so workers stay
# balanced even though already-measured indices are near-free skips.
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
[ "$#" -gt 0 ] || { echo "usage: sbatch ... $0 <dataset,shard,count> ..." >&2; exit 2; }
: "${TRAIN_SEED_OFFSET:=100000}"
: "${EPOCHS:=200}"
: "${GLOBAL_START:=0}"
: "${GLOBAL_END:=600}"

PROJECT="$HOME/real_training_functional_screen_20260923"
PYTHON=/project/prj-sis01/xuxiaoyu/graph_select/envs/graphcov-py311/bin/python3
GRAPH=/project/prj-sis01/xuxiaoyu/graph_select
EMBED=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_20260922/cache/embeddings_img224_smokefull
DATA=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/data/medmnist
MEDMNIST=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/data/medmnist
DYN=/project/prj-sis01/xuxiaoyu/graph_select/cache
mkdir -p "$PROJECT/logs"

cd "$PROJECT"
echo "start=$(date -Is) host=$(hostname) job=$SLURM_JOB_ID"
echo "work=$*"

# Map worker i onto the i-th GPU SLURM actually gave us, rather than onto
# physical device i. nvidia-smi inside these jobs lists every GPU on the node,
# so this cluster is not isolating devices by cgroup: hardcoding
# CUDA_VISIBLE_DEVICES=0,1,2 would happily run on another user's GPUs whenever
# SLURM allocated us anything other than the first three.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -ra DEVS <<< "$CUDA_VISIBLE_DEVICES"
else
  echo "WARNING: SLURM set no CUDA_VISIBLE_DEVICES; falling back to 0..n-1" >&2
  DEVS=(); for ((d = 0; d < $#; d++)); do DEVS+=("$d"); done
fi
echo "allocated gpus=${DEVS[*]}  workers=$#"
[ "${#DEVS[@]}" -ge "$#" ] || {
  echo "only ${#DEVS[@]} GPUs for $# work items" >&2; exit 2; }

gpu=0
pids=()
names=()
for item in "$@"; do
  ds="${item%%,*}"; rest="${item#*,}"
  sidx="${rest%%,*}"; scnt="${rest##*,}"
  [ "$ds,$sidx,$scnt" = "$item" ] || {
    echo "malformed work item: $item" >&2; exit 2; }
  out="$PROJECT/results/seed_${TRAIN_SEED_OFFSET}/mg_${SLURM_JOB_ID}_g${gpu}_${ds}_s${sidx}of${scnt}"
  mkdir -p "$out"
  echo "  worker$gpu -> physical gpu ${DEVS[$gpu]} : $ds shard $sidx/$scnt"

  CUDA_VISIBLE_DEVICES="${DEVS[$gpu]}" "$PYTHON" "$PROJECT/real_training_functional_screen.py" \
    --embedding-root "$EMBED" --data-root "$DATA" --medmnist-root "$MEDMNIST" \
    --dynamics-root "$DYN" --graphcov-root "$GRAPH" --output "$out" \
    --datasets "$ds" --blocks 2 --budget 25 \
    --n-random 60 --n-perturb-seeds 3 --train-seed-offset "$TRAIN_SEED_OFFSET" \
    --global-start "$GLOBAL_START" --global-end "$GLOBAL_END" \
    --shard-index "$sidx" --shard-count "$scnt" \
    --skip-existing "$PROJECT/results" \
    --epochs "$EPOCHS" --batch-size 128 --lr 0.1 --weight-decay 5e-4 \
    --size 224 --num-workers 4 --progress-every 5 \
    > "$PROJECT/logs/w-${SLURM_JOB_ID}-g${gpu}-${ds}-s${sidx}.out" 2>&1 &

  pids+=($!)
  names+=("$ds:$sidx/$scnt")
  gpu=$((gpu + 1))
done

# Wait on each worker individually so one crash is reported with its identity
# instead of collapsing into a single non-zero exit from `wait`.
rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "ok   ${names[$i]}"
  else
    echo "FAIL ${names[$i]} (see logs/w-${SLURM_JOB_ID}-g${i}-*.out)" >&2
    rc=1
  fi
done
echo "finish=$(date -Is) rc=$rc"
exit "$rc"
