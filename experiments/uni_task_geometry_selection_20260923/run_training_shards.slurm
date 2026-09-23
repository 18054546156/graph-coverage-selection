#!/bin/bash
# Real-training factorial for the selection x weighting interaction.
# One job holds 6 GPUs and runs 6 shards concurrently, one process per GPU.
# Submit twice with SHARD_BASE=0 and SHARD_BASE=6 (N_SHARDS=12 total), because
# MaxJobsPU=5 and three slots are held by an unrelated Table-1 run.
#SBATCH --job-name=wint
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
PROJECT="$HOME/uni_task_geometry_selection_20260923"
PYTHON=/project/prj-sis01/xuxiaoyu/graph_select/envs/graphcov-py311/bin/python3
GRAPH=/project/prj-sis01/xuxiaoyu/graph_select
: "${SHARD_BASE:=0}"
: "${N_SHARDS:=12}"
: "${LOCAL:=6}"
OUT="$PROJECT/results/training_$SLURM_JOB_ID"
mkdir -p "$OUT" "$PROJECT/logs"

echo "start=$(date -Is) job=$SLURM_JOB_ID host=$(hostname) shards=${SHARD_BASE}..$((SHARD_BASE+LOCAL-1))/$N_SHARDS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

pids=()
for i in $(seq 0 $((LOCAL-1))); do
  SHARD=$((SHARD_BASE + i))
  CUDA_VISIBLE_DEVICES=$i "$PYTHON" "$PROJECT/train_weighted.py" \
    --cells "$PROJECT/cells/training_cells.jsonl" \
    --output "$OUT/shard_${SHARD}.jsonl" \
    --shard "$SHARD" --n-shards "$N_SHARDS" \
    --epochs 200 --batch-size 128 --lr 0.1 --size 224 \
    --num-workers 6 --graphcov-root "$GRAPH" \
    > "$PROJECT/logs/shard_${SHARD}_${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!)
  echo "launched shard $SHARD on local gpu $i pid ${pids[-1]}"
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || { echo "shard pid $pid FAILED"; fail=1; }
done
echo "finish=$(date -Is) fail=$fail"
exit $fail
