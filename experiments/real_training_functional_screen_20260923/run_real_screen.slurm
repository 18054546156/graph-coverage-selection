#!/bin/bash
# Three identical jobs are submitted under three accounts with different
# train-seed offsets. Each job uses five GPUs, one dataset process per GPU.
# The endpoint is official-test BA from real ResNet-18 training, equal weights.
#SBATCH --job-name=realfs
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=30
#SBATCH --mem=60G
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
PROJECT="$HOME/real_training_functional_screen_20260923"
PYTHON=/project/prj-sis01/xuxiaoyu/graph_select/envs/graphcov-py311/bin/python3
GRAPH=/project/prj-sis01/xuxiaoyu/graph_select
EMBED=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_20260922/cache/embeddings_img224_smokefull
DATA=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/data/medmnist
DYN=/project/prj-sis01/xuxiaoyu/graph_select/cache
MEDMNIST=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/data/medmnist
: "${TRAIN_SEED_OFFSET:=0}"
: "${EPOCHS:=200}"
: "${N_RANDOM:=60}"
: "${N_PERTURB_SEEDS:=3}"

mkdir -p "$PROJECT/results" "$PROJECT/logs"
OUT="$PROJECT/results/run_$SLURM_JOB_ID"
mkdir -p "$OUT"
DATASETS=(bloodmnist organamnist organsmnist pathmnist tissuemnist)
pids=()
for i in "${!DATASETS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i "$PYTHON" "$PROJECT/real_training_functional_screen.py" \
    --embedding-root "$EMBED" --data-root "$DATA" --dynamics-root "$DYN" \
    --graphcov-root "$GRAPH" --medmnist-root "$MEDMNIST" \
    --output "$OUT/${DATASETS[$i]}" \
    --datasets "${DATASETS[$i]}" --budget 25 --blocks 2 \
    --n-random "$N_RANDOM" --n-perturb-seeds "$N_PERTURB_SEEDS" \
    --train-seed-offset "$TRAIN_SEED_OFFSET" --epochs "$EPOCHS" \
    --batch-size 128 --lr 0.1 --weight-decay 5e-4 --size 224 \
    --num-workers 4 \
    > "$PROJECT/logs/${DATASETS[$i]}_${SLURM_JOB_ID}.log" 2>&1 &
  pids+=("$!")
  echo "launched ${DATASETS[$i]} on gpu $i pid ${pids[-1]}"
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || { echo "pid $pid FAILED"; fail=1; }
done
echo "finish=$(date -Is) fail=$fail"
exit "$fail"
