#!/bin/bash
# One dataset per GPU. This deliberately uses 1 GPU so jobs can pack onto
# fragmented free capacity across the two 8-GPU nodes.
#SBATCH --job-name=realfs1
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${DATASET:?DATASET must be set}"
: "${TRAIN_SEED_OFFSET:=0}"
: "${EPOCHS:=200}"
PROJECT="$HOME/real_training_functional_screen_20260923"
PYTHON=/project/prj-sis01/xuxiaoyu/graph_select/envs/graphcov-py311/bin/python3
GRAPH=/project/prj-sis01/xuxiaoyu/graph_select
EMBED=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_20260922/cache/embeddings_img224_smokefull
DATA=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/data/medmnist
MEDMNIST=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/data/medmnist
DYN=/project/prj-sis01/xuxiaoyu/graph_select/cache
OUT="$PROJECT/results/seed_${TRAIN_SEED_OFFSET}/${DATASET}_${SLURM_JOB_ID}"
mkdir -p "$OUT" "$PROJECT/logs"

echo "start=$(date -Is) dataset=$DATASET seed_offset=$TRAIN_SEED_OFFSET host=$(hostname)"
python_args=(
  --embedding-root "$EMBED" --data-root "$DATA" --medmnist-root "$MEDMNIST"
  --dynamics-root "$DYN" --graphcov-root "$GRAPH" --output "$OUT"
  --datasets "$DATASET" --blocks 2 --budget 25
  --n-random 60 --n-perturb-seeds 2 --train-seed-offset "$TRAIN_SEED_OFFSET"
  --epochs "$EPOCHS" --batch-size 128 --lr 0.1 --weight-decay 5e-4
  --size 224 --num-workers 4 --progress-every 1
)
# Slurm sets CUDA_VISIBLE_DEVICES to the GPU allocated to this one-GPU job.
# Do not overwrite it: several one-GPU jobs intentionally share one node.
"$PYTHON" "$PROJECT/real_training_functional_screen.py" "${python_args[@]}"
echo "finish=$(date -Is)"
