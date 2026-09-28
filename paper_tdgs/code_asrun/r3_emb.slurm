#!/bin/bash
#SBATCH --job-name=r3emb
#SBATCH --partition=gpu-a100
#SBATCH --qos=qos-high-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=hpcgpu108
#SBATCH --output=%x_%A_%a.out
#
# Round 3, stage 0: frozen DINOv2-L / CLIP-L embeddings for train+val of the five
# benchmark datasets. One array task per dataset. Inference only; nothing trains,
# the test split is never loaded. Writes into the round-3 tree only.
set -uo pipefail
umask 002
S=/mnt/prj01/hgrp-1502-5TB/tdgs_shared
PY=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/envs/medmnistc-py311/bin/python
export HF_HOME=$S/hf_cache HF_HUB_OFFLINE=1
DSL=(bloodmnist organamnist organsmnist pathmnist tissuemnist)
D=${DSL[${SLURM_ARRAY_TASK_ID:-0}]}
echo "host=$(hostname) ds=$D start=$(date -Is)"
"$PY" "$S/code/extract_emb.py" --datasets "$D" --out "$S/r3/emb" --encoders dinov2 clip
"$PY" "$S/code/extract_emb.py" --datasets "$D" --out "$S/r3/emb" --encoders uni --splits val
echo "done=$(date -Is)"
