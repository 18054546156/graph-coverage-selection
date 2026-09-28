#!/bin/bash
#SBATCH --job-name=r3sel
#SBATCH --partition=gpu-a100
#SBATCH --qos=qos-high-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --exclude=hpcgpu108
#SBATCH --output=%x_%A_%a.out
#
# Round 3 selection staging (probe-free, no training, test split never loaded).
# One array task per dataset; tissuemnist needs the full 240G because three
# k=50 A+A^2 kernels (~615M nnz each on UNI) are resident at once.
set -uo pipefail
umask 002
S=/mnt/prj01/hgrp-1502-5TB/tdgs_shared
PY=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/envs/medmnistc-py311/bin/python
DSL=(bloodmnist organamnist organsmnist pathmnist tissuemnist)
D=${DSL[${SLURM_ARRAY_TASK_ID:-0}]}
R=${RATIO:-0.02}
cd "$S/code"
mkdir -p "$S/r3/sel" "$S/r3/report"
echo "host=$(hostname) ds=$D ratio=$R start=$(date -Is)"
# embeddings may be extracted by another account's job: wait for the last file
# the extractor writes for this dataset (atomic rename, so existence == complete)
LAST=$S/r3/emb/${D}_val_uni_224.npz
w=0; while [[ ! -f $LAST && $w -lt 360 ]]; do sleep 60; w=$((w + 1)); done
[[ -f $LAST ]] || { echo "FAILED $D: embeddings never appeared"; exit 3; }
"$PY" mv_select.py --dataset "$D" --ratio "$R" \
      --out-dir "$S/r3/sel" --report "$S/r3/report" \
      ${ARMS:+--arms $ARMS} ${TAG:+--tag $TAG} \
  || echo "FAILED $D r$R"
echo "done=$(date -Is)"
