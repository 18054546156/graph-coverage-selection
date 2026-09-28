#!/bin/bash
#SBATCH --job-name=valeval
#SBATCH --partition=gpu-a100
#SBATCH --qos=qos-high-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=hpcgpu108
#SBATCH --output=%x_%j.out
#
# Amendment 2: val BA of every already-trained round-1/2 cell (read-only on the
# run trees; outputs go to $S/val_eval). Resumable: re-submit to continue.
set -uo pipefail
umask 002
S=/mnt/prj01/hgrp-1502-5TB/tdgs_shared
P=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab
PY=$P/envs/medmnistc-py311/bin/python
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$S/code"
echo "host=$(hostname) start=$(date -Is)"
"$PY" val_eval.py --out "$S/val_eval" \
      --roots "$P/tdgs_round1_20260927/formal" "$S/runs/round2_20260927/formal" \
      ${ROOTS_EXTRA:-} ${DATASETS:+--datasets $DATASETS}
echo "end=$(date -Is)"
