#!/bin/bash
#SBATCH --job-name=corrS1

#SBATCH --qos=qos-high-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-h100,gpu-a100
#SBATCH --output=corrS1_%x_%j.log
# MedMNIST-C inference on existing checkpoints (proposal_shift_robustness_20261002).
# Pure inference -- trains nothing.
#
# Usage: sbatch --job-name=corrS1_path corr_s1.slurm pathmnist [ratio]
#   optional 2nd arg = one ratio, to split datasets whose grid exceeds the wall.
# Env:
#   TREE=s1|s2          s1 = original checkpoints (default), s2 = colour-aug retrains
#   CORR_OUT_ROOT=...   output tree; REQUIRED on qiangzeng/danranwang (cannot write /project)
#   CODE=...            code dir (default: shared tdgs_shared/code/shiftbench)
#
# Cost (~3000 img/s on A100): pathmnist 90 ckpt x 55 x 7180 -> ~3.3h; blood ~1.6h;
#   organS ~4.8h; organA ~4.8h per ratio; tissue ~8h per ratio.
# Resumable per (run, corruption): resubmitting skips finished work.
set -euo pipefail
umask 002
DS="${1:?dataset argument required}"
RATIOS="${2:-0.02 0.05}"
TREE="${TREE:-s1}"
CODE="${CODE:-/mnt/prj01/hgrp-1502-5TB/tdgs_shared/code/shiftbench}"
PY=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/envs/medmnistc-py311/bin/python
cd "$CODE"
# The pipeline's graphcov guard runs git on a repo owned by xiaoyuxu2; on the other
# accounts git refuses it ("dubious ownership") and the guard fails spuriously.
SRC=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/sources/table1_clean_only_20260922
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$(realpath "$SRC")"
# hpcgpu108 has had a dead GPU (memory: luhpc-resource-limits). Fail fast, don't hang.
if ! "$PY" -c "import torch; torch.zeros(1).cuda()" > /dev/null 2>&1; then
    echo "DEAD GPU on $(hostname) dev=${CUDA_VISIBLE_DEVICES:-?} -- resubmit" >&2; exit 3
fi
echo "host=$(hostname) user=$(whoami) gpu=${CUDA_VISIBLE_DEVICES:-?} ds=$DS ratios=$RATIOS tree=$TREE out=${CORR_OUT_ROOT:-default} start=$(date -Is)"
"$PY" corr_rescore_s1.py \
      --tree "$TREE" \
      --datasets "$DS" \
      --ratios $RATIOS \
      --seeds 42 43 44 45 46
echo "done=$(date -Is)"
