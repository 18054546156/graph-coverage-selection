#!/bin/bash
#SBATCH --job-name=s0colour
#SBATCH --partition=cpu-amd9754
#SBATCH --qos=qos-normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=3800M
#SBATCH --time=08:00:00
#SBATCH --output=s0colour_%j.log
# S0 colour-gap check, CPU only. qos-normal caps memory at 4G per user, so the
# script streams npz rows instead of loading arrays.
set -euo pipefail
PY=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/envs/medmnistc-py311/bin/python
CODE=${CODE:-/mnt/prj01/hgrp-1502-5TB/tdgs_shared/code/shiftbench}
cd "$CODE"
echo "host=$(hostname) start=$(date -Is)"
"$PY" s0_colour_gap.py --datasets pathmnist bloodmnist --per-set 3000 --out "$CODE/s0_colour_gap.json"
echo "done=$(date -Is)"
