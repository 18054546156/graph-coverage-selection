#!/bin/bash
#SBATCH --job-name=proxyreg
#SBATCH --partition=gpu-a100
#SBATCH --qos=qos-high-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --exclude=hpcgpu108
#SBATCH --output=proxyreg_%j.log
# Compares the full-pool demand proxy against a budget-matched one, scored by
# Spearman(per-class proxy demand, per-class 1-recall of the FINAL ResNet-18)
# read off the archived Table-1 predictions. Trains no ResNet.
set -euo pipefail
PY=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/envs/medmnistc-py311/bin/python
cd "$HOME/tdgs"
echo "host=$(hostname) start=$(date -Is)"
"$PY" proxy_regime.py --out "$HOME/proxy_regime.json"
echo "done=$(date -Is)"
