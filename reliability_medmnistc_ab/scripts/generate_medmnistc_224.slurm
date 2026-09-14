#!/bin/bash
# Generate MedMNIST-C from verified clean 224-pixel MedMNIST files.
# The upstream medmnistc-api source is called by generate_medmnistc.py.
#SBATCH --job-name=medmnistc-generate
#SBATCH --account=hpc-sis
# The account currently exposes the required memory through this QOS.
# No GPU is requested; this is a CPU/ImageMagick generation job.
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/logs/medmnistc_generate_%j.out
#SBATCH --error=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/logs/medmnistc_generate_%j.err

set -euo pipefail

ROOT=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab
PYTHON="$ROOT/envs/medmnistc-py311/bin/python"
export MAGICK_HOME="$ROOT/envs/imagemagick"
export LD_LIBRARY_PATH="$MAGICK_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$ROOT"
exec "$PYTHON" scripts/generate_medmnistc.py
