#!/bin/bash
#SBATCH --job-name=robust-step1-pilot
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/logs/step1_representation_drift_%j.out
#SBATCH --error=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/logs/step1_representation_drift_%j.err

set -euo pipefail
ROOT=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab
PY="$ROOT/envs/medmnistc-py311/bin/python"
MEDC_ROOT="$ROOT/sources/medmnistc-api"
GRAPH_ROOT="$ROOT/sources/repos/graphcov"
if [ ! -d "$GRAPH_ROOT/graphcov" ]; then GRAPH_ROOT="$ROOT/sources/graph-coverage-selection-codex-reliability-audit-v29/repos/graphcov"; fi
OUT="$ROOT/results/representation_drift/pathmnist_pilot1000"

mkdir -p "$ROOT/logs" "$OUT"
export PYTHONPATH="$MEDC_ROOT:$GRAPH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MAGICK_HOME="$ROOT/envs/imagemagick"
export LD_LIBRARY_PATH="$MAGICK_HOME/lib"
export PATH="$MAGICK_HOME/bin:$PATH"
export WAND_MAGICK_LIBRARY_SUFFIX="-7.Q16HDRI"
export LD_PRELOAD="$MAGICK_HOME/lib/libMagickCore-7.Q16HDRI.so:$MAGICK_HOME/lib/libMagickWand-7.Q16HDRI.so"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "===== step 1 representation drift pilot ====="
date -Is; hostname; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
"$PY" -c 'import torch, medmnist, medmnistc; print("torch", torch.__version__, "cuda", torch.cuda.is_available()); print("medmnist", medmnist.__version__)'
"$PY" -c 'from wand.image import Image; from medmnistc.corruptions.registry import CORRUPTIONS_DS; print("wand/registry", len(CORRUPTIONS_DS["pathmnist"]))'

"$PY" "$ROOT/scripts/diagnose_representation_drift.py" \
  --graphcov-root "$GRAPH_ROOT" \
  --data-root "$ROOT/data/medmnist" \
  --selection-root "$ROOT/artifacts/selections" \
  --cache-root "/project/prj-sis01/xuxiaoyu/graph_select/cache/embeddings" \
  --output-root "$OUT" \
  --datasets pathmnist \
  --methods random,herding,fps,graph_a2 \
  --ratios 0.05 \
  --selection-seed 42 \
  --dynamics-seed 42 \
  --dynamics-size 28 \
  --dynamics-epochs 200 \
  --pilot-n 1000 \
  --views all \
  --severities 1,2,3,4,5 \
  --batch-size 128 \
  --workers 8 \
  --knn-k 50 \
  --save-knn

echo "===== step 1 completed ====="
find "$OUT" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
