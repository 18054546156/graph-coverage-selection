#!/bin/bash
#SBATCH --job-name=robust-step2-smoke
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/logs/step2_training_variance_smoke_%j.out
#SBATCH --error=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/logs/step2_training_variance_smoke_%j.err

set -euo pipefail
ROOT=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab
PY="$ROOT/envs/medmnistc-py311/bin/python"
MEDC_ROOT="$ROOT/sources/medmnistc-api"
GRAPH_ROOT="$ROOT/sources/repos/graphcov"
if [ ! -d "$GRAPH_ROOT/graphcov" ]; then GRAPH_ROOT="$ROOT/sources/graph-coverage-selection-codex-reliability-audit-v29/repos/graphcov"; fi
CHECKPOINT_ROOT="$ROOT/artifacts/checkpoints/variance_smoke"
PREDICTION_ROOT="$ROOT/artifacts/predictions/variance_smoke"
METRICS_ROOT="$ROOT/results/raw_metrics/variance_smoke"
SUMMARY_ROOT="$ROOT/results/summary_tables/variance_smoke"
PLAN="$ROOT/protocol/variance_smoke_plan.json"

mkdir -p "$ROOT/logs" "$CHECKPOINT_ROOT" "$PREDICTION_ROOT" "$METRICS_ROOT" "$SUMMARY_ROOT"
export PYTHONPATH="$MEDC_ROOT:$GRAPH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MAGICK_HOME="$ROOT/envs/imagemagick"
export LD_LIBRARY_PATH="$MAGICK_HOME/lib"
export PATH="$MAGICK_HOME/bin:$PATH"
export WAND_MAGICK_LIBRARY_SUFFIX="-7.Q16HDRI"
export LD_PRELOAD="$MAGICK_HOME/lib/libMagickCore-7.Q16HDRI.so:$MAGICK_HOME/lib/libMagickWand-7.Q16HDRI.so"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "===== step 2 training variance smoke ====="
date -Is; hostname; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
"$PY" -c 'import torch, medmnist, medmnistc; print("torch", torch.__version__, "cuda", torch.cuda.is_available()); print("medmnist", medmnist.__version__)'
"$PY" -c 'from wand.image import Image; from medmnistc.corruptions.registry import CORRUPTIONS_DS; print("wand/registry", len(CORRUPTIONS_DS["pathmnist"]))'

"$PY" "$ROOT/scripts/run_training_variance.py" \
  --mode run \
  --dataset pathmnist \
  --method graph_a2 \
  --ratio 0.05 \
  --selection-seed 42 \
  --training-seed 42 \
  --data-root "$ROOT/data/medmnist" \
  --corrupted-root "$ROOT/data/medmnistc" \
  --selection-root "$ROOT/artifacts/selections" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --prediction-root "$PREDICTION_ROOT" \
  --metrics-root "$METRICS_ROOT" \
  --output-root "$SUMMARY_ROOT" \
  --plan "$PLAN" \
  --epochs 1 \
  --batch-size 256 \
  --workers 8 \
  --log-every 1

echo "===== step 2 smoke completed ====="
find "$CHECKPOINT_ROOT" "$METRICS_ROOT" -type f -printf '%p %s bytes\n' | sort
