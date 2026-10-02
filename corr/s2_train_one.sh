#!/bin/bash
# S2: train ONE (dataset, ratio, arm, seed) cell with colour-only augmentation,
# replaying the exact selection of the matching S1 run. Derived from
# ~/td_run_one.sh; differences: TD_SEL_FILE = S1 run's selected_local_indices.npy,
# augment: true, num_workers 8, and the s2_colour_pipeline.py wrapper.
#
# The caller sets CUDA_VISIBLE_DEVICES to a single device.
# Usage: s2_train_one.sh <dataset> <ratio> <arm> <seed> <selection.npy>
set -euo pipefail
umask 002

DATASET=$1; RATIO=$2; ARM=$3; SEED=$4; SEL=$5
CODE=${CODE:-/mnt/prj01/hgrp-1502-5TB/tdgs_shared/code/shiftbench}

PROJECT_ROOT=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab
SOURCE_ROOT=$PROJECT_ROOT/sources/table1_clean_only_20260922
REPO_ROOT=$SOURCE_ROOT
PYTHON=$PROJECT_ROOT/envs/medmnistc-py311/bin/python
EXPERIMENT_ROOT=/mnt/prj01/hgrp-1502-5TB/tdgs_shared/runs/s2_coloraug_20261002
# Per-account writable cache (only xiaoyuxu2 can write /project); HF cache is read-only + offline.
CACHE_ROOT=${S2_CACHE_ROOT:-$HOME/tdgs_cache}
HF_HOME=$PROJECT_ROOT/table1_clean_only_20260922/hf_cache

RUN_ROOT=$EXPERIMENT_ROOT/formal/ratio_${RATIO}/$DATASET/$ARM/seed_$SEED

if find "$RUN_ROOT/results" -name predictions_clean.npz 2>/dev/null | grep -q .; then
    echo "SKIP  $DATASET r$RATIO $ARM s$SEED  (already has predictions)"
    exit 0
fi
[[ -f "$SEL" ]] || { echo "MISSING SELECTION $SEL" >&2; exit 3; }
export TD_SEL_FILE=$SEL

mkdir -p "$RUN_ROOT"
cat > "$RUN_ROOT/s2_provenance.json" <<EOF
{"source_selection": "$SEL", "augmentation": "ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), no crop/flip",
 "wrapper": "s2_colour_pipeline.py", "started": "$(date -Is)", "host": "$(hostname)"}
EOF
CONFIG=$RUN_ROOT/config.yaml
cat > "$CONFIG" <<EOF
name: s2colour_${ARM}_${DATASET}_ratio${RATIO}_seed_${SEED}
datasets: [$DATASET]
methods: [precomputed]
ratios: [$RATIO]
selection_seed: 42
training_seeds: [$SEED]
epochs: 1000
dynamics_epochs: 200
image_size: 224
dynamics_image_size: 28
batch_size: 256
num_workers: 8
embedding_source: uni
augment: true
download_medmnist: false
corruption_hash: false
checkpoint_rule: final_epoch
EOF

echo "START $DATASET r$RATIO $ARM s$SEED gpu=${CUDA_VISIBLE_DEVICES:-?} $(date -Is)"

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DETERMINISTIC_TRAINING=1
export CLEAN_ONLY=1
export PREP_ROOT=$SOURCE_ROOT
export GRAPH_ROOT=$SOURCE_ROOT
export MEDC_ROOT=$SOURCE_ROOT/third_party/medmnistc
export MEDMNIST_ROOT=$PROJECT_ROOT/data/medmnist
export MEDMNISTC_ROOT=$PROJECT_ROOT/data/medmnistc
export RELIABILITY_OUT=$RUN_ROOT/results
export SELECTION_OUT=$RUN_ROOT/selections
export GRAPH_CACHE=$CACHE_ROOT
export HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1
export TRAIN_SOURCE=clean
export TRAIN_ON_CORRUPTED=0
export RUN_FULL_TRAIN=0
export CORR_HASH=0
export AUTO_CONSOLIDATE=1
export SMOKE_N=0
export IMAGEMAGICK_ROOT=$PROJECT_ROOT/envs/imagemagick

source "$SOURCE_ROOT/reliability_medmnistc/slurm/common_env.sh"
configure_imagemagick

cd "$SOURCE_ROOT"
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="$(realpath "$SOURCE_ROOT")"
"$PYTHON" "$CODE/s2_colour_pipeline.py" --config "$CONFIG" --phase full

echo "DONE  $DATASET r$RATIO $ARM s$SEED $(date -Is)"
