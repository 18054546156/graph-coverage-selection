#!/bin/bash
# Train + evaluate ONE Table-1 baseline cell on the AUTHOR seed grid (42-46).
#
# WHY THIS EXISTS. The Table-1 archive ran the 7 baselines on seeds 0-4; the
# author (graphcov/run/__main__.py --seed 42, seed = seed + trial, --trials 5)
# used 42-46. graph_a2 and tdgs_cls already sit on 42-46 (round 1/2), so this
# puts every Table-1 column on the same seed grid.
#
# PROTOCOL IS THE ARCHIVE'S, UNCHANGED: same source tree, same pipeline.py, same
# config keys as table1_clean_only_20260922/run_clean_only.slurm (verified: the
# archive's seed_42 and seed_0 configs differ only in the seed), same
# DETERMINISTIC_TRAINING=1, and -- like the archive and like the author --
# selection_seed == training_seed. Only the seed values change.
#
# Selections must already exist (t1_select.slurm). This runner never selects:
# it runs PHASE=train then PHASE=evaluate, both of which LOAD the verified
# selection artifact. PHASE=full would redo selection per cell, which for
# tissuemnist facility on CPU is hours.
#
# Usage: t1_run_one.sh <dataset> <ratio> <method> <seed>
set -euo pipefail

DATASET=$1; RATIO=$2; METHOD=$3; SEED=$4

PROJECT_ROOT=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab
SOURCE_ROOT=$PROJECT_ROOT/sources/table1_clean_only_20260922
REPO_ROOT=$SOURCE_ROOT
PYTHON=$PROJECT_ROOT/envs/medmnistc-py311/bin/python

T1_ROOT=${T1_ROOT:?T1_ROOT is required}
T1_CACHE=${T1_CACHE:?T1_CACHE is required}
HF_HOME=$PROJECT_ROOT/table1_clean_only_20260922/hf_cache

SEED_ROOT=$T1_ROOT/formal/$DATASET/seed_$SEED
SEL_ROOT=$T1_ROOT/sel/seed_$SEED
RUN_DIR=$SEED_ROOT/results/$DATASET/$METHOD/ratio_$RATIO/selection_seed_$SEED/train_seed_$SEED/aug_0
SEL_DIR=$SEL_ROOT/$METHOD/$DATASET/ratio_$RATIO/selection_seed_$SEED

if [[ -f $RUN_DIR/predictions_clean.npz ]]; then
    echo "SKIP  $DATASET r$RATIO $METHOD s$SEED  (already has predictions)"
    exit 0
fi
# Staging is split across the three accounts, so a cell can reach a GPU before
# another account has staged its dataset. Wait (bounded) instead of failing.
SEL_WAIT=${SEL_WAIT:-21600}
waited=0
while [[ ! -f $SEL_DIR/index_order_sha256.txt && $waited -lt $SEL_WAIT ]]; do
    [[ $((waited % 1800)) -eq 0 ]] && echo "WAIT  $DATASET r$RATIO $METHOD s$SEED selection not staged yet (${waited}s)"
    sleep 60; waited=$((waited + 60))
done
if [[ ! -f $SEL_DIR/index_order_sha256.txt ]]; then
    echo "MISSING SELECTION  $DATASET r$RATIO $METHOD s$SEED -> $SEL_DIR" >&2
    exit 3
fi

# One config per cell, so concurrent cells of the same (dataset, seed) never
# rewrite each other's file. Keys mirror run_clean_only.slurm formal mode.
CONFIG=$SEED_ROOT/config_${METHOD}_r${RATIO}.yaml
mkdir -p "$SEED_ROOT"     # 09-28 fix: without this every first cell of a (ds, seed) died here
cat > "$CONFIG" <<EOF
name: table1_s4246_${DATASET}_${METHOD}_r${RATIO}_seed_${SEED}
datasets: [$DATASET]
methods: [$METHOD]
ratios: [$RATIO]
selection_seed: $SEED
training_seeds: [$SEED]
epochs: 1000
dynamics_epochs: 200
image_size: 224
dynamics_image_size: 28
batch_size: 256
num_workers: 4
embedding_source: uni
augment: false
download_medmnist: false
corruption_hash: false
method_config:
  facility: {global_selection: false, execution_device: auto, cpu_min_class_size: 40000}
  graph_a2: {global_selection: true, k_neighbors: 50, k_hops: 2}
checkpoint_rule: final_epoch
EOF

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DETERMINISTIC_TRAINING=1
export CLEAN_ONLY=1
export PREP_ROOT=$SOURCE_ROOT
export GRAPH_ROOT=$SOURCE_ROOT
export MEDC_ROOT=$SOURCE_ROOT/third_party/medmnistc
export MEDMNIST_ROOT=$PROJECT_ROOT/data/medmnist
export MEDMNISTC_ROOT=$PROJECT_ROOT/data/medmnistc
export RELIABILITY_OUT=$SEED_ROOT/results
export SELECTION_OUT=$SEL_ROOT
export GRAPH_CACHE=$T1_CACHE
export HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1
export TRAIN_SOURCE=clean
export TRAIN_ON_CORRUPTED=0
export RUN_FULL_TRAIN=0
export CORR_HASH=0
export AUTO_CONSOLIDATE=0     # cells of one seed run concurrently; never race on metrics.csv
export SMOKE_N=0
export IMAGEMAGICK_ROOT=$PROJECT_ROOT/envs/imagemagick

source "$SOURCE_ROOT/reliability_medmnistc/slurm/common_env.sh"
configure_imagemagick

cd "$SOURCE_ROOT"
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="$(realpath "$SOURCE_ROOT")"

echo "START $DATASET r$RATIO $METHOD s$SEED gpu=${CUDA_VISIBLE_DEVICES:-?} $(date -Is)"
"$PYTHON" reliability_medmnistc/scripts/run_pipeline.py --config "$CONFIG" --phase train
"$PYTHON" reliability_medmnistc/scripts/run_pipeline.py --config "$CONFIG" --phase evaluate
[[ -f $RUN_DIR/predictions_clean.npz ]] || { echo "NO PREDICTIONS after evaluate: $RUN_DIR" >&2; exit 4; }
echo "DONE  $DATASET r$RATIO $METHOD s$SEED $(date -Is)"
