#!/bin/bash
# Train ONE (dataset, ratio, arm, seed) cell of the TD-Cover matrix.
#
# Split out of run_td_screen.slurm so that several cells can run concurrently
# inside a single slurm allocation. Each training uses ~3.8GB RSS and one GPU,
# while qos-high-gpu allows 12 GPUs but only 5 running JOBS per user -- so the
# only way to reach the GPU ceiling is to pack multiple trainings per job.
#
# The caller sets CUDA_VISIBLE_DEVICES to a single device before invoking this.
#
# Usage: td_run_one.sh <dataset> <ratio> <arm> <seed>
set -euo pipefail

DATASET=$1; RATIO=$2; ARM=$3; SEED=$4

PROJECT_ROOT=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab
SOURCE_ROOT=$PROJECT_ROOT/sources/table1_clean_only_20260922
REPO_ROOT=$SOURCE_ROOT          # common_env.sh reads this and runs under set -u
PYTHON=$PROJECT_ROOT/envs/medmnistc-py311/bin/python

EXPERIMENT_ROOT=${TD_EXP_ROOT:-$PROJECT_ROOT/td_screen_20260925}
CACHE_ROOT=${TD_CACHE_ROOT:-$PROJECT_ROOT/table1_clean_only_20260922/cache}
HF_HOME=${TD_HF_ROOT:-$PROJECT_ROOT/table1_clean_only_20260922/hf_cache}

RUN_ROOT=$EXPERIMENT_ROOT/formal/ratio_${RATIO}/$DATASET/$ARM/seed_$SEED

# Idempotent: the matrix is submitted whole and re-submitted after failures, so
# a cell that already produced predictions must not be retrained. Retraining is
# not just wasted GPU -- it would silently overwrite a result that has already
# been read into an analysis.
if compgen -G "$RUN_ROOT/results/**/predictions_clean.npz" > /dev/null 2>&1 || \
   find "$RUN_ROOT/results" -name predictions_clean.npz 2>/dev/null | grep -q .; then
    echo "SKIP  $DATASET r$RATIO $ARM s$SEED  (already has predictions)"
    exit 0
fi

# Selections staged before 2026-09-25 live in the project (readable by all three
# accounts); newly staged ones land in the staging account's own home, because
# only xiaoyuxu2 can write the project and queueing every account behind it
# costs ~50min. Search the local dir first, fall back to the shared one.
SEL_NAME=${DATASET}_r${RATIO}_${ARM}_s42.npy
LOCAL_SEL=${TD_SEL_DIR:-$HOME/td_sel}/$SEL_NAME
SHARED_SEL=$PROJECT_ROOT/gamma_beta_gate_20260924/td_cover_sel/$SEL_NAME

# Staging runs concurrently with training: the whole matrix is submitted at once
# and the worklist is ordered so that already-staged cells come first. A cell
# whose selection is still being staged waits rather than failing, otherwise the
# pack would burn through the not-yet-staged half of the matrix in seconds and
# need a second submission round. Bounded, so a genuinely missing cell still
# surfaces as an error instead of holding a GPU forever.
SEL_WAIT=${SEL_WAIT:-2700}
waited=0
while [[ ! -f "$LOCAL_SEL" && ! -f "$SHARED_SEL" && $waited -lt $SEL_WAIT ]]; do
    [[ $((waited % 300)) -eq 0 ]] && echo "WAIT  $DATASET r$RATIO $ARM  selection not staged yet (${waited}s)"
    sleep 60; waited=$((waited + 60))
done
TD_SEL_FILE=$LOCAL_SEL
[[ -f "$TD_SEL_FILE" ]] || TD_SEL_FILE=$SHARED_SEL
if [[ ! -f "$TD_SEL_FILE" ]]; then
    echo "MISSING SELECTION  $DATASET r$RATIO $ARM  -> $TD_SEL_FILE" >&2
    exit 3
fi
export TD_SEL_FILE

mkdir -p "$RUN_ROOT"
CONFIG=$RUN_ROOT/config.yaml
cat > "$CONFIG" <<EOF
name: tdmatrix_${ARM}_${DATASET}_ratio${RATIO}_seed_${SEED}
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
num_workers: 4
embedding_source: uni
augment: false
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
"$PYTHON" reliability_medmnistc/scripts/run_pipeline.py --config "$CONFIG" --phase full

echo "DONE  $DATASET r$RATIO $ARM s$SEED $(date -Is)"
