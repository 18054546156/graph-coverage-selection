#!/bin/bash
# Isolated 5% Table 1 run. One task = one dataset + one paired seed.
# The source tree and reusable caches are the audited, completed 2% snapshot;
# all 5% configs, selections, results, and logs go to this experiment root.
# The caller supplies DATASET, SEED, MODE, and --mem.
# MODE=smoke tests all methods on an 8192-example subset for one epoch.
# MODE=formal runs the full 5% protocol for 1000 epochs.
#SBATCH --job-name=table1-5pct
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --output=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_5pct_20260922/logs/%x_%j.out
#SBATCH --error=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_5pct_20260922/logs/%x_%j.err

set -euo pipefail
umask 0002

: "${DATASET:?DATASET is required}"
: "${SEED:?SEED is required}"

MODE="${MODE:-smoke}"
PROJECT_ROOT=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab
SOURCE_ROOT=$PROJECT_ROOT/sources/table1_clean_only_20260922
REPO_ROOT=$SOURCE_ROOT
PYTHON=$PROJECT_ROOT/envs/medmnistc-py311/bin/python
DATA_ROOT=$PROJECT_ROOT/data/medmnist
CORR_ROOT=$PROJECT_ROOT/data/medmnistc
EXPERIMENT_ROOT=$PROJECT_ROOT/table1_clean_only_5pct_20260922
# Reuse the completed, seed-keyed embeddings/dynamics caches; 5% selection
# changes the budget, not these feature/dynamics artifacts.
CACHE_ROOT=$PROJECT_ROOT/table1_clean_only_20260922/cache
HF_HOME=$PROJECT_ROOT/table1_clean_only_20260922/hf_cache
HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
HF_HUB_CACHE=$HUGGINGFACE_HUB_CACHE
TRANSFORMERS_CACHE=$HF_HOME/transformers

case "$MODE" in
  smoke)
    RUN_ROOT=$EXPERIMENT_ROOT/smoke/$DATASET/seed_$SEED
    EPOCHS=1
    DYNAMICS_EPOCHS=20
    SMOKE_N=8192
    BATCH_SIZE=64
    NUM_WORKERS=0
    AUTO_CONSOLIDATE=0
    ;;
  formal)
    RUN_ROOT=$EXPERIMENT_ROOT/formal/$DATASET/seed_$SEED
    EPOCHS=1000
    DYNAMICS_EPOCHS=200
    SMOKE_N=0
    BATCH_SIZE=256
    NUM_WORKERS=4
    AUTO_CONSOLIDATE=1
    ;;
  *)
    echo "unknown MODE=$MODE; expected smoke or formal" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_ROOT" "$EXPERIMENT_ROOT/logs" \
  "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"
CONFIG=$RUN_ROOT/config.yaml

cat > "$CONFIG" <<EOF
name: table1_clean_only_5pct_${DATASET}_seed_${SEED}_${MODE}
datasets: [$DATASET]
methods: [random, el2n_top, forgetting, eva, facility, fps, herding, graph_a2]
ratios: [0.05]
selection_seed: $SEED
training_seeds: [$SEED]
epochs: $EPOCHS
dynamics_epochs: $DYNAMICS_EPOCHS
image_size: 224
dynamics_image_size: 28
batch_size: $BATCH_SIZE
num_workers: $NUM_WORKERS
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
export MEDMNIST_ROOT=$DATA_ROOT
export MEDMNISTC_ROOT=$CORR_ROOT
export RELIABILITY_OUT=$RUN_ROOT/results
export SELECTION_OUT=$RUN_ROOT/selections
export GRAPH_CACHE=$CACHE_ROOT
export HF_HOME
export HUGGINGFACE_HUB_CACHE
export HF_HUB_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_CACHE
export TRAIN_SOURCE=clean
export TRAIN_ON_CORRUPTED=0
export RUN_FULL_TRAIN=0
export CORR_HASH=0
export AUTO_CONSOLIDATE=$AUTO_CONSOLIDATE
export SMOKE_N=$SMOKE_N

export IMAGEMAGICK_ROOT=$PROJECT_ROOT/envs/imagemagick
source "$SOURCE_ROOT/reliability_medmnistc/slurm/common_env.sh"
configure_imagemagick

cd "$SOURCE_ROOT"
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="$(realpath "$SOURCE_ROOT")"
"$PYTHON" reliability_medmnistc/scripts/run_pipeline.py --config "$CONFIG" --phase full
