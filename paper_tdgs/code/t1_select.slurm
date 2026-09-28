#!/bin/bash
#SBATCH --job-name=t1sel
#SBATCH --partition=gpu-a100
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=hpcgpu108
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#
# Stage the Table-1 baseline selections on the author seed grid 42-46.
# ONE ARRAY TASK PER SEED; DS is a comma-separated dataset list and the
# pipeline loops over it itself. --mem=64G covers tissuemnist, whose facility
# runs on CPU over classes of >40k points (the archive ran it at 48G).
#
# PHASE=select of the archive's own pipeline.py, 7 baselines x {2%, 5%} in one
# pass, so a seed's training dynamics (el2n/forgetting/eva) are computed once and
# shared by both budgets. Dynamics for s42/s43 (and s44 for organa/organs) are
# reused from the archive cache through symlinks; the rest are computed here
# and written to the SHARED cache, so no account ever recomputes them.
#
# selection_seed == training_seed, as in the archive and in the author's code.
# graph_a2 is NOT staged: it is seed-independent (archive sha identical for
# seeds 0/1/42/43) and already trained on 42-46 in rounds 1/2.
set -euo pipefail
umask 002

: "${DS:?DS is required}"
: "${T1_ROOT:?T1_ROOT is required}"
: "${T1_CACHE:?T1_CACHE is required}"
SEED=$((42 + ${SLURM_ARRAY_TASK_ID:-0}))
# sbatch --export splits on commas, so a list passed as DS=a,b arrives as DS=a
# (09-28: blood/organS were silently never staged). Pass DS=a:b instead.
DS=${DS//:/,}

PROJECT_ROOT=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab
SOURCE_ROOT=$PROJECT_ROOT/sources/table1_clean_only_20260922
REPO_ROOT=$SOURCE_ROOT
PYTHON=$PROJECT_ROOT/envs/medmnistc-py311/bin/python
HF_HOME=$PROJECT_ROOT/table1_clean_only_20260922/hf_cache

TAG=${DS//,/_}
[[ $DS == *,* ]] && TAG=multi
SEL_ROOT=$T1_ROOT/stage/seed_$SEED
mkdir -p "$SEL_ROOT"
CONFIG=$SEL_ROOT/config_select_${TAG}.yaml
cat > "$CONFIG" <<EOF
name: table1_s4246_${TAG}_select_seed_${SEED}
datasets: [$DS]
methods: [random, el2n_top, forgetting, eva, facility, fps, herding]
ratios: [0.02, 0.05]
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
export RELIABILITY_OUT=$SEL_ROOT/results      # PHASE=select trains nothing
export SELECTION_OUT=$T1_ROOT/sel/seed_$SEED   # <method>/<ds>/ratio_<r>/selection_seed_<s>/
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
export AUTO_CONSOLIDATE=0
export SMOKE_N=0
export IMAGEMAGICK_ROOT=$PROJECT_ROOT/envs/imagemagick

source "$SOURCE_ROOT/reliability_medmnistc/slurm/common_env.sh"
configure_imagemagick

cd "$SOURCE_ROOT"
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="$(realpath "$SOURCE_ROOT")"

echo "host=$(hostname) user=$(whoami) ds=$DS seed=$SEED start=$(date -Is)"
"$PYTHON" reliability_medmnistc/scripts/run_pipeline.py --config "$CONFIG" --phase select
echo "=== staged ==="
find "$SELECTION_OUT" -name index_order_sha256.txt | sort | while read -r f; do
    echo "$(cut -c1-16 "$f")  ${f#$SELECTION_OUT/}"
done
echo "done=$(date -Is)"
