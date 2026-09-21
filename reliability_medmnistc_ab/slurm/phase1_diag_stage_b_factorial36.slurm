#!/bin/bash
# Diagnostic plan item 3: 3x3 seed factorial, 36 new training units.
# {random, graph_a2} x {pathmnist, bloodmnist} x selection_seed{42,43,44}
# x training_seed{42,43,44}. clean->clean arm only, 1000 epochs, final
# checkpoint evaluated on the official TEST split (pipeline.py's built-in
# behavior -- no val-best selection). Evaluates clean BA + all MedMNIST-C
# corruption BA per cell, matching item 4's "clean BA / corruption BA /
# worst-class recall / ECE15, no cherry-picking" requirement directly from
# metrics.jsonl.
#SBATCH --job-name=phase1-diag-train
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=0-35%5
#SBATCH --output=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/formal_runs/phase1_diag_20260920_xiaoyuxu2/logs/train_%A_%a.out
#SBATCH --error=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/formal_runs/phase1_diag_20260920_xiaoyuxu2/logs/train_%A_%a.err

set -euo pipefail

ROOT=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab
REPO_ROOT=$ROOT/sources/graph-coverage-selection-medmnistc-reliability-clean-d74
PYTHON=$ROOT/envs/medmnistc-py311/bin/python
CONFIG=$REPO_ROOT/reliability_medmnistc/configs/full_5datasets_8methods.yaml
BASE_OUT=${VERIFY_ROOT:-$ROOT/formal_runs/phase1_diag_20260920_xiaoyuxu2}

# NOTE: local index arrays must NOT be named METHODS/DATASETS/SEEDS -- those
# names collide with the exported env vars pipeline.py reads via
# os.environ.get(). Bash cannot export an array to a child process; doing
# `export METHODS=$METHOD` on a variable that is still an array silently
# fails (the child process sees no METHODS key at all), so pipeline.py fell
# back to its full 8-method/5-dataset defaults every cell -- this is why the
# first run of this script trained on all 5 datasets x 8 methods per array
# task instead of the intended single cell, causing both the timeouts and
# the concurrent shared_cache SHA256 corruption (5 parallel tasks racing to
# write the same full-default cache files with non-atomic np.savez writes).
METHOD_LIST=(random graph_a2)
DATASET_LIST=(pathmnist bloodmnist)
SEED_LIST=(42 43 44)

IDX=$SLURM_ARRAY_TASK_ID
METHOD_IDX=$(( IDX / 18 ))
REM1=$(( IDX % 18 ))
DATASET_IDX=$(( REM1 / 9 ))
REM2=$(( REM1 % 9 ))
SEL_IDX=$(( REM2 / 3 ))
TRAIN_IDX=$(( REM2 % 3 ))

METHOD=${METHOD_LIST[$METHOD_IDX]}
DATASET=${DATASET_LIST[$DATASET_IDX]}
SEL_SEED=${SEED_LIST[$SEL_IDX]}
TRAIN_SEED=${SEED_LIST[$TRAIN_IDX]}

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DETERMINISTIC=1
export PREP_ROOT=$REPO_ROOT
export GRAPH_ROOT=$REPO_ROOT
export MEDC_ROOT=$REPO_ROOT/third_party/medmnistc
export MEDMNIST_ROOT=$ROOT/data/medmnist
export MEDMNISTC_ROOT=$ROOT/data/medmnistc
export METHODS=$METHOD
export DATASETS=$DATASET
export RATIOS=0.02
export SELECTION_SEED=$SEL_SEED
export TRAINING_SEEDS=$TRAIN_SEED
export EPOCHS=${VERIFY_EPOCHS:-1000}
export DYNAMICS_EPOCHS=200
export IMAGE_SIZE=224
export BATCH_SIZE=256
export NUM_WORKERS=4
export EMBEDDING_SOURCE=uni
export AUGMENT=0
export RUN_FULL_TRAIN=0
export AUTO_CONSOLIDATE=0
export DOWNLOAD_MEDMNIST=0
export CORR_HASH=0
export TRAIN_SOURCE=clean
export TRAIN_ON_CORRUPTED=0

ARM_OUT=$BASE_OUT/factorial36/$METHOD/$DATASET/sel_seed_$SEL_SEED/train_seed_$TRAIN_SEED
mkdir -p "$ARM_OUT" "$ARM_OUT/logs" "$ARM_OUT/selections"
export RELIABILITY_OUT=$ARM_OUT
export SELECTION_OUT=$ARM_OUT/selections
export GRAPH_CACHE=$BASE_OUT/shared_cache

cd "$REPO_ROOT"
source "$REPO_ROOT/reliability_medmnistc/slurm/common_env.sh"
configure_imagemagick

echo "cell method=$METHOD dataset=$DATASET selection_seed=$SEL_SEED training_seed=$TRAIN_SEED array_index=$IDX"
"$PYTHON" reliability_medmnistc/scripts/select.py --config "$CONFIG"
"$PYTHON" reliability_medmnistc/scripts/train.py --config "$CONFIG"
"$PYTHON" reliability_medmnistc/scripts/evaluate.py --config "$CONFIG"
echo "cell complete method=$METHOD dataset=$DATASET selection_seed=$SEL_SEED training_seed=$TRAIN_SEED"
