#!/bin/bash
# One task runs one dataset-method cell for the clean-selection baseline.
# Submit separate arrays with TASK_OFFSET and ACCOUNT_TAG.
#SBATCH --job-name=medmnistc-cell
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --qos=YOUR_SLURM_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00

set -euo pipefail

ROOT=/project/PROJECT_ROOT/USER/reliability_medmnistc_ab
REPO_ROOT=$ROOT/sources/graph-coverage-selection-medmnistc-reliability-clean-d74
CONFIG=$REPO_ROOT/reliability_medmnistc/configs/full_5datasets_8methods.yaml
SELECTION_ROOT=$ROOT/formal_runs/single_seed_sel42_train42_20260915/selections
TASK_OFFSET=${TASK_OFFSET:?set global task offset}
ACCOUNT_TAG=${ACCOUNT_TAG:?set account tag}
METHOD_INDEX=$((TASK_OFFSET + SLURM_ARRAY_TASK_ID))

DATASETS=(organsmnist organamnist pathmnist tissuemnist bloodmnist)
METHODS_LIST=(random el2n_top forgetting eva facility fps herding graph_a2)
METHOD=${METHODS_LIST[$METHOD_INDEX]}
DATASET=all5

case "$ACCOUNT_TAG" in
  HPC_USER_A)
    PYTHON=$ROOT/envs/medmnistc-py311/bin/python
    OUTROOT=$ROOT/formal_runs/single_seed_5datasets_8methods_ratio2_clean_20260918_HPC_USER_A
    ;;
  HPC_USER_B)
    PYTHON=$ROOT/envs/medmnistc-py311/bin/python
    OUTROOT=/home/HPC_USER_B/prj_HPC_USER_B/single_seed_5datasets_8methods_ratio2_clean_20260918
    ;;
  *) echo "unsupported ACCOUNT_TAG=$ACCOUNT_TAG" >&2; exit 2;;
esac

TASK_ROOT=$OUTROOT/method_tasks/$METHOD
mkdir -p "$TASK_ROOT" "$OUTROOT/logs" "$OUTROOT/manifests"

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DETERMINISTIC=1
export REPO_ROOT
export MEDMNIST_ROOT=$ROOT/data/medmnist
export MEDMNISTC_ROOT=$ROOT/data/medmnistc
export MEDC_ROOT=$REPO_ROOT/third_party/medmnistc
export GRAPH_ROOT=$REPO_ROOT
export PREP_ROOT=$REPO_ROOT
export METHODS=$METHOD
export RATIOS=0.02
export SELECTION_SEED=42
export TRAINING_SEEDS=42
export EPOCHS=1000
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
export RELIABILITY_OUT=$OUTROOT
export SELECTION_OUT=$SELECTION_ROOT
export GRAPH_CACHE=$ROOT/formal_runs/single_seed_sel42_train42_20260915/cache

cd "$REPO_ROOT"
source "$REPO_ROOT/reliability_medmnistc/slurm/common_env.sh"
configure_imagemagick

echo "account=$ACCOUNT_TAG method_index=$METHOD_INDEX dataset_scope=all5 method=$METHOD"
echo "python=$PYTHON"
"$PYTHON" -c 'import sys, torch, torchvision, medmnist; print(sys.version); print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available()); print("torchvision", torchvision.__version__, "medmnist", medmnist.__version__)'
echo "selection_root=$SELECTION_ROOT"
echo "output_root=$OUTROOT"

"$PYTHON" reliability_medmnistc/scripts/train.py --config "$CONFIG"
"$PYTHON" reliability_medmnistc/scripts/evaluate.py --config "$CONFIG"

cat > "$TASK_ROOT/submission_cell.txt" <<EOF
account=$ACCOUNT_TAG
method_index=$METHOD_INDEX
dataset=$DATASET
method=$METHOD
ratio=0.02
selection_seed=42
training_seed=42
epochs=1000
selection_root=$SELECTION_ROOT
source_head=$(git -C "$REPO_ROOT" rev-parse HEAD)
EOF

echo "complete dataset=$DATASET method=$METHOD"
