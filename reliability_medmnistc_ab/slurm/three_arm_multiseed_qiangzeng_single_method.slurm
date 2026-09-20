#!/bin/bash
# Multi-seed three-arm audit: one method and one isolated output tree per task.
# Starts only after the five single-seed repair tasks succeed.
#SBATCH --job-name=medmnistc-qz-ms20
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --array=0-7%5
#SBATCH --output=/home/qiangzeng/prj_qiangzeng/three_arm_multiseed_20260920_qiangzeng/logs/msmethod_%A_%a.out
#SBATCH --error=/home/qiangzeng/prj_qiangzeng/three_arm_multiseed_20260920_qiangzeng/logs/msmethod_%A_%a.err

set -euo pipefail

ROOT=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab
REPO_ROOT=/home/qiangzeng/prj_qiangzeng/graph-coverage-selection-medmnistc-reliability-clean-repair-20260920
PYTHON=$ROOT/envs/medmnistc-py311/bin/python
CONFIG_DIR=$REPO_ROOT/reliability_medmnistc/configs/multiseed_qiangzeng
METHODS=(random el2n_top forgetting eva facility fps herding graph_a2)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
CONFIG=$CONFIG_DIR/${METHOD}.yaml
BASE_OUT=/home/qiangzeng/prj_qiangzeng/three_arm_multiseed_20260920_qiangzeng
CLEAN_SELECTION=$ROOT/formal_runs/single_seed_sel42_train42_20260915/selections
SHARED_CACHE=$ROOT/formal_runs/three_arm_single_seed_20260919_xiaoyuxu2/selection_corrupted_clean_train_pixelate_s3/cache

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DETERMINISTIC=1
export PREP_ROOT=$REPO_ROOT
export GRAPH_ROOT=$REPO_ROOT
export MEDC_ROOT=$REPO_ROOT/third_party/medmnistc
export MEDMNIST_ROOT=$ROOT/data/medmnist
export MEDMNISTC_ROOT=$ROOT/data/medmnistc
export DATASETS=organsmnist,organamnist,pathmnist,tissuemnist,bloodmnist
export RATIOS=0.02
export SELECTION_SEED=42
export TRAINING_SEEDS=42,43,44,45,46
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
export TRAIN_CORRUPTION=pixelate
export TRAIN_SEVERITY=3
export IMAGEMAGICK_ROOT=$ROOT/envs/imagemagick
export GRAPH_CACHE=$SHARED_CACHE

# The one-method YAML is the only source of METHODS.
unset METHODS
cd "$REPO_ROOT"
git config --global --add safe.directory "$REPO_ROOT" 2>/dev/null || true
source "$REPO_ROOT/reliability_medmnistc/slurm/common_env.sh"
configure_imagemagick

echo "account=qiangzeng method=$METHOD config=$CONFIG"
echo "training_seeds=$TRAINING_SEEDS base_out=$BASE_OUT"
"$PYTHON" -c 'import sys, torch, torchvision, medmnist; print(sys.version); print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available()); print("torchvision", torchvision.__version__, "medmnist", medmnist.__version__)'

for ARM in clean selection_corrupted_clean_train selection_corrupted_corrupted_train; do
  case "$ARM" in
    clean)
      TRAIN_SOURCE=clean
      TRAIN_ON_CORRUPTED=0
      ARM_OUT=$BASE_OUT/clean_selection_clean_train/$METHOD
      ARM_SELECTION=$CLEAN_SELECTION
      ;;
    selection_corrupted_clean_train)
      TRAIN_SOURCE=corrupted
      TRAIN_ON_CORRUPTED=0
      ARM_OUT=$BASE_OUT/selection_corrupted_clean_train_pixelate_s3/$METHOD
      ARM_SELECTION=$ARM_OUT/selections
      ;;
    selection_corrupted_corrupted_train)
      TRAIN_SOURCE=corrupted
      TRAIN_ON_CORRUPTED=1
      ARM_OUT=$BASE_OUT/selection_corrupted_corrupted_train_pixelate_s3/$METHOD
      ARM_SELECTION=$ARM_OUT/selections
      ;;
  esac

  mkdir -p "$ARM_OUT/logs" "$ARM_OUT/manifests"
  export TRAIN_SOURCE TRAIN_ON_CORRUPTED
  export RELIABILITY_OUT=$ARM_OUT
  export SELECTION_OUT=$ARM_SELECTION
  echo "begin arm=$ARM method=$METHOD train_source=$TRAIN_SOURCE train_on_corrupted=$TRAIN_ON_CORRUPTED"
  if [ "$ARM" != clean ]; then
    "$PYTHON" reliability_medmnistc/scripts/select.py --config "$CONFIG"
  fi
  "$PYTHON" reliability_medmnistc/scripts/train.py --config "$CONFIG"
  "$PYTHON" reliability_medmnistc/scripts/evaluate.py --config "$CONFIG"
  echo "complete arm=$ARM method=$METHOD"
done

echo "complete all arms method=$METHOD training_seeds=$TRAINING_SEEDS"
