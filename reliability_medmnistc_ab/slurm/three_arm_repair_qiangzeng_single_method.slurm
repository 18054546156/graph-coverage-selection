#!/bin/bash
# Five missing single-seed cells; one method per array element.
# facility/herding: corrupted selection -> clean training.
# el2n_top/forgetting/eva: corrupted selection -> corrupted training.
#SBATCH --job-name=medmnistc-qz-repair20
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/home/qiangzeng/prj_qiangzeng/three_arm_single_seed_20260920_qiangzeng_repair/logs/repair_%A_%a.out
#SBATCH --error=/home/qiangzeng/prj_qiangzeng/three_arm_single_seed_20260920_qiangzeng_repair/logs/repair_%A_%a.err

set -euo pipefail

ROOT=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab
REPO_ROOT=/home/qiangzeng/prj_qiangzeng/graph-coverage-selection-medmnistc-reliability-clean-repair-20260920
PYTHON=$ROOT/envs/medmnistc-py311/bin/python
CONFIG_DIR=$REPO_ROOT/reliability_medmnistc/configs/repair_qiangzeng
METHODS=(facility herding el2n_top forgetting eva)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
CONFIG=$CONFIG_DIR/${METHOD}.yaml
OUT_ROOT=/home/qiangzeng/prj_qiangzeng/three_arm_single_seed_20260920_qiangzeng_repair

if [ "$METHOD" = facility ] || [ "$METHOD" = herding ]; then
  ARM_OUT=$OUT_ROOT/selection_corrupted_clean_train_pixelate_s3
  TRAIN_ON_CORRUPTED=0
else
  ARM_OUT=$OUT_ROOT/selection_corrupted_corrupted_train_pixelate_s3
  TRAIN_ON_CORRUPTED=1
fi

mkdir -p "$ARM_OUT/logs" "$ARM_OUT/manifests"
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
export TRAIN_SOURCE=corrupted
export TRAIN_CORRUPTION=pixelate
export TRAIN_SEVERITY=3
export TRAIN_ON_CORRUPTED
export RELIABILITY_OUT=$ARM_OUT
export SELECTION_OUT=$ARM_OUT/selections
export GRAPH_CACHE=$ROOT/formal_runs/three_arm_single_seed_20260919_xiaoyuxu2/selection_corrupted_clean_train_pixelate_s3/cache
export IMAGEMAGICK_ROOT=$ROOT/envs/imagemagick

# The YAML is the sole source of METHODS; remove stale Slurm exports.
unset METHODS
cd "$REPO_ROOT"
git config --global --add safe.directory "$REPO_ROOT" 2>/dev/null || true
source "$REPO_ROOT/reliability_medmnistc/slurm/common_env.sh"
configure_imagemagick

echo "account=qiangzeng method=$METHOD arm_out=$ARM_OUT"
echo "config=$CONFIG"
echo "cache=$GRAPH_CACHE"
"$PYTHON" -c 'import sys, torch, torchvision, medmnist; print(sys.version); print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available()); print("torchvision", torchvision.__version__, "medmnist", medmnist.__version__)'

"$PYTHON" reliability_medmnistc/scripts/select.py --config "$CONFIG"
"$PYTHON" reliability_medmnistc/scripts/train.py --config "$CONFIG"
"$PYTHON" reliability_medmnistc/scripts/evaluate.py --config "$CONFIG"

cat > "$ARM_OUT/manifests/repair_${METHOD}_submission_cell.txt" <<EOF
account=qiangzeng
method=$METHOD
datasets=organsmnist,organamnist,pathmnist,tissuemnist,bloodmnist
ratio=0.02
selection_seed=42
training_seed=42
epochs=1000
train_source=corrupted
train_corruption=pixelate
train_severity=3
train_on_corrupted=$TRAIN_ON_CORRUPTED
graph_cache=$GRAPH_CACHE
source_head=$(git rev-parse HEAD)
EOF

echo "complete method=$METHOD"
