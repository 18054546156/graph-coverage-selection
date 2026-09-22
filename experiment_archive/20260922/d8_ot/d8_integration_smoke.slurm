#!/bin/bash
#SBATCH --job-name=d8-pipeline-smoke
#SBATCH --account=hpc-sis
#SBATCH --partition=gpu-a100
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/d8_class_conditional_ot_20260922/logs/d8-integration-smoke-%j.out
#SBATCH --error=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/d8_class_conditional_ot_20260922/logs/d8-integration-smoke-%j.err

set -euo pipefail
umask 0002

PROJECT_ROOT=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab
export D8_ROOT="$PROJECT_ROOT/d8_class_conditional_ot_20260922"
export SOURCE_ROOT="$D8_ROOT/source_snapshot"
export REPO_ROOT="$SOURCE_ROOT"
export PYTHON_BIN="$PROJECT_ROOT/envs/medmnistc-py311/bin/python"
export PYTHON="$PYTHON_BIN"
export PREP_ROOT="$SOURCE_ROOT"
export GRAPH_ROOT="$SOURCE_ROOT"
export MEDC_ROOT="$SOURCE_ROOT/third_party/medmnistc"
export MEDMNIST_ROOT="$PROJECT_ROOT/data/medmnist"
export MEDMNISTC_ROOT="$PROJECT_ROOT/data/medmnistc"
export HF_HOME="$PROJECT_ROOT/table1_clean_only_20260922/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_OFFLINE=1
export IMAGEMAGICK_ROOT="$PROJECT_ROOT/envs/imagemagick"
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DETERMINISTIC_TRAINING=1
export TRAIN_SOURCE=clean
export TRAIN_ON_CORRUPTED=0
export RUN_FULL_TRAIN=0
export CORR_HASH=0
export CLEAN_ONLY=1
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="$(realpath "$SOURCE_ROOT")"

source "$SOURCE_ROOT/reliability_medmnistc/slurm/common_env.sh"
configure_imagemagick

cd "$D8_ROOT"
"$PYTHON_BIN" d8_benchmark_worker.py --smoke
