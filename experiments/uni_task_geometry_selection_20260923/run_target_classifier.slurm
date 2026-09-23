#!/bin/bash
# Separate target-classifier entry point. It consumes a frozen linear-pilot
# selection_manifest.jsonl and never changes the selected IDs.
#SBATCH --job-name=uni-task-geometry-target
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/xiaoyuxu2/uni_task_geometry_selection_20260923/logs/%x-%j.out
#SBATCH --error=/home/xiaoyuxu2/uni_task_geometry_selection_20260923/logs/%x-%j.err

set -euo pipefail
PROJECT=/home/xiaoyuxu2/uni_task_geometry_selection_20260923
GRAPH=/project/prj-sis01/xuxiaoyu/graph_select
DATA=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/data/medmnist
PYTHON=/project/prj-sis01/xuxiaoyu/graph_select/envs/medmnistc-py311/bin/python

mkdir -p "$PROJECT/logs" "$PROJECT/target_classifier"
echo "Target-classifier launcher is intentionally separate from the linear pilot."
echo "Implement/submit only after the frozen linear-pilot manifest is reviewed."
echo "PROJECT=$PROJECT GRAPH=$GRAPH DATA=$DATA PYTHON=$PYTHON"
