#!/bin/bash
# Isolated frozen-UNI selection pilot. No official validation/test is opened.
# The target-classifier stage has a separate launcher.
#SBATCH --job-name=uni-task-geometry-linear
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-normal
#SBATCH --partition=cpu-amd9754
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/xiaoyuxu2/uni_task_geometry_selection_20260923/logs/%x-%j.out
#SBATCH --error=/home/xiaoyuxu2/uni_task_geometry_selection_20260923/logs/%x-%j.err

set -euo pipefail
PROJECT=/home/xiaoyuxu2/uni_task_geometry_selection_20260923
GRAPH=/project/prj-sis01/xuxiaoyu/graph_select
EMBED=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_20260922/cache/embeddings_img224_smokefull
DATA=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/data/medmnist
PYTHON=/project/prj-sis01/xuxiaoyu/graph_select/envs/graphcov-py311/bin/python3

mkdir -p "$PROJECT/logs" "$PROJECT/results"
echo "start=$(date -Is) job=$SLURM_JOB_ID host=$(hostname)"
"$PYTHON" "$PROJECT/run_linear_probe.py" \
  --embedding-root "$EMBED" \
  --data-root "$DATA" \
  --graphcov-root "$GRAPH" \
  --output-dir "$PROJECT/results/run_$SLURM_JOB_ID" \
  --size 224 --max-per-class 1000 --blocks 6 --budget-per-class 50 \
  --pilot-per-class 10 --audit-fraction 0.20 \
  --graph-k-neighbors 10 --graph-k-hops 2 --tau 0.5 \
  --gradient-projection-dim 64 --seed 20260923
echo "finish=$(date -Is)"
