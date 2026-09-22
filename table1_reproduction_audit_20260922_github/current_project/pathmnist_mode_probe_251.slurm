#!/bin/bash
#SBATCH --job-name=path-mode251
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --qos=YOUR_SLURM_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/project/PROJECT_ROOT/USER/reliability_medmnistc_ab/results/pathmnist_mode_probe_torch251_20260918/logs/%A_%a.out
#SBATCH --error=/project/PROJECT_ROOT/USER/reliability_medmnistc_ab/results/pathmnist_mode_probe_torch251_20260918/logs/%A_%a.err

set -euo pipefail

ROOT=/project/PROJECT_ROOT/USER/reliability_medmnistc_ab
AUTHOR=$ROOT/sources/graph-coverage-selection-author-diagnostic-b66-20260918
PYTHON=$ROOT/envs/medmnistc-py311/bin/python
SELROOT=$ROOT/formal_runs/single_seed_sel42_train42_20260915/selections
OUTROOT=$ROOT/results/pathmnist_mode_probe_torch251_20260918
DRIVER=$OUTROOT/pathmnist_mode_probe_driver.py

case "${SLURM_ARRAY_TASK_ID:?}" in
  0) METHOD=random; MODE=iteration;;
  1) METHOD=graph_a2; MODE=iteration;;
  2) METHOD=random; MODE=epoch;;
  3) METHOD=graph_a2; MODE=epoch;;
  *) echo "invalid task id" >&2; exit 2;;
esac

SEL=$SELROOT/$METHOD/pathmnist/ratio_0.05/selection_seed_42/selected_indices.npy
TASK=$OUTROOT/tasks/task_${SLURM_ARRAY_TASK_ID}_${METHOD}_${MODE}
mkdir -p "$TASK" "$OUTROOT/logs"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DETERMINISTIC=1
export PYTHONPATH="$AUTHOR:$OUTROOT:${PYTHONPATH:-}"
export MEDMNIST_DATA_DIR=$ROOT/data/medmnist

echo "dataset=pathmnist ratio=0.05 method=$METHOD mode=$MODE selection_seed=42 training_seed=42"
echo "torch=$($PYTHON -c 'import torch; print(torch.__version__, torch.version.cuda)')"
echo "selection=$SEL"

"$PYTHON" -u "$DRIVER" \
  --dataset pathmnist --method "$METHOD" --ratio 0.05 --mode "$MODE" \
  --selection "$SEL" --output "$TASK" --repo "$AUTHOR" \
  --epochs 1000 --iterations 1000 --seed 42 --batch-size 256 --num-workers 4 \
  2>&1 | tee "$TASK/task.log"
