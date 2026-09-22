#!/bin/bash
# Author-code clean Table-1 audit: 5 datasets x 2 ratios x two methods.
#SBATCH --job-name=table1-author
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --qos=YOUR_SLURM_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-19%5
#SBATCH --output=/project/PROJECT_ROOT/USER/reliability_medmnistc_ab/formal_runs/table1_author_single_seed_20260918/logs/table1_%A_%a.out
#SBATCH --error=/project/PROJECT_ROOT/USER/reliability_medmnistc_ab/formal_runs/table1_author_single_seed_20260918/logs/table1_%A_%a.err

set -euo pipefail

ROOT=/project/PROJECT_ROOT/USER/reliability_medmnistc_ab
REPO_ROOT=$ROOT/sources/graph-coverage-selection-180545-b66
AUTHOR_ROOT=$ROOT/sources/graph-coverage-selection-author-8cf
OUTPUT_ROOT=$ROOT/formal_runs/table1_author_single_seed_20260918
CACHE_ROOT=$ROOT/formal_runs/single_seed_sel42_train42_20260915/cache/img224_dyn200_sel42_smokefull
PYTHON=$ROOT/envs/medmnistc-py311/bin/python

DATASETS=(organsmnist organamnist pathmnist tissuemnist bloodmnist)
TASK_ID=${SLURM_ARRAY_TASK_ID:?array task id is required}
DATASET=${DATASETS[$((TASK_ID / 4))]}
SUBTASK=$((TASK_ID % 4))
case "$SUBTASK" in
  0) METHOD=graph_a2; RATIO=0.02 ;;
  1) METHOD=random;   RATIO=0.02 ;;
  2) METHOD=graph_a2; RATIO=0.05 ;;
  3) METHOD=random;   RATIO=0.05 ;;
  *) echo "invalid task id: $TASK_ID" >&2; exit 2 ;;
esac

TASK_OUTPUT=$OUTPUT_ROOT/tasks/task_${TASK_ID}_${DATASET}_${METHOD}_r${RATIO}
mkdir -p "$TASK_OUTPUT"

cd "$REPO_ROOT"
export PYTHONHASHSEED=0
export MEDMNIST_DATA_DIR=/home/HPC_USER_A/.medmnist
git config --global --add safe.directory "$REPO_ROOT" 2>/dev/null || true
git config --global --add safe.directory "$(readlink -f "$REPO_ROOT")" 2>/dev/null || true
git config --global --add safe.directory "$AUTHOR_ROOT" 2>/dev/null || true
git config --global --add safe.directory "$(readlink -f "$AUTHOR_ROOT")" 2>/dev/null || true

echo "task=$TASK_ID dataset=$DATASET method=$METHOD ratio=$RATIO seed=42"
echo "fork_commit=$(git rev-parse HEAD)"
echo "author_commit=$(git -C "$AUTHOR_ROOT" rev-parse HEAD)"
echo "cache=$CACHE_ROOT"

"$PYTHON" -u -m graphcov.run \
  --datasets "$DATASET" \
  --embeddings uni \
  --methods "$METHOD" \
  --ratios "$RATIO" \
  --trials 1 \
  --seed 42 \
  --deterministic \
  --global \
  --sparse-cpu \
  --k-neighbors 50 \
  --k-hops 2 \
  --training-paradigm epoch \
  --epochs 1000 \
  --test-every-n-epochs 10 \
  --batch-size 256 \
  --lr 0.1 \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --size 224 \
  --num-workers 4 \
  --cache "$CACHE_ROOT" \
  --output "$TASK_OUTPUT" \
  -v 2>&1 | tee "$TASK_OUTPUT/task.log"

"$PYTHON" - "$TASK_OUTPUT" "$REPO_ROOT" "$AUTHOR_ROOT" "$DATASET" "$METHOD" "$RATIO" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

out = Path(sys.argv[1])
repo = Path(sys.argv[2])
author = Path(sys.argv[3])
dataset = sys.argv[4]
method = sys.argv[5]
ratio = float(sys.argv[6])

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def commit(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()

files = sorted(p for p in out.rglob('*') if p.is_file())
record = {
    'dataset': dataset,
    'method': method,
    'ratio': ratio,
    'seed': 42,
    'fork_commit': commit(repo),
    'author_commit': commit(author),
    'files': {str(p.relative_to(out)): sha(p) for p in files if p.name != 'provenance.json'},
}
(out / 'provenance.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps(record, indent=2, sort_keys=True))
PY
