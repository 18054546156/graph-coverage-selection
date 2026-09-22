#!/bin/bash
# One isolated end-to-end GPU smoke: selection -> train -> clean/MedMNIST-C evaluation.
#SBATCH --job-name=medmnistc-smoke
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/integration_smoke_%j.out
#SBATCH --error=logs/integration_smoke_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
PYTHON="${PYTHON:-python}"
SMOKE_ROOT="${SMOKE_ROOT:-/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/smoke_outputs/clean_branch_p0p1_v2}"
cd "$REPO_ROOT"
source "$REPO_ROOT/reliability_medmnistc/slurm/common_env.sh"
configure_imagemagick

export MEDMNIST_ROOT="${MEDMNIST_ROOT:-/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/data/medmnist}"
export MEDMNISTC_ROOT="${MEDMNISTC_ROOT:-/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/data/medmnistc}"
export RELIABILITY_OUT="$SMOKE_ROOT/runs"
export SELECTION_OUT="$SMOKE_ROOT/selections"
export GRAPH_CACHE="$SMOKE_ROOT/cache"
export DATASETS=pathmnist
export METHODS=random
export RATIOS=0.02
export TRAINING_SEEDS=42
export EPOCHS=1
export SMOKE_N=1000
export RUN_FULL_TRAIN=0
export ALLOW_MISSING_CLASSES=1
export AUTO_CONSOLIDATE=0

"$PYTHON" reliability_medmnistc/scripts/select.py \
  --config reliability_medmnistc/configs/full_5datasets_8methods.yaml
"$PYTHON" reliability_medmnistc/scripts/train.py \
  --config reliability_medmnistc/configs/full_5datasets_8methods.yaml
"$PYTHON" reliability_medmnistc/scripts/evaluate.py \
  --config reliability_medmnistc/configs/full_5datasets_8methods.yaml
"$PYTHON" reliability_medmnistc/scripts/summarize.py \
  --config reliability_medmnistc/configs/full_5datasets_8methods.yaml

RUN_DIR="$SMOKE_ROOT/runs/pathmnist/random/ratio_0.02/selection_seed_42/train_seed_42/aug_0"
test -f "$RUN_DIR/final.pt"
test -f "$RUN_DIR/training_history.json"
test -f "$RUN_DIR/run_complete.json"
RUN_DIR="$RUN_DIR" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
completion = json.loads((run_dir / "run_complete.json").read_text())
rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line]
prediction_files = sorted(run_dir.glob("predictions_*.npz"))
assert completion["status"] == "complete"
assert completion["conditions"] == 56, completion
assert len(rows) == 56, len(rows)
assert len(prediction_files) == 12, [path.name for path in prediction_files]
assert {row["corruption"] for row in rows if row["severity"]} == set(completion["prediction_sha256"]) - {"clean"}
print({"conditions": len(rows), "prediction_files": len(prediction_files)})
PY
echo "integration smoke complete: $SMOKE_ROOT"
