#!/bin/bash
# Controlled OrganSMNIST 2% reproduction audit.
# Fixed selection_seed=42; Random/Graph-A2; training_seed=42..46.
#SBATCH --job-name=organ-repro
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --qos=YOUR_SLURM_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-9%5
#SBATCH --output=/project/PROJECT_ROOT/USER/reliability_medmnistc_ab/formal_runs/repro_organsmnist_2pct_random_grapha2_d74_train5_20260918/logs/organ_repro_%A_%a.out
#SBATCH --error=/project/PROJECT_ROOT/USER/reliability_medmnistc_ab/formal_runs/repro_organsmnist_2pct_random_grapha2_d74_train5_20260918/logs/organ_repro_%A_%a.err

set -euo pipefail

ROOT=/project/PROJECT_ROOT/USER/reliability_medmnistc_ab
REPO_ROOT=$ROOT/sources/graph-coverage-selection-medmnistc-reliability-clean-d74
OLD_FORMAL=$ROOT/formal_runs/single_seed_sel42_train42_20260915
AUDIT_ROOT=$ROOT/formal_runs/repro_organsmnist_2pct_random_grapha2_d74_train5_20260918
PYTHON=$ROOT/envs/medmnistc-py311/bin/python
CONFIG=$REPO_ROOT/reliability_medmnistc/configs/full_5datasets_8methods.yaml

cd "$REPO_ROOT"
source "$REPO_ROOT/reliability_medmnistc/slurm/common_env.sh"
configure_imagemagick
# The shared checkout is owned by HPC_USER_A. Register both the symlink path
# and its resolved NFS path for HPC_USER_B's Git safety check.
git config --global --add safe.directory "$REPO_ROOT" 2>/dev/null || true
git config --global --add safe.directory "$(readlink -f "$REPO_ROOT")" 2>/dev/null || true

TASK_ID=${SLURM_ARRAY_TASK_ID:?array task id is required}
if (( TASK_ID < 5 )); then
  METHOD=random
  TRAINING_SEED=$((42 + TASK_ID))
else
  METHOD=graph_a2
  TRAINING_SEED=$((42 + TASK_ID - 5))
fi

export REPO_ROOT
export METHODS=$METHOD
export DATASETS=organsmnist
export RATIOS=0.02
export SELECTION_SEED=42
export TRAINING_SEEDS=$TRAINING_SEED
export DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export RUN_FULL_TRAIN=0
export AUTO_CONSOLIDATE=0
export MEDMNIST_ROOT=$ROOT/data/medmnist
export MEDMNISTC_ROOT=$ROOT/data/medmnistc
export SELECTION_OUT=$OLD_FORMAL/selections
export GRAPH_CACHE=$OLD_FORMAL/cache
export RELIABILITY_OUT=$AUDIT_ROOT/runs
export GRAPH_ROOT=$REPO_ROOT
export MEDC_ROOT=$REPO_ROOT/third_party/medmnistc

echo "task=$TASK_ID method=$METHOD ratio=0.02 selection_seed=42 training_seed=$TRAINING_SEED"
echo "repo_commit=$(git rev-parse HEAD)"

"$PYTHON" reliability_medmnistc/scripts/validate_selections.py \
  --config "$CONFIG" --phase validate
"$PYTHON" reliability_medmnistc/scripts/train.py \
  --config "$CONFIG"
"$PYTHON" reliability_medmnistc/scripts/evaluate.py \
  --config "$CONFIG"

"$PYTHON" - "$ROOT" "$OLD_FORMAL" "$AUDIT_ROOT" "$REPO_ROOT" "$METHOD" "$TRAINING_SEED" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

root, old_formal, audit_root, repo_root = map(Path, sys.argv[1:5])
method = sys.argv[5]
train_seed = int(sys.argv[6])
selection_dir = old_formal / "selections" / method / "organsmnist" / "ratio_0.02" / "selection_seed_42"
run_dir = audit_root / "runs" / "organsmnist" / method / "ratio_0.02" / "selection_seed_42" / f"train_seed_{train_seed}" / "aug_0"

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

required = [
    selection_dir / "selected_indices.npy",
    selection_dir / "selected_local_indices.npy",
    selection_dir / "index_order_sha256.txt",
    selection_dir / "class_counts.json",
    selection_dir / "selection_config.json",
    run_dir / "run_config.json",
    run_dir / "training_history.json",
    run_dir / "metrics.jsonl",
    run_dir / "final.pt",
    run_dir / "predictions_clean.npz",
    run_dir / "run_complete.json",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("missing required outputs: " + ", ".join(missing))

selected = np.load(selection_dir / "selected_indices.npy", allow_pickle=False)
local_selected = np.load(selection_dir / "selected_local_indices.npy", allow_pickle=False)
labels = np.asarray(np.load(root / "data" / "medmnist" / "organsmnist.npz", allow_pickle=False)["train_labels"]).reshape(-1)
class_counts = {str(int(label)): int((labels[selected] == label).sum()) for label in np.unique(labels)}
record = {
    "dataset": "organsmnist",
    "method": method,
    "ratio": 0.02,
    "selection_seed": 42,
    "training_seed": train_seed,
    "code_commit": os.popen(f"git -C {repo_root} rev-parse HEAD").read().strip(),
    "selection_dir": str(selection_dir),
    "run_dir": str(run_dir),
    "selection_index_order_sha256_recorded": (selection_dir / "index_order_sha256.txt").read_text(encoding="utf-8").strip(),
    "selection_indices_npy_sha256": sha256_file(selection_dir / "selected_indices.npy"),
    "selection_local_indices_npy_sha256": sha256_file(selection_dir / "selected_local_indices.npy"),
    "selection_n": int(selected.size),
    "selection_local_n": int(local_selected.size),
    "class_counts_recorded": load_json(selection_dir / "class_counts.json"),
    "class_counts_recomputed": class_counts,
    "selection_config": load_json(selection_dir / "selection_config.json"),
    "checkpoint_sha256": sha256_file(run_dir / "final.pt"),
    "prediction_sha256": {path.name: sha256_file(path) for path in sorted(run_dir.glob("predictions_*.npz"))},
    "training_history_sha256": sha256_file(run_dir / "training_history.json"),
    "metrics_sha256": sha256_file(run_dir / "metrics.jsonl"),
    "run_config": load_json(run_dir / "run_config.json"),
    "run_complete": load_json(run_dir / "run_complete.json"),
}
if record["class_counts_recorded"] != record["class_counts_recomputed"]:
    raise SystemExit("class count mismatch between artifact and recomputation")
if record["selection_n"] != record["selection_local_n"]:
    raise SystemExit("global/local selected index lengths differ")
out = audit_root / "audit" / f"{method}_train_seed_{train_seed}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
print("audit_record=", out)
print("checkpoint_sha256=", record["checkpoint_sha256"])
print("prediction_files=", len(record["prediction_sha256"]))
PY
