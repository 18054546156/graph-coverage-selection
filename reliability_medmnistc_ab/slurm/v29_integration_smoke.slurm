#!/bin/bash
#SBATCH --job-name=v29-int-smoke
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/logs/v29_integration_smoke_%j.out
#SBATCH --error=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/logs/v29_integration_smoke_%j.err

set -euo pipefail

ROOT=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab
SRC="$ROOT/sources/graph-coverage-selection-codex-reliability-audit-v29"
NB="$SRC/reliability_full_experiment.ipynb"
PY="$ROOT/envs/medmnistc-py311/bin/python"
DATA_ROOT="$ROOT/data/medmnist"
CORR_ROOT="$ROOT/data/medmnistc"
RUN_ROOT="$ROOT/smoke_outputs/v29_integration/pathmnist_seed42"
CACHE_ROOT="$ROOT/smoke_outputs/v29_integration/cache"
SELECTION_ROOT="$ROOT/smoke_outputs/v29_integration/selections"

mkdir -p "$ROOT/logs" "$RUN_ROOT" "$CACHE_ROOT" "$SELECTION_ROOT"
export MAGICK_HOME="$ROOT/envs/imagemagick"
export LD_LIBRARY_PATH="$MAGICK_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$MAGICK_HOME/bin:$PATH"
export WAND_MAGICK_LIBRARY_SUFFIX="-7.Q16HDRI"
export LD_PRELOAD="$MAGICK_HOME/lib/libMagickCore-7.Q16HDRI.so:$MAGICK_HOME/lib/libMagickWand-7.Q16HDRI.so${LD_PRELOAD:+:$LD_PRELOAD}"
cd "$SRC"

test "$(git rev-parse HEAD)" = "7f5f2e3bdcb14b8d3750700dcd3951be3871ee3a"
test "$(git -C repos/graphcov rev-parse HEAD)" = "8cf757adc4c333dc1427d511f0de2f246d15ebac"
test "$(git -C repos/medmnistc rev-parse HEAD)" = "8acfd2710c6e0e8b2745be8b1fa1c17b94b8f8a7"

echo "===== v29 integration smoke ====="
date -Is
hostname
git rev-parse HEAD
git -C repos/graphcov rev-parse HEAD
git -C repos/medmnistc rev-parse HEAD
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$PY" -c 'import sys, torch, torchvision, medmnist; print(sys.version); print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available()); print("torchvision", torchvision.__version__, "medmnist", medmnist.__version__)'

# Reuse the verified full UNI cache for the first SMOKE_N source rows. The
# dynamics cache is intentionally not preloaded: DYNAMICS_EPOCHS=1 must run.
"$PY" - "$CACHE_ROOT" <<'PY'
from pathlib import Path
import hashlib
import json
import numpy as np
import os

cache_root = Path(os.environ.get("CACHE_ROOT", __import__("sys").argv[1]))
namespace = cache_root / "img224_dyn1_sel42_smoke1000"
namespace.mkdir(parents=True, exist_ok=True)
source = Path("/project/prj-sis01/xuxiaoyu/graph_select/cache/embeddings/pathmnist_train_uni_224.npz")
with np.load(source, allow_pickle=False) as package:
    embeddings = np.asarray(package["embeddings"][:1000], dtype=np.float32)
target = namespace / "pathmnist_train_uni_224.npz"
np.savez_compressed(target, embeddings=embeddings)
manifest = {
    "source": str(source),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "target": str(target),
    "shape": list(embeddings.shape),
    "dtype": str(embeddings.dtype),
    "rows": 1000,
}
(namespace / "smoke_embedding_cache_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

export PREP_ROOT="$SRC"
export GRAPH_ROOT="$SRC/repos/graphcov"
export MEDC_ROOT="$SRC/repos/medmnistc"
export MEDMNIST_ROOT="$DATA_ROOT"
export MEDMNISTC_ROOT="$CORR_ROOT"
export RELIABILITY_OUT="$RUN_ROOT"
export GRAPH_CACHE="$CACHE_ROOT"
export SELECTION_OUT="$SELECTION_ROOT"
export DATASETS=pathmnist
export RATIOS=0.05
export SELECTION_SEED=42
export TRAINING_SEEDS=42
export EPOCHS=1
export DYNAMICS_EPOCHS=1
export SMOKE_N=1000
export BATCH_SIZE=256
export NUM_WORKERS=4
export EMBEDDING_SOURCE=uni
export AUGMENT=0
export RUN_FULL_TRAIN=0
export CORR_HASH=0
export DOWNLOAD_MEDMNIST=0

echo "===== experiment environment ====="
env | grep -E '^(PREP_ROOT|GRAPH_ROOT|MEDC_ROOT|MEDMNIST|RELIABILITY|GRAPH_CACHE|SELECTION_OUT|DATASETS|RATIOS|SELECTION_SEED|TRAINING_SEEDS|EPOCHS|DYNAMICS_EPOCHS|SMOKE_N|BATCH_SIZE|NUM_WORKERS|EMBEDDING_SOURCE|AUGMENT|RUN_FULL_TRAIN|CORR_HASH|DOWNLOAD_MEDMNIST)=' | sort

# The published artifact is a notebook. Execute its code cell directly with
# the pinned Python so no notebook server or untracked core-code wrapper is
# involved in the experiment.
"$PY" - "$NB" <<'PY'
import json
import sys
from pathlib import Path

notebook = Path(sys.argv[1])
payload = json.loads(notebook.read_text(encoding="utf-8"))
code_cells = ["".join(cell.get("source", [])) for cell in payload["cells"] if cell.get("cell_type") == "code"]
if len(code_cells) != 1:
    raise RuntimeError(f"expected exactly one executable code cell, found {len(code_cells)}")
compiled = compile(code_cells[0], str(notebook), "exec")
namespace = {"__name__": "__main__", "__file__": str(notebook)}
exec(compiled, namespace, namespace)
PY

echo "===== smoke output inventory ====="
find "$RUN_ROOT" -type f -printf '%P %s bytes\n' | sort
echo "===== smoke completed ====="
