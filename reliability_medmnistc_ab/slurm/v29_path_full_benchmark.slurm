#!/bin/bash
#SBATCH --job-name=v29-path-benchmark
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/logs/v29_path_benchmark_%j.out
#SBATCH --error=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/logs/v29_path_benchmark_%j.err

set -euo pipefail

METHOD="${METHOD:?submit with METHOD=random, herding, fps, or graph_a2}"
case "$METHOD" in
    random|herding|fps|graph_a2) ;;
    *) echo "unsupported METHOD=$METHOD" >&2; exit 2 ;;
esac

RATIO="${RATIO:-0.05}"
RATIO_TAG="${RATIO_TAG:-005}"
case "$RATIO_TAG:$RATIO" in
    001:0.01|005:0.05|010:0.10) ;;
    *)
        echo "unsupported ratio: RATIO=$RATIO RATIO_TAG=$RATIO_TAG (use 001:0.01, 005:0.05, or 010:0.10)" >&2
        exit 2
        ;;
esac

ROOT=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab
SRC="$ROOT/sources/graph-coverage-selection-codex-reliability-audit-v29"
NB="$SRC/reliability_full_experiment.ipynb"
PY="$ROOT/envs/medmnistc-py311/bin/python"
DATA_ROOT="$ROOT/data/medmnist"
CORR_ROOT="$ROOT/data/medmnistc"
OUT="$ROOT/results/raw_metrics/benchmark_pathmnist_${METHOD}_ratio${RATIO_TAG}_sel42_train42"
CACHE_ROOT="$ROOT/artifacts/cache/benchmark_pathmnist_${METHOD}_ratio${RATIO_TAG}_sel42_train42"
SELECTION_ROOT="$ROOT/artifacts/selections/benchmark_pathmnist_${METHOD}_ratio${RATIO_TAG}_sel42_train42"

mkdir -p "$ROOT/logs" "$OUT" "$CACHE_ROOT" "$SELECTION_ROOT"
export MAGICK_HOME="$ROOT/envs/imagemagick"
export LD_LIBRARY_PATH="$MAGICK_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$MAGICK_HOME/bin:$PATH"
export WAND_MAGICK_LIBRARY_SUFFIX="-7.Q16HDRI"
export LD_PRELOAD="$MAGICK_HOME/lib/libMagickCore-7.Q16HDRI.so:$MAGICK_HOME/lib/libMagickWand-7.Q16HDRI.so${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONDONTWRITEBYTECODE=1

cd "$SRC"
test "$(git rev-parse HEAD)" = "7f5f2e3bdcb14b8d3750700dcd3951be3871ee3a"
test "$(git -C repos/graphcov rev-parse HEAD)" = "8cf757adc4c333dc1427d511f0de2f246d15ebac"
test "$(git -C repos/medmnistc rev-parse HEAD)" = "8acfd2710c6e0e8b2745be8b1fa1c17b94b8f8a7"

echo "===== v29 full benchmark ====="
date -Is
hostname
echo "METHOD=$METHOD"
echo "RATIO=$RATIO"
echo "RATIO_TAG=$RATIO_TAG"
echo "SRC=$SRC"
echo "OUT=$OUT"
echo "CACHE_ROOT=$CACHE_ROOT"
echo "SELECTION_ROOT=$SELECTION_ROOT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unset}"
git rev-parse HEAD
git -C repos/graphcov rev-parse HEAD
git -C repos/medmnistc rev-parse HEAD
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$PY" -c 'import sys, torch, torchvision, medmnist; print(sys.version); print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available()); print("torchvision", torchvision.__version__, "medmnist", medmnist.__version__)'

export PREP_ROOT="$SRC"
export GRAPH_ROOT="$SRC/repos/graphcov"
export MEDC_ROOT="$SRC/repos/medmnistc"
export MEDMNIST_ROOT="$DATA_ROOT"
export MEDMNISTC_ROOT="$CORR_ROOT"
export RELIABILITY_OUT="$OUT"
export GRAPH_CACHE="$CACHE_ROOT"
export SELECTION_OUT="$SELECTION_ROOT"
export DATASETS=pathmnist
export RATIOS="$RATIO"
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
export SMOKE_N=0
export CORR_HASH=0
export DOWNLOAD_MEDMNIST=0

if [ "$METHOD" = graph_a2 ] || [ "$METHOD" = herding ] || [ "$METHOD" = fps ]; then
    "$PY" - "$ROOT" "$CACHE_ROOT/img224_dyn200_sel42_smokefull" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

root = Path(sys.argv[1])
namespace = Path(sys.argv[2])
source = Path("/project/prj-sis01/xuxiaoyu/graph_select/cache/embeddings/pathmnist_train_uni_224.npz")
target = namespace / source.name
expected_sha256 = "f04a3b6d1596fdff307f0f00ac89a4fb3ea182664bd0d52a9e0821c7612a2ac5"
manifest = json.loads((root / "protocol/medmnist_download_manifest.json").read_text())
path_data = next(row for row in manifest["datasets"] if row["dataset"] == "pathmnist")
if path_data["n_train"] != 89996 or path_data["md5"] != "2c51a510bcdc9cf8ddb2af93af1eadec":
    raise RuntimeError("PathMNIST data manifest differs from the recorded cache protocol")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if sha256(source) != expected_sha256:
    raise RuntimeError("existing UNI cache hash changed")
with np.load(source, allow_pickle=False) as package:
    if package.files != ["embeddings"] or package["embeddings"].shape != (path_data["n_train"], 1024) or package["embeddings"].dtype != np.float32:
        raise RuntimeError("existing UNI cache does not match full PathMNIST embeddings")
namespace.mkdir(parents=True, exist_ok=True)
if not target.exists():
    tmp = target.with_suffix(".tmp.npz")
    shutil.copyfile(source, tmp)
    os.replace(tmp, target)
if sha256(target) != expected_sha256:
    raise RuntimeError("staged UNI cache hash mismatch")
record = {
    "source": str(source), "target": str(target),
    "sha256": expected_sha256, "shape": [89996, 1024], "dtype": "float32",
    "medmnist_224_md5": path_data["md5"],
    "provenance_note": "Previously extracted UNI cache; row count verified against official MedMNIST, per-image extraction provenance unavailable.",
}
(namespace / "uni_cache_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
print("Reusing verified UNI cache:", json.dumps(record, sort_keys=True), flush=True)
PY
fi

echo "===== experiment configuration ====="
env | grep -E '^(PREP_ROOT|GRAPH_ROOT|MEDC_ROOT|MEDMNIST|RELIABILITY|GRAPH_CACHE|SELECTION_OUT|DATASETS|RATIOS|SELECTION_SEED|TRAINING_SEEDS|EPOCHS|DYNAMICS_EPOCHS|IMAGE_SIZE|BATCH_SIZE|NUM_WORKERS|EMBEDDING_SOURCE|AUGMENT|RUN_FULL_TRAIN|SMOKE_N|CORR_HASH|DOWNLOAD_MEDMNIST)=' | sort

# Run the published notebook body while narrowing its fixed eight-method list
# at runtime. This changes neither the notebook file nor either upstream source.
"$PY" - "$NB" <<'PY'
import json
import os
import sys
from pathlib import Path

notebook = Path(sys.argv[1])
method = os.environ["METHOD"]
payload = json.loads(notebook.read_text(encoding="utf-8"))
code_cells = ["".join(cell.get("source", [])) for cell in payload["cells"] if cell.get("cell_type") == "code"]
if len(code_cells) != 1:
    raise RuntimeError(f"expected exactly one executable code cell, found {len(code_cells)}")
source = code_cells[0]
all_methods = '["random", "el2n_top", "forgetting", "eva", "facility", "fps", "herding", "graph_a2"]'
old = f"METHODS = {all_methods}"
new = f"METHODS = [{method!r}]"
if source.count(old) != 1:
    raise RuntimeError("notebook METHODS declaration changed; refusing an implicit protocol change")
source = source.replace(old, new, 1)
compiled = compile(source, str(notebook) + f"[{method}]", "exec")
namespace = {"__name__": "__main__", "__file__": str(notebook)}
exec(compiled, namespace, namespace)
PY

echo "===== output inventory ====="
find "$OUT" -type f -printf '%P %s bytes\n' | sort
echo "===== benchmark completed: $METHOD ====="
