#!/bin/bash
#SBATCH --job-name=d8-organsmnist-select-smoke
#SBATCH --account=hpc-sis
#SBATCH --partition=cpu-amd9754
#SBATCH --qos=qos-normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/d8-smoke-%j.out
#SBATCH --error=logs/d8-smoke-%j.err

set -euo pipefail
umask 0002

PROJECT_ROOT=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab
D8_ROOT="$PROJECT_ROOT/d8_class_conditional_ot_20260922"
EMBEDDINGS="$PROJECT_ROOT/table1_clean_only_20260922/cache/embeddings_img224_smokefull/organsmnist_train_uni_224.npz"
DATA="$PROJECT_ROOT/data/medmnist/organsmnist_224.npz"
OUT="$D8_ROOT/smoke/organsmnist_seed0_20260922"
PYTHON="$PROJECT_ROOT/envs/medmnistc-py311/bin/python"

cd "$D8_ROOT"
mkdir -p logs "$OUT"

python - "$EMBEDDINGS" "$DATA" "$OUT" <<'PY'
import sys
from pathlib import Path
for value in sys.argv[1:]:
    if not Path(value).exists():
        raise SystemExit(f"required path does not exist: {value}")
PY

"$PYTHON" -c 'import numpy, scipy; print("numpy", numpy.__version__, "scipy", scipy.__version__)'
"$PYTHON" run_d8_hpc_smoke.py "$EMBEDDINGS" "$DATA" "$OUT"
