#!/bin/bash
# GPU-only preflight for Table 1 and v11. This job performs no selection or training.

#SBATCH --job-name=graphcov_check
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --qos=YOUR_SLURM_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/project/PROJECT_ROOT/USER/graph_bench/table1_reproduction/logs/preflight_%j.out
#SBATCH --error=/project/PROJECT_ROOT/USER/graph_bench/table1_reproduction/logs/preflight_%j.err

set -euo pipefail

PROJECT=${GRAPHCOV_PROJECT_ROOT:-/project/PROJECT_ROOT/USER/graph_bench}
PYTHON=${GRAPHCOV_PYTHON:-/project/PROJECT_ROOT/USER/graph_select/envs/graphcov-py311/bin/python}
TMP_ROOT=${TMPDIR:-/tmp}/graphcov_preflight_${SLURM_JOB_ID}

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p "${TMP_ROOT}"
cd "${PROJECT}"
test -x "${PYTHON}"

echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date --iso-8601=seconds)"
"${PYTHON}" - <<'PY'
from pathlib import Path
import faiss
import torch

from table1_reproduction.official_runtime import verify_official_vendor

assert torch.cuda.is_available(), "PyTorch cannot see the allocated GPU"
assert faiss.get_num_gpus() >= 1, "FAISS cannot see the allocated GPU"
verify_official_vendor()

home = Path.home()
for dataset in ("organsmnist", "organamnist", "pathmnist", "tissuemnist", "bloodmnist"):
    for suffix in ("", "_224"):
        archive = home / ".medmnist" / f"{dataset}{suffix}.npz"
        assert archive.is_file() and archive.stat().st_size > 0, f"missing archive: {archive}"

hf_cache = home / ".cache" / "huggingface" / "hub" / "models--MahmoodLab--UNI"
assert hf_cache.is_dir(), f"missing UNI cache: {hf_cache}"
print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
print(f"faiss={faiss.__version__} faiss_gpus={faiss.get_num_gpus()}")
print("vendor, datasets, and UNI cache: OK")
PY

"${PYTHON}" -m compileall -q \
  table1_reproduction/experiments table1_reproduction/official_runtime.py \
  v11/experiments v11/methods

"${PYTHON}" table1_reproduction/experiments/job1_select.py \
  --config table1_reproduction/configs/job1_table1.json --dry-run > "${TMP_ROOT}/table1_job1.json"
"${PYTHON}" table1_reproduction/experiments/job2_downstream.py \
  --config table1_reproduction/configs/job2_table1.json --dry-run > "${TMP_ROOT}/table1_job2.json"
"${PYTHON}" v11/experiments/job1_select.py \
  --config v11/configs/job1_table1_calibration.json --dry-run > "${TMP_ROOT}/v11_job1.json"
"${PYTHON}" v11/experiments/job2_downstream.py \
  --config v11/configs/job2_table1_validation_seed42.json --dry-run > "${TMP_ROOT}/v11_val1.json"
"${PYTHON}" v11/experiments/job2_downstream.py \
  --config v11/configs/job2_table1_validation_3seeds.json --dry-run > "${TMP_ROOT}/v11_val3.json"

"${PYTHON}" - "${TMP_ROOT}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected = {
    "table1_job1.json": 160,
    "table1_job2.json": 400,
    "v11_val1.json": 25,
    "v11_val3.json": 75,
}
for name, count in expected.items():
    jobs = json.loads((root / name).read_text(encoding="utf-8"))
    assert len(jobs) == count, f"{name}: {len(jobs)} != {count}"
    print(f"{name}: {len(jobs)} jobs")

v11_job1 = json.loads((root / "v11_job1.json").read_text(encoding="utf-8"))
selection_count = sum(len(job["ratios"]) * len(job["variants"]) for job in v11_job1)
assert len(v11_job1) == 5, f"v11_job1.json: {len(v11_job1)} dataset jobs != 5"
assert selection_count == 100, f"v11 Job 1 selections: {selection_count} != 100"
print(f"v11_job1.json: {len(v11_job1)} dataset jobs, {selection_count} selections")
PY

echo "PREFLIGHT PASSED end=$(date --iso-8601=seconds)"
