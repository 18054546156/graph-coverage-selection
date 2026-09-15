#!/bin/bash
# Validate the clean checkout, pinned submodule, and YAML configuration.
#SBATCH --job-name=medmnistc-preflight
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/medmnistc_preflight_%j.out
#SBATCH --error=logs/medmnistc_preflight_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
PYTHON="${PYTHON:-python}"
cd "$REPO_ROOT"
source "$REPO_ROOT/reliability_medmnistc/slurm/common_env.sh"
configure_imagemagick
git merge-base --is-ancestor 8cf757adc4c333dc1427d511f0de2f246d15ebac HEAD
git diff --quiet 8cf757adc4c333dc1427d511f0de2f246d15ebac -- graphcov
test "$(git -C third_party/medmnistc rev-parse HEAD)" = "8acfd2710c6e0e8b2745be8b1fa1c17b94b8f8a7"
"$PYTHON" -m py_compile reliability_medmnistc/reliability/pipeline.py reliability_medmnistc/scripts/*.py
"$PYTHON" -c 'import yaml; print("PyYAML", yaml.__version__)'
echo "preflight OK"
