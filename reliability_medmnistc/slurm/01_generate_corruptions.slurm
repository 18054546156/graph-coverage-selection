#!/bin/bash
# Generate all supported MedMNIST-C files from verified 224-pixel clean data.
#SBATCH --job-name=medmnistc-generate
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/medmnistc_generate_%j.out
#SBATCH --error=logs/medmnistc_generate_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
PYTHON="${PYTHON:-python}"
MEDMNIST_ROOT="${MEDMNIST_ROOT:-$REPO_ROOT/data/medmnist}"
MEDMNISTC_ROOT="${MEDMNISTC_ROOT:-$REPO_ROOT/data/medmnistc}"
cd "$REPO_ROOT"
exec "$PYTHON" reliability_medmnistc/scripts/generate_medmnistc.py \
  --medmnist-root "$MEDMNIST_ROOT" \
  --output-root "$MEDMNISTC_ROOT" \
  --source-root "$REPO_ROOT/third_party/medmnistc"
