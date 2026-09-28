#!/bin/bash
#SBATCH --job-name=tdgssel
#SBATCH --partition=gpu-a100
#SBATCH --qos=qos-high-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --exclude=hpcgpu108
#SBATCH --output=tdgssel_%j.log
# Stage all five TDGS arms for the datasets named in DS (default: the four
# smaller pools; tissuemnist gets its own job because its k=50 A+A^2 kernel is
# ~615M nonzeros and the demand kernel adds ~250M more).
#
# Selections land in ~/tdgs_sel, NOT ~/td_sel, so this cannot collide with the
# concurrent session's staging.
#
# Usage: DS="bloodmnist organsmnist organamnist pathmnist" sbatch tdgs_stage.slurm
set -uo pipefail
PY=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/envs/medmnistc-py311/bin/python
DS=${DS:-"bloodmnist organsmnist organamnist pathmnist"}
RATIO=${RATIO:-0.02}
cd "$HOME/tdgs"
echo "host=$(hostname) ds='$DS' ratio=$RATIO start=$(date -Is)"
for d in $DS; do
    echo "################ $d ################"
    "$PY" tdgs_select.py --dataset "$d" --ratio "$RATIO" \
          --out-dir "$HOME/tdgs_sel" --report "$HOME/tdgs_report" \
        || echo "FAILED $d"
done
echo "done=$(date -Is)"
