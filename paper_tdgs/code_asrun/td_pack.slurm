#!/bin/bash
# Packed runner for the TD-Cover matrix: N trainings per slurm job, one per GPU.
#
# WHY PACKED. qos-high-gpu allows a user 12 GPUs, 256G and 128 cpus, but only
# 5 RUNNING JOBS. One training needs 1 GPU, ~4GB and 4 cpus. Submitting one job
# per training therefore caps the account at 5 concurrent trainings and leaves
# 7 GPUs idle. Packing 3 per job reaches 4 jobs x 3 GPUs = 12 GPUs, the true
# ceiling, and triples throughput per account.
#
# WORK DISTRIBUTION. Each array task takes every NTASKS-th line of the worklist,
# then its GPU workers take every NGPU-th of those. No shared state, no lock
# file, no coordinator -- a task that dies takes only its own share down, and
# re-submitting the whole array is safe because td_run_one.sh skips cells that
# already have predictions.
#
# Usage:
#   WORKLIST=~/tdwork.txt NGPU=3 sbatch --array=0-3 td_pack.slurm
#
# Worklist format, one cell per line:  <dataset> <ratio> <arm> <seed>
#SBATCH --job-name=tdpack
#SBATCH --partition=gpu-a100
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=18
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=hpcgpu108
#SBATCH --output=tdpack_%A_%a.out
#SBATCH --error=tdpack_%A_%a.err

set -uo pipefail          # NOT -e: one failed cell must not kill the whole pack

: "${WORKLIST:?WORKLIST is required}"
NGPU=${NGPU:-3}
RUNNER=${RUNNER:-$HOME/td_run_one.sh}
TASK=${SLURM_ARRAY_TASK_ID:-0}
NTASKS=${SLURM_ARRAY_TASK_COUNT:-1}

mapfile -t ALL < <(grep -vE '^\s*(#|$)' "$WORKLIST")
echo "host=$(hostname) task=$TASK/$NTASKS ngpu=$NGPU worklist=${#ALL[@]} cells start=$(date -Is)"
nvidia-smi --query-gpu=index,name --format=csv,noheader || true

# this task's share of the worklist
MINE=()
for ((i = TASK; i < ${#ALL[@]}; i += NTASKS)); do MINE+=("${ALL[$i]}"); done
echo "this task owns ${#MINE[@]} cells"

worker() {
    local g=$1 ok=0 fail=0 skip=0
    for ((j = g; j < ${#MINE[@]}; j += NGPU)); do
        read -r ds ratio arm seed <<< "${MINE[$j]}"
        out=$(CUDA_VISIBLE_DEVICES=$g "$RUNNER" "$ds" "$ratio" "$arm" "$seed" 2>&1)
        rc=$?
        if   [[ $rc -ne 0 ]];              then fail=$((fail+1)); echo "[gpu$g] FAIL rc=$rc $ds r$ratio $arm s$seed"; echo "$out" | tail -15
        elif grep -q '^SKIP' <<< "$out";   then skip=$((skip+1)); echo "[gpu$g] $(grep '^SKIP' <<< "$out")"
        else ok=$((ok+1)); echo "[gpu$g] ok $ds r$ratio $arm s$seed"; fi
    done
    echo "[gpu$g] finished: ok=$ok skip=$skip fail=$fail"
}

for ((g = 0; g < NGPU; g++)); do worker "$g" & done
wait
echo "task=$TASK done=$(date -Is)"
