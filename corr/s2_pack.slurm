#!/bin/bash
#SBATCH --job-name=s2pack

#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu-h100,gpu-a100
#SBATCH --output=s2pack_%j.log
# S2 work-stealing packer: one training per allocated GPU. Cells come from
# $CODE/s2_worklist.tsv and are claimed with an atomic `mkdir` under
# $S2_EXP/claims, so packers on different accounts / GPU types share one queue.
# s2_train_one.sh is idempotent (skips cells that already have predictions).
set -uo pipefail
umask 002
CODE=${CODE:-/mnt/prj01/hgrp-1502-5TB/tdgs_shared/code/shiftbench}
S2_EXP=/mnt/prj01/hgrp-1502-5TB/tdgs_shared/runs/s2_coloraug_20261002
LIST=$CODE/s2_worklist.tsv
CL=$S2_EXP/claims
LOGS=$S2_EXP/logs
mkdir -p "$CL" "$LOGS" "$HOME/tdgs_cache"
# luhpc does not isolate GPUs: nvidia-smi shows all cards and CUDA_VISIBLE_DEVICES
# holds the ALLOCATED physical ids. Map worker g -> ${GPUS[g]} (memory:
# luhpc-gpu-mapping-no-isolation); never use nvidia-smi -L or a bare index.
IFS=, read -ra ALLOC <<< "${CUDA_VISIBLE_DEVICES:?no GPUs allocated}"
# Probe each allocated card; a dead one (hpcgpu108 history) would fail every cell.
PY=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/envs/medmnistc-py311/bin/python
GPUS=()
for dev in "${ALLOC[@]}"; do
    if CUDA_VISIBLE_DEVICES=$dev "$PY" -c "import torch; torch.zeros(1).cuda()" > /dev/null 2>&1; then
        GPUS+=("$dev")
    else
        echo "DEAD GPU $dev on $(hostname), skipping"
    fi
done
NGPU=${#GPUS[@]}
(( NGPU > 0 )) || { echo "no usable GPU"; exit 3; }
echo "host=$(hostname) user=$(whoami) job=${SLURM_JOB_ID:-} allocated=${CUDA_VISIBLE_DEVICES} start=$(date -Is)"

worker() {
    local gpu=$1
    while IFS=$'\t' read -r ds ratio arm seed sel; do
        [[ -z ${ds:-} ]] && continue
        local key="${ds}_r${ratio}_${arm}_s${seed}"
        mkdir "$CL/$key" 2>/dev/null || continue
        echo "$(hostname) $(whoami) gpu$gpu job${SLURM_JOB_ID:-} $(date -Is)" > "$CL/$key/owner"
        if CUDA_VISIBLE_DEVICES=$gpu bash "$CODE/s2_train_one.sh" "$ds" "$ratio" "$arm" "$seed" "$sel" \
                > "$LOGS/$key.log" 2>&1; then
            echo "ok   gpu$gpu $key $(date -Is)"; touch "$CL/$key/done"
        else
            echo "FAIL gpu$gpu $key (see $LOGS/$key.log) $(date -Is)"; touch "$CL/$key/failed"
        fi
    done < "$LIST"
}

# Colour jitter makes training CPU-bound (GPU ~idle, 10 GB of 80 used), so run
# WORKERS_PER_GPU trainings per card; each is an independent deterministic process.
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
for ((g = 0; g < NGPU; g++)); do
    for ((w = 0; w < WORKERS_PER_GPU; w++)); do worker "${GPUS[$g]}" & done
done
wait
echo "done=$(date -Is)"
