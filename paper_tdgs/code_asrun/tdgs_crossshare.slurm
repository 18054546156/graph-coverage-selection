#!/bin/bash
#SBATCH --job-name=tdgsxshare
#SBATCH --partition=gpu-a100
#SBATCH --qos=qos-high-gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --exclude=hpcgpu108
#SBATCH --output=%x_%A_%a.out
#
# Per-class cross-class edge share, one array task per dataset. No training, no
# probe, NO GPU REQUEST -- graphcov's build_knn_graph falls back to faiss-CPU.
#
# WHY NOT A CPU PARTITION. cpu-amd9754 is idle with 1.5TB nodes, but this account
# is only associated with qos-high-gpu and qos-normal, and qos-normal caps the
# user at mem=4G. So qos-high-gpu is the only usable queue even for CPU work.
#
# WHY IT QUEUES. qos-high-gpu allows 5 RUNNING jobs per user and the 15 round-2
# training jobs already hold 5 on each of the three accounts. This array is
# therefore submitted to wait and starts as training array tasks drain. That is
# fine: nothing downstream is blocked by it today, and dropping a training task
# to free a slot would orphan that shard's cells until the next relaunch.
#
# 100G, not 240G: the peak is the A_sym @ A_sym product for tissuemnist, whose
# output kernel is ~615M nonzeros (~5GB) with a comparable scipy workspace.
set -uo pipefail
umask 002
S=/mnt/prj01/hgrp-1502-5TB/tdgs_shared
PY=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/envs/medmnistc-py311/bin/python
DSL=(pathmnist organamnist bloodmnist organsmnist tissuemnist)
D=${DSL[${SLURM_ARRAY_TASK_ID:-0}]}
cd "$S/code"
echo "host=$(hostname) ds=$D cpus=${SLURM_CPUS_PER_TASK:-?} start=$(date -Is)"
"$PY" perclass_crossshare.py --dataset "$D" --out-dir "$S/archive/round2/crossshare"
echo "done=$(date -Is)"
