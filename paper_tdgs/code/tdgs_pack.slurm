#!/bin/bash
# Packed runner for the TDGS matrix, THREE-ACCOUNT edition.
#
# DIFFERENCES FROM td_pack.slurm (which this is derived from):
#
#  1. umask 002. All three accounts write into ONE shared result tree on
#     /mnt/prj01/hgrp-1502-5TB (group hgrp-1502, setgid 2770). Without umask 002
#     the first account to create a directory makes it g-w and the other two fail
#     halfway through the matrix.
#
#  2. gres=gpu:2 with --array=0-4 per account. qos-high-gpu allows 12 GPUs but
#     only 5 RUNNING JOBS per user, so 5 jobs x 2 GPUs = 10 GPUs is the shape
#     that maximises JOB count at 15 across three accounts (the stated target).
#     If GPU count matters more than job count, use --array=0-3 with NGPU=3
#     (4 jobs x 3 = 12 GPUs, the true per-user ceiling, 36 slots total).
#
#  3. Cross-account deduplication is free. td_run_one.sh skips any cell that
#     already has predictions_clean.npz, and the path is under the SHARED
#     TD_EXP_ROOT -- so a cell finished by qiangzeng is skipped by danranwang
#     without any coordinator, lock file, or post-hoc merge. This is why the
#     shared tree matters: with per-account trees, the same cell would be
#     trained up to three times and the merge step could pick either copy.
#
#  4. Every account is given the SAME worklist in the SAME order, and shards it
#     by (account_offset + array_task) so the three accounts do not start on the
#     same cells and immediately SKIP each other.
#
# Usage (normally via launch_3accounts.sh):
#   WORKLIST=... SHARD=0 NSHARD=3 NGPU=2 TD_SEL_DIR=... TD_EXP_ROOT=... \
#   sbatch --array=0-4 tdgs_pack.slurm
#
# Worklist format, one cell per line:  <dataset> <ratio> <arm> <seed>
#SBATCH --job-name=tdgspack
#SBATCH --partition=gpu-a100
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=hpcgpu108
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -uo pipefail          # NOT -e: one failed cell must not kill the pack
umask 002                 # inherited by every child, including the python run

: "${WORKLIST:?WORKLIST is required}"
: "${TD_EXP_ROOT:?TD_EXP_ROOT is required (must be the SHARED tree)}"
: "${TD_SEL_DIR:?TD_SEL_DIR is required}"
export TD_EXP_ROOT TD_SEL_DIR
export TD_CACHE_ROOT=${TD_CACHE_ROOT:-$HOME/tdgs_cache}
export TD_HF_ROOT=${TD_HF_ROOT:-/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_20260922/hf_cache}
export SEL_WAIT=${SEL_WAIT:-600}

NGPU=${NGPU:-2}
SHARD=${SHARD:-0}
NSHARD=${NSHARD:-1}
RUNNER=${RUNNER:-$HOME/td_run_one.sh}
TASK=${SLURM_ARRAY_TASK_ID:-0}
NTASKS=${SLURM_ARRAY_TASK_COUNT:-1}

mapfile -t ALL < <(grep -vE '^\s*(#|$)' "$WORKLIST")
NALL=${#ALL[@]}

# Global stride: every (shard, task) pair gets a distinct residue class over the
# whole worklist, so the 15 jobs tile the matrix without overlap and all of them
# begin near the FRONT of the list -- which is ordered by scientific priority, so
# an early stop still leaves P0 complete.
STRIDE=$((NSHARD * NTASKS))
OFFSET=$((SHARD * NTASKS + TASK))

echo "host=$(hostname) user=$(whoami) shard=$SHARD/$NSHARD task=$TASK/$NTASKS"
echo "worklist=$NALL cells  stride=$STRIDE offset=$OFFSET  ngpu=$NGPU"
echo "exp_root=$TD_EXP_ROOT"
echo "sel_dir=$TD_SEL_DIR"
echo "start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

mkdir -p "$TD_CACHE_ROOT"
MINE=()
for ((i = OFFSET; i < NALL; i += STRIDE)); do MINE+=("${ALL[$i]}"); done
echo "this job owns ${#MINE[@]} cells"

worker() {
    local g=$1 ok=0 fail=0 skip=0
    for ((j = g; j < ${#MINE[@]}; j += NGPU)); do
        read -r ds ratio arm seed <<< "${MINE[$j]}"
        local t0=$SECONDS
        out=$(CUDA_VISIBLE_DEVICES=$g "$RUNNER" "$ds" "$ratio" "$arm" "$seed" 2>&1)
        rc=$?
        if   [[ $rc -ne 0 ]]; then
            fail=$((fail+1))
            echo "[gpu$g] FAIL rc=$rc $ds r$ratio $arm s$seed"
            echo "$out" | tail -20
        elif grep -q '^SKIP' <<< "$out"; then
            skip=$((skip+1)); echo "[gpu$g] $(grep '^SKIP' <<< "$out")"
        else
            ok=$((ok+1))
            echo "[gpu$g] ok $ds r$ratio $arm s$seed  ($((SECONDS-t0))s)"
        fi
    done
    echo "[gpu$g] finished: ok=$ok skip=$skip fail=$fail"
}

for ((g = 0; g < NGPU; g++)); do worker "$g" & done
wait
echo "shard=$SHARD task=$TASK done=$(date -Is)"
