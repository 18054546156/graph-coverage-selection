#!/bin/bash
#SBATCH --job-name=tdgssel2
#SBATCH --partition=gpu-a100
#SBATCH --qos=qos-high-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --exclude=hpcgpu108
#SBATCH --output=%x_%A_%a.out
#
# Stage every round-2 selection. ONE ARRAY TASK PER DATASET -- datasets are not
# looped inside a task, because tissuemnist needs the whole 240G on its own: the
# k=50 A+A^2 kernel is ~615M nonzeros and the per-class block kernel for
# a2_perclass adds more on top.
#
# A GPU is requested even though the round-2 arms are probe-free (no OOF head is
# fitted), because this qos is the only one with the memory, and holding one A100
# for ~15 min is cheaper than debugging a partition change. The 5% arms DO fit the
# probe, since tdgs_cls at a new budget is regenerated with the frozen round-1
# selector.
set -uo pipefail
umask 002                     # so all three accounts can read/replace these
S=/mnt/prj01/hgrp-1502-5TB/tdgs_shared
PY=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/envs/medmnistc-py311/bin/python
DSL=(pathmnist organamnist bloodmnist organsmnist tissuemnist)
D=${DSL[${SLURM_ARRAY_TASK_ID:-0}]}
cd "$S/code"
mkdir -p "$S/sel" "$S/archive/round2"
echo "host=$(hostname) ds=$D start=$(date -Is)"

# ---- 2%: the five round-2 arms -------------------------------------------
# --assert-a2-identity PROVES the 2x2 factorial's (no-mask, uniform-weight) cell
# equals graph_a2, instead of assuming it. If that identity fails, the factorial's
# reuse of the already-trained graph_a2 cells is invalid and the job says so.
"$PY" tdgs_select_r2.py --dataset "$D" --ratio 0.02 \
      --out-dir "$S/sel" --report "$S/archive/round2" --assert-a2-identity \
  || echo "FAILED $D r0.02 round2arms"

# ---- 5%: the two headline arms at the second budget ----------------------
"$PY" tdgs_select.py --dataset "$D" --ratio 0.05 --arms graph_a2 tdgs_cls \
      --out-dir "$S/sel" --report "$S/archive/round2" \
  || echo "FAILED $D r0.05 headline"

echo "done=$(date -Is)"
ls -la "$S/sel" | grep "$D" || true
