#!/bin/bash
#SBATCH --job-name=alsel
#SBATCH --partition=gpu-h100,gpu-a100
#SBATCH --qos=qos-high-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --exclude=hpcgpu108
#SBATCH --output=/mnt/prj01/hgrp-1502-5TB/tdgs_shared/acs/logs/%x_%j.out
# E1 of report/proposal_acs_20261002.md: TypiClust / ProbCover / MaxHerding selections (author code, author
# defaults; per-class quota main version + label-free appendix version). Selections only; nothing is trained.
# The GPU is used for the selectors' own distance/kernel computations.
# Submit: sbatch --mem=<..> --export=ALL,DSL="<ds ...>" al_baselines_stage.slurm
set -uo pipefail
umask 002
S=/mnt/prj01/hgrp-1502-5TB/tdgs_shared
PY=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/envs/medmnistc-py311/bin/python
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
E=embeddings_img224_smokefull
cd $S/code
echo "host=$(hostname) start=$(date -Is)"; nvidia-smi --query-gpu=name --format=csv,noheader
sha256sum al_baselines_select.py al_baselines_stage.slurm
for D in $DSL; do
  case $D in
    retinamnist|breastmnist) EMB=$S/ambig/t1cache/$E/${D}_train_uni_224.npz ;;
    dermamnist|octmnist)     EMB=$S/knnf/t1cache/$E/${D}_train_uni_224.npz ;;
    *)                       EMB=$S/t1cache/$E/${D}_train_uni_224.npz ;;
  esac
  for M in ${METHODS:-typiclust probcover maxherding}; do
    "$PY" al_baselines_select.py --dataset $D --method $M --emb $EMB \
        --out-dir $S/acs/al_sel --report $S/acs/al_report || echo "FAILED $D $M"
  done
done
ls $S/acs/al_sel | wc -l
echo "done=$(date -Is)"
