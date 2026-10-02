#!/bin/bash
#SBATCH --job-name=acssel
#SBATCH --partition=gpu-h100,gpu-a100
#SBATCH --qos=qos-high-gpu
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --exclude=hpcgpu108
#SBATCH --output=/mnt/prj01/hgrp-1502-5TB/tdgs_shared/acs/logs/%x_%j.out
# E0 of report/proposal_acs_20261002.md: ACS selections + identity checks + descriptives. CPU only
# (CPU FAISS, as the W1/ambig staging that produced cls_uni). Nothing is trained.
# Submit:  sbatch --mem=<..> --export=ALL,D=<dataset> acs_stage.slurm   (optionally GPUK=1 + --gres=gpu:1)
set -uo pipefail
umask 002
S=/mnt/prj01/hgrp-1502-5TB/tdgs_shared
PY=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/envs/medmnistc-py311/bin/python
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} GRAPHCOV_USE_FAISS_GPU=0 TDGS_UNI_EMB_DIRS=$S/r3/emb
E=embeddings_img224_smokefull
case $D in
  retinamnist|breastmnist)
    EMB=$S/ambig/t1cache/$E/${D}_train_uni_224.npz; HREF=$S/runs/ambig_20261002/sel/seed_42/herding/$D
    CLS=$S/ambig/sel/${D}_r{R}_cls_uni_s42.npy ;;
  dermamnist|octmnist)
    EMB=$S/knnf/t1cache/$E/${D}_train_uni_224.npz; HREF=$S/runs/knnf_20261002/sel/seed_42/herding/$D
    CLS=$S/w1/sel/${D}_r{R}_cls_uni_s42.npy ;;
  tissuemnist)
    EMB=$S/t1cache/$E/${D}_train_uni_224.npz; HREF=$S/runs/table1_s4246_20260928/sel/seed_42/herding/$D
    CLS=$S/sel/${D}_r{R}_tdgs_cls_s42.npy ;;
  *) echo "unknown D=$D"; exit 2 ;;
esac
EXTRA=""; [[ "${GPUK:-0}" == 1 ]] && EXTRA="--gpu-faiss-kernel"
mkdir -p $S/acs/sel $S/acs/report
cd $S/code
echo "host=$(hostname) ds=$D start=$(date -Is)"; sha256sum acs_select.py acs_stage.slurm mv_select.py tdgs_select.py
for R in ${RATIOS:-0.02 0.05}; do
  "$PY" acs_select.py --dataset $D --ratio $R --author-emb $EMB --herding-ref $HREF \
      --cls-ref ${CLS/\{R\}/$R} --knnf-report $S/knnf/td_sel/${D}_knnf_report.json \
      --out-dir $S/acs/sel --report $S/acs/report $EXTRA || echo "FAILED $D $R"
done
ls $S/acs/sel | grep "^${D}_" | sed 's/^/staged /'
echo "done=$(date -Is)"
