#!/bin/bash
#SBATCH --job-name=r3smoke --partition=gpu-a100 --qos=qos-high-gpu --gpus=1 --cpus-per-task=8 --mem=48G --time=00:40:00 --exclude=hpcgpu108 --output=%x_%j.out
umask 002
S=/mnt/prj01/hgrp-1502-5TB/tdgs_shared
PY=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/envs/medmnistc-py311/bin/python
export HF_HOME=$S/hf_cache HF_HUB_OFFLINE=1
for d in bloodmnist organamnist; do
$PY $S/code/extract_emb.py --datasets $d --out $S/r3/smoke --encoders uni dinov2 clip --splits train --limit 512
done
$PY - <<'P'
import numpy as np
B='/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_20260922/cache/embeddings_img224_smokefull'
for d in ['bloodmnist','organamnist']:
    ref=np.load(f'{B}/{d}_train_uni_224.npz')['embeddings'][:512]
    new=np.load(f'/mnt/prj01/hgrp-1502-5TB/tdgs_shared/r3/smoke/{d}_train_uni_224_smoke512.npz')['embeddings']
    cos=(ref*new).sum(1)/np.linalg.norm(ref,axis=1)/np.linalg.norm(new,axis=1)
    print(d,'UNI reproduce: cos min %.6f mean %.6f  maxabs %.3g'%(cos.min(),cos.mean(),np.abs(ref-new).max()))
    for e in ['dinov2','clip']:
        z=np.load(f'/mnt/prj01/hgrp-1502-5TB/tdgs_shared/r3/smoke/{d}_train_{e}_224_smoke512.npz')['embeddings']; print(d,e,z.shape,np.isfinite(z).all())
P
