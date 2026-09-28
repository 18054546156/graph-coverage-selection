#!/bin/bash
#SBATCH --job-name=tdgssmoke
#SBATCH --partition=gpu-a100
#SBATCH --qos=qos-high-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --exclude=hpcgpu108
#SBATCH --output=tdgssmoke_%j.log
# TDGS smoke: organsmnist only (n=13932, the smallest pool) at 2%.
# Verifies all five arms run, the per-class quota holds, the graph_a2 arm
# reproduces the archived selection, and reports whether the OOF proxy
# recovers the final models' confusion structure.
# Writes selections to a SEPARATE dir so it cannot collide with the concurrent
# session's ~/td_sel staging.
set -euo pipefail
PY=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/envs/medmnistc-py311/bin/python
cd "$HOME/tdgs"
echo "host=$(hostname) start=$(date -Is)"
"$PY" tdgs_select.py --dataset organsmnist --ratio 0.02 \
      --out-dir "$HOME/tdgs_sel" --report "$HOME/tdgs_report"

echo "=== validity: does the graph_a2 arm reproduce the archive? ==="
"$PY" - <<'EOF'
import glob, numpy as np, os
P="/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_20260922/formal"
fs=[f for f in glob.glob(f"{P}/**/graph_a2/ratio_0.02/**/selected_indices.npy",recursive=True) if "/organsmnist/" in f]
arch=np.load(sorted(fs)[0]).ravel().astype(int)
mine=np.load(os.path.expanduser("~/tdgs_sel/organsmnist_r0.02_graph_a2_s42.npy")).ravel().astype(int)
print("archived",len(arch),"mine",len(mine))
print("order_match",bool(np.array_equal(arch,mine)),"set_match",set(arch.tolist())==set(mine.tolist()))
EOF
echo "done=$(date -Is)"
