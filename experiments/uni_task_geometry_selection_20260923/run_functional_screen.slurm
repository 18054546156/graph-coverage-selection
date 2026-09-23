#!/bin/bash
# Functional screen: which functional of a selected set predicts accuracy at fixed budget.
# 4 processes, one GPU each. tissuemnist and pathmnist get their own because their
# pools are the largest; the two organ sets share one because they are the same
# 201 LiTS volumes and run back to back cheaply.
#
# Memory is capped at 36G because qos-high-gpu counts 256G PER USER and the four
# in-flight jobs already hold 216G.
#SBATCH --job-name=fscreen
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=30
#SBATCH --mem=60G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
PROJECT="$HOME/uni_task_geometry_selection_20260923"
PYTHON=/project/prj-sis01/xuxiaoyu/graph_select/envs/graphcov-py311/bin/python3
EMBED=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_20260922/cache/embeddings_img224_smokefull
DATA=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/data/medmnist
# 200-epoch per-sample training dynamics (forgetting / AUM / EL2N), 28px, seed 42.
# All five datasets have the 28px variant; pathmnist also has 224 but mixing
# resolutions across datasets would confound the cross-dataset comparison.
DYN=/project/prj-sis01/xuxiaoyu/graph_select/cache
OUT="$PROJECT/results/screen_$SLURM_JOB_ID"
mkdir -p "$OUT" "$PROJECT/logs"
cd "$PROJECT"

# NOT named GROUPS: that is a readonly bash builtin array of the user's GIDs, and
# assigning to it fails silently under `set -u`, so the loop launched group ids.
DS_GROUPS=("tissuemnist" "pathmnist" "organamnist organsmnist" "bloodmnist")
pids=()

# GPU 0: full-scale priors. This runs first-class rather than as an afterthought
# because it gates two live decisions -- whether pathmnist was rightly excluded from
# the training factorial (that exclusion cites a headroom measured on an 8% pool),
# and whether the cross-dataset ordering prediction in PLAN_conditional_coreset.md
# rests on a real ordering or on a subsampling artefact.
CUDA_VISIBLE_DEVICES=0 "$PYTHON" measure_priors_fullscale.py \
  --embedding-root "$EMBED" --data-root "$DATA" \
  --output "$OUT/priors_fullscale.json" \
  > "$PROJECT/logs/priors_fullscale_${SLURM_JOB_ID}.log" 2>&1 &
pids+=($!)
echo "launched full-scale priors on gpu 0 pid ${pids[-1]}"

for i in "${!DS_GROUPS[@]}"; do
  CUDA_VISIBLE_DEVICES=$((i+1)) "$PYTHON" functional_screen.py \
    --embedding-root "$EMBED" --data-root "$DATA" \
    --output "$OUT/screen_${i}.jsonl" \
    --datasets ${DS_GROUPS[$i]} --dynamics-root "$DYN" \
    --budget 25 --blocks 2 --n-random 120 --n-perturb-seeds 8 \
    > "$PROJECT/logs/screen_${i}_${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!)
  echo "launched [${DS_GROUPS[$i]}] on gpu $((i+1)) pid ${pids[-1]}"
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || { echo "pid $pid FAILED"; fail=1; }; done
echo "finish=$(date -Is) fail=$fail"
exit $fail
