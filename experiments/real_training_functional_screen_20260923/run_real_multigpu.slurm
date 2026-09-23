#!/bin/bash
# N independent single-GPU workers inside ONE slurm job.
#
# Why: qos-high-gpu caps a user at 5 running JOBS but 12 GPUs. One dataset per
# job at --gres=gpu:1 therefore used 5 of the 12 GPUs we are entitled to and left
# 7 idle. Packing 3 workers per job turns the same 4-5 job slots into 12 GPUs.
#
# Each worker is a plain single-GPU run of real_training_functional_screen.py,
# pinned with CUDA_VISIBLE_DEVICES and given its own --output directory (the
# driver opens "<output>/<dataset>.jsonl" with mode "w", so sharing an output
# directory between workers would truncate).
#
# Submit with:
#   sbatch --gres=gpu:3 --cpus-per-task=18 --mem=60G \
#     --partition=gpu-a100 --job-name=rfs-mg-a \
#     run_real_multigpu.slurm bloodmnist,0,2 pathmnist,0,3 tissuemnist,0,3
#
# Work items are POSITIONAL ARGUMENTS, one "dataset,shard_index,shard_count" per
# allocated GPU. They must NOT be passed through --export: sbatch splits an
# --export value on commas, so --export=ALL,WORK="bloodmnist,0,2 pathmnist,0,3"
# silently sets WORK=bloodmnist and invents variables named "0" and "2 pathmnist".
#
# Shards are STRIDED (index % count == shard), not sliced, so workers stay
# balanced even though already-measured indices are near-free skips.
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
[ "$#" -gt 0 ] || { echo "usage: sbatch ... $0 <dataset,shard,count> ..." >&2; exit 2; }
# Each worker sweeps every train-seed offset in order, on its own GPU. Listing
# the analysis-critical offset first means the whole fleet -- across all three
# accounts -- finishes that seed before starting the next, instead of each
# account owning one seed and the fleet finishing all three at the same late
# time. Combined with --skip-existing this is also what makes the run
# self-balancing: a worker whose shard is mostly done simply moves on.
: "${OFFSETS:=100000 0 200000}"
: "${EPOCHS:=200}"
: "${GLOBAL_START:=0}"
: "${GLOBAL_END:=600}"

# Resolve from the submission directory, never from $HOME. The screen now runs
# under three accounts (xiaoyuxu2 / qiangzeng / danranwang), and HOME is a
# variable the ssh client forwards -- on this setup it arrives as the *client's*
# Windows path. SLURM_SUBMIT_DIR is set by sbatch to the directory the job was
# submitted from, which is the project directory by construction.
PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
[ -f "$PROJECT/real_training_functional_screen.py" ] || {
  echo "PROJECT=$PROJECT has no driver; submit from the project directory" >&2
  exit 2; }
PYTHON=/project/prj-sis01/xuxiaoyu/graph_select/envs/graphcov-py311/bin/python3
GRAPH=/project/prj-sis01/xuxiaoyu/graph_select
EMBED=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_20260922/cache/embeddings_img224_smokefull
DATA=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/data/medmnist
MEDMNIST=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/data/medmnist
DYN=/project/prj-sis01/xuxiaoyu/graph_select/cache
mkdir -p "$PROJECT/logs"

cd "$PROJECT"
echo "start=$(date -Is) host=$(hostname) job=$SLURM_JOB_ID"
echo "work=$*"

# Map worker i onto the i-th GPU SLURM actually gave us, rather than onto
# physical device i. nvidia-smi inside these jobs lists every GPU on the node,
# so this cluster is not isolating devices by cgroup: hardcoding
# CUDA_VISIBLE_DEVICES=0,1,2 would happily run on another user's GPUs whenever
# SLURM allocated us anything other than the first three.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -ra DEVS <<< "$CUDA_VISIBLE_DEVICES"
else
  echo "WARNING: SLURM set no CUDA_VISIBLE_DEVICES; falling back to 0..n-1" >&2
  DEVS=(); for ((d = 0; d < $#; d++)); do DEVS+=("$d"); done
fi
echo "allocated gpus=${DEVS[*]}  workers=$#"
[ "${#DEVS[@]}" -ge "$#" ] || {
  echo "only ${#DEVS[@]} GPUs for $# work items" >&2; exit 2; }

gpu=0
pids=()
names=()
for item in "$@"; do
  ds="${item%%,*}"; rest="${item#*,}"
  sidx="${rest%%,*}"; scnt="${rest##*,}"
  [ "$ds,$sidx,$scnt" = "$item" ] || {
    echo "malformed work item: $item" >&2; exit 2; }
  echo "  worker$gpu -> physical gpu ${DEVS[$gpu]} : $ds shard $sidx/$scnt over offsets [$OFFSETS]"
  export OFFSETS PROJECT PYTHON GRAPH EMBED DATA MEDMNIST DYN
  export GLOBAL_START GLOBAL_END EPOCHS SLURM_JOB_ID
  export GPU="$gpu" DS="$ds" SIDX="$sidx" SCNT="$scnt"

  # One worker = one GPU sweeping the offsets sequentially.
  # --skip-existing scans BOTH this account's results and skipseed/, the merged
  # key list from all three accounts, so no worker repeats what another account
  # already measured. Without it offset 100000 would be re-run from 74% done.
  CUDA_VISIBLE_DEVICES="${DEVS[$gpu]}" bash -c '
    set -uo pipefail
    for off in $OFFSETS; do
      out="$PROJECT/results/seed_${off}/mg_${SLURM_JOB_ID}_g${GPU}_${DS}_s${SIDX}of${SCNT}"
      mkdir -p "$out"
      echo "[$DS s$SIDX/$SCNT] === offset $off ==="
      "$PYTHON" "$PROJECT/real_training_functional_screen.py" \
        --embedding-root "$EMBED" --data-root "$DATA" --medmnist-root "$MEDMNIST" \
        --dynamics-root "$DYN" --graphcov-root "$GRAPH" --output "$out" \
        --datasets "$DS" --blocks 2 --budget 25 \
        --n-random 60 --n-perturb-seeds 3 --train-seed-offset "$off" \
        --global-start "$GLOBAL_START" --global-end "$GLOBAL_END" \
        --shard-index "$SIDX" --shard-count "$SCNT" \
        --skip-existing "$PROJECT/results" "$PROJECT/skipseed" \
        --epochs "$EPOCHS" --batch-size 128 --lr 0.1 --weight-decay 5e-4 \
        --size 224 --num-workers 4 --progress-every 5
      rc=$?
      [ "$rc" -eq 0 ] || echo "[$DS s$SIDX/$SCNT] offset $off exited rc=$rc" >&2
    done
  ' > "$PROJECT/logs/w-${SLURM_JOB_ID}-g${gpu}-${ds}-s${sidx}.out" 2>&1 &

  pids+=($!)
  names+=("$ds:$sidx/$scnt")
  gpu=$((gpu + 1))
done

# Wait on each worker individually so one crash is reported with its identity
# instead of collapsing into a single non-zero exit from `wait`.
rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "ok   ${names[$i]}"
  else
    echo "FAIL ${names[$i]} (see logs/w-${SLURM_JOB_ID}-g${i}-*.out)" >&2
    rc=1
  fi
done
echo "finish=$(date -Is) rc=$rc"
exit "$rc"
