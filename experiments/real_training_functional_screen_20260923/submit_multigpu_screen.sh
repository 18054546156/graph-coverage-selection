#!/bin/bash
# Fill the whole 12-GPU quota with 4 jobs of 3 workers each.
#
# The binding constraint on qos-high-gpu is jobs, not GPUs:
#     MaxJobsPU=5   MaxSubmitPU=15   cpu=128,gres/gpu=12,mem=256G
# One dataset per job at --gres=gpu:1 spends all 5 job slots to run 5 GPUs and
# strands 7. Three workers per job runs 12 GPUs in 4 slots and leaves one spare.
#
# Resource arithmetic against the per-user caps (4 jobs x 3 GPUs):
#     gpu  4 * 3  = 12   == cap 12
#     mem  4 * 60 = 240G <= cap 256G
#     cpu  4 * 18 = 72   <= cap 128
# Asking 64G/job would hit the memory cap exactly and risk never starting, so
# workers get 20G each rather than the 24G a solo job used.
#
# Datasets are weighted by cost: pathmnist and tissuemnist train slower
# (~82s/run vs ~69s) and get 3 workers, the rest get 2. Each job mixes fast and
# slow datasets so no single job becomes the straggler.
#
# Jobs are split across gpu-a100 and gpu-rtx4090 because neither partition has
# 12 free GPUs alone. Combined with stride sharding this randomises GPU model
# across selection indices instead of aligning it with a contiguous range -- the
# same selection has been observed to move 2.58pp across hosts, so alignment
# would have been a confound. `gpu_name` is recorded in every row regardless.
#
# Usage:
#   ./submit_multigpu_screen.sh              # submit
#   DRY_RUN=1 ./submit_multigpu_screen.sh    # print only
set -euo pipefail

: "${TRAIN_SEED_OFFSET:=100000}"
: "${DRY_RUN:=0}"
: "${MEM:=60G}"
: "${CPUS:=18}"
: "${GPUS:=3}"

cd "$HOME/real_training_functional_screen_20260923"
SCRIPT=run_real_multigpu.slurm
[ -f "$SCRIPT" ] || { echo "missing $SCRIPT -- scp it from the local folder" >&2; exit 1; }

# Fail loudly rather than silently reproducing an old library or re-running work.
grep -q -- "--n-perturb-seeds 3" "$SCRIPT" || {
  echo "$SCRIPT does not ask for n-perturb-seeds 3" >&2; exit 1; }
grep -q -- "--skip-existing" "$SCRIPT" || {
  echo "$SCRIPT is missing --skip-existing" >&2; exit 1; }
grep -q -- "--shard-index" "$SCRIPT" || {
  echo "$SCRIPT is missing --shard-index" >&2; exit 1; }
python3 - <<'PY' || { echo "driver lacks --shard-count; re-copy it" >&2; exit 1; }
import sys, pathlib
src = pathlib.Path("real_training_functional_screen.py").read_text(encoding="utf-8")
sys.exit(0 if "--shard-count" in src and "shard_count" in src else 1)
PY

# job name | partition | 3 work items "dataset,shard_index,shard_count"
JOBS=(
  "rfs-mg-a|gpu-a100|bloodmnist,0,2 pathmnist,0,3 tissuemnist,0,3"
  "rfs-mg-b|gpu-a100|bloodmnist,1,2 pathmnist,1,3 tissuemnist,1,3"
  "rfs-mg-c|gpu-rtx4090|organamnist,0,2 organsmnist,0,2 pathmnist,2,3"
  "rfs-mg-d|gpu-rtx4090|organamnist,1,2 organsmnist,1,2 tissuemnist,2,3"
)

echo "offset=$TRAIN_SEED_OFFSET  ${#JOBS[@]} jobs x $GPUS gpus = $(( ${#JOBS[@]} * GPUS )) workers"
echo

for spec in "${JOBS[@]}"; do
  name="${spec%%|*}"; rest="${spec#*|}"
  part="${rest%%|*}"; work="${rest#*|}"
  echo "  $name  $part  [$work]"
  # $work is deliberately unquoted: it must word-split into one positional
  # argument per worker. It must NOT go through --export, which splits its own
  # value on commas and would reduce "bloodmnist,0,2 pathmnist,0,3" to
  # WORK=bloodmnist. TRAIN_SEED_OFFSET is exported instead and picked up by
  # --export=ALL (the sbatch default), since it contains no commas.
  if [ "$DRY_RUN" = "1" ]; then
    echo "    DRY_RUN: sbatch --gres=gpu:$GPUS --cpus-per-task=$CPUS --mem=$MEM --partition=$part --job-name=$name $SCRIPT $work"
  else
    TRAIN_SEED_OFFSET="$TRAIN_SEED_OFFSET" \
    sbatch --gres=gpu:"$GPUS" --cpus-per-task="$CPUS" --mem="$MEM" \
      --partition="$part" --job-name="$name" \
      "$SCRIPT" $work
  fi
done

echo
echo "Workers stride over [0,600) so every (dataset, block) is covered by all of"
echo "its shards; already-measured selections are skipped at run time."
