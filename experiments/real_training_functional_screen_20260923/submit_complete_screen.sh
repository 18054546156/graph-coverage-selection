#!/bin/bash
# Complete the real-training functional screen for ONE train-seed offset, both blocks.
#
# Why this exists: the first wave of shards was launched with hand-picked index
# ranges that happened to cover only a middle slice of block 1 (see logs:
# blood [242,346), organa [256,353), ...). Block 0 has essentially zero rows, and
# the screen's central guard -- sign and significance must agree on BOTH independent
# pool/audit blocks -- cannot be evaluated at all without it.
#
# Two things changed along with this script:
#   1. --n-perturb-seeds is now 3, not 2. At 2 every (dataset, block, family, m)
#      group has exactly 2 rows, and analyze_screen.py drops groups smaller than 3,
#      so the entire perturbation ladder -- 73% of each block -- contributed nothing
#      to the decision cell. Running more of the old design would not have fixed it.
#   2. --skip-existing makes re-submission idempotent. Selections are matched on
#      (dataset, block, name, train_seed), NOT on global index, because the index
#      depends on n_perturb_seeds and therefore shifted; the selections themselves
#      did not (perturb() seeds on seed + 7919*m + s, independent of how many seeds
#      the library asks for). So every row measured so far is still valid and is
#      reused rather than recomputed.
#
# Library layout at n_perturb_seeds=3:
#   60 random + 15 anchors x (1 unperturbed + 5 m-levels x 3 seeds) = 300 per block
#   global index: block 0 = [0,300), block 1 = [300,600)
#
# Usage:
#   TRAIN_SEED_OFFSET=100000 PARTITION=a100 ./submit_complete_screen.sh
#   TRAIN_SEED_OFFSET=100000 PARTITION=a100 DRY_RUN=1 ./submit_complete_screen.sh
set -euo pipefail

: "${TRAIN_SEED_OFFSET:=100000}"
: "${PARTITION:=a100}"          # a100 | 4090
: "${SHARDS_PER_DATASET:=2}"    # 2 -> shard 0 is exactly block 0, shard 1 is block 1
: "${DATASETS:=bloodmnist organamnist organsmnist pathmnist tissuemnist}"
: "${DRY_RUN:=0}"
# Submit one shard index across all datasets, rather than all shards of one dataset.
# qos-high-gpu runs 5 jobs per user at a time, so submission order IS priority order:
# block 0 has no coverage at all and must fill those 5 slots before block 1, which is
# already ~40% done, competes for them.
: "${ONLY_SHARD:=}"

LIBRARY=300                     # per block, at n_perturb_seeds=3
TOTAL=$((LIBRARY * 2))          # both blocks
SCRIPT="run_real_shard_${PARTITION}.slurm"

cd "$HOME/real_training_functional_screen_20260923"
[ -f "$SCRIPT" ] || { echo "no such launcher: $SCRIPT" >&2; exit 1; }

# Fail loudly if the launcher was not updated, rather than silently reproducing the
# 2-seed library that caused the problem this script exists to fix.
grep -q -- "--n-perturb-seeds 3" "$SCRIPT" || {
  echo "$SCRIPT still asks for n-perturb-seeds != 3; re-copy it from the local folder" >&2
  exit 1; }
grep -q -- "--skip-existing" "$SCRIPT" || {
  echo "$SCRIPT is missing --skip-existing; re-copy it from the local folder" >&2
  exit 1; }

span=$(( (TOTAL + SHARDS_PER_DATASET - 1) / SHARDS_PER_DATASET ))
echo "offset=$TRAIN_SEED_OFFSET partition=$PARTITION library=$LIBRARY/block"
echo "each dataset split into $SHARDS_PER_DATASET shards of $span indices over [0,$TOTAL)"
echo

for ((i = 0; i < SHARDS_PER_DATASET; i++)); do
  [ -n "$ONLY_SHARD" ] && [ "$i" != "$ONLY_SHARD" ] && continue
  for ds in $DATASETS; do
    start=$((i * span))
    end=$((start + span))
    [ "$end" -gt "$TOTAL" ] && end=$TOTAL
    [ "$start" -ge "$TOTAL" ] && break
    blocks="block0"
    [ "$start" -ge "$LIBRARY" ] && blocks="block1"
    [ "$start" -lt "$LIBRARY" ] && [ "$end" -gt "$LIBRARY" ] && blocks="block0+1"
    cmd=(sbatch --job-name="rfs-${ds:0:6}-s$i" "$SCRIPT")
    echo "  $ds shard $i [$start,$end)  $blocks"
    if [ "$DRY_RUN" = "1" ]; then
      echo "    DRY_RUN: DATASET=$ds START_INDEX=$start END_INDEX=$end SHARD_ID=c$i ${cmd[*]}"
    else
      DATASET="$ds" START_INDEX="$start" END_INDEX="$end" SHARD_ID="c$i" \
        TRAIN_SEED_OFFSET="$TRAIN_SEED_OFFSET" "${cmd[@]}"
    fi
  done
done

echo
echo "Already-measured selections are skipped at run time, so shards overlapping the"
echo "first wave's [242,358) band cost only the library build, not the training."
