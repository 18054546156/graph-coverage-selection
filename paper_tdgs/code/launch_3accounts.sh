#!/bin/bash
# Launch the TDGS round-2 matrix across three accounts: 3 x 5 = 15 slurm jobs,
# 30 concurrent trainings, ONE shared result tree, no post-hoc merge.
#
# Run this from the workstation (Git Bash). It is idempotent: re-running it after
# a crash re-submits the same 15 jobs and every already-finished cell is skipped
# by td_run_one.sh, because the SKIP check reads the SHARED tree.
#
# WHY A SHARED TREE. Only xiaoyuxu2 can write /project/prj-sis01/xuxiaoyu, so the
# earlier multi-account work wrote into three separate homes and had to merge
# done-keys by hand before each launch -- and forgetting to do that re-ran
# finished work. /mnt/prj01/hgrp-1502-5TB is group hgrp-1502 with setgid, and all
# three accounts are in that group, so one tree + umask 002 removes the merge
# step entirely and makes cross-account deduplication automatic.
#
# ACCOUNT LIMITS (per user, qos-high-gpu): 12 GPUs, 256G, 5 running jobs,
# 15 submitted jobs. 5 jobs x 2 GPUs = 10 GPUs and 5 running jobs -> legal, and
# gives the requested 15 jobs across three accounts.
#   Prefer GPU count over job count?  NJOB=4 NGPU=3  -> 12 GPUs/account, 36 slots.
#
# Usage:
#   ./launch_3accounts.sh stage          # selections only (CPU, ~15 min)
#   ./launch_3accounts.sh train          # submit the 15 training jobs
#   ./launch_3accounts.sh status         # queue + progress across all accounts
#   ./launch_3accounts.sh harvest        # collect results into archive/
#   ./launch_3accounts.sh cancel         # cancel only OUR jobs, all accounts
#   ./launch_3accounts.sh worklist2b     # write work/round2b.txt (40 cells, no staging needed)
#   ./launch_3accounts.sh train2b        # submit round 2b; REFUSES while any tdgspack job
#                                        # is still queued/running on any account
#   status for 2b:  WORKLIST=/mnt/prj01/hgrp-1502-5TB/tdgs_shared/work/round2b.txt ./launch_3accounts.sh status
#
# TABLE 1 on the author seed grid 42-46 (7 baselines x 5 ds x 2 budgets = 350 cells):
#   ./launch_3accounts.sh t1stage        # seed the shared cache, write the worklist,
#                                        # submit selection staging (5 seeds per account)
#   ./launch_3accounts.sh t1train        # 15 t1pack jobs; each account's jobs wait
#                                        # (afterany) for that account's staging
#   ./launch_3accounts.sh t1status
#   ./launch_3accounts.sh t1harvest      # also runs the bit-identity check vs the archive
set -uo pipefail

S=/mnt/prj01/hgrp-1502-5TB/tdgs_shared
ACCOUNTS=(luhpc luhpc-qiangzeng luhpc-danranwang)
OWNER=luhpc                       # the only account that can write $PROJECT
NJOB=${NJOB:-5}                   # array tasks per account
NGPU=${NGPU:-2}                   # GPUs (concurrent trainings) per job
WORKLIST=${WORKLIST:-$S/work/round2.txt}
WORKLIST_2B=$S/work/round2b.txt
T1_ROOT=$S/runs/table1_s4246_20260928
T1_CACHE=$S/t1cache
WORKLIST_T1=$S/work/table1_s4246.txt
ARCH=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab
# staging split: the pipeline loops datasets inside one task, one task per seed
T1_STAGE_DS=(tissuemnist pathmnist,bloodmnist organamnist,organsmnist)
T1_STAGE_MEM=(48G 32G 32G)      # archive ran tissue at 48G including training
EXP_ROOT=$S/runs/round2_20260927
SEL_DIR=$S/sel
PY=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/envs/medmnistc-py311/bin/python

CMD=${1:-status}

# ---------------------------------------------------------------- helpers
push_code() {
    echo "== syncing code to $S/code (owner account) =="
    for f in tdgs_select.py tdgs_select_r2.py tdgs_pack.slurm tdgs_stage_r2.slurm \
             perclass_crossshare.py tdgs_crossshare.slurm class_mechanism.py \
             make_worklist.py harvest.py ladder.py launch_3accounts.sh \
             t1_select.slurm t1_run_one.sh t1_harvest.py make_table1.py; do
        [ -f "$(dirname "$0")/$f" ] && \
          scp -q "$(dirname "$0")/$f" "$OWNER:$S/code/$f"
    done
    ssh "$OWNER" "umask 002; chmod g+rwX $S/code/* 2>/dev/null; chmod +x $S/code/*.sh 2>/dev/null; ls -la $S/code"
}

# Each non-owner account needs a WRITABLE graph cache. The real embedding npz
# files live in the read-only project cache, so the per-account cache is seeded
# with symlinks: reads hit the originals, any write the pipeline attempts lands
# in the account's own home instead of failing on a read-only mount.
prep_caches() {
    for H in "${ACCOUNTS[@]}"; do
        echo "== preparing cache on $H =="
        ssh "$H" 'bash -s' <<'EOF'
umask 002
SRC=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_20260922/cache
DST=$HOME/tdgs_cache
mkdir -p "$DST"
for d in "$SRC"/*; do
    b=$(basename "$d")
    if [ -d "$d" ]; then
        mkdir -p "$DST/$b"
        for f in "$d"/*; do ln -sfn "$f" "$DST/$b/$(basename "$f")" 2>/dev/null; done
    else
        ln -sfn "$d" "$DST/$b" 2>/dev/null
    fi
done
echo "  $(whoami): $(find "$DST" -maxdepth 2 | wc -l) cache entries, writable=$([ -w "$DST" ] && echo YES || echo NO)"
EOF
    done
}

# ---------------------------------------------------------------- commands
case "$CMD" in

stage)
    push_code
    ssh "$OWNER" "bash -s" <<EOF
set -uo pipefail
umask 002
mkdir -p $S/work $EXP_ROOT
$PY $S/code/make_worklist.py --only P0 P1 --out $WORKLIST
wc -l < $WORKLIST
EOF
    echo "== submitting selection staging (one array task per dataset) =="
    ssh "$OWNER" "bash -s" <<EOF
umask 002
mkdir -p $S/archive/round2 $S/logs/staging
cd $S/logs/staging
sbatch --array=0-4 $S/code/tdgs_stage_r2.slurm
EOF
    ;;

train)
    prep_caches
    ssh "$OWNER" "ls $WORKLIST >/dev/null" || { echo "no worklist; run stage first"; exit 1; }
    i=0
    for H in "${ACCOUNTS[@]}"; do
        echo "== submitting shard $i on $H ($NJOB jobs x $NGPU gpu) =="
        ssh "$H" "bash -s" <<EOF
umask 002
U=\$(whoami)
mkdir -p $S/logs/\$U && cd $S/logs/\$U
sbatch --array=0-$((NJOB-1)) --gres=gpu:$NGPU \
  --export=ALL,WORKLIST=$WORKLIST,TD_EXP_ROOT=$EXP_ROOT,TD_SEL_DIR=$SEL_DIR,\
TD_CACHE_ROOT=\$HOME/tdgs_cache,NGPU=$NGPU,SHARD=$i,NSHARD=${#ACCOUNTS[@]},SEL_WAIT=3600 \
  $S/code/tdgs_pack.slurm
EOF
        i=$((i+1))
    done
    echo "== 15 jobs submitted; 'status' to watch =="
    ;;

status)
    total=$(ssh "$OWNER" "grep -cvE '^\s*(#|\$)' $WORKLIST 2>/dev/null" || echo 0)
    done_n=$(ssh "$OWNER" "find $EXP_ROOT -name predictions_clean.npz 2>/dev/null | wc -l" || echo 0)
    echo "progress: $done_n / $total cells"
    for H in "${ACCOUNTS[@]}"; do
        echo "--- $H ---"
        ssh "$H" 'squeue -u $(whoami) -o "%.10i %.10P %.10j %.2t %.10M %.4D %R" 2>/dev/null | tail -20'
    done
    ssh "$OWNER" "bash -s" <<EOF
echo "--- per-arm completion ---"
for a in a2_perclass tdgs_mask tdgs_wcls tdgs_cls_perm tdgs_lam1 graph_a2 tdgs_cls; do
  n=\$(find $EXP_ROOT -path "*/\$a/*" -name predictions_clean.npz 2>/dev/null | wc -l)
  printf "  %-16s %3d\n" "\$a" "\$n"
done
echo "--- failures in logs ---"
grep -h "FAIL rc=" $S/logs/*/*.out 2>/dev/null | sort | uniq -c | head -20
EOF
    ;;

harvest)
    ssh "$OWNER" "bash -s" <<EOF
umask 002
mkdir -p $S/archive/round2
$PY $S/code/harvest.py --exp-root $EXP_ROOT \
    --out $S/archive/round2/round2_harvest.json
$PY $S/code/ladder.py --harvest $S/archive/round2/round2_harvest.json \
    --round1 $S/archive/round1/tdgs_round1_harvest.json \
    | tee $S/archive/round2/round2_ladder.txt
EOF
    ;;

cancel)
    for H in "${ACCOUNTS[@]}"; do
        echo "--- cancelling tdgspack/tdgssel2/t1sel/t1pack on $H (leaves other jobs alone) ---"
        ssh "$H" 'for j in tdgspack tdgssel2 t1sel t1pack; do scancel -u $(whoami) --name=$j 2>/dev/null; done; squeue -u $(whoami) -h | wc -l'
    done
    ;;

worklist2b)
    push_code
    ssh "$OWNER" "bash -s" <<EOF
umask 002
$PY $S/code/make_worklist.py --only P2 --out $WORKLIST_2B
grep -cvE '^\s*(#|\$)' $WORKLIST_2B
EOF
    ;;

train2b)
    # Round 2b must not overlap round 2: td_run_one.sh only skips cells that are
    # FINISHED, so a cell another job is still training would be trained twice.
    # Round 2b cells are disjoint from round 2 by construction, but the guard also
    # stops a second copy of 2b being submitted on top of a running one.
    busy=0
    for H in "${ACCOUNTS[@]}"; do
        n=$(ssh "$H" 'squeue -u $(whoami) -h -n tdgspack 2>/dev/null | wc -l')
        echo "  $H: $n tdgspack job(s) queued/running"
        busy=$((busy + n))
    done
    if [ "$busy" -gt 0 ]; then
        echo "REFUSING: $busy tdgspack job(s) still in the queue. Wait for round 2 to finish."
        exit 1
    fi
    ssh "$OWNER" "test -s $WORKLIST_2B" || { echo "no $WORKLIST_2B; run worklist2b first"; exit 1; }
    WORKLIST=$WORKLIST_2B exec "$0" train
    ;;

t1stage)
    push_code
    # Shared, group-writable cache seeded with SYMLINKS to the archive's UNI
    # embeddings and its seed-42/43/44 training dynamics. Dynamics the archive
    # never computed (s44 for path/blood/tissue, all of s45/s46) are written
    # here once and read by all three accounts; the archive tree is never
    # written to.
    ssh "$OWNER" "bash -s" <<EOF
set -uo pipefail
umask 002
AC=$ARCH/table1_clean_only_20260922/cache
mkdir -p $T1_CACHE $T1_ROOT $S/work $S/logs/t1
for d in embeddings_img224_smokefull img224_dyn200_sel42_smokefull \
         img224_dyn200_sel43_smokefull img224_dyn200_sel44_smokefull; do
    mkdir -p $T1_CACHE/\$d
    for f in \$AC/\$d/*.npz; do ln -sfn "\$f" $T1_CACHE/\$d/\$(basename "\$f"); done
done
find $T1_CACHE -type l | wc -l
$PY $S/code/make_worklist.py --only P3 --out $WORKLIST_T1
EOF
    i=0
    for H in "${ACCOUNTS[@]}"; do
        echo "== staging ${T1_STAGE_DS[$i]} x seeds 42-46 on $H =="
        ssh "$H" "bash -s" <<EOF
umask 002
mkdir -p $S/logs/t1/\$(whoami) && cd $S/logs/t1/\$(whoami)
sbatch --parsable --array=0-4 --mem=${T1_STAGE_MEM[$i]} \
  --export=ALL,DS=${T1_STAGE_DS[$i]},T1_ROOT=$T1_ROOT,T1_CACHE=$T1_CACHE \
  $S/code/t1_select.slurm | tee $S/logs/t1/\$(whoami)/stage_jobid
EOF
        i=$((i+1))
    done
    ;;

t1train)
    busy=0
    for H in "${ACCOUNTS[@]}"; do
        n=$(ssh "$H" 'squeue -u $(whoami) -h -n t1pack 2>/dev/null | wc -l')
        busy=$((busy + n))
    done
    [ "$busy" -gt 0 ] && { echo "REFUSING: $busy t1pack job(s) already queued/running"; exit 1; }
    ssh "$OWNER" "test -s $WORKLIST_T1" || { echo "no $WORKLIST_T1; run t1stage first"; exit 1; }
    i=0
    for H in "${ACCOUNTS[@]}"; do
        echo "== t1pack shard $i on $H ($NJOB jobs x $NGPU gpu) =="
        ssh "$H" "bash -s" <<EOF
umask 002
U=\$(whoami)
mkdir -p $S/logs/t1/\$U && cd $S/logs/t1/\$U
DEP=\$(cat stage_jobid 2>/dev/null)
sbatch --job-name=t1pack --array=0-$((NJOB-1)) --gres=gpu:$NGPU \
  \${DEP:+--dependency=afterany:\$DEP} \
  --export=ALL,WORKLIST=$WORKLIST_T1,TD_EXP_ROOT=$T1_ROOT,TD_SEL_DIR=$T1_ROOT/sel,\
T1_ROOT=$T1_ROOT,T1_CACHE=$T1_CACHE,RUNNER=$S/code/t1_run_one.sh,\
NGPU=$NGPU,SHARD=$i,NSHARD=${#ACCOUNTS[@]},SEL_WAIT=21600 \
  $S/code/tdgs_pack.slurm
EOF
        i=$((i+1))
    done
    ;;

t1status)
    total=$(ssh "$OWNER" "grep -cvE '^\s*(#|\$)' $WORKLIST_T1 2>/dev/null" || echo 0)
    ssh "$OWNER" "bash -s" <<EOF
echo "progress: \$(find $T1_ROOT/formal -name predictions_clean.npz 2>/dev/null | wc -l) / $total cells"
echo "selections staged: \$(find $T1_ROOT/sel -name index_order_sha256.txt 2>/dev/null | wc -l) / 350"
for r in 0.02 0.05; do for m in random el2n_top forgetting eva facility fps herding; do
  printf "  r%s %-11s %3d\n" \$r \$m \$(find $T1_ROOT/formal -path "*/\$m/ratio_\$r/*" -name predictions_clean.npz 2>/dev/null | wc -l)
done; done
grep -h "FAIL rc=\|MISSING SEL\|Traceback" $S/logs/t1/*/*.out $S/logs/t1/*/*.err 2>/dev/null | sort | uniq -c | head -20
EOF
    for H in "${ACCOUNTS[@]}"; do
        echo "--- $H ---"
        ssh "$H" 'squeue -u $(whoami) -o "%.10i %.8j %.2t %.10M %R" 2>/dev/null | tail -20'
    done
    ;;

t1harvest)
    ssh "$OWNER" "bash -s" <<EOF
umask 002
mkdir -p $S/archive/table1
$PY $S/code/t1_harvest.py --root $T1_ROOT --out $S/archive/table1/t1_s4246_harvest.json \
    --compare-archive $ARCH/table1_clean_only_20260922 $ARCH/table1_clean_only_5pct_20260922
$PY $S/code/make_table1.py --baselines $S/archive/table1/t1_s4246_harvest.json \
    --round2 $S/archive/round2/round2_harvest.json \
    --round1 $S/archive/round1/tdgs_round1_harvest.json \
    | tee $S/archive/table1/table1_s4246.txt
EOF
    ;;

*)
    sed -n '1,30p' "$0"; exit 1 ;;
esac
