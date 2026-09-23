#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT=/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_20260922
LAUNCHER=$EXPERIMENT_ROOT/run_clean_only.slurm
QUEUE_FILE=${QUEUE_FILE:?QUEUE_FILE is required}
LOG_FILE=${LOG_FILE:-$EXPERIMENT_ROOT/logs/formal_queue_${USER}.log}
JOB_PREFIX=table1-formal-${USER}
LOCK_FILE=${QUEUE_FILE}.lock

mkdir -p "$(dirname "$LOG_FILE")"
exec >>"$LOG_FILE" 2>&1

echo "queue worker started user=$USER queue=$QUEUE_FILE at=$(date -Is)"

active_jobs() {
    squeue -h -u "$USER" -o "%j|%T" \
      | awk -F'|' -v prefix="$JOB_PREFIX" \
        '$1 ~ ("^" prefix "-") && ($2 == "PENDING" || $2 == "RUNNING") { n++ } END { print n + 0 }'
}

claim_next() {
    local line
    exec 9>>"$LOCK_FILE"
    flock -x 9
    line=$(sed -n '1p' "$QUEUE_FILE" 2>/dev/null || true)
    if [[ -n "$line" ]]; then
        tail -n +2 "$QUEUE_FILE" >"$QUEUE_FILE.tmp.$$"
        mv "$QUEUE_FILE.tmp.$$" "$QUEUE_FILE"
    fi
    flock -u 9
    exec 9>&-
    printf '%s' "$line"
}

while true; do
    active=$(active_jobs)
    if (( active >= 15 )); then
        sleep 30
        continue
    fi

    line=$(claim_next)
    if [[ -z "$line" ]]; then
        if (( active == 0 )); then
            echo "queue worker finished at=$(date -Is)"
            exit 0
        fi
        sleep 30
        continue
    fi

    read -r dataset seed mem <<<"$line"
    job_name="${JOB_PREFIX}-${dataset}-s${seed}"
    if ! job_id=$(DATASET="$dataset" SEED="$seed" MODE=formal \
        sbatch --parsable \
          --partition=gpu-a100 \
          --time=5-00:00:00 \
          --job-name="$job_name" \
          --mem="$mem" \
          "$LAUNCHER"); then
        exec 9>>"$LOCK_FILE"
        flock -x 9
        { printf '%s\n' "$line"; cat "$QUEUE_FILE"; } >"$QUEUE_FILE.tmp.$$"
        mv "$QUEUE_FILE.tmp.$$" "$QUEUE_FILE"
        flock -u 9
        exec 9>&-
        echo "submission failed; returned task to queue: $line"
        sleep 60
        continue
    fi
    echo "submitted job=$job_id dataset=$dataset seed=$seed mem=$mem active_before=$active at=$(date -Is)"
done
