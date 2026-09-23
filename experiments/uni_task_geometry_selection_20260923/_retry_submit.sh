#!/bin/bash
# Non-destructive: waits for a free slot under MaxSubmitPU=15 instead of
# cancelling anyone else's queued work.
cd "$HOME/uni_task_geometry_selection_20260923"
for i in $(seq 1 240); do
  if out=$(sbatch run_functional_screen.slurm 2>&1); then
    echo "$(date -Is) SUBMITTED $out" >> logs/retry_submit.log; exit 0
  fi
  echo "$(date -Is) waiting: $out" >> logs/retry_submit.log
  sleep 120
done
echo "$(date -Is) GAVE UP" >> logs/retry_submit.log
