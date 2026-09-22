# Paired seed plan

This is a plan only. It does not submit or start jobs.

```text
seed 0: selection_seed=0, training_seed=0
seed 1: selection_seed=1, training_seed=1
seed 2: selection_seed=2, training_seed=2
seed 3: selection_seed=3, training_seed=3
seed 4: selection_seed=4, training_seed=4
seed 5: selection_seed=5, training_seed=5
seed 6: selection_seed=6, training_seed=6
seed 2026: selection_seed=2026, training_seed=2026
```

For each dataset and paired seed, the eventual task will be clean-only. It will not use MedMNIST-C. The task may process both 2% and 5% only after the ratio decision is confirmed.

No Slurm command is recorded here because the RAM allocation and final task granularity are still pending.
