# Missing public Table 1 k=10 runs

This isolated supplement completes exactly the twelve missing downstream runs
from the public author-runner snapshot:

* dataset: `bloodmnist`;
* methods: `el2n`, `eva`, `fps`;
* training seeds: `45`, `46`;
* ratios: `0.02`, `0.05`.

The runner reads the already frozen selection files under
`table1_reproduction/outputs/job1_selection/` and writes into the historical
public output root `table1_reproduction/outputs/job2_table1/`. It calls the
author evaluator with test evaluation every 10 epochs, preserving the A-layer
`final` and `best` fields. It does not use the current strict validation-
checkpoint runner, change any indices, or regenerate a selection.

Six independent Slurm jobs each run one method and one missing seed, covering
both ratios. `METHOD` and `TRAINING_SEED` are passed as positional arguments to
`sbatch`, because this HPC environment does not reliably preserve custom
`--export` variables inside the batch shell. Existing result files are skipped
unless `--force` is explicitly used.

Completion record: Slurm jobs `29238`--`29243` all completed with exit code
`0:0`. The twelve result files are locally synchronized under
`table1_reproduction/outputs/job2_table1/bloodmnist/`; the authoritative
394-result summary is under
`reports/table1_group_meeting/evidence/public_k10_394run_*`.
