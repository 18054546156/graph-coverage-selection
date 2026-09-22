# Table 1 clean-only transition

Created on 2026-09-22 before starting the new experiment. This directory is independent of the existing MedMNIST-C three-arm output trees.

## Current scope

- New experiment: clean train selection -> clean selected training -> clean test.
- No MedMNIST-C, pixelate, corrupted selection, or corrupted training.
- Paired seeds: `selection_seed == training_seed`.
- Confirmed seed count: 8. Candidate list: `0,1,2,3,4,5,6,2026`.
- No Slurm submission has been made from this directory.

## Existing clean result inventory

`clean_clean_test_inventory_20260922.csv` contains the 136 clean-arm model cells that passed the read-only HPC audit as of the 2026-09-21 snapshot. It is an inventory of existing results, not a new run.

The 136 cells are distributed as follows:

```text
random      25
el2n_top    25
forgetting  25
eva         25
facility    25
graph_a2    11
```

The source audit is copied under `old_three_arm_resume/HPC_USER_B_inventory_20260921.json`.

## Old three-arm experiment

The old outputs remain in their original HPC directories. They are not moved or overwritten. The currently running legacy tasks were `32712_4` and `32712_7`; their final state is recorded in the transition handoff.

## Pending decisions before submission

- Whether to run ratio `0.02` only or both `0.02` and `0.05`.
- RAM per concurrent task. Ten 48G jobs would request 480G, above the previously recorded 256G total RAM quota.
- The task launcher and exact Slurm array layout.
