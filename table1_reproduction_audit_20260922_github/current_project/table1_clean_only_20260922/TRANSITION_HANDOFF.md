# Transition handoff: old MedMNIST-C audit to clean-only Table 1 study

Date: 2026-09-22 Asia/Shanghai

## Completed before stopping

1. Read-only status checks were performed for both SSH accounts.
2. The audited clean-arm inventory was exported to `clean_clean_test_inventory_20260922.csv`.
3. The old HPC_USER_B audit inventory was copied to `old_three_arm_resume/HPC_USER_B_inventory_20260921.json`.
4. A separate output directory was created for the new clean-only study.

## Old jobs and outputs

- `HPC_USER_A`: no active Slurm jobs at the status check.
- `HPC_USER_B`: legacy Job `32712_4` (Facility) and `32712_7` (Graph-A2) were running at the status check and were then cancelled explicitly.
- Final Slurm states after the stop: `32712_4` cancelled after `1-18:52:22`; `32712_7` cancelled after `14:31:52`. Existing output files were not deleted.
- Completed legacy outputs stay in `/home/HPC_USER_B/prj_HPC_USER_B/three_arm_multiseed_20260920_HPC_USER_B/`.
- The old code, YAML files, logs, cache paths, and existing run directories must not be overwritten by the new study.
- The 136-cell clean-arm CSV is the last validated inventory; the two cancelled jobs were not reclassified from partial markers into valid results after stopping.

## New study contract

```text
clean train -> clean selection -> clean selected training -> clean test
selection_seed == training_seed
paired seeds: 0, 1, 2, 3, 4, 5, 6, 2026
```

No MedMNIST-C data may enter selection, training, checkpoint selection, or evaluation for this Table 1 replication.

## Submission gate

No new `sbatch` may be submitted until the ratio choice, RAM per task, task granularity, and final output root are explicitly confirmed.
