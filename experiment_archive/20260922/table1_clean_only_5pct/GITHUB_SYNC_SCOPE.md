# GitHub Sync Scope

Snapshot date: 2026-09-23 (HKT)

This directory is the independent 5% clean-only Table 1 reproduction archive. It contains the executed launcher and queue worker, smoke audit, actual formal run configurations, Slurm stdout/stderr logs, `run_config.json`, `metrics.jsonl`, `run_complete.json`, selection metadata, and the current status workbook copied from the local output archive.

The shared executed source snapshot is already archived at `../table1/code_snapshot/`. The 5% launcher changes only the ratio and experiment root; it reuses that audited source snapshot and cache protocol.

## Deliberately excluded from the GitHub working tree

- Model checkpoints: `*.pt`, `*.pth`, `*.ckpt`.
- Dataset files and pretrained weights.
- Embedding/dynamics caches and other cache directories.
- Generated tensor artifacts: `*.npy`, `*.npz`, and prediction tensors.
- Temporary lock files and Python bytecode.

These files are large derived artifacts and are not needed for a code/config/log diff against the author's repository. The committed JSON metrics, completion markers, selection metadata, configs, logs, Excel workbook, and source provenance preserve the reproducibility audit. The original HPC paths and the exclusion policy are recorded here so the omitted files are not confused with failed runs.

## Current snapshot

- Planned model cells: 320 (5 datasets x 8 methods x 8 paired seeds).
- At the snapshot used for the workbook: 304 complete, 16 incomplete.
- The 16 incomplete cells are all TissueMNIST.
- The corresponding workbook is `Table1_clean_only_5pct_progress_20260923.xlsx`.
