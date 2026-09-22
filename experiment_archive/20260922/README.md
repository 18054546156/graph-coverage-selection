# Experiment Archive: 2026-09-22

This directory preserves the source, executed configurations, launch scripts, Slurm logs, and final Table 1 workbook for two isolated experiments.

## Table 1 clean-only benchmark

- Protocol: 2% clean selection, selected-clean training, official clean-test evaluation.
- Matrix: five datasets x eight methods x eight paired seeds = 320 model cells.
- Seeds: 0, 1, 2, 3, 4, 5, 6, 2026; selection seed equals training seed.
- Status at 2026-09-22: 320/320 complete; the workbook audit found no metric-hash mismatches.
- `table1/code_snapshot/`: executed project source, runtime metadata, launcher, and queue worker.
- `table1/configs/`: 40 actual dataset/seed configuration snapshots.
- `table1/logs/`: 160 formal Slurm stdout/stderr files.
- `table1/Table1_clean_only_results_20260922.xlsx`: audited results and paper comparison.
- `table1/verification_summary.txt` and `table1/HANDOFF.md`: audit details and protocol caveats.

The comparison to paper Table 1 is descriptive, not an exact-protocol reproduction claim: this benchmark used eight paired seeds and final-epoch checkpoints, while the audited strict runner uses five trials and validation-selected checkpoints.

## D8 class-conditional OT benchmark

- Matrix: five datasets x eight paired seeds = 40 model cells.
- Status at 2026-09-22: 40/40 `run_complete.json` markers found on HPC.
- `d8_ot/implementation/`: method implementation, tests, documentation, and experiment scripts.
- `d8_ot/source_snapshot_clean/`: executed source snapshot without Git internals, caches, notebooks, or embedded binary assets.
- `d8_ot/configs/training/`: 40 Slurm task configs and 40 per-run runtime configs.
- `d8_ot/logs/`: all 43 captured Slurm and worker trace files.
- `d8_ot/d8_benchmark.slurm`, `d8_benchmark_worker.py`, `d8_integration_smoke.slurm`, `d8_ot.py`, `d8_registry.py`: exact files copied from the HPC execution directory.
- `d8_ot/FINAL_STATUS.md`: source locations and archive counts.

Large or non-source artifacts are intentionally excluded: datasets, pretrained weights, embeddings, model checkpoints, per-image predictions, and caches. No API keys, SSH private keys, passwords, or tokens are included.
