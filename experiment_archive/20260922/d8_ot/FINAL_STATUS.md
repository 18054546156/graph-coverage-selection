# D8 Execution Archive

Snapshot date: 2026-09-22 (HPC)

## Verified execution state

- Formal benchmark: five datasets x eight paired seeds = 40 cells.
- HPC completion-marker count: 40 `run_complete.json` files.
- Captured log count: 43 files under the D8 execution `logs/` directory, including Slurm stdout/stderr and worker JSONL traces.
- Captured config count: 40 `config.yaml` files and 40 per-run `run_config.json` files.
- This archive contains source/configuration/log artifacts only; it does not contain trained weights, datasets, embedding caches, selected-index arrays, or prediction tensors.

## HPC locations at execution time

- Experiment root: `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/d8_class_conditional_ot_20260922/`
- Executed source snapshot: `source_snapshot/`
- Formal configs and results: `training/<dataset>/seed_<seed>/`
- Slurm and worker logs: `logs/`

## Archive layout

- `implementation/`: method and experiment documentation, local implementation, integration patch, and tests.
- `source_snapshot_clean/source_snapshot/`: code snapshot used for execution, excluding nested Git metadata, generated caches, notebooks with embedded outputs, and binary media/data assets.
- `configs/training/`: exact per-task config files and runtime config snapshots.
- `logs/`: copied HPC execution logs.
- Root-level launcher and method files are copies from the HPC execution directory.

The earlier `D8_BENCHMARK_EXECUTION.md` describes a pre-completion snapshot. For final completion counts, use this file and verify against the captured run configs/logs if reproducing.
