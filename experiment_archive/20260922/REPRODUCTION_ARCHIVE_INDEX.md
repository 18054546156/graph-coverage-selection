# Table 1 Reproduction Archive

This GitHub archive keeps the reproducibility evidence separate from the author's upstream source.

## Packages

- `table1/`: 2% clean-only Table 1 benchmark. Contains the audited source snapshot, 40 actual configs, 160 Slurm logs, the 320-run workbook, and the synced non-checkpoint formal metadata/metrics snapshot.
- `table1_clean_only_5pct/`: independent 5% clean-only benchmark. Contains the 5% launcher/dispatcher, smoke audit, formal configs, Slurm logs, non-checkpoint formal metadata/metrics, and the current workbook.

## What is included

- Executed Python source snapshots and shell/Slurm launchers.
- Actual per-dataset/per-seed YAML configs.
- Slurm `.out` and `.err` logs.
- `metrics.jsonl`, `run_config.json`, `run_complete.json`, training history, selection metadata, and SHA256 provenance.
- Excel summaries and audit notes.

## What is excluded

Model checkpoints (`*.pt`, `*.pth`, `*.ckpt`), datasets, pretrained weights, embedding/dynamics caches, prediction tensors, and large generated arrays (`*.npy`, `*.npz`) are excluded. They are generated artifacts rather than source/config/log evidence, and the raw cache/tensor volume is too large for a normal GitHub repository. The archive packages document this scope explicitly.

## Diff workflow

1. Compare `table1/code_snapshot/` with the pinned author source under `official-graph-coverage-8cf757a/`.
2. Read `table1/code_snapshot/UPSTREAM_COMPARISON.md` and `PROVENANCE.md` for commit and SHA information.
3. Compare `table1/run_clean_only.slurm` with `table1_clean_only_5pct/run_clean_only_5pct.slurm` to isolate the 2% to 5% protocol change.
4. Use each package's workbook and `formal/` metadata to connect a reported metric to its config, completion marker, and log.

The current working branch is `codex/table1-v11-execution`; the archive push is kept separate from unrelated research files in the worktree.
