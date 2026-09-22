# Table 1 clean-only benchmark handoff

Updated: 2026-09-22 18:28 HKT, Asia/Shanghai

## Scope

This is a separate clean-only benchmark: select 2% of the clean training set, train on those selected clean images, and evaluate on the official clean test split. No MedMNIST-C corruption, pixelation, severity condition, corrupted selection, or corrupted training is used.

- Datasets: OrganSMNIST, OrganAMNIST, PathMNIST, TissueMNIST, BloodMNIST
- Methods: Random, EL2N, Forgetting, EVA, Facility, FPS, Herding, Graph-A2
- Paired seeds: 0, 1, 2, 3, 4, 5, 6, 2026; selection seed equals training seed
- Planned size: 5 datasets x 8 methods x 8 seeds = 320 model results
- Main comparison metric: clean-test balanced accuracy (BA)
- Training: 1000 epochs; final-epoch checkpoint; deterministic training enabled
- Graph-A2: global selection, k=50, hops=2

## Live status

All 320/320 result cells have a complete marker and a matching `metrics.jsonl` SHA256. The final cell, TissueMNIST / seed 2026 / Graph-A2, completed in Slurm job `33378` on `hpcgpu109` with exit code 0 after 6 h 23 min. No Table 1 metric hash mismatches were found.

The workbook is refreshed from the complete 320/320 Table 1 benchmark. All five datasets and all eight methods have eight completed paired seeds. The refresh uses completion markers plus SHA256 checks; a partial checkpoint is not counted.

## Files in this package

- `Table1_clean_only_results_20260922.xlsx`: summary, per-run Table 1 metrics, and paper Table 1 comparison. It contains only the 320 clean-only benchmark cells.
- `configs/`: all 40 actual Slurm-run `config.yaml` snapshots, organized by dataset and seed.
- `logs/`: all 160 captured formal `.out`/`.err` Slurm logs, including the completed job `33378` log.
- `code_snapshot/`: executed pipeline, phase scripts, Slurm launchers, queue worker, runtime manifest, and upstream comparison guide copied from the HPC execution tree; `PROVENANCE.md` records source paths and checksums.
- `verification_summary.txt`: marker counts, hash checks, formula checks, and paper comparison for the workbook snapshot.

To diff against the author's exact source version, start with `code_snapshot/UPSTREAM_COMPARISON.md`. The experiment branch is based on upstream GraphCov commit `8cf757adc4c333dc1427d511f0de2f246d15ebac`; the upstream `graphcov/` tree is unchanged, while the clean-only experiment wrapper and its execution artifacts are added separately.

The exact submitted launcher and queue worker are preserved in `code_snapshot/`; the clean-only pipeline and helper scripts are preserved under `code_snapshot/reliability_medmnistc/`. The root-level `table1_clean_only_20260922/` directory contains earlier local planning artifacts and is not a substitute for the executed source snapshot.

## Paper Table 1 comparison

The workbook reports each dataset-method position as current mean BA, sample SD and sample variance across eight paired seeds, alongside the paper mean and reported SD. Variance is in squared percentage points (pp^2), uses ddof=1 for this benchmark, and is derived as reported SD squared for the paper. Deltas are current minus paper.

Across the 40 dataset-method positions, the equal-weight mean signed BA gap is +0.28 pp and mean absolute BA gap is 1.74 pp; 34/40 positions are within 3 pp. Mean current sample variance is 6.68 pp^2. The equal-weight mean SD gap is +0.30 pp and mean variance gap is +0.69 pp^2. These summarize different dataset-method cells equally; they are not a pooled patient-level estimate.

The current benchmark uses eight paired seeds and a final-epoch checkpoint. The audited strict runner uses five trials and a validation-selected checkpoint. These are descriptive gaps, not exact-protocol reproduction errors or evidence that a method is superior to the paper.

## HPC paths

- Table 1 source: `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/sources/table1_clean_only_20260922/`
- Table 1 formal results: `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_20260922/formal/`
- Table 1 logs: `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_20260922/logs/`

Model checkpoints, datasets, and embedding caches are intentionally not copied to GitHub.
