# Table 1 clean-only benchmark handoff

Updated: 2026-09-22 18:03 HKT, Asia/Shanghai

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

All 320/320 result cells now have a complete marker and a matching `metrics.jsonl` SHA256. The final cell, TissueMNIST / seed 2026 / Graph-A2, completed in Slurm job `33378` on `hpcgpu109` with exit code 0 after 6 h 23 min. No Table 1 or D8 metric hash mismatches were found.

The workbook was refreshed at 17:56 HKT and contains the complete 320/320 Table 1 benchmark. All five datasets and all eight methods have eight completed paired seeds. The refresh used completion markers plus SHA256 checks; a partial checkpoint is not counted.

## Files in this package

- `Table1_and_D8_live_results_20260922.xlsx`: summary, per-run Table 1 metrics, paper Table 1 comparison, and paired D8 vs Graph-A2 comparison. This snapshot contains 320/320 Table 1 cells and 40 exact D8/Graph-A2 pairs.
- `configs/`: 40 actual Slurm-run `config.yaml` snapshots, organized by dataset and seed.
- `logs/`: 160 formal `.out`/`.err` Slurm logs, including the completed job `33378` log.
- `code_snapshot/`: run code and launch/config source copied from the HPC execution tree; `PROVENANCE.md` records source locations and checksums.
- `verification_summary.txt`: marker counts, hash checks, paper comparison, and D8 paired comparison for the workbook snapshot.

The exact submitted launcher and queue worker are preserved in `code_snapshot/`; the clean-only pipeline and helper scripts are preserved under `code_snapshot/reliability_medmnistc/`. The root-level `table1_clean_only_20260922/` directory contains earlier local planning artifacts and is not a substitute for the executed source snapshot.

## Paper Table 1 comparison

The workbook reports each dataset-method position as current mean BA, sample SD and sample variance across eight paired seeds, alongside the paper mean and reported SD. Variance is in squared percentage points (pp^2), uses ddof=1 for this benchmark, and is derived as reported SD squared for the paper. Deltas are current minus paper.

Across the 40 dataset-method positions, the equal-weight mean signed BA gap is +0.28 pp and mean absolute BA gap is 1.74 pp; 34/40 positions are within 3 pp. Mean current sample variance is 6.68 pp^2. The equal-weight mean SD gap is +0.30 pp and mean variance gap is +0.69 pp^2. These summarize different dataset-method cells equally; they are not a pooled patient-level estimate.

The current benchmark uses eight paired seeds and a final-epoch checkpoint. The audited strict runner uses five trials and a validation-selected checkpoint. These are descriptive gaps, not exact-protocol reproduction errors or evidence that a method is superior to the paper.

## D8 comparison

D8 finished its 40/40 training cells. It has no active Slurm jobs and will not be extended. All 40 D8 results have an exact Graph-A2 match. The paired BA difference (D8 minus Graph-A2) is -1.13 percentage points (95% paired t interval: -1.95 to -0.30), with D8 higher on 10/40 pairs. This supports no claim that D8 beats Graph-A2; it is an internal comparison, not a field-wide SOTA claim.

## HPC paths

- Table 1 source: `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/sources/table1_clean_only_20260922/`
- Table 1 formal results: `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_20260922/formal/`
- Table 1 logs: `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_20260922/logs/`
- D8 results: `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/d8_class_conditional_ot_20260922/training/`

Model checkpoints, datasets, and embedding caches are intentionally not copied to GitHub.
