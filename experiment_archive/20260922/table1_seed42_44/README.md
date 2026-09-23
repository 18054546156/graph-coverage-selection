# Table 1 clean-only seed 42/43/44 progress archive

Snapshot: 2026-09-23 15:14 HKT.

This archive is independent from the completed Table 1 seed 0/1/2/3/4/5/6/2026 runs. It tracks the new paired seed sweep:

- Ratios: 2% and 5%
- Datasets: OrganSMNIST, OrganAMNIST, PathMNIST, TissueMNIST, BloodMNIST
- Methods: random, el2n_top, forgetting, eva, facility, fps, herding, graph_a2
- Paired seeds: 42, 43, 44
- Expected model units: 2 x 5 x 3 x 8 = 240
- Protocol: clean selection -> selected clean training -> clean test
- `selection_seed == training_seed`

At this snapshot, 37/240 model units have a valid completion marker and clean severity-0 metric. They are all from the new 2% sweep. The new 5% sweep has no completed target-seed metrics at this snapshot. Incomplete rows remain in the workbook with blank ACC, BA, and worst-class recall fields.

## Files

- `Table1_clean_only_seed42_44_progress_20260923.xlsx`: workbook with `Summary`, `Runs`, and `Paper comparison` sheets.
- `AUTHOR_VS_CURRENT_CODE_AUDIT.html`: code-level comparison of the original author repository and the current reproduction implementation.
- `workbook_inspection.ndjson`: artifact-tool inspection output used for workbook verification.

## HPC roots

- 2%: `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_20260922`
- 5%: `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_5pct_20260922`

The workbook records the remote run path, completion-marker path, metrics path, config hash, and selection hash for completed rows. Checkpoint files and large derived tensors are not included in this GitHub archive.
