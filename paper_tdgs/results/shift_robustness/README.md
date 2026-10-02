# Shift-robustness benchmark (proposal: `paper_tdgs/report/proposal_shift_robustness_20261002.md`)

Question: methods tied on clean BA -- do they stay tied under acquisition shift (MedMNIST-C)?

## prelim/ -- exploratory only, NOT citable as evidence
- `precheck_all_corr.json`, `precheck_all_clean.json` (+ path-only subsets): Oct 1 precheck.
  Usable grid: seeds 0,1,2 x 8 bench methods x blood/path x 2%/5% x 11 corruptions x 5 severities.
  No mv_mean. Seeds 0-2 are not the formal seeds. The seed-42/43 rows in it are a 20-row smoke
  test (1 corruption, 2 methods) and are ignored by the battery.
- `battery_prelim_seeds012.md`: full metric battery on the above.
  H1 gate passes at cell level (tau 3/4, #1 flip 3/4), but blood 2% #1-under-EXP = herding comes
  from the seed-2 outlier (memory `medmnistc-exposure-reversal-is-one-seed`). Use only as motivation.

## s1/ -- confirmatory S1 (formal seeds)
- Jobs: luhpc 36178 (corrS1_path), 36179 (corrS1_blood), submitted 2026-10-02 on hpcgpu107 (A100).
- Grid: 9 methods (8 bench + mv_mean) x seeds 42-46 x blood/path x 2%/5% x 11 corruptions x 5 severities;
  180/180 checkpoints discovered (`s1_discover.txt`).
- Checkpoint provenance (training hardware differs, inference is all A100): 7 baselines ~half
  `table1_a100_20260929` / half `table1_s4246_20260928`; graph_a2 from `round2_20260927` (14) +
  `tdgs_round1_20260927` (6); mv_mean from `mvf_20260930` (16) + `plus_a100_20260930` (4).
- Cluster output: `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_corruption_s1_20261002/`.
- Harvest: on cluster `python ~/corr/harvest_s1.py` -> `s1_corr.json`, `s1_clean.json`; copy here, then
  `python corr/shift_battery.py --corr .../s1/s1_corr.json --clean .../s1/s1_clean.json --out .../s1/battery_s1.md`

## Code (corr/)
- `corr_rescore_s1.py` inference (`--discover-only` lists checkpoints), `corr_s1.slurm` launcher,
  `harvest_s1.py` collector, `shift_battery.py` metric battery (works on prelim and S1 alike).

## s0/ -- colour-gap check (DONE 10-02, CPU job 36219)
See `s0/FINDINGS_s0.md`. Path real cross-centre gap = 0.64 (hue), EXP sev1 0.41-2.19 (no hue);
blood has no real shift (0.035 = floor). Matched-magnitude S1 re-analysis included there.

## PREREG_s1b_s2_20261002.md -- decision rules for S1b and S2, written before their results.

## s1b/ -- organA/organS/tissue inference (RUNNING, qiangzeng, RTX 4090 hpcgpu103)
Jobs 36235 organS, 36236 organA r.02, 36237 tissue r.02, 36238 organA r.05 (after 36235),
36239 tissue r.05 (after 36236). Output: tdgs_shared/runs/corr_s1b_20261002.
Harvest: `python harvest_s1.py --tag s1b --datasets organamnist organsmnist tissuemnist --out-root <that>`.

## s2/ -- colour-aug retraining (RUNNING, danranwang, H100 hpcgpu108; GPU0 dead, 3 cards x 2 trainings)
180 cells = S1 selections replayed via `precomputed`, ColorJitter(.4,.4,.4,.1), no crop/flip,
DETERMINISTIC_TRAINING=1, 1000 epochs. Packers 36233/36234 (mkdir claims in
tdgs_shared/runs/s2_coloraug_20261002/claims). Inference 36240/36241 queued afterany.
Harvest: `python harvest_s1.py --tag s2 --tree s2`.

## Code
Shared copy for all 3 accounts: `/mnt/prj01/hgrp-1502-5TB/tdgs_shared/code/shiftbench/` (logs/ inside).

## Not yet done
S3 DRO coreset -- only if S2 gates pass (see PREREG).
