# corr/ — shift-robustness benchmark code

Supports `paper_tdgs/report/proposal_shift_robustness_20261002.md` and
`paper_tdgs/results/shift_robustness/`. Question: methods tied on clean test
BA — do they stay tied under acquisition shift (MedMNIST-C corruptions)?
Pure inference + one controlled retrain; nothing here edits `graphcov/`.

## S0 — is MedMNIST-C's EXP corruption a realistic-size shift?

- `s0_colour_gap.py` — computes 8 colour statistics (R/G/B mean, HSV sat/val,
  circular hue cos/sin, luminance sd) per image, then the standardized mean
  difference between train/val (sampling floor), train/test (real shift), and
  clean-test/corrupted-test at each EXP severity. Streams the zipped `.npy`
  arrays in fixed-size chunks so it fits the 4G CPU-queue memory cap.
- `s0_colour_gap.slurm` — CPU-only launcher (`qos-normal`).

## S1 / S1b — corruption inference on existing checkpoints

- `corr_rescore_s1.py` — runs every MedMNIST-C corruption x severity against
  archived `final.pt` checkpoints. Pure inference, trains nothing.
  `--tree s1` (original Table-1 checkpoints) or `--tree s2` (colour-aug
  retrains, see below); `--discover-only` lists which checkpoints exist
  without touching the GPU.
- `corr_s1.slurm` — SLURM launcher: dead-GPU pre-flight probe, and the
  `GIT_CONFIG safe.directory` fix needed when the pipeline's
  `assert_graphcov_source_unchanged()` guard runs git on a repo owned by a
  different cluster account (git otherwise refuses with "dubious ownership").
- `harvest_s1.py` — collects `metrics_corr.jsonl` + each run's own clean
  metrics into the two flat JSON lists `shift_battery.py` reads.

## S2 — does standard colour augmentation remove the problem?

Retrains the same 180 (dataset x ratio x method x seed) selections from S1,
unchanged, with one difference: `ColorJitter(brightness=.4, contrast=.4,
saturation=.4, hue=.1)` added to the training transform. No crop/flip, so
colour is isolated from geometry.

- `s2_colour_pipeline.py` — monkeypatches `graphcov.run.data.get_train_transform`
  before the pipeline imports it, then runs the pipeline unmodified.
- `s2_worklist.py` — reads S1's discovered checkpoints and each one's own
  `selected_local_indices.npy`, writes the 180-cell worklist.
- `s2_train_one.sh` — trains one cell, replaying its S1 selection via
  `TD_SEL_FILE` (method arm = `precomputed`).
- `s2_pack.slurm` — work-stealing packer: workers `mkdir` a claim per cell so
  multiple jobs/accounts/GPU types can drain one shared worklist without
  double-processing. Runs `WORKERS_PER_GPU` trainings per card (ColorJitter is
  CPU-bound, so one training per GPU leaves most of the GPU idle).

## Metrics

- `shift_battery.py` — the full metric battery: rank correlation
  (clean vs EXP/STR), regret of the clean-BA #1 pick, H1 gate
  (tau_EXP < tau_STR, #1 flip), and the method x corruption interaction
  `I(m) = [BA(m)-BA(random)]_EXP - [BA(m)-BA(random)]_clean` with a paired
  t-stat across seeds. Works on prelim, S1, S1b, or S2 output alike.

## Order

S0 (CPU, no gate) -> S1 (confirmatory, blood+path) -> S1b (replication check,
organA/organS/tissue, inference only) -> S2 (decisive: does colour
augmentation remove the shift effect?) -> S3 (robust-selection method design,
gated on S2 passing — see `PREREG_s1b_s2_20261002.md`).
