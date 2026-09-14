# v29 reliability code preparation

This directory contains the prepared and runnable medical data-selection
reliability experiment.

## What is included

- `repos/`: pinned Git repositories used for the experiment or metric audit.
- `sources/google_uq/`: pinned source fallback for the Google UQ files that
  cannot be checked out on Windows.
- `reliability_code_prep.ipynb`: the only experiment code file. It imports
  upstream functions, defines the small adapters inline, prints commit status,
  runs tiny metric checks, and records the full-run command templates.
- `reliability_full_experiment.ipynb`: the long-run experiment code.
  It calls the pinned GraphCov selectors and MedMNIST-C dataset class, trains
  final-epoch ResNet-18 checkpoints, saves per-sample predictions, and writes
  per-run clean/corruption metrics. It resumes only after checking a complete
  run marker and all expected prediction files.
- `clone_manifest.json`: URLs, commits, roles, and the Google Windows fallback.

## Important status

The Google Research monorepo contains a Linux-style filename that Windows
cannot check out. Its clone metadata is retained in `repos/google-research-sparse`
at commit `08a8d6736475776f42ffac23b2c13111a28e5795`; the four required UQ
source files are copied into `sources/google_uq/` and are used as a fallback
for manual verification. This is recorded rather than hidden.

The preparation notebook intentionally does not train. The full notebook does
download official 224-pixel MedMNIST files when absent, creates missing
MedMNIST-C files through the pinned `DatasetManager`, trains, and evaluates.
It never uses the validation split. The current data-preparation package and
its status record are in `reliability_medmnistc_ab/`.

## Fixed experiment contract

Use official train as `D`, official test as `E`, and do not use MedMNIST `val`
for selection, checkpoint choice, temperature fitting, or threshold choice.
Compare `random`, `el2n_top`, `forgetting`, `eva`, `facility`, `fps`, `herding`,
and `graph_a2` at 2% first, then 5%, with identical downstream training,
final-epoch evaluation, and shared seeds. Save `sample_id`, `y_true`,
`logits`, and `probs` for every clean and corrupted test pass.

## Long-run environment variables

The full notebook defaults to all five target datasets, ratios `0.02,0.05`,
selection seed `42`, training seeds `42,43,44,45,46`, 1000 epochs, 224-pixel
inputs, and the `uni` embedding source. Set `DATASETS`, `RATIOS`,
`SELECTION_SEED`, `TRAINING_SEEDS`, `EPOCHS`, `DYNAMICS_EPOCHS`,
`MEDMNIST_ROOT`, `MEDMNISTC_ROOT`, `RELIABILITY_OUT`, `SELECTION_OUT`, and
`GRAPH_CACHE` before execution. `SEEDS` remains accepted as a legacy alias for
`TRAINING_SEEDS`. The `uni` source requires the same `timm`/model weights used
by the pinned GraphCov release. A failed dependency or missing model must be
fixed rather than silently replaced by another encoder.

Example remote execution from this directory:

```bash
DATASETS=pathmnist,organsmnist RATIOS=0.02,0.05 \
  SELECTION_SEED=42 TRAINING_SEEDS=42,43,44,45,46 \
  jupyter nbconvert --to notebook --execute reliability_full_experiment.ipynb \
  --output reliability_full_experiment.executed.ipynb
```

Each run writes to a unique path below `RELIABILITY_OUT/<dataset>/<method>/`
with separate selection and training seed components. Selection artifacts are
also written below `SELECTION_OUT/<method>/<dataset>/`. The notebook evaluates
all corruption names in the pinned registry by default; use `CORRUPTIONS` only
for an explicitly labelled smoke or pilot run.

## Related source repositories

The `repos/` directory contains only sources pinned in `clone_manifest.json`:
GraphCov, MedMNIST-C, fd-shifts, Google Research metadata/fallback,
temperature-scaling, and uncertainty_ICLR. They are references and are not
reimplemented. Each upstream license is retained. The benchmark's primary
selection implementation remains GraphCov; the other repositories are used
for corruption generation or metric cross-checks.

## Metric convention

`BA` is the mean of per-class recall. `NLL` and multiclass Brier use the true
class probability and one-hot vector respectively. `ECE` is top-label ECE with
15 equal-width bins. `Risk@80%` accepts exactly the highest-confidence 80% (up
to integer rounding). Local `AURC` is in `[0, 1]`; fd-shifts' display AURC is
scaled by 1000, so do not subtract those values without rescaling.
