# Real-training functional screen (2026-09-23)

Replaces the linear-probe endpoint of
[`../uni_task_geometry_selection_20260923/`](../uni_task_geometry_selection_20260923/)
with one real ResNet-18-from-scratch training run per selected set.

## Why this exists

The probe screen answers *"which functional of a selected set predicts **linear-probe**
balanced accuracy"*. That is the wrong question: this project has four recorded events
where an offline/probe criterion passed and real training disagreed, and the probe's
within-method rank agreement with real training measured only Spearman +0.119
(underpowered, `n=10/cell`, `se(rho)≈0.378`). See
[`../uni_task_geometry_selection_20260923/ANALYSIS.md`](../uni_task_geometry_selection_20260923/ANALYSIS.md)
§2–§3 for the full diagnosis.

This screen asks the question that actually matters:

```
selected set  ->  raw 224px MedMNIST images  ->  equal-weight ResNet-18 from scratch
              ->  official test split  ->  ba_real
```

## What runs

`real_training_functional_screen.py` reuses the probe screen's selection library and
functional definitions verbatim (`functional_screen.build_library`,
`.functionals`, `.dyn_functionals`) and swaps only the endpoint:
`run_linear_probe.metrics(...)` becomes `train_weighted.train_one_cell(...)`.
Everything about *which* sets get measured is therefore identical to the probe run,
which is what makes the two directly comparable.

Each output row carries all 26 functionals (`F_*`), the real endpoint `ba_real`, and
`endpoint = "resnet18_official_test_equal_weight_final_epoch"`, plus per-run
diagnostics (`train_seconds`, `ece15`, `worst_class_recall`, `per_class_recall`,
`selection_sha256`).

Weights are deliberately all-equal: the Voronoi/weighted-training branch was REFUTED
(ANALYSIS.md §2), so weighting is held fixed and only the *selection* varies.

## Design

| | |
|---|---|
| datasets | 5 (blood, organa, organs, path, tissue) |
| budget | 25 images per class |
| blocks | 2 independent pool/audit splits |
| selections per block | **225** = 60 random + 15 anchors x (1 unperturbed + 5 perturbation levels x 2 seeds) |
| training seeds | 3 (`--train-seed-offset` 0 / 100000 / 200000) |
| **total real training runs** | **6,750** |

Three seeds is sized to the measured noise floor, not picked by habit: per
`(dataset, arm)` cell, sd-from-selection (0.9-1.4pp) and sd-from-training-noise
(0.5-2.0pp) are the same order of magnitude (ANALYSIS.md §3). At `k=3` a true
`r=0.4` functional observes as `r≈0.32`, which reaches `t=3` at `n≈90`; every cell
here has 225.

## How to launch

`run_real_one_dataset.slurm` is the unit of work: **one dataset, one GPU**, so jobs
pack onto fragmented free capacity instead of waiting for a whole node.

```bash
DATASET=bloodmnist TRAIN_SEED_OFFSET=0 sbatch run_real_one_dataset.slurm
```

Output goes to `results/seed_${TRAIN_SEED_OFFSET}/${DATASET}_${SLURM_JOB_ID}/${DATASET}.jsonl`,
flushed per row, so a killed job loses at most one training run and partial results
are always analyzable.

`run_real_screen.slurm` is the multi-dataset-per-node variant, kept for when a whole
node is free.

## Runtime dependency note

On the cluster this directory also contains byte-identical copies of
`functional_screen.py`, `run_linear_probe.py`, and `train_weighted.py` from
`../uni_task_geometry_selection_20260923/`, because the driver imports them as
top-level modules. They are **not** duplicated in git — copy them in before running:

```bash
cp ../uni_task_geometry_selection_20260923/{functional_screen,run_linear_probe,train_weighted}.py .
```

`--graphcov-root` must point at a `graph_select` checkout (for
`graphcov.run.data.get_transform`); the driver sets `sys.modules["faiss"] = None`
first so `graphcov.run.graph`'s FAISS-GPU probe falls back to its CPU path instead of
crashing on this cluster's CUDA driver mismatch.

## Status at time of commit

Code is wired and smoke-tested; first real rows confirmed on disk with correct
`ba_real` / `endpoint` fields. The full 6,750-run measurement is **in flight**, not
complete. No functional-vs-`ba_real` conclusion exists yet, and none should be quoted
until the runs finish and `analyze_screen.py` is re-run against `ba_real`.
