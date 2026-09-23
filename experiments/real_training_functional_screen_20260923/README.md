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

`real_training_functional_screen.py` reuses the probe screen's selection library
*code* and functional definitions (`functional_screen.build_library`,
`.functionals`, `.dyn_functionals`) and swaps only the endpoint:
`run_linear_probe.metrics(...)` becomes `train_weighted.train_one_cell(...)`.

The library *parameters* differ, so "identical to the probe run" is not true and the
earlier wording here ("verbatim") was wrong. The probe used
`--n-random 120 --n-perturb-seeds 8` (735 selections/block); this screen uses
`60 / 3` (300/block). Because `build_library` seeds each draw on quantities that do
not depend on those counts -- random draws on `seed + s`, perturbations on
`seed + 7919*m + s` -- the real library is a strict *subset* of the probe library:
`random_0..59` and every `s <= 2` perturbation denote the same selections in both.
Joining on `name` (not on index) therefore still gives a paired probe-vs-real
comparison, at n=300/block rather than the n=225 originally claimed.

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
| selections per block | **300** = 60 random + 15 anchors x (1 unperturbed + 5 perturbation levels x **3** seeds) |
| training seeds | 3 (`--train-seed-offset` 0 / 100000 / 200000) |
| **total real training runs** | **9,000** |

Three training seeds is sized to the measured noise floor, not picked by habit: per
`(dataset, arm)` cell, sd-from-selection (0.9-1.4pp) and sd-from-training-noise
(0.5-2.0pp) are the same order of magnitude (ANALYSIS.md §3). At `k=3` a true
`r=0.4` functional observes as `r≈0.32`, which reaches `t=3` at `n≈90`.

### Why `--n-perturb-seeds` must be >= 3 (fixed 2026-09-23, was 2)

`analyze_screen.py` centres the functional and the endpoint within every
`(dataset, block, family, m)` group and skips groups smaller than 3. At
`n_perturb_seeds = 2` **every** perturbation group has exactly 2 rows, so the entire
perturbation ladder was silently dropped from analyses C / D / D2 -- the
decision-relevant cell. Measured on the first 379 rows: 207 (55%) contributed
nothing, and the only family reaching the decision cell was `random`.

This does not fix itself with more compute. At full scale, 165 of every 225 rows
(73%) are anchor ladders sitting in groups of 1 or 2, so finishing the original
design would have left the perturbation sweep -- the whole reason the library is
built this way (ANALYSIS.md §5) -- contributing zero.

Raising the count is additive rather than invalidating: `perturb()` seeds on
`seed + 7919*m + s`, independent of `n_perturb_seeds`, so `s = 0, 1` denote the same
selections as before and only `s = 2` is new work. What *does* change is
`global_screen_index` (library length 225 -> 300), which is why `--skip-existing`
matches on `(dataset, block, name, train_seed)` and never on the index.

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

`submit_complete_screen.sh` lays out the shards for one train-seed offset across both
blocks and is idempotent (see `--skip-existing` above), so it is safe to re-run after
a job dies:

```bash
TRAIN_SEED_OFFSET=100000 ONLY_SHARD=0 ./submit_complete_screen.sh   # block 0
TRAIN_SEED_OFFSET=100000 ONLY_SHARD=1 ./submit_complete_screen.sh   # block 1
```

It submits one shard index across all five datasets rather than all shards of one
dataset, because `qos-high-gpu` runs 5 jobs per user at a time and submission order is
therefore priority order.

## The training protocol is NOT the paper's, deliberately

This matters when reading `ba_real`, and was undocumented until 2026-09-23:

| | paper §3.1 / `Table1_Reproduction_By_Method` | this screen |
|---|---|---|
| epochs | 1,000 | **200** |
| batch size | 256 | **128** |
| cosine `eta_min` | 0 | **1e-4** |
| lr / momentum / weight decay | 0.1 / 0.9 / 5e-4 | same |
| resolution, augmentation | 224, none | same |

At budget 25/class the training set is only 200-275 images, so in optimizer *steps*
this screen runs 2.5-5x shorter than Table 1 (blood: ~400 steps vs ~1,000; organa:
~600 vs ~3,000). Two consequences:

1. `ba_real` is **not** comparable in level to any Table 1 number.
2. Whether the *ranking* of selections is stable between ~400 and ~1,000 steps is an
   untested assumption. This project has four recorded cases of a conclusion flipping
   when the endpoint changed (ANALYSIS.md §2); probe -> real training is now measured,
   but 200ep/bs128 -> 1000ep/bs256 is not.

Testing (2) is cheap and has not been done: re-train ~40 selections from one
(dataset, block) at `EPOCHS=1000 --batch-size 256` and rank-correlate against the
200-epoch `ba_real`. ~3 GPU-h.

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

## Status 2026-09-23 20:10

**In flight, ~6% complete. Nothing here is a conclusion.**

597 rows measured. The first wave of shards was launched with hand-picked index
ranges that covered only a middle slice of block 1 (`[242,346)` on blood,
`[256,353)` on organa, ...), so **block 0 had 3 rows in total** and the screen's
central guard -- sign and significance must agree on both independent pool/audit
blocks -- could not be evaluated at all. Jobs 33884-33893 now cover both blocks for
`--train-seed-offset 100000`.

A provisional read of the 379 rows from the first wave, with every caveat above
(single block, single training seed, `random` the only family reaching the decision
cell, pathmnist almost empty):

- Of the **10 probe-endpoint survivors**, **9 do not survive at `ba_real`**.
  `dist_mean` is the only one left (`t=-3.93`); `moment1` `-2.35`, `dist_p90`
  `-2.02`, `mmd2` `-1.67`, `aum_mean` `+1.44`, `purity` `+0.54`, `mass_gini` `-0.54`,
  `moment2` `+0.05` all fall below the `|t|>3` bar. This is the measurement the
  thread was built to make, and it says the probe manufactured false positives --
  which is why no method was built on that list.
- Controlling for `dist_mean`, seven functionals still add signal: `logdet` `+3.75`,
  `mmd2` `-3.74`, `dist_p90` `-3.69`, `xclass_sep` `+3.32`, `spread` `+3.32`,
  `moment1` `-3.27`, `density` `-3.21`. Mean coverage alone is not sufficient.
- The pre-registered prediction that `difficulty_ks` would add signal beyond
  geometry is **falsified**: partial `t=+0.39`, sign reversed. `difficulty_w1`
  `-1.32`, `el2n_early_mean` `-0.01`.
- `kcenter` (the FPS max-min objective) sits **-3.12 sd** below the random draws in
  the baseline leaderboard, `-7.53` on tissuemnist. With Table 1 (FPS loses to
  Facility Location on 5/5 datasets) and the probe screen (`dist_max` never a
  survivor), that is three independent lines against max-coverage objectives.

None of this has cleared two-block replication yet. Treat it as a hypothesis set.
