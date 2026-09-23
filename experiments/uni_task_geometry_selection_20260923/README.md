# UNI Task-Geometry Selection Pilot

This directory is a new, isolated experiment. It does not modify the prior
FD, UNI-manifold, WP1/WP3, Graph-A2, or test-only comparison directories.

## Purpose

The experiment has two stages:

1. A cheap frozen-UNI linear-probe audit. For each dataset and block, a seeded
   capped train candidate pool is split into a selection pool and a held-out
   train audit pool. Four methods select 50 examples per class: balanced
   random, herding, Graph-A2 (`k_graph=10`, `h_graph=2`), and direct UNI
   facility coverage.
2. A target-classifier stage. It consumes the saved selection manifest and
   trains the same image classifier for every method on the same source-train
   audit split. It is a separate Slurm entry point and is not run by the
   linear-probe job.

The train audit split is not the official validation or test split. This pilot
does not read either official split. The target-classifier script evaluates on
the held-out portion of the train candidate pool unless a future protocol
explicitly supplies a source-validation manifest.

## Round 3: allocation comparator

The later allocation study must include a direct empirical baseline in
addition to mass-only and dimension-derived allocation. For a frozen region
`r`, define the calibration distortion of `k` representatives as

```text
D_r(k) = sum_i w_i min_j d(z_i, z_j)
Delta_r(k) = D_r(k-1) - D_r(k)
```

The direct baseline allocates a fixed total budget by the largest measured
next-example gains `Delta_r(k)`, subject to the same pre-registered regional
floor and candidate caps as the other methods. Curves are estimated on a
train-calibration split and final selection is performed on a disjoint
train-selection split. This tests the proposal's central claim directly: if
the empirical distortion curve predicts useful allocations as well as or
better than the fitted finite-scale dimension, the dimension fit has no
incremental value for allocation.

The implementation is in `allocation_baselines.py`. It is not mixed into the
fixed per-class Round 1/2 pilot and is not submitted until those rounds show a
reproducible signal.

## Round 1

Methods use equal class budgets and equal training weights. The direct geometry
method minimizes average nearest-center chordal distance in normalized UNI
space through greedy facility location. Graph-A2 is imported from the pinned
GraphCov repository, with the FAISS import blocked so the existing CPU path is
used on this cluster.

## Round 2

Round 2 uses a common 10-example-per-class pilot inside the 50-example budget.
It compares random fill, pure geometry, task geometry, and a within-class
permuted-gradient control. The task feature is a regularized multinomial
linear-head gradient from the pilot. To keep CPU memory bounded, the flattened
gradient is projected once per dataset/block to a fixed 64 dimensions; this is
recorded in metadata and is not tuned against outcomes.

## Frozen defaults

- datasets: bloodmnist, organamnist, organsmnist, pathmnist, tissuemnist
- image/embedding size: 224
- maximum candidate examples per class: 1,000
- blocks: 6
- class budget: 50
- audit fraction: 20%
- Graph-A2: `k_graph=10`, `h_graph=2`, class-local selection
- task-geometry mixing: `tau=0.5`
- task-gradient projection: 64 dimensions
- linear probe: multinomial logistic regression, `C=1`, `lbfgs`, max 1,000 iterations

## HPC paths

The intended isolated HPC directory is:

```text
/home/xiaoyuxu2/uni_task_geometry_selection_20260923/
```

The shared inputs are read-only:

```text
/project/prj-sis01/xuxiaoyu/graph_select/
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_20260922/cache/embeddings_img224_smokefull/
```

## Outputs

`run_linear_probe.py` writes `round1/` and `round2/`, including result CSVs,
selection JSONL, per-block hashes, input hashes, and run metadata. The target
classifier writes to a separate `target_classifier/` directory.

## Interpretation

This pilot tests whether local/task geometry preserves classification-relevant
information under a fixed class budget. It does not test a dimension-to-quota
law. Dimension-based allocation is a later experiment only if direct geometry
has a reproducible signal and the allocation curve is validated on held-out
budgets.
