# D8 Method and Migration Into the Clean Benchmark

## Research question

At a fixed 2% per-class sample budget, can a subset chosen to preserve the source-training class-conditional feature distribution train the same downstream classifier better than Graph-A2's graph-coverage subset?

The mechanism hypothesis is narrow: a representative set can cover many neighborhoods yet assign the wrong *implicit mass* to within-class modes once every selected image is trained with equal weight. D8 asks whether explicitly matching class-conditional mass reduces that mismatch. This is not yet a novelty or performance claim.

## Mathematical target

For frozen source-training embeddings (z_i=f(x_i)/\|f(x_i)\|_2), class (c), class quota (q_c), and selected original indices (S_c\subseteq V_c):

\[
P_c=\frac1{n_c}\sum_{i\in V_c}\delta_{z_i},\quad
Q_{S,c}=\frac1{q_c}\sum_{j\in S_c}\delta_{z_j},\quad
\min_{S:|S_c|=q_c}\frac1C\sum_cW_1(P_c,Q_{S,c}).
\]

The transport plan (T_c\) for fixed (S_c) obeys (T_c\mathbf1=\mathbf1/n_c), (T_c^T\mathbf1=\mathbf1/q_c), (T_c\ge0), with cost (\|z_i-z_j\|_2). The selector receives the already-audited per-class quota from the benchmark; it must not silently alter the benchmark's rounding.

## Implemented heuristic

The isolated `d8_ot.py` currently:

1. L2-normalizes the provided embeddings and partitions only by source-train labels.
2. Initializes (q_c) distinct source-train examples from the supplied seed.
3. Computes a classwise entropic OT coupling to uniform source and selected-point masses using log-domain Sinkhorn.
4. Updates each support toward its coupling-weighted geometric median.
5. Projects the updated supports to distinct actual source-train rows; repeats to a fixed iteration limit or stable indices.
6. Returns original row indices and per-class numerical diagnostics.

This is a **heuristic approximation**. Epsilon, initialization, outer iteration count, geometric-median updates, and projection can affect the selected set. Do not describe its output as the exact minimum of (J(S)). Parameters must be frozen from source-train/source-validation evidence before official clean-test evaluation.

## Difference from current methods

| Method | Directly optimized signal | What the selected examples represent |
|---|---|---|
| Graph-A2 | Greedy facility coverage over (K=A_{sym}+A_{sym}^2) on a kNN graph | Points with broad graph reach/coverage |
| FDMat | OT-derived ranking against an estimated feature target built using class information and class-center approximation | Samples favored by the paper's estimated target/transport score |
| Herding | Greedy feature/kernel mean matching | First-moment or kernel-mean representation |
| D8 prototype | Entropic OT between the full source-train empirical class measure and a uniform measure on selected real points | A fixed-budget, equal-mass approximation to each source class distribution |

The D8 difference is mathematically inspectable, but it is close to existing OT representative-subsampling work. It may collapse to an adaptation rather than a new algorithm after full prior-art comparison.

## Isolated integration recipe

Do not patch the active benchmark source or its `formal/` outputs. Make a separate code snapshot named `sources/table1_d8_ot_20260922/` and a separate result root named `table1_d8_ot_20260922/`. Reuse only immutable inputs after verifying identity and permissions:

- clean MedMNIST data: `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/data/medmnist/`;
- the existing Python environment: `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/envs/medmnistc-py311/`;
- UNI weights / embeddings only if the checkpoint, preprocessing, sample ordering, feature dimension, and cache namespace match the frozen benchmark exactly.

The executable registry adapter is in `d8_registry.py` and is unit-tested against the GraphCov registry contract. In a *copy* of `graphcov/run/selection.py`, copy `d8_ot.py` and `d8_registry.py` beside it, then append:

```python
from .d8_registry import register_d8_method
register_d8_method(register_method)
```

The import/registration smoke must run in that isolated source copy before sync. Add `d8_ot` to the *new* config's method list only.

The exact import path and keyword override behavior must be tested against the pinned clean-only source snapshot before syncing. Add `d8_ot` to the *new* config's method list only. Keep `CLEAN_ONLY=1`, ratio `0.02`, `selection_seed=training_seed`, final-epoch checkpoint rule, 224 input, 1000 training epochs, deterministic runtime, and all other frozen settings identical to the benchmark. Set the D8 output root explicitly; never let it write under the existing benchmark root.

## Optimization route without test leakage

1. First benchmark the current heuristic unchanged; do not tune to the clean-test scores.
2. Use only source-train/official source-validation to compare a small predeclared epsilon set and choose one global setting across the five datasets. Prefer validation BA of the frozen downstream recipe plus an independent source distribution diagnostic; do not optimize only the same Sinkhorn score D8 already minimizes.
3. Compare a disclosed FDMat-UNI adaptation, Graph-A2, Herding, balanced Random, and (if source-defined strata exist) subtype-proportional Random. This distinguishes OT contribution from center sampling and simple within-class mode proportions.
4. Keep the objective single-purpose. Do not stack multi-view robustness, difficulty, boundary bonuses, or Graph-A2 kernels into D8 before the base mechanism survives.
5. Freeze code/configuration, then evaluate the official clean test once for the paired 40-cell comparison. A later corruption study is a separate experiment and cannot retroactively tune D8.

## What "beat Graph-A2" must mean

Report paired (\Delta BA_{d,s}=BA_{D8,d,s}-BA_{GraphA2,d,s}) for every dataset (d) and seed (s\in\{0,1,2,3,4,5,6,2026\}); include per-dataset means and intervals, all class recalls, worst-class recall, selection/runtime/memory diagnostics, and failed cells. Do not treat the 40 cells as 40 independent datasets. Predeclare a practical-effect threshold and analysis before the test comparison. If gains are confined to one dataset/seed, are within noise, or are matched by FDMat/Herding/simple stratified sampling, do not claim D8 beats the method.
