# The instrument — why every number is reported the way it is

Read this before interpreting any table. Most wrong conclusions in this project
came from reading a real number against the wrong noise scale.

## Determinism

Training is **bit-deterministic per seed**: `DETERMINISTIC_TRAINING=1`,
`PYTHONHASHSEED=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` → 48/48 repeats identical.

This was **not** true before 2026-09-21: the env var was dead code and replicate
spread was 2–7pp. A "noise floor" of 2.05–2.35pp that circulated earlier was an
artifact of pre-fix rows (`gpu_name=None`) and is **retracted**.

Practical consequence: the same training seed on two arms shares its
initialisation exactly, so **pair by training seed** and the difference is
attributable to the subset. Also: `graph_a2`, `herding`, `facility`, `fps` are
deterministic *selectors* (`n_distinct_selections=1`), so there is no
selection-seed variance to average over.

## Noise scales — the ruler

| quantity | value |
|---|---|
| σ_run (same subset, different seed) | **1.37pp** |
| σ_subset (different subset inside the good region) | **0.82pp** |
| paired σ_d | 2–3pp |
| MDE per cell, n=5 | 1.2–8.8pp |
| **MDE per cell, n=3** | **2.4–17.6pp** (×2.0: SE ×√(5/3)=1.29, t-quantile 4.30/2.78=1.55) |
| random-draw BA spread (929 subsets) | 14–25pp |

Two things follow, and both are enforced in `code/ladder.py`:

1. **`p > 0.05` at n=3 carries no information.** The effect we are chasing outside
   tissuemnist is +0.81pp against a per-cell MDE of 2.4–17.6pp. Sign consistency
   across datasets is the primary evidence; the dataset-level exact sign test has
   its own p (5/5 → 1/32 = 0.031).
2. **Pairing by training seed buys less than it looks** — σ_d ≈ √2·σ, i.e. the
   arms' seed effects are near-independent. It is still the right design, but it
   is not a variance-reduction miracle.

Retired endpoints: `ba_drop` is the worst endpoint measured and is biased toward
weak methods. Do not use it.

## Effective sample size is not 5 datasets

- **organamnist and organsmnist are the same 201 LiTS CT volumes** — one vote, not
  two.
- pathmnist has only 0.61pp of headroom, so it cannot pass any 1pp gate on its own.
- Effective independent N is **2–3**. A pooled SE over 15 cells is printed but must
  never be read as 15 independent observations.
- MedMNIST training arrays are **shuffled**, so patient/volume stratification is
  unavailable (adjacency gap ≈ 0 on 5/5). No patient-level splits are possible.

## Dataset priors (all measurable before training)

| dataset | n | C | bpc @2% | subset-effect sd | **cross-class edge share** |
|---|---|---|---|---|---|
| pathmnist | 89,996 | 9 | 199 | 4.07pp | **0.6%** |
| organamnist | 34,561 | 11 | 62 | 2.29pp | 20.3% |
| organsmnist | 13,932 | 11 | 25 | 0.80pp | 32.1% |
| bloodmnist | 11,959 | 8 | 29 | 1.42pp | 9.9% |
| tissuemnist | 165,466 | 8 | 413 | 1.07pp | **58.5%** |

`Spearman(cross-class share, subset-effect sd) = −0.80`. This was long read as
"datasets with lots of cross-class material have no headroom". Act 4 of the story
argues the opposite reading, and the round-2 mask arms test it directly.

Report effects as a fraction of headroom, not in raw pp, whenever comparing across
datasets.

## The zero-training replayer, and its limit

929 random subsets replay **bit-exactly** from labels alone, which makes
random-family questions free. But `R²_max` is only 0.48–0.72 and tissuemnist is a
structural null, so the replayer **cannot** be used for flatness tests or for any
question about a non-random selector.

## Why frozen-UNI proxies keep failing

A linear probe on frozen UNI has `rho = +0.085 (p=0.135)` against real subset
quality, and — the decisive part — **its dynamic range is inversely proportional
to the real subset effect**:

| dataset | probe OOF BA | `frac(d<0.05)` | subset-effect sd |
|---|---|---|---|
| pathmnist | 0.9974 | **0.981** | 4.07pp |
| bloodmnist | 0.9796 | 0.856 | 1.42pp |
| organamnist | 0.9355 | 0.617 | 2.29pp |
| organsmnist | 0.8403 | 0.540 | 0.80pp |
| tissuemnist | 0.4787 | — | 1.07pp |

`Spearman(frac(d<0.05), subset-effect sd) = +0.8`. The probe is most confident
exactly where the final learner has the most room to differ. This one table
explains every L1/L2/L3 surrogate failure, the K-epoch proxy's half-result, and
round 1's `tdgs_d` = −2.51pp — and it is why the `proxy_regime` diagnostic was
retired rather than resubmitted.

Probe validity check, run at staging time and reported (never used to build the
selector): worst class matches the final models' on **3/5** exactly (path 7,
blood 3, tissue 1), organamnist recovers the same confusion pair {4,5} with the
members reversed, organsmnist is the only genuine miss — and is also the dataset
where the final models themselves only agree 0.55 on the worst class.

## Metric discipline

The metric battery is **closed**: 182 paired tests, 6 nominal p<0.05 against 9.1
expected, **0 survive BH q<0.10**. The apparent "10/10 calibration unanimity" is
fake — that group has **1.31 effective dimensions** (the selective group, 1.11).

The robustness axis is also closed: `r(corruption error, clean error) = 0.970–0.996`
in 4/4 cells, so corruption reproduces the clean ranking. **Do not** run the 68
GPU-h organ/tissue corruption rescore.

**Conclusion: stop adding metrics, add seeds.** Main text reports 5 metrics (worst
recall, BA, per-class recall, mean±SD over n seeds, rBE); 12 more live in the
appendix.
