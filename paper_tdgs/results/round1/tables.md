# Round 1 — all numbers

Job **34970**, 4 array tasks (2:32–2:58 h each), **75/75 cells COMPLETED**.
ratio = 0.02, training seeds 42/43/44, `selection_seed` = 42 for every cell.
Source: `tdgs_round1_harvest.json` (75 rows, includes per-class recalls).

Validity gate passed before any of this was read: the `graph_a2` arm reproduces
the archived organsmnist `selected_indices.npy` **order-exact**; per-class quota
asserts pass on all 25 arm×dataset selections.

---

## 1. Balanced accuracy, mean over 3 seeds (%)

| dataset | graph_a2 | **tdgs_cls** | tdgs_d | tdgs_du | tdgs_perm |
|---|---|---|---|---|---|
| pathmnist | 82.74 | **83.16** | 78.94 | 78.08 | 79.56 |
| organamnist | 87.19 | **87.42** | 86.98 | 86.84 | 86.25 |
| bloodmnist | 80.85 | **82.57** | 81.53 | 81.93 | 81.08 |
| organsmnist | 63.12 | **63.98** | 59.70 | 59.95 | 61.74 |
| tissuemnist | 42.86 | **48.41** | 45.83 | 46.70 | 45.89 |

## 2. Worst-class recall, mean over 3 seeds (%)

| dataset | graph_a2 | tdgs_cls | tdgs_d | tdgs_du | tdgs_perm |
|---|---|---|---|---|---|
| pathmnist | **41.09** | 39.59 | 35.47 | 37.13 | 36.34 |
| organamnist | 69.27 | **71.38** | 66.39 | 59.98 | 59.29 |
| bloodmnist | 54.69 | **56.30** | 49.28 | 45.71 | 49.80 |
| organsmnist | 33.58 | **37.56** | 26.70 | 30.45 | 33.09 |
| tissuemnist | 28.76 | **40.28** | 38.77 | 36.99 | 36.65 |

## 3. Paired ladder (pp, paired by training seed, n=15 cells / 5 datasets)

| contrast | BA pooled (se) | worst pooled (se) | reading |
|---|---|---|---|
| `tdgs_cls − graph_a2` | **+1.76** (0.60) | **+3.54** (1.64) | **5/5 datasets same sign**, 11/15 cells |
| `tdgs_d − tdgs_cls` | **−2.51** (0.53) | −5.70 (1.40) | 5/5 negative — difficulty as demand weight is harmful |
| `tdgs_du − tdgs_d` | +0.10 (0.49) | −1.27 (1.63) | direction term adds nothing |
| **`tdgs_du − tdgs_perm`** | **−0.20** (0.67) | −0.98 (1.58) | **decisive null: direction structure is inert** |
| `tdgs_du − graph_a2` | −0.65 (0.97) | −3.43 (2.20) | the full TDGS loses to the author's method |
| `tdgs_d − graph_a2` | −0.76 (0.75) | −2.16 (1.94) | |

Per-dataset BA deltas for the two that matter:

```
                       tdgs_cls-graph_a2    tdgs_d-tdgs_cls    tdgs_du-tdgs_perm
pathmnist   ( 2.49% x)       +0.43              -4.22               -1.48
bloodmnist  (26.52% x)       +1.71              -1.04               +0.85
organamnist (42.41% x)       +0.24              -0.44               +0.60
organsmnist (53.77% x)       +0.86              -4.28               -1.80
tissuemnist (66.06% x)       +5.55              -2.59               +0.81
```
(`x` = cross-class edge share of `K = A_sym + A_sym²`, the a-priori moderator,
measured by job 35062. An earlier prior table quoted 0.6 / 9.9 / 20.3 / 32.1 /
58.5% — that was the **1-hop `A`** share, the wrong variable, since credit flows
along `K`'s entries and `A` omits the 2-hop mass. The five datasets are
**order-identical under both definitions**, so every rank statistic survives; the
absolute values are not interchangeable and only the `K` definition is used here.)

## 4. `tdgs_cls − graph_a2` per cell, against the noise scale

| dataset | s42 | s43 | s44 | within-arm 3-seed sd (graph_a2 / tdgs_cls) |
|---|---|---|---|---|
| pathmnist | −0.62 | +1.47 | +0.43 | 1.12 / 2.16 |
| organamnist | +0.58 | +1.03 | −0.90 | 1.29 / 0.38 |
| bloodmnist | +0.36 | +2.12 | +2.66 | 3.21 / 3.43 |
| organsmnist | −0.01 | +2.77 | −0.18 | 0.54 / 1.21 |
| **tissuemnist** | **+3.94** | **+6.28** | **+6.43** | 1.21 / 1.45 |

Worst-class recall, tissuemnist: **+11.26 / +11.99 / +11.30** — seed sd ≈ 0.4pp.
The cleanest signal measured in this project.

**Pooled excluding tissuemnist: +0.81pp** (sd 1.22, n=12). Both numbers must be
reported; tissuemnist carries the pooled figure. This is gate G5.

## 5. Selection-space report (zero GPU cost, from staging)

| dataset | n | C | bpc | J(a2, cls) | J(a2, du) | J(du, perm) | J(d, du) |
|---|---|---|---|---|---|---|---|
| pathmnist | 89,996 | 9 | 199 | **0.7741** | 0.1814 | 0.2301 | 0.3954 |
| bloodmnist | 11,959 | 8 | 29 | 0.2853 | 0.0918 | 0.2178 | 0.2747 |
| organamnist | 34,561 | 11 | 62 | 0.2514 | 0.1678 | 0.3230 | 0.4283 |
| organsmnist | 13,932 | 11 | 25 | 0.1702 | 0.1270 | 0.2249 | 0.3285 |
| tissuemnist | 165,466 | 8 | 413 | **0.0863** | 0.0881 | 0.2028 | 0.2772 |

**`J(graph_a2, tdgs_cls)` is perfectly monotone in cross-class edge share —
Spearman = −1.00** (unchanged under the corrected `K` definition, since the five
datasets are order-identical under both). So the mask's *intervention strength* is
fully predictable a priori: it barely moves pathmnist's set (2.49% cross-class →
J=0.77) and almost completely replaces tissuemnist's (66.06% → J=0.09).

`J(du, perm)` = 0.20–0.32 confirms the permutation genuinely bit — the two arms
select largely different sets and still score the same, which is why "inert" is the
right word for the direction term and not "untested".

## 5b. Class-level mechanism instrument (`code/class_mechanism.py`, zero GPU)

Descending from dataset to class raises N from 5 (effective 2–3) to **47**, since
both sides of the mechanism claim are defined per class. Measured properties of
the resulting instrument:

| quantity | value |
|---|---|
| class-level `Δrecall_c` | mean **+1.57pp**, sd 4.15pp, range [−6.69, +13.05], 29/47 positive |
| variance decomposition | between-dataset **20.7%** / within-dataset **79.3%** |
| within-dataset variance | observed 13.39 = **real signal 4.91** + seed noise 8.48 |
| as sd | real class heterogeneity **2.22pp** vs seed noise **2.91pp** |
| `R²` ceiling for any `x` | **0.37** |
| class-points exceeding own \|t\|=2 | **13/47** |
| saturated points (both arms recall≡1.000) | 1 (pathmnist/1) — excluded; zero headroom by construction |

**Noise exceeds signal**, so the class-level regression is supporting evidence,
not decisive: extrapolating the dataset-level slope ((5.55−0.43)/(0.6606−0.0249)
≈ 8.1 pp per unit) predicts only `|t| ≈ 2.6` at N=47.

It is nonetheless a **legal** regression: noise in `y` inflates the slope's se but
does not bias it, and `x` is counted exactly from `K`, so there is no
errors-in-variables attenuation.

Going from 3 to 5 seeds (already planned as P1-c/P1-d) shrinks the noise variance
by 3/5 and lifts the `R²` ceiling from 0.37 to **0.49** — the seed expansion buys
main-table power and Fig-3 power at the same time.

The signal is concentrated, which is itself a mechanism prediction to be tested:

| dataset | class | Δrecall | \|t\| over 3 seeds |
|---|---|---|---|
| tissuemnist | 5 | **+13.05** | 10.96 |
| tissuemnist | 7 | +7.84 | 10.64 |
| organamnist | 10 | +7.18 | 5.49 |
| tissuemnist | 4 | +6.73 | 4.90 |
| tissuemnist | 0 | +9.65 | 4.69 |
| bloodmnist | 6 | +5.71 | 4.41 |
| pathmnist | 0 | +6.25 | 3.17 |

organamnist/10 is the interesting one: a 5.5-sigma per-class gain inside a dataset
whose pooled ΔBA is only +0.24pp. Full y-side table in
`class_mechanism_yonly.json`; the x side (per-class cross-class share) comes from
job **35076**.

## 6. Probe validity (reported, never used to build the selector)

| dataset | OOF BA | in-sample | proxy worst / final worst | `frac(d<0.05)` | subset-effect sd |
|---|---|---|---|---|---|
| pathmnist | 0.9974 | 0.9988 | 7 / 7 **MATCH** | **0.981** | 4.07pp |
| bloodmnist | 0.9796 | 0.9875 | 3 / 3 **MATCH** | 0.856 | 1.42pp |
| organamnist | 0.9355 | 0.9649 | 5 / 4 — same pair {4,5}, reversed | 0.617 | 2.29pp |
| organsmnist | 0.8403 | 0.9350 | 2 / 4 **MISMATCH** | 0.540 | 0.80pp |
| tissuemnist | 0.4787 | 0.6172 | 1 / 1 **MATCH** | — | 1.07pp |

Worst class matches exactly on 3/5; organamnist recovers the right confusion pair
with the members swapped (the validity check compared the *ordered* pair, so it
printed MISMATCH — a reporting bug, not a proxy failure); organsmnist is the only
genuine miss, and is also the dataset where the final models themselves agree only
0.55 on the worst class.

**`Spearman(frac(d<0.05), subset-effect sd) = +0.8`** — the probe is most confident
exactly where the final learner has the most room to differ. This is why
`tdgs_d` = −2.51pp is structural and not a tuning problem, and why the
`proxy_regime` diagnostic was retired instead of resubmitted.

## 7. Where the raw data is

- `results/round1/tdgs_round1_harvest.json` — 75 rows: `ds, ratio, arm, seed, ba,
  worst, acc, recalls[], path`
- `results/round1/selection_reports/*.json` — per dataset: bpc, probe validity,
  full Jaccard matrix, per-arm first/last greedy gain
- `logs/round1/tdpack_34970_*.{out,err}` — the 4 array tasks
- cluster, 89 MB:
  `/mnt/prj01/hgrp-1502-5TB/tdgs_shared/archive/round1/cells/ratio_0.02/<ds>/<arm>/seed_<s>/`
  with `metrics.jsonl`, `predictions_clean.npz`, `run_config.json`,
  `run_complete.json`, `selected_indices.npy`
- the 75 `final.pt` (3.3 GB) stay in `$PROJECT/tdgs_round1_20260927`; paths are in
  `archive/round1/checkpoints_MANIFEST.txt`
