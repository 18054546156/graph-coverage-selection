# Task-Geometry Selection: History, Diagnosis, and the Functional-Screen Redesign

**Scope.** This document covers one continuous investigation thread inside the
graph-coverage-selection project: attempts to design a selection method for
MedMNIST subsets (bloodmnist, organamnist, organsmnist, pathmnist, tissuemnist)
that beats the eight baselines already benchmarked in `Table1_Reproduction_By_Method/`
(Random, Herding, Forgetting, EL2N, FPS, Facility Location, EVA, GraphCov/Graph-A2).
It synthesizes ~2 weeks of dated experiment logs into one narrative: what was tried,
what killed each attempt, the root-cause diagnosis that emerged, and the concrete
next step it points to.

Everything below is measured, not aspirational. Where a number is a single-seed
estimate it says so; where a claim is underpowered it says so.

---

## 1. Timeline: eight ideas, eight deaths, one recurring shape

| # | Idea | Killed by | Core number |
|---|---|---|---|
| 1 | Bootstrap-stability gating (`s_c`) | Ruled out twice | stability ≠ informativeness |
| 2 | Margin vs. purity as a difficulty proxy | Redundant, not complementary | OOF logistic margin doesn't beat purity |
| 3 | Coverage-equalization (`cov_c`) | Barely correlates with real accuracy | 4th proposed fix for the same non-signal, closed |
| 4 | FRACTAL-Select (heterogeneity-gated) | Null control beats real on 4/5 datasets | only tissuemnist exception has a known false-positive precedent |
| 5 | Component A: label-aware LDA metric | 12/12 comparisons `\|t\|<1.6`, 9/12 negative | the organsmnist "+3.9pp" reading was max-selection bias |
| 6 | Neyman / σ_c quota reweighting | Deferred (target mismatch: estimation variance ≠ training BA), not falsified | correct balanced-accuracy derivation exists (`n_c ∝ σ_c`, no `N_c`) but was never trained |
| 7 | Task-geometry gradient-based selection | Pre-registered gate closed | +0.19pp vs. permutation control |
| 8 | Voronoi weighting × selection interaction | **REFUTED** by real-training gate (§3) | pooled −0.064pp vs. probe's predicted +1.05/+2.92/+1.60 |

The pattern across all eight: **invent a score → build a method around it → gate
it → fail**, without ever testing, at fixed budget, whether the score has *any*
relationship to accuracy before committing to a method built on it. When that link
was finally tested directly and honestly (item 8, and the `results/ladder.json`
covering-distortion regression before it: `r=+0.014, t=0.21` against a test powered
to see `r=0.36`), the answer was no. The "107/120 sign agreement" that looked like
supporting evidence for covering distortion turned out to be herding being better
than random at *everything*, not distortion causing BA — a method-identity confound,
not a functional effect.

This is why the current work (`functional_screen.py`) inverts the methodology:
**measure which functional of a selected set predicts accuracy first, design a
method around it second.**

---

## 2. The REFUTED gate (item 8), in full

`gate_training.py` on 360 real-training runs (bloodmnist / organsmnist / tissuemnist,
random and herding arms, 2 train seeds each), frozen before results:

| dataset | probe-predicted DiD (pre-registered) | real-training DiD |
|---|---|---|
| bloodmnist | +1.05 ± 0.44, 4/4 blocks agree | **−0.090 ± 1.082** |
| organsmnist | +2.92 ± 0.13, 4/4 blocks agree | **+0.382 ± 0.700** |
| tissuemnist | +1.60 ± 0.40, 4/4 blocks agree | **−0.484 ± 0.330** |

Pooled real-training DiD: **−0.064pp**. Permuted (null) control: **+0.057pp**.
The real effect is indistinguishable from — and numerically smaller than — its own
permutation control. **REFUTED.** Weighted training does not beat equal weighting
differently for random vs. herding under real training, despite a probe signal that
was internally consistent (4/4 blocks, same sign, `\|t\|>2`) on all three datasets.

This is the **fourth** offline-gate-passes / real-training-disagrees event in this
project (after FRACTAL, a tissuemnist Voronoi false positive, and the task-geometry
gate). All four share the same shape: the criterion is computed on UNI embeddings
with a deterministic linear-probe endpoint; the disagreeing endpoint is a ResNet-18
trained from scratch on pixels.

---

## 3. Root-cause diagnosis: the instrument, not the method

`probe_vs_training.py` tested the probe directly against the 60 distinct selections
behind the 360 real-training runs above.

**Level 1 — arms pooled, dataset fixed:** Spearman +0.21 / +0.80 / +0.48. Looks
strong, but this is mostly the *arm* effect — the probe correctly knows herding is
different from random. It is not evidence the probe ranks selections *within* a
method.

**Level 2 — within arm (the test a screen actually needs):** mean Spearman
**+0.119** (attenuation-corrected +0.166) over 6 (dataset × arm) cells, 2 of 6
negative.

**But this test is underpowered.** `n=10` selections per cell → `se(ρ) ≈ 0.378` →
a true `\|ρ\|` has to exceed **~0.76** to be distinguishable from zero at this sample
size. The correct reading is *not* "the probe carries no within-method signal" — it
is "the probe is definitely not the pooled +0.80 the Level-1 number suggests, and a
design with `n=10` cannot say more than that." Treating the pooled correlation as
if it applied within a method family is the same confound that produced the false
"107/120 sign agreement" for covering distortion — between-method variation
masquerading as within-method signal, twice.

**The number that explains why every gate keeps failing this way** — per
`(dataset, arm)` cell, decomposing training-BA variance across 10 replicates using
the two training seeds:

| cell | sd from selection | sd from training noise (2-seed mean) | reliability |
|---|---|---|---|
| blood/random | 0.000 | 0.020 | **0.00** |
| blood/herding | 0.0138 | 0.0144 | 0.48 |
| organs/random | 0.0103 | 0.0069 | 0.69 |
| organs/herding | 0.0118 | 0.0106 | 0.56 |
| tissue/random | 0.0130 | 0.0054 | 0.85 |
| tissue/herding | 0.0094 | 0.0079 | 0.59 |

**Selection-quality signal (0.9–1.4pp) and training noise (0.5–2.0pp for a 2-seed
mean) are the same order of magnitude.** Two seeds is not enough — bloodmnist/random
has reliability **0.00**, meaning at this budget *which random subset you draw does
not measurably affect trained accuracy at all*, so no functional can be discovered
there no matter how many seeds you add.

Note this is genuine seed-to-seed variance, not leftover nondeterminism: an earlier
diagnostic found `DETERMINISTIC_TRAINING` was dead code (fixed 2026-09-21), but
`train_weighted.py` calls `make_deterministic()` unconditionally, and all 360 of
these runs went through it. The 0.5–2.0pp is the real noise floor of ResNet-18
trained from scratch on 200 images, not a bug.

**Conclusion:** every failed gate in this project is consistent with a measurement
instrument (the probe) as coarse as the effect it was being used to detect. Do not
attribute the next null result to the method under test until the instrument itself
is sized to the effect.

---

## 4. What this project is actually benchmarked against

`Table1_Reproduction_By_Method/` already contains a faithful reproduction (394/400
runs within tolerance, 91.4% within ±4pp of the paper, 2.13pp mean error) of eight
methods: **Random, Herding, Forgetting, EL2N, FPS, Facility Location, EVA, GraphCov
(Graph-A2)**. The functional screen (§5) is not a ninth competitor — it is a
measurement layer over the same hypothesis space these eight methods already encode,
run *before* committing to a ninth method's design.

| baseline | property it selects on | functional-screen equivalent |
|---|---|---|
| FPS | max–min coverage (`dist_max`) | `dist_max` |
| Facility Location | mean coverage (`dist_mean`) | `dist_mean`; current `kmedoids` anchor is a **Lloyd-medoid approximation**, not true submodular greedy facility location — gap, see §6 |
| Herding | matches class-mean moment | `moment1`, `herding` anchor (exact) |
| Forgetting (Toneva) | forgetting-event count | `forget_mean`/`forget_sd`, `forget_high`/`forget_low` anchors (exact) |
| EL2N (Paul) | early-epoch gradient-norm proxy | `el2n_early_mean`, `el2n_high` anchor (exact) |
| EVA | cross-epoch error **variance** (dual-window) | **not yet implemented** — gap, see §6 |
| CCS (Zheng) | difficulty-stratified coverage | `difficulty_ks`/`difficulty_w1`, `ccs` anchor (exact) |
| Moderate (Xia) | nearest-median-distance-to-centroid | `moderate` anchor (exact) |
| GraphCov / Graph-A2 | this project's own graph-coverage method | already trained, not re-derived here |

Five of the eight are now exact functionals/anchors in `functional_screen.py`.
Two gaps remain and should be closed before claiming a result "beats all eight":

1. **EVA's own score** (variance of the per-sample error/loss trajectory across a
   dual epoch window) is not computed. `forget`/`el2n`/`aum` are related but not
   equivalent — EVA is explicitly a different combination.
2. **True submodular Facility Location** (greedy maximization of coverage, e.g. via
   lazy greedy) is approximated by Lloyd-medoid refinement (`k_medoids` in
   `functional_screen.py`), which converges to a similar but not identical
   objective.

---

## 5. Design principle of the functional screen

`functional_screen.py` measures **26 functionals** of a selected set (15 geometric:
`dist_mean/p90/max`, `moment1/2`, `mmd2`, `mass_gini`, `spread`, `centrality`,
`logdet`, `purity`, `margin`, `density`, `sel_margin_min`, `xclass_sep`; 11
dynamics: forgetting mean/sd/frac-zero, AUM mean/sd, EL2N final/early mean,
difficulty-KS/W1, hardest/easiest-decile fraction) against real balanced accuracy
at a **fixed budget** (25/class), with three design choices that directly target
the failure modes in §1–3:

- **Perturbation sweeps inside a fixed method family** (`m ∈ {1,2,5,10,25}` points
  per class swapped for random same-class pool points), so a functional's own
  effect can be read independent of which generator produced the set. Comparing
  *methods* (herding vs. random) confounds every functional with "which method" —
  this is exactly the confound that produced the false covering-distortion signal.
- **Two independent pool/audit blocks, always** — nothing is reported as real
  without agreeing in sign and significance on both.
- **Class-equal averaging**, not point-weighted, because the endpoint is balanced
  accuracy; a frequency-weighted functional would measure a different target than
  the one being predicted.

`analyze_screen.py` reports four nested analyses (A: whole library, dominated by
pathological sets; B: competitive-family covariate; C: within-family, method
identity removed — the analogue of the test that killed covering distortion; D:
within-family restricted to competitive families, the decision-relevant cell) plus
a fifth, explicitly flagged as biased (E: outcome-conditioned, collider bias from
selecting on the endpoint). It also now reports D2 (D minus pure hard-example
families) and a baseline leaderboard (every anchor's unperturbed BA as a z-score
against the random draws in the same dataset/block) — see the code for exact
definitions.

**Status as of 2026-09-23 (probe run COMPLETE):** job 33767 finished in 19m12s and
produced **7,350 selections** across all 5 datasets x 2 blocks x 26 functionals
(`results/screen_33767/screen_{0..3}.jsonl.gz`, committed here). Analysis output is
`results/screen_33767/analyze_screen.json`.

A real bug was found and fixed while running the analysis: `analyze_screen.py` called
`collections.groupby`, which does not exist (only `itertools` has `groupby`), so
analysis E raised `AttributeError` before producing anything. Fixed to
`itertools.groupby`.

**Probe-endpoint survivors** (`|t|>3` in both D and D2, same sign) — 10 functionals:
`dist_mean`, `dist_p90`, `mmd2`, `moment1`, `moment2`, `mass_gini`, `purity`,
`spread`, `aum_mean`, `el2n_early_mean`.

**These are candidates, not conclusions.** They are what predicts *linear-probe*
balanced accuracy. Per §3, the probe is not validated as a within-method ranker for
real training, so this list is a hypothesis set to be re-tested at the real endpoint
(§7) — not a result to build a method on. That mistake is exactly what killed items
1-8 in §1.

---

## 6. Immediate gaps (not yet done)

1. ~~**Endpoint is still the probe, not real training.**~~ **CLOSED (code) / IN
   FLIGHT (measurement).** The real-training driver is implemented in
   [`../real_training_functional_screen_20260923/`](../real_training_functional_screen_20260923/)
   and is running — see §7.
2. **EVA's own score and true submodular Facility Location** are not in the
   functional/anchor set (§4). Still open. Needed before any claim of the form
   "beats all eight baselines" is fully honest.
3. ~~**Full-scale dataset priors** are queued but not landed.~~ **CLOSED.**
   `measure_priors_fullscale.py` completed as part of job 33767;
   `results/screen_33767/priors_fullscale.json` is committed here. The pathmnist
   exclusion from the training factorial cited a headroom measured on an 8%-capped
   pool and should now be rechecked against this file.

---

## 7. Next step, now RUNNING: real-training functional screen

> **Status 2026-09-23 16:20.** Implemented, smoke-tested, and submitted. Code lives
> in [`../real_training_functional_screen_20260923/`](../real_training_functional_screen_20260923/)
> (see its README for the exact design and launch commands). 15 one-GPU jobs are in
> flight across three cluster accounts; first rows confirmed on disk with correct
> `ba_real` and `endpoint="resnet18_official_test_equal_weight_final_epoch"` fields.
> **The measurement is not finished — no functional-vs-`ba_real` result exists yet.**
> The rest of this section is the design rationale, which is unchanged.

Move the screen's endpoint from probe BA to real training BA, and size the seed
count to the reliability numbers in §3 rather than reusing the 2-seed design that
made the REFUTED gate underpowered.

**Why this is affordable, not a compromise:** the 360 completed real-training runs
averaged **52.1s each** (median 47.8s; 5.21 GPU-h total for all 360) — at
budget=25/class the training set is only ~200–1400 images, so training cost is
negligible next to evaluation/bookkeeping. There is no need to shrink the design to
afford real training.

**Proposed sizing:**

| | |
|---|---|
| datasets × blocks | 5 × 2 = 10 cells |
| selections per block | **225** (60 random + 15 anchors × (1 unperturbed + 5 perturbation levels × 2 seeds)) |
| training seeds | 3 (`--train-seed-offset` 0 / 100000 / 200000) |
| **total runs** | **6,750 × ~52s ≈ 97 GPU-h** |

Three seeds is justified from the organsmnist/herding cell (`σ_sel=1.18pp`,
single-seed `σ=1.50pp`): at k=3, reliability ≈0.65, a true `r=0.4` functional
observes as `r≈0.32`, and with n≈90+ selections per cell that reaches `t=3`. Every
cell in the proposed design has 200+ selections, which comfortably covers this.

**One cell should get zero budget, not an equal share:** `blood/random` measured
`σ_sel=0.000` — at budget=25 on bloodmnist, which random subset you draw does not
move trained accuracy at all, so no amount of seed averaging will produce a signal
there. Allocate seeds unevenly rather than splitting evenly across all 10 cells.

This run simultaneously produces three things the project currently lacks:

1. **The actual functional screen** (geometric + dynamics, at the real endpoint).
2. **A properly powered probe-vs-training validation** (`n≈214`/cell instead of
   `n=10`) — resolving the open question in §3 instead of leaving it bounded.
3. **A real-training baseline leaderboard** — which of Herding / CCS / Moderate /
   Forgetting / EL2N / GraphCov actually wins under real training, not under the
   probe.

The completed probe screen (job 33767) is not wasted: the real-training driver
reuses its selection library *verbatim* (same `build_library`, same seeds), so the
two runs measure the same 225 sets per block at two different endpoints. That makes
the probe-vs-real-training comparison a paired, `n=225`/cell test instead of the
`n=10` one that left §3 unresolved. The probe's role changes from *decision-maker*
to *selection-library generator and comparison baseline*.

---

## 8. File index

### Committed measurement outputs

| file | what it is |
|---|---|
| `results/screen_33767/screen_{0..3}.jsonl.gz` | **The probe screen itself** — 7,350 selections x 26 functionals x 5 datasets x 2 blocks (job 33767, 19m12s). `gunzip -c` to read; one JSON object per line |
| `results/screen_33767/analyze_screen.json` | `analyze_screen.py` output on the above: the A/B/C/D/D2/E analyses, baseline leaderboard, and the 10-functional `survivors` list (§5) |
| `results/screen_33767/priors_fullscale.json` | Full-pool, uncapped dataset priors — closes §6 gap 3 |
| `results/gate_training.json` | The frozen REFUTED gate (§2) |
| `results/probe_vs_training.json` | The probe-vs-real-training diagnosis (§3), incl. the per-cell reliability decomposition |
| `results/ladder.json` | The covering-distortion regression (`r=+0.014, t=0.21`) referenced in §1 |
| `results/dataset_priors.json` | Earlier capped-pool priors, superseded by `priors_fullscale.json` |

Raw training logs and embedding caches are deliberately **not** committed (they are
GB-scale and regenerable); everything needed to re-run the analyses is here.

### Code

| file | role |
|---|---|
| `functional_screen.py` | Measures 26 functionals of a selected set against probe BA at fixed budget; perturbation-based library generation; dynamics anchors wired in 2026-09-23 |
| `analyze_screen.py` | Four/five nested analyses (whole library → competitive → within-family → within-family-competitive → outcome-conditioned-biased) plus baseline leaderboard. Set `ENDPOINT = "ba_real"` to re-run against the real-training screen |
| `HANDOFF_NEXT_SESSION_20260923.md` | Operational handoff: paths, cluster layout, standing constraints, what to do next |
| `../real_training_functional_screen_20260923/` | **The real-training endpoint screen** (§7) — same library, `ba_real` instead of probe BA |
| `probe_vs_training.py` | Tests probe-BA vs. real-training-BA rank agreement on the 60 selections behind the 360-run gate |
| `gate_training.py` | The frozen, pre-registered REFUTED gate (§2) |
| `measure_dataset_priors.py` | Capped-pool (1000/class) dataset priors — superseded by `measure_priors_fullscale.py`, not yet landed |
| `measure_priors_fullscale.py` | Full-pool, no-cap dataset priors (queued) |
| `build_training_cells.py` | Builds the pool/audit/selection cells used by the 360-run training factorial |
| `train_weighted.py` | Real ResNet-18-from-scratch training with optional per-sample weighting |
| `run_linear_probe.py` | The deterministic linear-probe endpoint (`R.metrics`, `R.herding`, `R.normalize`, etc.) shared by most scripts here |
| `run_functional_screen.slurm` | Cluster launcher: full-scale priors (GPU 0) + 4-way dataset-grouped functional screen (GPUs 1-4) |
| `run_training_shards.slurm` | Cluster launcher for the real-training factorial |
| `PLAN_conditional_coreset.md` | Pre-registered plan for a soft-label / Voronoi-conditional coreset method — gated behind §7 landing first |
| `ARXIV_*`, `*_PAPERS_*`, `LITERATURE_SEARCH_FINAL_REPORT.md`, `RELATED_METHODS_REFERENCE_TABLE.md`, `COMPARATIVE_ANALYSIS_VORONOI_CORESET.md`, `EVA_PAPER_ANALYSIS.md`, `DETAILED_METHOD_COMPARISON.md` | Novelty/prior-art audit for `PLAN_conditional_coreset.md`; see `TOP_PAPERS_FOR_GEOMETRY_SELECTION.md` and `SEARCH_EXECUTIVE_SUMMARY.md` for the condensed versions |
