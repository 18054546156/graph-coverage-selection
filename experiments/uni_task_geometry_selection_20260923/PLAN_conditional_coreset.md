# Conditional-Consistent Coresets — plan, 2026-09-23

Supersedes the geometric-selection line. Written after the measurements below, not
before; every number here is from this project's own data and is reproducible from
the scripts named.

---

## 1. What was measured

**(M1) Geometric coverage does not predict accuracy at fixed budget.**
On `results/ladder.json`, centring both the functional and the endpoint within every
(dataset, cap, budget, arm) cell — so budget, dataset and method are all held fixed
and only the pool draw varies:

| endpoint | n | slope (pp / unit distortion) | r | t |
|---|---|---|---|---|
| raw BA | 240 | −0.9 | −0.001 | −0.02 |
| BA − full-pool BA | 240 | +9.7 | +0.014 | +0.21 |

Per dataset: blood −0.054, organs +0.072, tissue −0.015, all |t| < 0.7, signs
alternating. The test had power to see r = 0.36 (t ≈ 5.9): the block-level
distortion residual is sd 0.00149, which at the naively implied slope of ~250 pp/unit
would have produced a 0.37 pp BA swing against an observed BA residual sd of 1.01 pp.

The apparent support — herding has both lower distortion and higher BA in 107 of 120
paired strata — is the method confound. Herding is better at everything. Within a
method, the geometry moves and the accuracy does not.

**(M2) The whole graveyard is one mistake.** Covering distortion, facility location,
k-center, graph coverage (Graph-A2), FRACTAL-Select, coverage equalisation,
task-geometry: all are objectives on the **x-marginal**. M1 says that family has no
purchase at fixed budget. These are not seven independent failures.

**(M3) Selection is fighting for ~6% of the available headroom.** bloodmnist, budget
25, full train pool: full-pool BA 95.3, random 87.9, best selection in a 51-member
library 88.3. Headroom 7.4 pp; best-over-random 0.45 pp.

**(M4) The hard label is a lie, and how big a lie is dataset-specific.** Assign every
pool point to its nearest selected point **ignoring class** (6000-point subsample,
herding, 25/class):

| dataset | C | cell purity | frac. mixed cells | H(q)/ln C |
|---|---|---|---|---|
| tissuemnist | 8 | 0.480 | 0.965 | 0.607 |
| organsmnist | 11 | 0.705 | 0.840 | 0.281 |
| organamnist | 11 | 0.759 | 0.745 | 0.218 |
| bloodmnist | 8 | 0.910 | 0.530 | 0.122 |
| pathmnist | 9 | 0.979 | 0.222 | 0.029 |

Random gives essentially the same numbers (last column of the measurement), so this
is a property of the data, not of the selector.

---

## 2. The claim

Let π assign each pool point to a selected point, n_i = |cell i|, w_i = n_i / N, and
q_i = the empirical label distribution of cell i. The pool risk is

    R_pool(f) = (1/N) Σ_j CE(onehot(y_j), f(x_j))
              = Σ_i w_i · E_{j ∈ cell i} CE(onehot(y_j), f(x_j))
              = Σ_i w_i · CE(q_i, f(x_i))  +  D_x  +  o(1)

where D_x collects the x-displacement ‖f(x_j) − f(x_i)‖ inside a cell and is what
covering distortion bounds.

So the surrogate a coreset should minimise is `Σ_i w_i CE(q_i, f(x_i))`. Standard
practice minimises `(1/k) Σ_i CE(onehot(y_i), f(x_i))` — wrong on both factors.

**Hard-label coresets are inconsistent estimators of the pool risk, and the bias is
the mass-weighted mean cell label entropy.** The two error channels are:

* **D_x** — the x-marginal channel. Every prior method optimises this. **M1 measured
  it to be null at fixed budget.**
* **q_i vs onehot(y_i)** — the conditional channel. **M4 measured it to be large**,
  and it is not addressed by any method in the graveyard.

The two positive results this project has both sit on the neglected side: Voronoi
mass weighting (fixes w_i) and the weighting × selection interaction. Soft labels
fix the other factor of the same term.

---

## 3. The method

One objective, three coupled design choices:

    min_{S}  Σ_i n_i · H(q_i)        subject to   Σ_i w_i q_i[c] = 1/C  for all c

1. **Selection** = minimise the mass-weighted label entropy of the *induced
   partition*. This is not a per-point purity score (that is the `purity` /
   `easy` anchor already in the screen, and per-point purity is in this project's
   graveyard). It is a property of the partition the selection creates.
2. **Weights** w_i = cell mass. Already positive on the probe; under real-training
   test right now as jobs 33697/33702.
3. **Soft labels** q_i = cell label distribution; train with `Σ_i w_i KL(q_i ‖ f)`.

Class balance moves from per-class **point count** to per-class **soft-label mass**.
That resolves the open item in memory (`per-class-quota-is-the-untouched-lever`):
equal quota was load-bearing because it was the only thing enforcing frequency-flat
training, and mass balance enforces the same thing on the quantity that actually
enters the loss.

---

## 4. Experiments

**Stage 0 — functional screen (queued, job pending on MaxSubmitPU).**
`functional_screen.py`: ~558 selections × 2 blocks × 5 datasets, 15 functionals of
the selected set, probe BA as a deterministic endpoint, perturbation-generated so the
functionals vary *within* a method family. `analyze_screen.py` reports the within-
family regression (the analogue of M1) plus cross-block replication.
*This can overturn the plan.* If some x-marginal functional does survive among
competitive selections on both blocks, section 2's premise is wrong.

**Stage 1 — probe, cheap.** Soft-label vs hard-label at budget 25, full pool,
2 blocks × 5 datasets, with all four controls below.

**Stage 2 — real training.** Only the arms that survive Stage 1, using the frozen
`train_weighted.py` path (final-epoch model, official test once, real determinism).

### Pre-registered controls

| id | control | what it kills |
|---|---|---|
| C1 | soft labels permuted among selected points within class | "q_i carries local information" vs "unequal targets change training at all" |
| C2 | **uniform label smoothing at matched mean entropy** | the single most likely killer: if plain smoothing at the same entropy gets the same gain, the Voronoi part does nothing |
| C3 | soft labels from a KNN ball at the selected point, not from its cell | "partition-consistent" vs "any local label averaging" |
| C4 | hard labels + Voronoi weights (the arm already in flight) | isolates the soft-label factor from the mass factor |

C2 is the one to run first. It is cheap and it is the control a reviewer will demand.

### Pre-registered prediction (written before any soft-label result exists)

Gain over the hard-label baseline is monotone in H(q)/ln C, i.e. ranks
**tissue > organs ≈ organa > blood > path**, with pathmnist < 0.3 pp.
Falsified if Spearman(H(q)/ln C, gain) ≤ 0 across the five datasets, or if pathmnist
shows a gain comparable to tissuemnist.

---

## 5. Risks, stated in advance

* **TissueMNIST is the predicted top beneficiary and is exactly where this project
  has already had a false positive** (`tissuemnist-voronoi-false-positive-gate`: an
  offline gate passed, real training regressed −2.4 pp and lost to its own permutation
  control). A tissue-only win is *not* evidence. The ordering prediction is what makes
  the result falsifiable rather than cherry-picked.
* **PathMNIST is doubly blocked** — 0.61 pp headroom and 0.979 cell purity. It cannot
  move. That makes it a useful negative control, not a failure.
* **The setting changes.** Soft labels use pool labels for points *not* in the
  coreset. Legal under the training-cost framing this project and MedMNIST coreset
  work already assume (Graph-A2 builds a graph over the whole labelled pool; UNI
  embeds it). Illegal under an annotation-cost framing. This must be stated in the
  paper and the baselines must be given the same access.
* **Novelty is not established.** Teacher-distilled soft labels are standard in
  dataset distillation (SRe2L / RDED / FKD family). The distinguishing claim is that
  q_i is the *empirical* pool conditional under the induced partition, with no trained
  teacher — but that claim needs a prior-art check before any GPU commitment.
* **M3 caps the prize.** If selection only reaches 6% of headroom, the honest upper
  bound on any selection-side method is small. The soft-label channel is a different
  channel and is not obviously subject to that cap — but that is an assumption, not a
  measurement, until Stage 1 runs.
