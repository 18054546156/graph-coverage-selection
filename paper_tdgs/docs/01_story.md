# The paper, as a story

Working title direction: **Coverage credit must respect the decision boundary —
class-compatible facility location for budget-constrained medical data selection**

The credibility structure is deliberate: **the negative results are the skeleton,
and the positive result is fenced in by a mechanism that predicts where it works
before training.** A paper that led with "+1.76pp" and buried the flatness result
would be weaker and easier to shoot down, because a reviewer who reruns any
single cell will land inside the noise.

---

## Act 1 — Reproduce the benchmark, then build an instrument

**Claim.** The author's Table 1 is reproducible, and the measurement apparatus
underneath it can be characterised.

**Evidence.**
- Table 1 reproduced: 8 methods × 5 datasets × 2 ratios × 5 seeds = 400 cells,
  **394 done, 91.4% within ±4pp of the paper**, mean error 2.13pp.
- Training is made **bit-deterministic** per seed (`DETERMINISTIC_TRAINING=1`;
  48/48 repeats identical). Before that fix the env var was dead code and
  replicate spread was 2–7pp.
- σ_run = 1.37pp; σ_subset inside the competitive region = **0.82pp**.
- Three archive defects found and reported as such: `selection_seed ≡
  training_seed` (confounds subset identity with initialisation), corruption rows
  pre-pooled across severity and single-seed (so rBE was never computed at power
  and CEC is not computable), and bloodmnist's archived selection is not
  reproducible under any tested config.

**Why it belongs in the paper.** Everything in Act 2 is a statement about
measurement. It is only admissible if the measurement was characterised first.

---

## Act 2 — The structural negative result (this is the scientific spine)

**Claim.** The benchmark cannot distinguish selection criteria, because balanced
accuracy is **flat over the region where all the good methods live**.

**Evidence.**

| quantity | value | reading |
|---|---|---|
| mean pairwise Jaccard, `graph_a2` / `herding` / `facility` | **0.049** | three statistically tied methods share 5% of their picks |
| random-draw Jaccard floor | 0.010 (2%) / 0.026 (5%) | so 0.049 is only 1.9–5.7× the floor |
| triple intersection | 0.2–3.0% | union is 2.7× the budget |
| reverse-solve J ≈ n/(2G−n) | **G ≈ 20% of the pool** | the "good subset" region |
| σ_subset vs σ_run | 0.82pp vs 1.37pp | subset choice matters *less* than the seed |
| span of four criteria vs four random draws | 1.48pp vs 1.78pp (obs/exp **0.83**) | criterion differences are *smaller* than sampling differences |
| `graph_a2` best in | 3 of 10 cells | |

**Mechanism.** The author's method's entire contribution is reliably landing in
that ~20% region (worth ≈1.75pp, the same as Herding). Inside the region there is
nothing left to discriminate. That single statement explains **20 independently
closed threads** (see [06_closed_threads.md](06_closed_threads.md)) at once — they
were all reparameterisations of the same covering geometry, evaluated on a flat
surface.

**Supporting falsification.** The Graph-A2 objective is not saturating (0.0%
exactly-zero marginal gains across 7 validity-passing cells), and the selection is
*locally* decisive (only 8–22 of ~10k candidates within 5% of the best at each
step) yet *globally* degenerate (a 1e-3 embedding perturbation changes 15–82% of
the picks, head included). Decisive per step, arbitrary in aggregate.

**Why it belongs.** This is a reusable, testable claim about the whole coverage-
selection literature, and it is the reason the paper's positive result is shaped
the way it is.

---

## Act 3 — Where the surface is *not* flat

**Claim.** Two axes survive Act 2, and only one of them is a legal lever under the
Table 1 protocol.

1. **Worst-class recall is a separate axis.** In the competitive band the tail gap
   exceeds the mean gap in **89.8% of 215 controlled pairs** (p = 3e-33).
   `herding` is +0.20pp on BA and **−1.12pp** on worst-class recall. BA hides
   2.3–13× cancellation between classes. At fixed budget *and* fixed quota,
   exemplar choice moves individual class recalls 5–24pp reproducibly.

2. **Objective/metric alignment, not the criterion.** This is the lever Act 4
   pulls. It is not "a better score for candidates"; it is *who is allowed to
   generate demand, and what relationship counts as covering it*.

**Excluded here, honestly.** Target-demand selection (TD-Cover) gives +2.35pp on
pathmnist with both controls passing and +8.22pp worst-class recall, and its
mechanism is resolved (it works exactly where the pool holds target-unneeded mass
that is unidentifiable train-only: demand R² .116 win / .457 / .669 no gain). But
it reads **unlabeled target inputs**, so it is not a legal entry in Table 1 and
appears only as an additional-setting result.

---

## Act 4 — The correction, and the result

**Claim.** Two specific misalignments between the facility-location objective and
balanced accuracy can be repaired without touching the scoring criterion, and
repairing them helps — by an amount predictable in advance from the data.

**The two misalignments.** `graph_a2` maximises `Σ_{i∈P} max_{j∈S} K_ij` where:

1. **`K_ij` ignores labels.** A candidate earns credit for covering a neighbour of
   a *different class*. The author's code says so explicitly ("boundary samples
   get credit for covering nearby other-class samples"). Budget is spent buying
   coverage that cannot help the decision boundary. The size of the leak is
   measurable per dataset — cross-class edge share runs from 0.6% (pathmnist) to
   **58.5%** (tissuemnist).
2. **Demand is pool-frequency weighted.** The sum runs over pool points, so a
   class with 10× the mass has 10× the say — while balanced accuracy gives every
   class equal say.

**The fix (`tdgs_cls`).** Confine coverage credit to within-class edges
(`1[y_i=y_j]`), weight each demand point `1/(C·n_c)`, blend λ=0.5 with the
author's original global objective. **No scoring criterion is changed**, which is
why it is not on Act 2's flat surface.

**Result (75 real cells, 3 seeds, ratio 0.02, paired by training seed):**

| | BA | worst-class recall |
|---|---|---|
| pooled | **+1.76pp** (se 0.60, n=15) | **+3.54pp** (se 1.64) |
| datasets same sign | **5/5** | 4/5 |
| cells positive | 11/15 | 9/15 |
| pooled excluding tissuemnist | **+0.81pp** (n=12) | — |

**The part that makes it a mechanism rather than a fit.** Effect size tracks
cross-class edge share, which is computable before any training:
tissuemnist (58.5%) → **+5.55pp BA / +11.5pp worst recall, seed sd 0.4pp**;
pathmnist (0.6%) → +0.43pp. The prediction was stated first and then measured.

**It also re-explains a previously-misread prior.** Spearman(cross-class share,
subset-effect sd) = −0.80 was read as "datasets with lots of cross-class material
have no headroom". The correct reading is the opposite: they *had* headroom, and
the original method was leaking budget there, which made every method look
similar and bad.

---

## Act 5 — What the pre-registered ablation falsified

**Claim.** The contribution is exactly the two alignment corrections — not a
richer story about class-pair-specific demand.

- **Difficulty as a demand weight: −2.51pp** (5/5 negative), worst recall
  −5.70pp. The cause is structural, not fixable with a better proxy: the frozen-
  UNI probe's dynamic range is *inversely* related to the real subset effect
  (Spearman +0.8 between `frac(d<0.05)` and subset-effect sd). On pathmnist the
  probe reaches OOF BA .9974 while the 1791-sample ResNet-18 reaches 81.5%, so
  `d_i` concentrates weight on 1.9% of the pool and destroys the class balance
  `tdgs_cls` had just installed.
- **Class-pair direction structure: inert.** `tdgs_du − tdgs_perm = −0.20pp`
  (se 0.67) — permuting `u` within class changes nothing. Per the pre-registered
  rule, the novelty is therefore *not* "class-pair demand". `u_i` is built from
  softmax tails that are numerical noise wherever the probe is accurate.

**Why Act 5 strengthens the paper.** It converts a vague "we model task demand"
into a narrow, defended claim, and it demonstrates that the ablation ladder had
real power to say no.

---

## What is still open before Act 4 can be written as stated

One rival explanation, and it is not rhetorical. The author's **release default is
`global_selection=False` = per-class scope**, which *also* confines coverage credit
within class, and `compare_global.py` is the author's own ablation cell.

- If per-class `graph_a2` at k=50 recovers most of +1.76pp → `tdgs_cls` is inside
  an occupied cell, Act 4 collapses, and the paper becomes Acts 1–3 plus a
  diagnostic.
- If per-class is worse (as the paper reports) while `tdgs_cls` is better → the
  novel ingredient is the **λ blend** (keep global geometry *and* add
  class-compatible credit) plus class-flat weighting.

That is gate **G1** in [05_future_experiments.md](05_future_experiments.md) and
the first 15 cells of the round-2 worklist. Act 4 must not be written until it
returns.
