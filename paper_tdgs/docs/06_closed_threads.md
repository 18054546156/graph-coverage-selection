# Closed threads — what was tried, what it cost, and the one mechanism behind all of it

**The single mechanism.** Balanced accuracy is **flat over the ~20% of the pool
where every good method lives**. The author's method's whole contribution is
reliably landing in that region (≈1.75pp, the same as Herding); inside it there is
nothing left to discriminate. Every thread below was a reparameterisation of the
same covering geometry, evaluated on a flat surface — so ≈0 was the predictable
outcome, and it took 20 measurements to see it.

Evidence for the mechanism itself: three statistically tied methods share Jaccard
0.049 (random floor 0.010/0.026); reverse-solving `J ≈ n/(2G−n)` gives
G ≈ 20% of the pool; σ_subset 0.82pp < σ_run 1.37pp; four criteria span 1.48pp
against 1.78pp expected from four random draws (obs/exp **0.83**).

The corollary that produced TDGS: **stop changing how candidates are scored inside
the objective; change the objective's alignment** — who may generate demand, and
what relationship counts as covering it.

---

## Criterion / scoring family — all ≈0 by construction

| thread | outcome |
|---|---|
| Difficulty as criterion | −6.72pp; within-class permutation control at exactly zero |
| Difficulty as **demand weight** (`tdgs_d`) | **−2.51pp**, 5/5 negative (round 1) |
| Margin vs purity difficulty proxy | redundant, not complementary |
| Bootstrap stability gating (`s_c`) | ruled out **twice** |
| Component A — label-aware LDA metric | 12/12 comparisons \|t\|<1.6, 9/12 negative; the organsmnist "+3.9pp" was max-selection bias |
| 26 functional features | 8545 real runs; every feature fails inside the competitive regime |
| D² Pruning | closed as difficulty-demand |
| `el2n_top` (= NUCS's Hardest-CP) | −28.9pp |
| Class-pair **direction** structure (`tdgs_du`) | inert vs its own permutation control (−0.20pp) |

## Coverage-geometry family

| thread | outcome |
|---|---|
| Kernel variants (`A+A²`, `A+A³`, `A+A²+A³`) | scoped out; GSS-G1 scores Graph-A2's own objective at 0.062 vs perm p95 0.076, so k/kernel/hops are predicted ≈0 |
| Voronoi loss reweighting | organsmnist +4.5pp screened positive → **refuted** at the frozen gate (pooled −0.064pp vs probe's +1.05/+2.92/+1.60) |
| Coverage-equalisation (`cov_c`) | 3 attempts; `cov_c` barely correlates with real accuracy |
| Per-class quota reallocation | 850 runs; held-out best-of-12 = −0.93pp, and the +3.15pp selection bias exceeds any real effect measured here |
| Quota slope ceiling | per-class slopes differ 11× (KKT story falsified) but the 1/C factor caps any reallocation under the noise floor (oracle +0.18/+0.68pp) |
| `hub_hop` (A: hub-indegree; B: h=1) | A: ρ=−0.39. B: offline said h\*=1 everywhere; real training gives −2.98pp worst recall (p=1e-5), zero BA upside |
| FRACTAL-Select | null control beats real on 4/5 datasets |
| GSS Gate G1 — 44 set-level kernel features | LODO ρ=0.032 vs permutation p95=0.076 |
| Per-class recall from coverage features | 36 train-only features LODO −0.133 vs null p95 +0.201; **oracle** coverage features also fail (+0.069) |

## Surrogate / proxy family — one root cause

| thread | outcome |
|---|---|
| UNI-space linear probe | ρ=+0.085 (p=0.135); dynamic range **inversely** proportional to the real subset effect |
| K-epoch proxy | +0.171 (p=0.348); fires only where headroom is absent |
| `K_acq` acquisition prior | UNI already encodes stain/focus/exposure (within-class R² 0.50–0.95); only true exception is pathmnist stain — the one dataset with no headroom |
| Proposal C — augmentation-consistency graph | crop+flip pooled −0.63pp over 7 pairs |
| HFlip / mirror-confusion | flip IS the mirror-class map (specificity .9157) but touches 1.4% of points; gate FAILED |
| Task-geometry gradient selection | +0.19pp vs permutation control |
| `proxy_regime` (budget-matched demand proxy) | **retired unrun** — `d_i` measured at −2.51pp makes proxy quality moot |

**Root cause, stated once:** the probe is most confident exactly where the final
learner has the most room to differ (`frac(d<0.05)` vs subset-effect sd,
Spearman +0.8). See [04_instrument.md](04_instrument.md).

## Metric / axis family

| thread | outcome |
|---|---|
| Metric battery (182 paired tests) | 6 nominal p<0.05 vs 9.1 expected; **0 survive BH q<0.10** |
| "10/10 calibration unanimity" | fake — 1.31 effective dimensions |
| Robustness axis (M49) | r(corruption, clean) = 0.970–0.996 in 4/4 → corruption reproduces the clean ranking. **Do not** run the 68 GPU-h rescore |
| Grad-CAM attention | 15/15 between-arm pairs inside the within-arm seed null |
| L2 flips | GROSS exceeds the seed null, NET stays inside — reshuffle, not reduce; ~40 seeds/arm needed to detect 1.6pp |
| `ba_drop` as an endpoint | worst endpoint measured, biased toward weak methods |

## Pre-experiments for this round

| id | prediction | outcome |
|---|---|---|
| **A (D1)** | >15% of the budget has exactly-zero marginal gain on ≥2 of 3 live units | **0.0% on all 7** validity-passing cells → does not proceed. The 22.4% that motivated it was TD-Cover's small *target* demand set, which saturates; `graph_a2`'s whole-pool demand set does not |
| **A2** | head stable (J>0.90), tail a coin flip (J<0.70) | premise false: a 1e-3 perturbation destabilises the **whole** selection (J_head 0.18–0.97). Locally decisive (8–22 of ~10k within 5%), globally degenerate |
| **Arm overlap** | — | the mechanism above; zero GPU cost, largest structural finding of the project |

## Retractions worth remembering

- **Noise floor 2.05–2.35pp: retracted.** Artifact of pre-determinism-fix rows.
- **Pooled ρ 0.295: retracted.** Aggregation artifact (correct: 1.579 / centred 0.821).
- **WP1's m95 max-over-classes ≠ steep coverage curve** — the curve is concave.
- **TD-Cover's +3.15pp was partly a bug.** Repairing `A+A³ → A+A²` drops it to
  +1.47pp (p=0.123); the worst-recall gain survives (+8.22pp, p=0.0038).
- **TD-Cover's worst-recall gain is pathmnist-only.** On organamnist it is
  −7.56pp (p=.002).
- **"CCS ties Random" is a reimplementation artifact** — the local `ccs_select` is
  stratified random sampling with no β cutoff.
- **Spearman(cross-class share, subset-effect sd) = −0.80 was misread** as "no
  headroom". Round 1's mask result argues the opposite; round 2's mask arms test it.
