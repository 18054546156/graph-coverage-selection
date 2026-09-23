# Research plan — selection x weighting interaction, real training

Status 2026-09-23. Supersedes the task-geometry branch, which the pre-registered
gate closed (`results/run_33659/gate_decision.json`, NO_GO, +0.19pp).

---

## 1. What is actually being claimed

**Claim.** The right way to weight a coreset during training is not a property
of the coreset alone. It depends on what the selection algorithm optimised.

- A selection that encodes **no** distributional structure (random) leaves the
  density information on the table, and assignment-mass weighting supplies it.
- A selection that already established an **unweighted moment invariant**
  (herding: `min || (1/k) sum z_i - mu_c ||`) is *damaged* by reweighting,
  because the weights destroy the very invariant it built.
- A selection that optimised **coverage** (facility location, graph_a2) is
  roughly orthogonal to weighting and shows little effect either way.

**Estimand.** The difference-in-differences

```
DiD = [ BA(random, voronoi) - BA(random, equal) ]
    - [ BA(herding, voronoi) - BA(herding, equal) ]
```

DiD is the estimand, not either main effect. It cancels any global effect of
reweighting (regularisation change, effective-LR change, class-prior shift) and
isolates the interaction, which is the claim.

**Why this is worth a paper-shaped effort and the previous eight ideas were
not.** Every prior branch in this project proposed a *better selection score*
and died. This one says the score is not the whole object: (selection, weight)
is a pair, and the field reports only the first half. It also makes a falsifiable
sign prediction per method rather than a "ours is higher" claim.

---

## 2. Evidence so far (linear probe, frozen UNI, train-only audit)

`results/run_33672/`, 5 datasets x 6 blocks x 8 arms x 3 weightings = 720 rows.

Coverage-repair only (`voronoi_within - equal`, class priors held fixed), paired
over 30 (dataset, block) cells:

| arm | delta (pp) | t | p |
|---|---|---|---|
| random_plus_pilot | +0.926 | +5.32 | <0.0001 |
| random | +0.848 | +4.65 | 0.0001 |
| permuted_task_geometry | +0.210 | +1.91 | 0.066 |
| graph_a2 | +0.143 | +1.50 | 0.14 |
| geometry_plus_pilot | +0.118 | +0.93 | 0.36 |
| task_geometry | +0.093 | +0.71 | 0.48 |
| geometry | -0.056 | -0.54 | 0.59 |
| **herding** | **-0.520** | **-3.49** | **0.0015** |

Probe DiD (random - herding) = **+1.37pp**.

Mechanism evidence (`results/run_33659/moment_matching.json`, pre-registered in
`PREDICTION_moment_matching.md` before `eps_mean` was computed):

- P3 **PASS 5/5**: herding attains the smallest first-moment error of all eight
  arms, by a factor of 2-3. Its defining property, confirmed.
- P1 **PASS 5/5** for herding: weighting degrades its moment error on every
  dataset, by 3-5x the magnitude of any other arm.
- P2 **PASS 4/5**: Spearman(delta_eps, delta_BA) < 0 per dataset.
- **Limit, stated plainly:** the moment channel explains the herding *damage*
  and NOT the random *benefit* (random's delta_eps is -0.0025 while its gain is
  +0.85pp; graph_a2's moment degrades yet its BA rises). Two channels, not one.
  The benefit channel tracks headroom, not the first moment.

---

## 3. Dataset priors (`results/dataset_priors.json`)

| | blood | organa | organs | path | tissue |
|---|---|---|---|---|---|
| headroom (pp) | 5.19 | 11.64 | 13.22 | **0.61** | 10.12 |
| probe BA full pool | 0.9531 | 0.8864 | 0.7809 | 0.9928 | 0.4812 |
| pool imbalance | 1.18 | 1.00 | **1.63** | 1.00 | 1.00 |
| near-dup (d<0.25) | **0.191** | 0.019 | 0.003 | 0.000 | 0.001 |
| intrinsic dim | 24.3 | 8.2 | 9.3 | 17.2 | 23.9 |

Three rules that follow and are now binding:

1. **OrganA and OrganS are the same 201 LiTS CT volumes**, differing only in
   slice plane. They are ONE replication unit. Every "n/5 datasets" count in
   this project's history overstates replication.
2. **PathMNIST is excluded**, with a stated reason: its entire headroom is
   0.61pp, so it cannot express a 1pp effect even in principle. Excluding it is
   a pre-registered decision made on a covariate, not on any outcome.
3. **Effects are reported as % of headroom**, not pp. Coverage-repair gain is
   6-13% of headroom across all five datasets (Pearson +0.950 vs headroom;
   Spearman +1.000 with the organ family pooled, n=4).

---

## 4. What changes when we move to real training

Four protocol changes, all one-way doors, all recorded here before the run:

1. **Official test split is opened.** Until now every number came from a
   train-only audit split and `official_test_accessed: false`. The interaction
   claim has to be made on the real endpoint. This happens **once**, with the
   analysis below fixed in advance. No second look, no re-tuning.
2. **Full training split, no per-class cap.** The 1000/class cap was a probe
   convenience. Selection runs on the whole train split.
3. **Final-epoch model, no checkpoint selection.** `graphcov`'s shared path can
   reload a "best" checkpoint chosen on an eval loader that falls back to the
   *test* loader. That is test-based model selection. `train_weighted.py`
   reports the final epoch, full stop.
4. **Determinism is actually on.** `graphcov.run.evaluation.set_seed` never
   calls `torch.use_deterministic_algorithms`, its in-process `PYTHONHASHSEED`
   write is a no-op, and its DataLoader has no `generator`/`worker_init_fn`.
   All four are handled in `train_weighted.py`, plus `CUBLAS_WORKSPACE_CONFIG`
   before torch import.

### Why a standalone runner

`graphcov.run.evaluation.train_and_evaluate` hardcodes `nn.CrossEntropyLoss()`
against a 2-tuple loader; per-sample weights cannot be passed. Patching the
shared package would put another user's in-flight Table-1 jobs at risk (jobs
33574 / 33611 / 33614 were running during this work). `train_weighted.py`
borrows only model and data loading.

---

## 5. Design

**Factorial.** 3 datasets (blood, organs, tissue -- one per independent source)
x 2 arms (random, herding) x 2 weightings (equal, voronoi_within) x 10 selection
replicates x 2 training seeds = **240 runs**.

**Selection replicates are not optional.** Herding is deterministic given a
pool, so with one pool it contributes a single subset while random contributes a
new one per seed. That asymmetry has already caused misattribution in this
project (`NONDETERMINISTIC_METHODS` + `selection_cache`). Each replicate draws
an 80% subsample of the train split and runs *both* arms on it, so both carry
selection variance and the weighting pair is matched within (replicate, seed).

**Budget.** Set from `results/ladder.json`, which sweeps
budget in {10, 25, 50, 100, 250} x cap in {1000, full}. The operating point is
chosen to maximise the probe DiD, on the probe only, before any GPU run. The
choice is recorded in `protocol_training.json` and then frozen.

**Primary endpoint.** Balanced accuracy on the official test split.
**Primary statistic.** DiD as defined in section 1, per dataset, with the
paired structure (replicate, seed).
**Decision rule.** Pre-registered in `gate_training.py` before results are read.

### Secondary, explicitly non-gating

- per-arm main effects (random alone, herding alone)
- worst-class recall and ECE15 (ECE is meaningful here, unlike on the probe
  where L2 shrinkage on unit-norm features made it degenerate at ~0.49)
- first-moment error measured on the real selections, for the mechanism
- effect as % of headroom, for cross-dataset comparison

### Built-in negative controls

- **bloodmnist is a predicted near-null** (5.19pp headroom, probe gain +0.33pp).
  If the effect appears there as strongly as on organs/tissue, the headroom
  story is wrong.
- **geometry / graph_a2 are predicted near-zero** and can be added as a third
  arm if the 2x2 lands positive. They are the "no interaction" cell.
- A **permuted-weight control** (the same weight multiset randomly reassigned
  within class) separates "weights carry Voronoi information" from "unequal
  weights change training at all". This is the control the task-geometry branch
  had and it is what made that NO_GO interpretable.

---

## 6. Known risk: the effect may be under the noise floor

Probe DiD is 1.37pp. Real-training paired std in this project is 2-3pp, and the
measured screen MDE at n=5 is 1.2-8.8pp. At n=20 paired units the MDE is roughly
1.3-1.9pp, i.e. the same order as the effect. Two mitigations, in order:

1. **Move to the regime where the effect is largest.** Gain tracks headroom and
   headroom grows as budget shrinks, so a smaller budget should amplify the
   effect. This is exactly what `ladder_locate_operating_point.py` measures.
   Choosing the operating point on the *probe* and then testing on real training
   is legitimate; choosing it on the real-training results would not be.
2. **Pair as tightly as possible.** Same subset, same init, same data order;
   only the loss weights differ. Selection noise is removed from the weighting
   contrast entirely.

If the ladder shows no budget where the probe DiD exceeds ~3pp, the honest move
is to say so and either enlarge n or report the probe result as a probe result.
It is not to run 240 GPU jobs at a known-underpowered effect size.

---

## 7. Artefacts

| file | role |
|---|---|
| `run_linear_probe.py` | probe experiment, 3 weightings, frozen tau |
| `gate_decision.py` | frozen probe -> GPU gate (task-geometry branch: NO_GO) |
| `test_moment_matching.py` | mechanism test against `PREDICTION_moment_matching.md` |
| `measure_dataset_priors.py` | dataset covariates -> `results/dataset_priors.json` |
| `ladder_locate_operating_point.py` | budget x scale sweep -> operating point |
| `train_weighted.py` | real ResNet-18 with per-sample loss weights, official test |
| `build_training_cells.py` | *to write*: emits the 240-cell jsonl |
| `gate_training.py` | *to write*: frozen decision rule for the real run |
| `protocol_training.json` | *to write*: the frozen protocol for phase B |

Cluster: `hpc-sis` account has **only** `qos-normal` (4G total) and
`qos-high-gpu` (128 cpu / 12 gpu / 256G). `qos-high-gpu` is **not permitted on
CPU partitions** -- CPU work must run on a GPU partition. `MaxJobsPU=5`, and
three slots were occupied by an unrelated in-flight Table-1 run.
