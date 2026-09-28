# Future experiments — round 2 (staged) and round 3 (gated)

Every block below is in `code/make_worklist.py` in the order given, because
`tdgs_pack.slurm` shards the worklist by a global stride: all 15 jobs start near
the front, so whatever finishes is a **prefix**, and the prefix is always the most
decisive experiment available.

---

## Round 2 — 145 cells, staged, ≈2.5–3 h on 30 slots

| block | arms | cells | decides |
|---|---|---|---|
| **P0-a** | `a2_perclass` | 15 | the only control that can kill the round-1 positive |
| **P0-b** | `tdgs_mask`, `tdgs_wcls` | 30 | completes the 2×2 factorial → both main effects + interaction |
| **P1-a** | `tdgs_cls_perm` | 15 | the positive's own matched null |
| **P1-b** | `tdgs_lam1` | 15 | is the global term dead weight? |
| **P1-c** | `graph_a2`, `tdgs_cls` @ seeds 45,46 | 20 | lifts the headline from n=3 to n=5 at 2% |
| **P1-d** | `graph_a2`, `tdgs_cls` @ ratio 0.05, 5 seeds | 50 | the second budget — a 2%-only result is half a table |

Selection staging is CPU-cheap: **all round-2 arms are probe-free**, so no OOF head
is fitted and nothing but the embeddings is needed.

### Pre-registered gates — written before round 2 ran

**G1 — KILL GATE.** If `(a2_perclass − graph_a2)` recovers **≥60%** of
`(tdgs_cls − graph_a2)` on **≥3/5** datasets, then `tdgs_cls` sits inside the
author's `compare_global.py` cell and is **not** a contribution.
*Consequence:* drop the coverage-alignment framing; the paper becomes Acts 1–3
(the flatness result) plus a diagnostic, which is still publishable but is a
different paper. Do not write Act 4 before this returns.

**G2 — NULL GATE.** If `(tdgs_cls − tdgs_cls_perm) ≤ 0` pooled, the gain comes
from confining credit within class **at all**, not from within-class kNN geometry.
*Consequence:* keep the result, but describe the method as a **class-scoping
correction**, not a graph method. This is a positioning change, not a retraction.

**G3 — FACTORIAL GATE.** If the mask main effect is **≥3×** the weight main
effect, report the method as class-compatible coverage and demote the weighting to
an implementation detail. If comparable, both are required, and the interaction
term decides whether they are separable.

**G4 — POWER GATE.** The headline must survive at **n=5 seeds** *and* at
**ratio 0.05** with the same sign on ≥4/5 datasets. If it holds only at 2%, the
claim is budget-specific and the abstract must say so.

**G5 — TISSUE GATE.** If excluding tissuemnist drops the pooled effect below
**+0.5pp**, the honest headline is "one dataset, mechanistically predicted, plus a
5-dataset directional trend" — **not** a 5-dataset mean improvement. Round 1
already sits at +0.81pp excluding tissue, so this gate is live.

### Predictions recorded now, so they cannot be rationalised later

- `tdgs_mask` should carry **most** of the mask main effect, and its per-dataset
  ordering should track cross-class edge share (tissue ≫ path).
- `tdgs_wcls` should be **small on bloodmnist and pathmnist** (least pool
  imbalance) and largest on organsmnist (pool imbalance 1.63) and tissuemnist.
- `a2_perclass` should be **worse than `graph_a2` on pathmnist** (0.6% cross-class
  — confining credit costs geometry and buys nothing) and could beat it on
  tissuemnist. If it beats `graph_a2` *everywhere*, that contradicts the author's
  own `compare_global.py` and the archive config needs re-auditing before anything
  else is believed.
- `tdgs_lam1` ≈ `a2_perclass` in *direction* on tissuemnist; if λ=1 ≥ λ=0.5 the
  method simplifies and the paper gets shorter.

---

## Round 3 — gated on round 2

### R3-a  Formal comparison against all 8 baselines (0 new baseline runs)

Winner vs the author's 8 methods, 2% and 5%, 5 seeds. **The baselines are not
re-run** — 394/400 already exist and are validated to ±4pp. Only the winner's
cells are new. Gated on G1 passing.

### R3-b  Worst-class recall as a declared second axis (0 new runs)

The data already exists. What is needed is the *medical* argument: why the worst
class's recall is the deployment-relevant quantity in these five tasks, and
whether the class it lands on is clinically meaningful. Supporting facts already
measured: the worst class is dataset-intrinsic (cross-arm agreement .88–.99),
errors concentrate 2.5–5.5× on one partner class, and in the competitive band the
tail gap beats the mean gap in 89.8% of 215 pairs. Caveat to state honestly: 6pp
on 1 of 11 classes is only 0.55pp of BA.

### R3-c  Targeted archive repairs (≈138 runs) — **needs explicit approval**

| id | defect | why it matters now |
|---|---|---|
| **D-A** | `selection_seed ≡ training_seed` | confounds subset identity with initialisation — exactly the question this paper is about |
| **D-B** | corruption rows pre-pooled over severity, single seed | rBE has never been computed at power; CEC is not computable at all |
| **D-C** | bloodmnist selection unreproducible; EVA off-config | test the `diag(K)`-not-zeroed hypothesis first (known to change 15% of blood's picks) — it is a 0-GPU check |

Do D-C's `diag(K)` test before spending any GPU on D-C.

### R3-d  Related work to integrate (named, must appear)

GIO (ICLR 2024), Uncertainty Herding (ICLR 2025), CRAIG (ICML 2020),
GRAD-MATCH (ICML 2021), DRoP (ICLR 2025); plus CCS, MaxHerding, D² Pruning,
ENRICH, confounder-aware medical selection, Tellez 2019.

**Positioning, precisely.** The NUCS / Class-Proportional line repairs
*class deletion* — global-hardest selection starving a class entirely. That
disease is **real** (organsmnist loses 6 of 11 classes to global-hardest) but it
**does not exist in this harness**, because every method is already capped
per-class; our `el2n_top` *is* their Hardest-CP and still scores −28.9pp. We repair
something different: given per-class caps, **coverage credit still flows across
classes and demand is still frequency-weighted**. That cell is unoccupied.

Also note for the related-work framing: the local `ccs_select` is stratified random
sampling with no β cutoff, so "CCS ties Random" here is a reimplementation
artifact, not a finding about CCS. D² Pruning, by contrast, is genuinely closed as
difficulty-demand.

---

## Explicitly excluded — do not reopen

- The kernel-variant line (`A+A³` vs `A+A²`, `graph_a3`) — scoped out.
- Anything not compared against the author's benchmark — scoped out.
- Adding new metrics (182 paired tests, 0 survive BH).
- The 68 GPU-h organ/tissue corruption rescore (M49 closed that axis:
  r = 0.970–0.996).
- Stability-based gating (rejected twice).
- `proxy_regime` round 2 / budget-matched demand proxy — `d_i` is measured at
  −2.51pp, so proxy quality is moot.
- Deferred, do not resume without asking: GLP-Select Phase 1, the 40 staged
  budget-curve jobs, pushing to `fork18`, the stalled TD-Cover worklist (16/72),
  the 24 Proposal C jobs, `proxy_hunt.py --ratio 0.05`, the stain precheck σ=0 null
  arm.
