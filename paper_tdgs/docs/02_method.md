# Method and the full arm inventory

## The author's method (`graph_a2`) — the λ=0 anchor

1. Frozen **UNI** embeddings of the training pool, L2-normalised.
2. **Global** cosine kNN graph, `k=50`, symmetrised `A ← max(A, Aᵀ)`.
3. `A_sym = D^(−1/2) A D^(−1/2)`.
4. `K = A_sym + A_sym²` (`k_hops=2`).
5. Lazy-greedy **facility location** `F_G(S) = Σ_{i∈P} max_{j∈S} K_ij` under an
   equal per-class quota, `importance='optional'` (uniform).

Budget: `bpc = int(N·ratio) // C`, then `|S| = bpc·C`. Verified on all five —
pathmnist 89996 @2% → `1799 // 9 = 199`, and `199×9 = 1791` = the archived
`n_selected`. Note this is 1791, **not** 1800; copying the formula loosely breaks
paired comparison.

Quota exhaustion follows the author's rule exactly: once a class is full, **every**
sample of that class becomes ineligible.

## TDGS (the round-1 design)

Out-of-fold proxy probabilities `p_i` from a linear head on frozen UNI (the fold
containing `i` is held out; in-fold probabilities would measure memorisation):

```
d_i    = 1 − p_i(y_i)                          demand STRENGTH ∈ [0,1)
v_i(c) = p_i(c) for c ≠ y_i, else 0            confusion direction
u_i    = v_i / (‖v_i‖₂ + ε)                    unit, non-negative
a_ij   = 1[y_i = y_j] · K̄_ij · ⟨u_i, u_j⟩      demand–support relation
F_task(S) = (1/C) Σ_c (1/|P_c|) Σ_{i∈P_c} d_i · max_{j∈S_c} a_ij
F_new(S)  = (1−λ)·F_G(S)/F_G(P) + λ·F_task(S)/F_task(P)        λ = 0.5
```

`K̄ = K / max(K)`. Both terms are monotone non-decreasing and submodular (each is
a sum of non-negatively-weighted max-coverage terms), so the blend is too, and
lazy greedy under the per-class quota is the same algorithm the author already
uses.

**Where `d_i` sits matters.** It is on the **demand point**, not the candidate. An
*easy* candidate that supports many hard demands still wins. That is what
distinguishes it from difficulty-as-criterion, which is separately closed on this
corpus (−6.72pp with a within-class permutation control at exactly zero).

**Information budget.** Source pool inputs + source labels only. No target inputs,
no target labels, no test confusion matrix. This is a *stricter* budget than
TD-Cover, which reads unlabeled target inputs — so TDGS is a legal entry in the
author's Table 1 and TD-Cover's numbers are **not** transferable to it.

---

## All 10 arms

### Round 1 — run, 75 cells (`code/tdgs_select.py`, frozen)

| arm | definition | isolates | result |
|---|---|---|---|
| `graph_a2` | λ=0 | the anchor; must reproduce the archive | reproduces **order-exact** on organsmnist |
| `tdgs_cls` | λ=0.5, `d≡1`, `⟨u,u⟩≡1`, weight `1/(C·n_c)` | within-class credit + class-flat demand | **+1.76pp BA, +3.54pp worst, 5/5** |
| `tdgs_d` | adds `d_i` | difficulty as a demand weight | **−2.51pp** vs `tdgs_cls`, 5/5 negative |
| `tdgs_du` | adds `⟨u_i,u_j⟩` | class-pair direction structure | +0.10pp vs `tdgs_d` |
| `tdgs_perm` | `u` permuted within class, `d` left aligned | **null for the direction term** | `tdgs_du − tdgs_perm = −0.20pp` → **inert** |

### Round 2 — staged, 145 cells (`code/tdgs_select_r2.py`)

All five are **free of `d_i` and `u_i`**, so round 2 never fits the probe, needs no
GPU for selection, and is deterministic given the embeddings (the only randomness
is `tdgs_cls_perm`'s seeded permutation).

| arm | definition | isolates |
|---|---|---|
| `a2_perclass` | `graph_a2` with the kNN graph built **inside each class**, k=50, λ=0 | **the kill control.** Is `tdgs_cls` just the author's `global_selection=False` cell? |
| `tdgs_mask` | within-class mask, **uniform** demand weight | the mask main effect |
| `tdgs_wcls` | **no** mask (global credit), class-flat weight | the weighting main effect |
| `tdgs_lam1` | λ=1 — pure class-compatible, no global blend | is the global term load-bearing? |
| `tdgs_cls_perm` | mask kept, candidate identity permuted **within class** | **the positive's own matched null** |
| `tdgs_mask_before_prop` | `(M⊙A)+(M⊙A)²` instead of `M⊙(A+A²)` | do cross-class intermediate nodes matter for the masked relation? |

#### The 2×2 factorial, with two cells already paid for

|  | uniform demand weight | class-flat weight `1/(C·n_c)` |
|---|---|---|
| **global credit** (no mask) | `graph_a2` — *already run* | `tdgs_wcls` — new |
| **within-class credit** (mask) | `tdgs_mask` — new | `tdgs_cls` — *already run* |

The top-left cell is `graph_a2` **by identity, not approximation**: with no mask and
uniform weights the demand term is `K/kmax` with `w=1`, i.e. proportional to `F_G`,
so `F_new` is a positive multiple of `F_G` and greedy returns the same set.
`tdgs_select_r2.py --assert-a2-identity` proves this at runtime rather than
asserting it in a comment.

The two new arms identify both main effects and an interaction contrast, but the
contrast is an estimate, not automatic evidence that the mechanisms interact:

```
mask effect   = (tdgs_mask − graph_a2)   and  (tdgs_cls − tdgs_wcls)
weight effect = (tdgs_wcls − graph_a2)   and  (tdgs_cls − tdgs_mask)
interaction   = tdgs_cls − tdgs_mask − tdgs_wcls + graph_a2
```

A large, reproducible interaction would mean the corrections are **not
separable**. In the current 2%/3-seed screen the mean interaction is +1.12pp,
so the paper should report it as provisional. Moreover, at the pure masked
`lambda=1` endpoint the class-flat row weight is theoretically inert under a
fixed filled quota; any weight effect here belongs to the `lambda=0.5` blend
against the global term.

The code now exposes two checks for this distinction:

```text
--assert-a2-identity
--assert-class-weight-identity
--arms tdgs_mask_before_prop
```

The first two are selection-ID identity checks. The optional third arm compares
`M ⊙ (A + A²)` with `(M ⊙ A) + (M ⊙ A)²`, while keeping the global term fixed.

#### Two implementation notes that change what the arms mean

**`a2_perclass` is not the same as masking a global graph.** Building the graph
inside a class counts `k=50` among *same-class* neighbours. Masking a globally-
built graph leaves a point with only as many neighbours as happened to share its
label — on tissuemnist (58.5% cross-class) that is roughly 20 of 50. These are
genuinely different operators, which is why both are run.

Implementation: block-diagonal `K` from per-class blocks, λ=0. Greedy over a
block-diagonal kernel with an equal per-class quota returns the **same set** as
independent per-class greedy — gains in one block never depend on coverage in
another, and the quota forces exactly `bpc` picks per class. Only the interleaving
order differs.

**`tdgs_cls_perm` permutes columns, not values.** For entries `(i,j)` with
`y_i = y_j`, `j` is replaced by `π_c(j)`. Demand rows are untouched, so every
demand point keeps its own degree and value multiset; only *which candidate
supplies which support* is randomised. Shuffling values instead would leave
adjacency intact, and within-class `K̄` values are fairly homogeneous, so it would
be a weak null. Every prior positive in this project that skipped its matched null
later died (Voronoi, task-geometry, FRACTAL), so this arm is not optional.
