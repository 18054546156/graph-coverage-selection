# TDGS follow-up: route comparison, preferred route, phased plan, and paper plan (2026-09-28)

Evidence basis: `evidence_ledger_20260928.md` (ledger). Pre-registration: `r3_prereg_20260928.md` (cluster copy sha256 743c4f70…, timestamp 17:42:59 HKT).
**Everything in this document except ledger §3–§5 is a plan or a hypothesis. None of it is a result.**

## 1. Starting from the ledger: what the actual problem is

1. The only significant positive result, tdgs_cls − graph_a2 at 2% (+1.57), is dominated by tissue.
   Excluding tissue it is +0.71 (CI not computed on its own), and it is ≈ 0 relative to the author's per-class scope (+0.03).
2. Tissue is exactly the dataset where the Graph-A2 authors admit that UNI does not fit: DAPI fluorescence versus H&E pretraining.
   It is also the dataset with the highest cross-class edge share (66%). **Encoder mismatch and cross-class leakage are fully confounded in the current data.**
3. The Act-2 result "BA is flat over the competitive region" (J=0.049, criterion span < random span) says that **changing the selection criterion on the same representation** almost never helps.
   All 20 closed threads are variants built on the same UNI geometry.
4. The real clinical/data setting: to build a new medical dataset, one chooses the samples to label from an unlabeled pool (the benchmark cheats slightly by using training labels for the quota).
   The modalities differ enormously (H&E, fluorescence, CT, blood smear), and **before any labels exist, nobody knows which foundation-model encoder is "right" for this modality.**

## 2. Three candidate routes

### R1: encoder-robust class-scoped coverage (multi-encoder robust coverage) ← **preferred**

- **Problem**: the selection geometry depends on one encoder, and encoders mismatch medical modalities (point 2).
- **Method**: every encoder v gets its own coverage objective F_v. Then maximize min_v F_v under the per-class quota, via a SATURATE-style truncated-sum greedy.
- **Testable gain**: on mismatched datasets (tissue / CT / blood), mv_rob > single-UNI TDGS; on the matched dataset (path), it should not lose.
  Also, max regret against the best single encoder (which is unknown in advance) should be the smallest.
- **Mechanism controls**:
  - mv_mean (is it just averaging?)
  - cls_cat (is it just feature concatenation?)
  - val-selected single view (is it just "picking the right encoder"?)
  - a2_<v> (is it just that Graph-A2 changes encoder?)
- **Novelty**: see prereg §2. The combination is not found in the retrieved literature; the algorithm itself is not new.
- **Risks**:
  - (a) The encoders may be selection-equivalent (S0-a).
  - (b) Flatness may also hold across encoders: every encoder falls inside the same "good region", so gains stay within noise.
  - (c) The tissue-dominated pattern may simply repeat.
- **Cost**: roughly 1 GPU-h for embeddings, 1–3 node-h for staging, and 70 GPU-h for 130 training cells.

### R2: matched hybrid per-class control (hpc_cls) — **mandatory, but not a stand-alone route**

- **Problem**: for TDGS to count as more than "Graph-A2 + Wei 2015", one must prove that "credit computed on the global graph" beats "credit computed on the per-class rebuilt graph" with everything else matched.
- **Method**: keep F_G (UNI global) + λ=0.5, and swap only the task branch's graph for the author's per-class rebuilt graph.
- **Gate**: G-R2 (prereg §4). If it fails, TDGS's methodological contribution shrinks to "a blend of known components".
- **Cost**: 25 cells, about 12 GPU-h. It runs as part of R3 (P4-a).

### R3': label-free cold start (no training labels) — **alternative, not started**

- **Problem**: in real annotation, the quota by class cannot be enforced because labels are not known yet. All 9 methods in this benchmark use training labels, so they are really "coreset / data pruning" methods, not annotation selection.
- **Method**: replace y with clusters on the multi-encoder graph (pseudo-labels), then compare against ProbCover / TypiClust / ε-AS.
- **Why not preferred**:
  - It changes the protocol (it no longer matches Graph-A2 Table 1), so the existing 400 + 260 cells cannot be reused.
  - The existing TD-Cover line already shows that label-free signals are unrecoverable on path (memory `td-cover-property-not-recoverable-train-only`).
- **When to take it up**: if R1 passes S0 but G-MV STOPs, and the user approves a protocol change.

### Excluded (not re-opened without new evidence)

Difficulty / direction demand, quota reallocation, Voronoi, stability gating, kernel variants on the same UNI, new metrics, corruption rescore (HANDOFF §11 and the memory list).

### Comparison

| | R1 | R2 | R3' |
|---|---|---|---|
| Starts from a real medical-data problem | Strong (modality ≠ encoder pretraining domain) | Weak (method attribution) | Strong (no labels) |
| Reuses existing results | tdgs_cls / graph_a2 / Table 1 used directly as comparators | Reused directly | Largely not reusable |
| Cheap early stop | S0 costs 0 GPU | — | No |
| Novelty | Candidate gap | Ablation only | Crowded field (ProbCover / TypiClust / ε-AS) |
| Risk of an overturning result | Medium–high (flatness) | Low | High |

## 3. The preferred route, R1: mathematics and algorithm

Notation and objective: see prereg §1. Additions:

**Normalization**: F_v(P) = 1, so F_v ∈ [0,1] and the views are on a common scale. This way no view is suppressed just because its kernel is denser, which is exactly the "calibration" requirement Xu 2026 emphasizes.

**Truncated surrogate**: for a fixed c, g_c(S) = Σ_v min(F_v(S), c) is monotone submodular, because min(·, c) of a monotone submodular function is still monotone submodular.
- If some S satisfies min_v F_v(S) ≥ c, then g_c(S) = |V|·c.
- Greedy under a partition matroid achieves ≥ 1/2 of the optimum of g_c.
- We keep the S with the largest min_v F_v over the c grid.
- We **do not claim** SATURATE's bicriteria guarantee, because the budget is not relaxed.

**Class scope**: every F_v contains the task branch 1[y_i = y_j]. So the multi-view part addresses "whose similarity to trust", and the class scope addresses "where credit is allowed to flow". The two are orthogonal.

**Algorithm (per dataset, one-off, CPU + 1 GPU for kNN)**

```
for v in V: Z_v ← frozen encoder(x) ; K_v ← kNN_50 → A_sym + A_sym²
            a_v ← 1[y_i=y_j]·K_v/max K_v ; normalize F_v(P)=1
for c in grid:  S_c ← lazy-greedy_{quota}( Σ_v min(F_v(S), c) )
S* ← argmax_c min_v F_v(S_c)
```

The gain of a candidate j is Σ_v [min(F_v + Δ_v(j), c) − min(F_v, c)], where Δ_v(j) is view v's marginal gain.
Lazy evaluation remains valid because the function is submodular.

**Complexity**
- Time: |V| × (single-view kernel O(N·k²) + FAISS kNN) + |grid| × |V| × single-view greedy.
- On tissue (N = 165k, 3 kernels at about 615M nnz each), memory is ≈ 3 × 7.4 GB for data + indices, plus the task kernels.
- Staging is given 200 GB; measured numbers will go into the report.

## 4. Phases and gates

Details are in prereg §4. Summary:

| Stage | Content | Cost | Continue | Stop / pivot |
|---|---|---|---|---|
| S0 selection space | Val kNN BA, cross-class share, Jaccard, cross-view coverage matrix | 0 training | S0-a J < 0.5 on ≥3/5 datasets | Views equivalent ⇒ stop R1 |
| Minimal validation + matched baselines | P4-a hpc_cls (n=5) + P4-b/c/d (n=3) | 130 cells | G-MV GO | STOP / PIVOT as in the prereg |
| Multi-seed | mv_rob / best single view / tdgs_cls up to n=5 | +30 cells | CI excludes 0 and excluding tissue ≥ +0.3 | Otherwise report as a screening result |
| Cross-budget | 5%, n=5 | +45 cells | Not significantly worse | — |
| Cross-representation | Already built in: the 3 encoders themselves; optionally add a medical encoder (e.g. BiomedCLIP; needs download and verification) | Staging only | — | — |
| Independent medical validation | DermaMNIST (the author's own imbalanced dataset) → CAMELYON17-WILDS | Needs data work | — | **Not executed** |

## 5. Paper plan (target: IEEE TMI / MedIA; TMI main text is 10 pages, so a 16-page version suits MedIA or TMI plus supplementary)

Main line (written according to how the evidence turns out; **if R1 stops, the "Method" section shrinks back to the audit + negative-results story**):

1. **Introduction**: in medical data selection, the representation does not match the modality. Graph-A2's own limitation, and MedCAL-Bench's encoder dependence.
2. **Related work**: coresets / cold-start AL / submodular (Wei 2015, SATURATE) / foundation-model embeddings / multi-view selection (Xu 2026).
3. **Benchmark and instruments**:
   - reproduction of the 400-cell archive
   - bit-level determinism
   - Table 1 on the author seeds 42–46
   - seed noise σ_train 1.37 / σ_subset 0.82
4. **Finding 1: flatness**. Changing the criterion on a single representation is ≈ random (J = 0.049, span 1.48 vs 1.78; 20 closed threads).
5. **Finding 2: the effect of class-scoped credit is concentrated on the mismatched dataset**. TDGS at 2%: +1.57, 5/5; excluding tissue +0.71; vs per-class +0.03; 5% fails. The full factorial and pre-registered falsifications.
6. **Method: encoder-robust coverage** (only if G-MV GO): model, algorithm, complexity.
7. **Experiments**:
   - Table 1 (9 methods + mv_rob)
   - regret table
   - ablations (mean / cat / val-select / hpc)
   - cross-budget
   - independent dataset
8. **Mechanism**: Spearman of cross-view coverage matrix, val kNN BA and cross-class share against ΔBA (reported as correlation, not causation).
9. **Limitations**: MedMNIST 28→224 resolution; no patient-level structure; the benchmark uses training labels; 5% is not discriminative.

Planned figures and tables:

| Item | Content | Data status |
|---|---|---|
| Fig 1 | Schematic: modality ↔ encoder mismatch; multi-view coverage | No dependency |
| Fig 2 | Flatness (Act 2) | Existing |
| Fig 3 | Forest plot: all contrasts in ledger §4.1 | Existing |
| Fig 4 | Heatmap of val kNN BA per encoder × dataset | After R3 S0 |
| Fig 5 | Cross-view coverage matrix F_v(S_arm) | After R3 S0 |
| Fig 6 | Regret: each arm vs the best single view | After R3 training |
| Table 1 | Main BA table, 9 + R3 columns | Table 1 rerun + R3 |
| Table 2 | Worst-class recall | Same |
| Table 3 | Pre-registered gates and falsifications | Existing + R3 |
