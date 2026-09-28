# TDGS evidence ledger (audit version, 2026-09-28)

**Scope**: all training results of TDGS / Graph-A2 on the 5 MedMNIST+ datasets up to 2026-09-28 18:00 HKT, plus configurations, pre-registered gates, literature and open items.
**Principles**: every number is traceable to a file plus hash. Each entry is labelled with one of three levels:
- **[Fact]**: already measured, with n and a CI.
- **[Unsupported]**: a hypothesis stated but either not tested or with a CI that includes 0.
- **[Post hoc]**: an explanation proposed after seeing the results. Not a conclusion.

The test set is used only for final evaluation and was never used for any selection or hyperparameter.

## 0. Sources and reproduction

| File | sha256 (first 16 chars) | Content |
|---|---|---|
| `results/round1/tdgs_round1_harvest.json` | d03c4819c28033b3 | R1: 75 cells, 2%, seeds 42–44 |
| `results/round2/round2_harvest_final_2b.json` | d44b9a2fa024163c | R2 + 2b: 185 cells (cluster `archive/round2/round2_harvest.json`, harvested 09-28) |
| `results/audit/contrasts_n5.txt` | 529797a206f3e6de | Output of the paired contrasts below |
| `code/audit_contrasts.py` | 77fd52a873d4f76b | Contrast script: later files override earlier ones, keyed by (ds, ratio, arm, seed); hierarchical bootstrap, B=20000 |

Reproduce: `python3 code/audit_contrasts.py results/round1/tdgs_round1_harvest.json results/round2/round2_harvest_final_2b.json`

Combined total: **260 cells** (the R1 graph_a2 / tdgs_cls cells are overridden by R2 cells with the same key; no double counting).

## 1. Configuration (verified)

| Item | Value | Basis |
|---|---|---|
| Datasets | path, blood, organA, organS, tissue (MedMNIST+ 224) | Author Table 1 |
| Budgets | 2%, 5%; bpc = ⌊N·r⌋ // C | Author formula; path 2% → 199×9 = 1791, matches archive |
| Embedding | UNI ViT-L/16, CLS, 1024-d, archived cache `embeddings_img224_smokefull` | Re-extracted 512 rows, bit-identical (job 35202, 09-28) |
| Graph | kNN k=50, global, symmetrized; K = A_sym + A_sym² | Archive replay (repo default k=10/per-class ≠ paper) |
| Training | ResNet-18 from scratch, 224, SGD lr 0.1, 1000 epochs, batch 256, cosine, no augmentation, `DETERMINISTIC_TRAINING=1` | Archive config, matches paper |
| Seeds | 42–46 (author convention); selection is seed-independent (all seeds read `_s42.npy`) | graphcov `seed + trial` |
| Primary metric | test balanced accuracy; secondary worst-class recall | Unchanged |
| TDGS | λ=0.5 fixed; class-flat weight 1/(C·n_c); quota exhaustion follows the author rule | `tdgs_select.py` |
| Checkpoint | final epoch (not chosen on val) | `checkpoint_rule: final_epoch` |

Built-in identity assertions (all passed during staging):
- (no mask, uniform weights) ≡ graph_a2, set-identical.
- λ=1 class weights are inert (set-identical).
- New for R3: the generic multi-term greedy reproduces `greedy_blended` exactly (local synthetic test; the cluster assertion runs at staging).

## 2. Test inventory

| Round | Arms | Budget / seeds | Cells | Status |
|---|---|---|---|---|
| Table 1 archive | 8 methods | 2%/5% × seeds 0–4 | 400 | Done; mean \|Δ\| vs paper 1.91pp |
| R1 (job 34970) | graph_a2, tdgs_cls, tdgs_d, tdgs_du, tdgs_perm | 2% × 42–44 | 75 | Done |
| R2 | + a2_perclass, mask, wcls, cls_perm, lam1; main arms 42–46 @2% & 5% | | 145 | Done |
| R2b | a2_perclass, cls_perm, mask, wcls @ 45/46 | 2% | 40 | Done (harvested 09-28) |
| Table 1 rerun (7 baselines × 42–46) | | 2%/5% | 350 | **0/350 trained**; see §6 incident |
| R3 (this route) | hpc_cls + multi-encoder arms | 2% | 130 (worklist) | Embedding extraction running; **no training results** |

## 3. Arm means (BA %, 2%; n in cell = number of seeds)

| Arm | path | blood | organA | organS | tissue | dataset mean |
|---|---|---|---|---|---|---|
| graph_a2 | 81.60/5 | 81.30/5 | 86.96/5 | 63.41/5 | 42.78/5 | 71.21 |
| a2_perclass | 82.01/5 | 83.39/5 | 87.45/5 | 63.12/5 | 42.38/5 | 71.67 |
| tdgs_mask | 81.85/5 | **84.69/5** | 86.98/5 | 62.62/5 | 46.95/5 | 72.62 |
| tdgs_wcls | 82.15/5 | 83.32/5 | 86.84/5 | 62.20/5 | 39.38/5 | 70.78 |
| **tdgs_cls** | 82.71/5 | 82.34/5 | 87.43/5 | 63.62/5 | 47.79/5 | **72.78** |
| tdgs_cls_perm | 81.80/5 | 84.13/5 | 86.60/5 | 62.89/5 | 41.44/5 | 71.37 |
| tdgs_lam1 | 79.86/3 | 85.50/3 | 87.89/3 | 61.43/3 | 48.24/3 | 72.59 |
| tdgs_d | 78.94/3 | 81.53/3 | 86.98/3 | 59.70/3 | 45.83/3 | 70.59 |
| tdgs_du | 78.08/3 | 81.93/3 | 86.84/3 | 59.95/3 | 46.70/3 | 70.70 |
| tdgs_perm | 79.56/3 | 81.08/3 | 86.25/3 | 61.74/3 | 45.89/3 | 70.90 |

At 5% (n=5): graph_a2 76.62 vs tdgs_cls 77.01.

Note: 4 of the 5 control arms have a higher blood BA than tdgs_cls; on blood the best is tdgs_mask (84.69).

## 4. Paired contrasts (pp; dataset mean [95% hierarchical bootstrap CI]; excl. tissue; same-sign datasets)

### 4.1 BA, 2%

| Contrast | Mean [CI] | Excl. tissue | Same sign | Level |
|---|---|---|---|---|
| tdgs_cls − graph_a2 | **+1.57 [+0.21, +3.40]** | +0.71 | 5/5 (16/25 cells) | **[Fact]** CI excludes 0; but the effect is dominated by tissue (+5.01) |
| a2_perclass − graph_a2 | +0.46 [−0.59, +1.70] | +0.68 | 3/5 | [Unsupported]: the author per-class scope also gives ≈ +0.7 on 4 datasets |
| **tdgs_cls − a2_perclass** | +1.11 [−0.75, +3.48] | **+0.03** | 3/5 | **[Unsupported]**: CI includes 0. Excluding tissue, ≈ 0 |
| tdgs_mask − a2_perclass | +0.95 [−0.71, +3.01] | +0.04 | 2/5 | [Unsupported] |
| tdgs_mask − graph_a2 | +1.41 [−0.50, +3.58] | +0.72 | 4/5 | [Unsupported] (CI includes 0) |
| tdgs_wcls − graph_a2 | −0.43 [−2.56, +1.60] | +0.31 | 2/5 | [Fact] tissue −3.40 is a single-dataset observation |
| tdgs_cls − tdgs_mask (weight given mask) | +0.16 [−1.59, +1.47] | −0.01 | 4/5 | [Unsupported]: no measurable independent contribution from the weight |
| tdgs_cls − tdgs_cls_perm (G2) | +1.41 [−0.83, +4.14] | +0.17 | 4/5 | [Unsupported]; tissue +6.35 |
| tdgs_lam1 − tdgs_cls (n=3) | −0.52 [−2.74, +1.69] | −0.61 | 2/5 | [Fact]: λ=1 is no better than λ=0.5 (it is not significantly worse either) |
| tdgs_d − tdgs_cls (n=3) | **−2.51 [−4.05, −0.99]** | −2.50 | 0/5 | **[Fact]**: difficulty demand is harmful |
| tdgs_du − tdgs_perm (n=3) | −0.20 [−1.75, +1.33] | −0.46 | 3/5 | [Fact]: the direction term shows no measurable effect |
| Interaction cls − mask − wcls + a2 (common basis, n=5) | +0.59 | — | 4/5 (blood −4.38) | [Unsupported] |

### 4.2 Worst-class recall, 2% (secondary endpoint)

| Contrast | Mean [CI] | Same sign |
|---|---|---|
| tdgs_cls − graph_a2 | +2.31 [−1.07, +5.75] | 4/5 (path −2.52) |
| tdgs_cls − a2_perclass | +2.38 [−0.61, +5.76] | **5/5** |
| tdgs_d − tdgs_cls | −5.70 [−9.36, −2.49] | 0/5 |

All are [Unsupported] except the last row, which is a [Fact].
tdgs_cls − a2_perclass is 5/5 in sign but the CI includes 0, and it is a secondary endpoint, so it **cannot be promoted** to a headline.

### 4.3 5% (G4)

| Contrast | Mean [CI] | Excl. tissue | Same sign |
|---|---|---|---|
| tdgs_cls − graph_a2 BA | +0.39 [−1.14, +2.39] | −0.55 | 2/5 |
| worst | −1.78 [−6.02, +2.45] | −3.53 | 2/5 |

**[Fact]**: G4 fails. At 5% there is no evidence of any gain.

### 4.4 Pre-registered gates, re-read at n=5

| Gate | Pre-registered criterion | n=3/mixed (HANDOFF §2.5) | **n=5 (this audit)** | Conclusion |
|---|---|---|---|---|
| Headline | 5/5 same sign | +1.57, 5/5 | Unchanged | Survives, tissue-dominated |
| G1 | a2_perclass reproduces ≥60% of the gain on ≥3/5 datasets ⇒ killed | 1/5, "passes" | **2/5** (blood 201%, organA 106%) | **Barely passes**. Excluding tissue, tdgs_cls − a2_perclass = +0.03, so the "difference from the author's per-class scope" holds only on tissue |
| G2 | cls − cls_perm pooled ≤ 0 ⇒ outcome B | +1.44 | +1.41, excl. tissue +0.17 | Not triggered; geometry matters only on tissue |
| G3 | Factorial | mask +1.33 / w −1.17 / interaction "+1.12" (mixed basis) | mask +1.41 / w −0.43 / interaction **+0.59** | The mask is the main intervention; the weight has no independent effect. Do not write "both are needed" |
| G4 | 5% ≥4/5 same sign | Fails | Fails | Outcome C: specific to 2% |
| G5 | Excl. tissue > +0.5 | +0.71 | +0.71 | Barely passes |

**What changed relative to HANDOFF §0/§2.5** (to be synced in §8):
1. G1 was written as "tdgs_cls − a2_perclass = +1.66". At n=5 it is **+1.11 with a CI including 0, and +0.03 excluding tissue**.
2. Interaction +1.12 → **+0.59** on a common n=5 basis.
3. The weight main effect −1.17 → **−0.43**.
4. Blood: tdgs_mask beats tdgs_cls by 2.35pp.

## 5. Three-level classification

### [Fact] (measured; CI excludes 0 or it is a structural identity)

1. At 2%, tdgs_cls beats graph_a2: BA +1.57 [+0.21, +3.40], 5/5 same sign.
2. Using difficulty as demand is harmful: −2.51 [−4.05, −0.99]; worst −5.70.
3. At 5%, the headline does not hold (2/5 same sign; CI includes 0).
4. At λ=1 the class weight is inert (proposition 3; runtime set-identity assertion).
5. With no mask and uniform weights, TDGS equals graph_a2 exactly.
6. Training is bit-deterministic; the harness replays bit for bit (48/48); the re-extracted UNI is bit-identical to the archive.
7. Cross-class edge share, K definition: path 2.5 / blood 26.5 / organA 42.4 / organS 53.8 / tissue 66.1%.
8. Graph-A2 names the UNI mismatch on TissueMNIST as its main limitation (primary text, verified).

### [Unsupported] (the CI includes 0, or the claim is untested)

1. "Global-graph mask ≠ per-class rebuilt graph, and the former is better": tdgs_cls − a2_perclass = +1.11, CI includes 0; excluding tissue +0.03.
2. "The class-flat weight is a necessary component": cls − mask = +0.16.
3. "Within-class geometry matters on all datasets": excluding tissue +0.17.
4. "Effect size scales monotonically with cross-class edge share": 2% BA-side Spearman = 0.00 (5 points).
5. "TDGS improves worst-class recall": +2.31, CI includes 0.
6. "The tissue gain comes from UNI mismatch rather than cross-class edges": the two are fully confounded (tissue is both the most mismatched dataset and the one with the highest cross-class share); R3 is the first test.

### [Post hoc] (explanations; not to be written as conclusions)

1. "5% fails because the larger bpc lets Graph-A2 cover the main modes itself": proposed after seeing the results; no direct test.
2. "At λ=1 path drops −3.30 because it loses the global term": single dataset, n=3.
3. "The weight alone is harmful because it pushes budget across the boundary while cross-class credit still flows": tissue only; n=5 main effect −0.43, CI includes 0.
4. "Blood per-class / mask being higher is related to boundary information": no test.

## 6. Infrastructure incidents and fixes (found in this audit)

- **The Table 1 rerun was not actually training.** In the first round of t1pack (35138/35139/35140), every cell failed with `rc=1`.
  - Cause: `t1_run_one.sh:59` wrote the config into `$SEED_ROOT` without creating it first (`No such file or directory`).
  - At audit time, **0/350** cells had been trained; 189/350 selections were staged.
  - Fix: add `mkdir -p "$SEED_ROOT"`. Cancelled 35138/39/40 and resubmitted as **35205 / 35206 / 35207** (09-28 ~17:40).
  - The "results on the morning of 09-29" in HANDOFF §8.4 no longer holds; re-estimate from actual progress.
- `organamnist r0.05 fps s46 FAIL`: same cause, not a separate problem.

## 7. Novelty (primary sources)

| Component | Closest prior work | Assessment |
|---|---|---|
| Class-conditional FL (task branch) | Wei, Iyer & Bilmes ICML 2015 eq. 6 | **Not new** |
| Global kNN + A+A² + per-class quota | Graph-A2 (the baseline itself) | Not new |
| Per-class vs global graph | Graph-A2 Table 3 (DermaMNIST: global +5.7 / +3.4pp) | Already ablated by the authors |
| Partition matroid ⇒ greedy 1/2 | Graph-A2 §2.3 | Not ours |
| "Global-graph similarity + class-scoped credit" blend (tdgs_cls) | No identical method found | Engineering-level difference; evidence exists only on tissue (§4.4 G1) |
| Multi-encoder robust coverage (R3) | Xu 2026 multi-view DPP (no real data); MedCAL-Bench (single encoder at a time); SATURATE (algorithm) | **Candidate gap**; search not exhaustive |

Conclusion: at the current evidence level, the stand-alone novelty of TDGS (tdgs_cls) is **not enough to support a high-level journal paper as the main method**. The contribution that can be written up is the "flatness / specific to 2% / measurement" story together with precise ablations, and the new method has to come from a new route (see `r3_prereg_20260928.md` and `research_plan_20260928.md`).

## 8. Open items

| Item | Status |
|---|---|
| Table 1 rerun, 350 cells | Resubmitted (35205–35207), 0 done |
| R3 embeddings | Job 35203 (blood done; others running or queued) |
| R3 staging | 35213–35215 (luhpc: blood/organA/organS), 35218/35219 (qiangzeng: path/tissue) |
| R3 training (130 cells) | Worklist `work/r3.txt` generated; **not submitted** (submit after the S0 gate and the Table 1 GPU load) |
| job 35076 (per-class cross-share) | Status still unconfirmed |
| DermaMNIST / CAMELYON17 independent validation | **Not executed**; data unavailable or unchecked |
| HANDOFF §0/§2.5 n=5 update | Done in this round (see HANDOFF) |
