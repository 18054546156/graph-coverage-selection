# Round 3 validation report (live document, 2026-09-28 18:55 HKT)

Status labels used throughout: **[Verified]** = actually run, with a traceable output. **[Interim]** = incomplete (datasets or seeds missing), no conclusion drawn. **[Not run]** = planned only.

## 1. Selection space S0: zero training [Verified, 4/5 datasets; tissue staging 35250 still running]

| Dataset | val kNN BA uni / dinov2 / clip | J(tdgs_cls, cls_dinov2 / clip) | J(tdgs_cls, hpc_cls) | Relative coverage of tdgs_cls in dinov2 / clip | mv_mean min_v F̂_v | mv_rob (A1) min_v F̂_v, c | J(mv_rob, mv_mean) |
|---|---|---|---|---|---|---|---|
| blood | 94.7 / 90.2 / 86.2 | 0.015 / 0.015 | 0.30 | 0.85 / 0.83 | 0.947 | 0.948, 0.95 | 0.46 |
| organA | 78.7 / 87.3 / 87.8 | 0.010 / 0.017 | 0.24 | 0.84 / 0.85 | 0.942 | 0.945, 0.95 | 0.56 |
| organS | 73.4 / 72.5 / 69.8 | 0.020 / 0.009 | 0.18 | 0.82 / 0.83 | 0.940 | 0.941, 0.94 | 0.52 |
| path | 99.6 / 96.7 / 96.9 | 0.011 / 0.014 | 0.73 | 0.87 / 0.85 | 0.955 | ≈0.955, 0.96 | 0.39 |

- **Identity checks** (generic greedy == `greedy_blended` order == the round 1/2 staged sets) pass on all 4 datasets. Quota assertions pass.
- **S0-a passes.**
- **S0-b:** on organA, UNI's val kNN BA is 8.6–9.1pp below the other two encoders. On blood / path, UNI leads by 2.7–4.5pp. On organS the three are within 3.6pp.
  - So no single encoder is best everywhere at the zero-training level.
  - **This is not BA.** Memory `uni-space-surrogates-are-structurally-blind` shows that surrogates in embedding space have repeatedly failed to predict subset effects.
- **mv_rob (after Amendment 1)** differs from mv_mean in the selected set (J 0.39–0.56), but its objective value is only 0.001–0.004 higher.
  - Prediction written before training: the BA difference between the two will be within noise.

## 2. P4 training (hpc_cls / mv_rob / mv_mean / cls_* / a2_*) [Not run: submitted, job 35258 queued]

No BA exists yet. Gates G-R2 and G-MV are still open.

## 3. Val evaluation fidelity [Verified]

`val_eval.py` re-runs inference with `final.pt`. Self-check: running the same code on the test split must reproduce the harness's `predictions_clean.npz`.

| Dataset | Label order | argmax agreement | \|ΔBA\| | max\|Δlogit\| | GPU |
|---|---|---|---|---|---|
| blood | ✓ | 1.000000 | 0 | 1.9e-2 | H100 |
| organA | ✓ | 0.999888 | 9.6e-5 | 3.4e-3 | H100 |
| organS | ✓ | 1.000000 | 0 | 6.8e-3 | H100 |
| path | ✓ | 1.000000 | 0 | 5.8e-2 | H100 |
| tissue | ✓ | **0.997716** | 8.4e-5 | 6.9e-2 | H100 → **fails the 0.999 criterion**; rerunning on A100 (the training GPU), job 35265 |

Conclusion: the evaluation code is correct. Differences across GPUs are TF32-level, and BA differs by < 0.01pp.

## 4. vsel: per-dataset configuration chosen on val [Interim, 4/5 datasets, no tissue]

Leave-one-seed-out. Test BA, pp, mean across datasets.

| Candidate set | vsel − tdgs_cls | vsel − graph_a2 | vsel − mean(𝒞) | Worst recall vs tdgs_cls | Gate |
|---|---|---|---|---|---|
| C5 = {a2, a2_perclass, mask, cls}, n = 5 | **−0.06** [−1.19, +1.00], 2/4 positive | +0.65 [−0.12, +1.77], 3/4 | +0.12 [−0.67, +0.74] | −1.09 | Not passed on 4/5 |
| C6 = C5 + lam1, n = 3 | −0.04 [−1.72, +2.19], 1/4 | +0.77, 1/4 | +0.37 | −1.29 | Not passed on 4/5 |

What val chose, by dataset (C5):

| Dataset | Val's choice | vsel − cls | oracle − cls | r(val, test) |
|---|---|---|---|---|
| blood | tdgs_mask 4/5 | +0.84 | +3.78 | 0.92 |
| organA | a2_perclass 5/5 | +0.03 | — | 0.36 |
| organS | tdgs_cls 5/5 | 0 | — | 0.50 |
| path | **graph_a2 5/5** | **−1.11** | — | 0.59 |

Reading of the table:

1. On blood, val reliably picks tdgs_mask. That matches the known fact that on blood, mask > cls. **Per-dataset choice helps where val and test are consistent (r = 0.92).**
2. On path, val picks graph_a2 5/5 times, but on test tdgs_cls is better (−1.11).
   - PathMNIST train/val come from NCT-CRC-HE-100K, while test is CRC-VAL-HE-7K from another centre. The memory `split-shift-audit` records that train|test AUC is 0.983.
   - **Under centre shift, val-based choice picks the wrong configuration.** This is a real problem with medical data, not a code issue.
3. The gap between oracle and tdgs_cls (0.5–3.8pp) is the upper bound of "per-dataset tuning". Val captures only a small part of it.
   - This matches the prior estimate in Amendment 2 (quota line: held-out best-of-12 was −0.93pp).
4. **Not a final conclusion.** Tissue has the largest tdgs_cls advantage, so adding it will change the dataset mean. The gate needs 5/5 datasets.

## 5. Undone / not run

- Tissue: staging (35250), val eval (35265), vsel.
- All R3 training BA (35258).
- Table 1 on seeds 42–46 (35205–7).
- 5% R3, DermaMNIST, CAMELYON17: **not run.**
