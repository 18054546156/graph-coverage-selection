# Round 3 validation report (live document, last updated 2026-09-28 22:45 HKT)

Status labels used throughout: **[Verified]** = actually run, with a traceable output. **[Interim]** = incomplete (datasets or seeds missing), no conclusion drawn. **[Not run]** = planned only.

## 1. Selection space S0: zero training [Verified, 5/5 datasets]

| Dataset | val kNN BA uni / dinov2 / clip | J(tdgs_cls, cls_dinov2 / clip) | J(tdgs_cls, hpc_cls) | Relative coverage of tdgs_cls in dinov2 / clip | mv_mean min_v F̂_v | mv_rob (A1) min_v F̂_v, c | J(mv_rob, mv_mean) |
|---|---|---|---|---|---|---|---|
| blood | 94.7 / 90.2 / 86.2 | 0.015 / 0.015 | 0.30 | 0.85 / 0.83 | 0.947 | 0.948, 0.95 | 0.46 |
| organA | 78.7 / 87.3 / 87.8 | 0.010 / 0.017 | 0.24 | 0.84 / 0.85 | 0.942 | 0.945, 0.95 | 0.56 |
| organS | 73.4 / 72.5 / 69.8 | 0.020 / 0.009 | 0.18 | 0.82 / 0.83 | 0.940 | 0.941, 0.94 | 0.52 |
| path | 99.6 / 96.7 / 96.9 | 0.011 / 0.014 | 0.73 | 0.87 / 0.85 | 0.955 | ≈0.955, 0.96 | 0.39 |
| tissue | 39.0 / 40.5 / 38.3 | 0.027 / 0.023 | 0.10 | 0.81 / 0.81 | 0.936 | ≈0.936, 0.94 | 0.34 |

- **Identity checks** (generic greedy == `greedy_blended` order == the round 1/2 staged sets) pass on all 5 datasets. Quota assertions pass.
- **S0-a passes.**
- **S0-b:** on organA, UNI's val kNN BA is 8.6–9.1pp below the other two encoders. On blood / path, UNI leads by 2.7–4.5pp. On organS the three are within 3.6pp.
  - So no single encoder is best everywhere at the zero-training level.
  - **This is not BA.** Memory `uni-space-surrogates-are-structurally-blind` shows that surrogates in embedding space have repeatedly failed to predict subset effects.
- **mv_rob (after Amendment 1)** differs from mv_mean in the selected set (J 0.39–0.56), but its objective value is only 0.001–0.004 higher.
  - Prediction written before training: the BA difference between the two will be within noise.

## 2. P4 training [Running: job 35293 (6×H100), 32/130 at 22:39; no BA has been read]

- 35258 was resubmitted as 35293 (Amendment 4). The hardware control P4-e (tdgs_cls × 25 on H100) is job 35306 and runs after 35293.
- Estimated completion: P4 about 09-29 02:00–03:00; P4-e about 04:30.
- Gates G-R2 and G-MV are still open.

## 3. Val evaluation fidelity [Verified]

`val_eval.py` re-runs inference with `final.pt`. Self-check: running the same code on the test split must reproduce the harness's `predictions_clean.npz`.

| Dataset | Label order | argmax agreement | \|ΔBA\| | max\|Δlogit\| | GPU |
|---|---|---|---|---|---|
| blood | ✓ | 1.000000 | 0 | 1.9e-2 | H100 |
| organA | ✓ | 0.999888 | 9.6e-5 | 3.4e-3 | H100 |
| organS | ✓ | 1.000000 | 0 | 6.8e-3 | H100 |
| path | ✓ | 1.000000 | 0 | 5.8e-2 | H100 |
| tissue | ✓ | 1.000000 | 0 | 0 | A100 (35265; on H100 it was 0.9977 and did not pass) |

Conclusion: the evaluation code is correct. Differences across GPUs are TF32-level, and BA differs by < 0.01pp.

## 4. vsel: per-dataset configuration chosen on val [Verified, 5/5 datasets, preregistered gate: **not passed**]

Leave-one-seed-out. Test BA in pp, dataset mean, with hierarchical bootstrap 95% CI. Raw output: `results/val_eval_vsel_r0.02_5ds.json`.

| Candidate set | vsel − tdgs_cls | vsel − graph_a2 | vsel − mean(𝒞) | worst recall vs tdgs_cls | Gate |
|---|---|---|---|---|---|
| C5, n = 5 | **−0.20** [−1.16, +0.73], 2/5 positive; excluding tissue −0.06 | +1.37 [+0.07, +2.99], 4/5 | +0.51 [−0.37, +1.43] | −1.21 | **FAIL** |
| C6 (+lam1), n = 3 | −0.12 [−1.50, +1.70], 1/5 | +1.64, 2/5 | +0.76 | −1.12 | **FAIL** |

Per dataset (C5):

| Dataset | val's choice | vsel − cls | oracle − cls | r(val, test) |
|---|---|---|---|---|
| blood | tdgs_mask 4/5 | **+0.84** | +3.78 | 0.92 |
| organA | a2_perclass 5/5 | +0.03 | +0.51 | 0.36 |
| organS | tdgs_cls 5/5 | 0 | +0.65 | 0.50 |
| path | graph_a2 5/5 | **−1.11** | +1.80 | 0.59 |
| tissue | tdgs_cls 4/5, mask 1/5 | −0.76 | +0.14 | 0.99 |

**Reading (verified):**

1. **Choosing a configuration per dataset on val does not beat the fixed `tdgs_cls`.** This is a preregistered negative result.
   - The per-dataset "best configuration" (oracle) is 0.1–3.8pp above tdgs_cls.
   - That headroom is only visible on test. Val can reliably recover it only on blood.
2. **path is the medical-data explanation.**
   - The path test set comes from another centre (CRC-VAL-HE-7K). val and train are from the same source.
   - val consistently picks graph_a2 (5/5), but it loses 1.11pp on test.
   - Under centre shift, a same-source val cannot select the configuration.
3. **The tissue loss comes from one seed.**
   - r(val, test) = 0.99, so val ranks correctly there.
   - But the val gap between cls and mask is below seed noise, so a single mis-pick costs 3.8pp.
4. vsel − graph_a2 is +1.37, but that is because vsel mostly picks cls/mask. It is not an independent benefit of "choosing per dataset".
5. **Consequence:** do not open a new hyperparameter grid.
   - A larger grid raises the oracle, but with the current val/test agreement the part that val can recover is ≈ 0.
   - The one exception is "which encoder" (Amendment 4), which is tested once, with the same gate, after P4.

## 5. Not done / not run

- R3 training BA (35293), the hardware control (35306), G-R2 / G-MV, and extended vsel.
- Table 1 on seeds 42–46: 110/350 at 22:36. Note: staging for blood/organS was **missed** because `sbatch --export` split on the comma. It was found and re-staged at 21:10 (35275/35276), and staging is now 350/350.
- R3 at 5%, DermaMNIST, CAMELYON17: **not run.**
