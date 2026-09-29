# Round 3 validation report (live document, last updated 2026-09-29 11:45 HKT)

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

## 2. P4 training and gates [Verified: 130/130 + P4-e 25/25; gates read 09-29 11:30]

- P4 35293 (6×H100) COMPLETED 02:42; P4-e 35306 (tdgs_cls × 25 on H100) COMPLETED 05:08. 0 failures.
- Harvests: `results/round3/r3_harvest.json`, `r3hw_harvest.json`. Gates: `code/r3_gates.py` → `results/round3/r3_gates.{txt,json}`.
- Reference `tdgs_cls` = H100 copy (Amendment 4). The A100 copy is the sensitivity row.

### 2.1 Hardware noise (P4-e) [Verified]

Same selection, same seed, same code, `DETERMINISTIC_TRAINING=1`; only the GPU type differs.

| | n | mean (H100 − A100) | mean \|Δ\| | sd | max \|Δ\| |
|---|---|---|---|---|---|
| tdgs_cls BA, pp | 25 | +0.22 | 1.67 | **2.38** | 6.74 (blood s43) |

- **Per-cell hardware noise is as large as seed noise (paired sd 2–3pp).** It is unbiased (mean +0.22), so it inflates variance but does not shift means.
- Consequence: any cross-hardware contrast carries this extra variance. Table 1 baselines (210 A100 + 140 H100 cells) vs R1/R2 arms (A100) are affected in variance only.

### 2.2 G-R2: `tdgs_cls − hpc_cls`, n = 5 [Verified, **FAIL**]

| Reference | mean | 95% CI | positive | excl. tissue |
|---|---|---|---|---|
| tdgs_cls (H100) | +0.57 | [−1.12, +2.26] | **3/5** | **−0.15** |
| tdgs_cls (A100, sensitivity) | +0.36 | [−0.86, +1.71] | 3/5 | −0.18 |

Per dataset (H100): blood −1.78, organA −0.25, organS +1.31, path +0.11, tissue +3.48. Worst-class recall −0.62.

- Gate requires ≥ +0.5 **and** ≥4/5 **and** excl. tissue > +0.3. It fails on the last two.
- **Preregistered consequence:** TDGS must be reported as "Graph-A2 global term + class-conditional FL (Wei et al. 2015)" and **must not be written as a new criterion**. Its advantage over the matched per-class hybrid exists only on tissue.

### 2.3 G-MV stage 1: `mv_rob − tdgs_cls`, seeds 42–44 [Verified, **STOP**]

| Contrast | mean | 95% CI | positive | excl. tissue |
|---|---|---|---|---|
| **mv_rob − tdgs_cls (H100)** | **−0.02** | [−1.18, +1.60] | 2/5 | +0.25 |
| mv_rob − val-selected single view | −0.24 | [−1.45, +1.46] | 2/5 | −0.04 |
| mv_rob − tdgs_cls (A100, sensitivity) | −0.29 | [−1.60, +0.94] | 2/5 | −0.21 |

- mean ≤ 0 ⇒ **STOP** by the preregistered rule. The multi-view route (R1) closes; stage 2 (n=5, 5%) and independent validation are not triggered.
- The prediction written before training (mv_rob ≈ mv_mean within noise) was **wrong in direction**: mv_mean − mv_rob = +1.04 [−0.04, +2.30], 4/5. The "robust" max-min aggregate is worse than plain averaging.
- Descriptive only (not gated; n=3, CIs all cross 0), arm − tdgs_cls[H100]:

| Arm | mean | per dataset (blood / organA / organS / path / tissue) |
|---|---|---|
| mv_mean | +1.02 | +4.57 / +0.26 / −0.06 / −0.05 / +0.36 |
| cls_dinov2 | +0.47 | +4.05 / +0.41 / −2.19 / +0.12 / −0.07 |
| cls_clip | +0.43 | +1.90 / +1.15 / −0.73 / +1.04 / −1.19 |
| hpc_cls | −0.17 | +2.22 / +0.65 / −0.48 / +0.34 / −3.60 |
| cls_cat | −0.53 | |
| a2_dinov2 | −1.21 | |
| a2_clip | −2.03 | |

  - Blood is the largest term in every positive row, and on blood seeds 42–44 the H100 reference copy is on average 1.1pp below its A100 copy (s43: −6.74). Read with the hardware noise of §2.1 in mind. These rows are **not** claims.

## 3. Val evaluation fidelity [Verified]

`val_eval.py` re-runs inference with `final.pt`. Self-check: running the same code on the test split must reproduce the harness's `predictions_clean.npz`.

| Dataset | Label order | argmax agreement | \|ΔBA\| | max\|Δlogit\| | GPU |
|---|---|---|---|---|---|
| blood | ✓ | 1.000000 | 0 | 1.9e-2 | H100 |
| organA | ✓ | 0.999888 | 9.6e-5 | 3.4e-3 | H100 |
| organS | ✓ | 1.000000 | 0 | 6.8e-3 | H100 |
| path | ✓ | 1.000000 | 0 | 5.8e-2 | H100 |
| tissue | ✓ | 1.000000 | 0 | 0 | A100 (35265; on H100 it was 0.9977 and did not pass) |
| R3 + P4-e, 5/5 datasets (job 35354, 155 ckpts) | ✓ | 1.000000 | 0 | **0** | H100 (same GPU type they trained on) |

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

## 5. Extended vsel (Amendment 4: encoder as a per-dataset choice) [Verified, **FAIL**]

𝒞 = C5 ∪ {cls_dinov2, cls_clip, mv_rob, mv_mean}. The R3 arms exist only on seeds 42–44 (stage 2 never triggered), so this is **n = 3**, not the n = 5 Amendment 4 names. Code: `vsel_analysis.py --h100-ref`; output `results/val_eval_r3/vsel_ext_r0.02.{txt,json}`. The unchanged C5/C6 result was regression-checked bit-identical after the edit.

| Reference | vsel − tdgs_cls | positive | excl. tissue | vsel − graph_a2 | vsel − mean(𝒞) | worst recall | Gate |
|---|---|---|---|---|---|---|---|
| tdgs_cls (H100) | **+0.44** [−0.53, +1.40] | 4/5 | +0.46 | +1.92 [−0.00, +4.24], 5/5 | +0.58 | +0.65 | **FAIL** (< +0.5) |
| tdgs_cls (A100, sensitivity) | +0.19 [−1.14, +1.37] | 3/5 | +0.03 | +1.95 | +0.57 | −2.05 | FAIL |

| Dataset | val's choice (s42 / s43 / s44) | vsel − cls | oracle − cls | r(val, test) |
|---|---|---|---|---|
| blood | cls_dinov2 / tdgs_mask / mv_rob | +0.84 | +5.98 | 0.96 |
| organA | cls_clip ×3 | **+1.15** | +1.33 | 0.74 |
| organS | cls_clip / tdgs_cls ×2 | −0.56 | +0.69 | 0.56 |
| path | cls_clip / cls_dinov2 ×2 | +0.40 | +2.23 | 0.44 |
| tissue | mv_mean ×3 | +0.36 | +0.61 | 0.99 |

- The gate fails narrowly (+0.44 vs +0.5), and the sensitivity row drops to +0.19 → the result depends on which hardware copy of the reference is used. It is **not** a positive result.
- organA is the one consistent case: val picks CLIP 3/3, and it matches S0-b (UNI val kNN BA 8.6–9.1pp below CLIP/DINOv2 on organA). Single dataset, n=3: a hypothesis, not a finding.

## 6. Table 1 on author seeds 42–46 [Verified, 350/350]

- Jobs 35205/35206/35207, last cell 09-29 09:17. Harvest `results/table1/t1_s4246_harvest.json`; table `results/table1/table1_s4246.txt` (`make_table1.py`).
- **Harness identity vs archive** (overlapping seed-cells 42–44): **40/40 A100 cells bit-identical**; the 16 that differ were all trained on H100, with **identical selection sha256**. So the harness is verified; the differences are GPU type (§2.1).
- Ours vs paper over 80 cells: mean |diff| 1.93pp, median 1.60, 70/80 within 4pp. Largest: path 5% facility 5.13, tissue 5% graph_a2 5.10.
- `tdgs_cls − baseline`, BA pp, dataset mean over 5 datasets, paired by seed (n = 5):

| Baseline | 2% (pos/5; excl. tissue) | 5% (pos/5; excl. tissue) |
|---|---|---|
| Graph-A2 | +1.57 (5/5; +0.71) | +0.39 (2/5; −0.55) |
| Random | +1.47 (4/5; +0.97) | +0.89 (3/5; +0.36) |
| Herding | +1.35 (3/5; +0.60) | −0.11 (1/5; −0.92) |
| Facility | +0.96 (4/5; +0.52) | +0.05 (2/5; −0.93) |
| FPS | +5.99 (5/5) | +3.36 (3/5) |
| EVA / Forgetting / EL2N | +17 / +18 / +31 (5/5) | +10 / +12 / +21 (5/5) |

- Caveat: tdgs_cls/graph_a2 cells are A100; 140/350 baseline cells are H100. §2.1 says that adds variance, not bias.
- At 2%, tdgs_cls has a positive dataset-mean margin against every baseline, but blood is negative against Random/Herding/Facility and the excl.-tissue margins are only 0.5–1.0pp. At 5% it ties Herding/Facility.

## 7. Where this leaves the paper (by the preregistered decision tree, HANDOFF §8.8)

- **G-MV STOP and G-R2 FAIL → "neither passes" branch:** stop R1. The paper is the audit + CSC (`tdgs_cls`) at 2% with its restrictions (tissue-dominated, not a new criterion vs Wei 2015 + per-class graph, not significant at 5%) + the negative results (demand terms, per-dataset val selection, multi-encoder robust coverage).
- R3 at 5%, DermaMNIST and CAMELYON17: **not run, and not triggered.**
