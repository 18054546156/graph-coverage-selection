# Round 3 pre-registration amendments (appended; the original `r3_prereg_20260928.md` sha256 743c4f70… is unchanged)

## Amendment 1: mv_rob truncation grid (2026-09-28 ~18:40 HKT, **no R3 training has been run and no BA exists at this point**)

**What was found.** In staging 35236 (blood / organA / organS), all 7 caps of mv_rob produced exactly the same selection, and J(mv_rob, mv_mean) = 1:

| Dataset | F_uni | F_dinov2 | F_clip |
|---|---|---|---|
| blood | 0.718 | 0.681 | 0.679 |
| organA | 0.691 | 0.657 | 0.695 |
| organS | 0.645 | 0.628 | 0.645 |

At a 2% budget the absolute coverage F_v(S) of every view is only 0.63–0.73, so the preregistered grid c ∈ [0.80, 1.0] **never truncates**. Σ_v min(F_v, c) then degenerates to Σ_v F_v, which gives the same argmax as mv_mean.

The preregistered mv_rob was therefore not the "max-min" it claimed to be. This is a design defect. It was found only from train-pool objective values, without looking at any val or test quantity.

**Change.**

- Normalize each view by its own single-view optimum:
  - F̂_v(S) = F_v(S) / F_v(S_v), where S_v is the single-view greedy selection cls_<v> (cls_uni = tdgs_cls).
  - F̂_v = 1 means "as good as selecting with that view alone".
- Grid c ∈ {0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0}. Keep the S with the largest min_v F̂_v.
- Still train-pool only. Code: `mv_select.py` (cluster md5 281f42f7…); jobs 35249 (blood / organA / organS rerun, report suffix `_mvrob_a1`) and 35250 (tissue, full).
- The pre-amendment files are kept as `*_mv_rob_v0_s42.npy`. They are identical to mv_mean and will not be trained.

**Prediction written down before the rerun.** Relative coverage of mv_mean is already 0.94–0.95 on every view (blood 0.953 / 0.947 / 0.948; organA 0.951 / 0.942 / 0.954; organS 0.947 / 0.940 / 0.940), so the room for mv_rob to raise min_v is estimated at ≤ 0.01.

**Added decision rule, applied after staging and before training:**

- If J(mv_rob, mv_mean) ≥ 0.8 on ≥ 4 of the 5 datasets, the two are treated as selection-equivalent. Only **mv_rob** is trained (it is the preregistered primary arm), and mv_mean is reported as "equivalent in selection space".
- Otherwise both are trained, as planned in P4-b.

## Recorded S0 results (3/5 datasets, 2026-09-28 18:05; path running, tissue queued)

| Dataset | val kNN BA uni / dinov2 / clip | Best view | J(tdgs_cls, cls_dinov2) | J(tdgs_cls, cls_clip) | xshare uni / dinov2 / clip |
|---|---|---|---|---|---|
| blood | **.947** / .902 / .862 | uni (+4.4pp) | 0.015 | 0.015 | .27 / .59 / .55 |
| organA | .787 / .873 / **.878** | clip (uni is 9.1pp lower) | 0.010 | 0.017 | .42 / .47 / .40 |
| organS | **.734** / .725 / .698 | uni (+0.9pp) | 0.020 | 0.009 | .54 / .63 / .61 |

- **Identity checks:** on all 3 datasets, graph_a2 and tdgs_cls reproduce both the greedy_blended order and the staged sets exactly.
- **S0-a (non-redundancy):** J < 0.03 on all 3 datasets, so it has already passed at 3/5.
  - Note: this is a *necessary* condition only. The flatness result (J = 0.049, yet BA is equivalent) shows that low J does not imply different BA.
- **S0-b (heterogeneity):** the view rankings differ across datasets. On organA, UNI is 9.1pp *below* the other two, while on blood UNI is 4.4pp above.
  - At the zero-training level, H_enc (no single encoder is best everywhere) **is not falsified**. The final verdict waits for path and tissue.
  - This is a train→val kNN measurement, not a result of the trained model.
- **Cross-view coverage:** a single-view selection covers the other views poorly. tdgs_cls reaches only 0.55–0.62 (absolute F) in the dinov2 / clip views, versus 0.67–0.73 for their own selections, which is 0.80–0.86 in relative terms.
  - The multi-view arms keep every view at ≥ 0.94 of its single-view optimum.
  - This is a statement in objective space. **It says nothing about BA.**

## Amendment 2: per-dataset configuration (added 2026-09-28, in response to user feedback; **not executed at the time of writing**)

**Motivation.** The five datasets differ in modality, cross-class share and best encoder, so a single fixed configuration is not necessarily right. Per-dataset configurations are allowed **only if chosen by val**, never by test.

**Existing assets.** Every trained cell saved `final.pt` (1000-epoch final weights). Val BA can therefore be obtained **without retraining**, by running inference on the val split.

The script `val_eval.py` is to be written. It must use exactly the same model construction and eval preprocessing as the test evaluation in the harness, and it must first pass a check: running it on the test split must reproduce `predictions_clean.npz` bit for bit.

**Arm `vsel` (val-selected configuration).**

- Candidate set 𝒞 (fixed now):
  - {graph_a2, a2_perclass, tdgs_mask, tdgs_cls} (2%, already trained on seeds 42–46)
  - plus, once R3 is trained, {cls_dinov2, cls_clip, mv_rob}
- For each dataset d and each held-out seed s:
  - c*(d, s) = argmax_{c∈𝒞} mean_{s'≠s} valBA(d, c, s')
  - Report testBA(d, c*(d, s), s).
- **Leave-one-seed-out**: this removes the "same seed trains well on both val and test" correlation.
- The comparators are fixed tdgs_cls and graph_a2, with paired seeds. The primary endpoint is still test BA.

**Selection-bias control.**

- Also report the "oracle" choice, c picked by test BA (used only as an upper bound), and the random choice (mean over 𝒞).
- vsel − mean(𝒞) is the true gain of "choosing by val".
- Prior evidence: in the quota line, the held-out result for best-of-12 was −0.93pp, against +3.15pp of selection bias. The prior probability of a positive gain is therefore **not high**.

**Gate.** vsel − tdgs_cls ≥ +0.5pp, with ≥ 3/5 datasets the same sign, and vsel − mean(𝒞) > 0 ⇒ report "per-dataset val configuration" as an effective strategy. Otherwise report it as a negative result.

**Metrics.** No new primary metric is added. BA stays primary and worst-class recall stays secondary.

- The reason is from memory `metric-battery-closed`: of 182 tests, 0 survive BH correction.
- Adding metrics only enlarges the multiple-comparison space.
- The val side adds val BA only as a *selection* quantity, not as an endpoint.

## Amendment 3 (2026-09-28 ~18:35 HKT; **no val BA has been read at this point**, and no R3 training has finished)

1. **Candidate set 𝒞 of Amendment 2 is expanded** to include `tdgs_lam1` (λ = 1, n = 3 on seeds 42–44).
   - 𝒞 now spans λ ∈ {0 (graph_a2), 0.5 (tdgs_cls), 1 (tdgs_lam1)} × scope {global, per-class (a2_perclass), mask (tdgs_mask)}.
   - Per-dataset hyperparameters are therefore chosen **by val** over this trained grid.
   - Because tdgs_lam1 only has n = 3, the leave-one-seed-out analysis runs on seeds 42–44 when tdgs_lam1 is in the set. A secondary analysis without tdgs_lam1 uses n = 5.
2. **mv_rob after Amendment 1 (4/5 datasets, before training):**

   | Dataset | J(mv_rob, mv_mean) | Chosen c |
   |---|---|---|
   | blood | 0.46 | 0.95 |
   | organA | 0.56 | 0.95 |
   | organS | 0.52 | 0.94 |
   | path | pending (35256) | — |

   - The equivalence threshold (≥ 0.8 on ≥ 4/5 datasets) is not met, so **both are trained**, as P4-b.
   - The measured gain in min_v F̂_v over mv_mean is only +0.001 to +0.004, matching the ≤ 0.01 predicted in Amendment 1.
   - mv_rob and mv_mean therefore differ **in their selection sets** (J ≈ 0.5) but are nearly identical **in objective value**. This is the "flat good region" phenomenon again.
3. **P4 training was submitted** as job 35258 into the independent tree `runs/r3_20260928`, with worklist `work/r3.txt` (130 cells), SEL_WAIT = 4 h (waiting for the tissue / path staging to finish).
   - The stale path `mv_rob` (v0) file was removed from the selection directory, so training cannot read the pre-amendment version.
4. **S0 at 4/5 datasets (tissue is still staging):**
   - S0-a: J(tdgs_cls, cls_dinov2 / cls_clip) is 0.009–0.020 on all 4 datasets, so it **passes**.
   - S0-b (val kNN BA, uni / dinov2 / clip):

     | Dataset | uni | dinov2 | clip |
     |---|---|---|---|
     | blood | 94.7 | 90.2 | 86.2 |
     | organA | 78.7 | 87.3 | 87.8 |
     | organS | 73.4 | 72.5 | 69.8 |
     | path | 99.6 | 96.7 | 96.9 |

     There is no single view that is best everywhere and leads by ≥ 2pp, so H_enc is **not falsified**. This is a zero-training kNN measurement.
