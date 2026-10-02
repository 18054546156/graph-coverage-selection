# Proposal: Does the choice of selection method matter under acquisition shift? (2026-10-02)

## 0. Why the clean-BA line stops here
- The top methods are tied on clean test BA: sd across subsets is 0.82, below the training sd of 1.37. mv_mean is first on 4 of 10 rows, which is about what ties would give by chance (4.08).
- Every train-only lever has been closed: frozen proxies, trajectory matching, learner view, view gate, a 4th encoder view, and the W1 law. A better proxy did not give better BA.
- Most remaining gains are smaller than the MDE (1.2–8.8pp). Another "select for higher clean BA" variant cannot reach the 8/10 bar.
- New question: methods that tie on clean data may still differ under acquisition shift. If they do, clean BA is the wrong criterion for choosing a selection method.

## 1. Hypotheses (falsifiable)
- **H1 (rank transfer):** Kendall τ between the clean ranking and the shifted ranking of 9 methods is clearly below 1 on the EXP family. The STR family should stay high, with τ ≥ 0.85.
- **H2 (variance amplification):** under EXP, the seed spread of paired gaps between methods is larger than on clean data. The p10/CVaR ordering differs from the mean-BA ordering.
- **H3 (consequence):** picking the method by clean BA has positive regret under shift. Regret here means BA lost compared with the best method under shift.

Evidence so far, all weak: seeds 0–2 only, blood and path only. τ(clean→EXP) is 0.64–0.79 versus 0.71–0.93 for STR. The #1 method changes in 3 of 4 cells. The earlier "graph_a2 loses to herding on blood" result was driven by a single seed (−18.1) and is retracted; see memory `medmnistc-exposure-reversal-is-one-seed`.

## 2. Shift proxy: MedMNIST-C
- **EXP family** (brightness±, contrast±, saturate) stands in for staining, scanner and exposure differences. **STR family** (blur, pixelate, jpeg, bubble, stain_deposit) covers sensor and compression artefacts, as a control.
- Literature anchor: Tellez et al. 2019 (MedIA, arXiv 1902.06543) use brightness, contrast and HSV perturbation to simulate multi-centre H&E stain variation.
- Caveat: BloodMNIST is Giemsa-stained, so on blood this is an analogy, not established.
- **Proxy check (S0, CPU only):** compare MedMNIST-C EXP severities 1–3 against the real colour differences between centres:
  - path: the train/test split is already from a different centre (AUC .99). Compare the per-channel and HSV mean/sd gap between train and test with the gap that each severity introduces.
  - blood: compare against Acevedo-20 vs Matek-19 colour statistics. This only works if the images can be obtained, because TCIA is blocked from all our networks. Otherwise path only.
  - If the real gap falls within severities 1–3, the proxy is quantitatively matched. If not, the argument can only be qualitative, and the paper must say so.

## 3. Metrics (existing framework, `table1-robustness-metric-framework`)
- Problem metrics: τ(clean→shift) and clean-pick regret.
- Main: BA and rBE reported separately for EXP and STR, severities 1–3 only (to avoid the floor effect); worst-class recall.
- Risk: p10/CVaR across seeds. Appendix: ECE under shift.

## 4. Methods
- The 8 benchmark methods plus mv_mean.
- Later stages:
  - training-side colour augmentation (HSV jitter; HED on path) as a confound check. If augmentation removes the gaps, the question is moot.
  - DRO coreset under covariate shift (arXiv 2501.14253) as the selection-side competitor.

## 5. Stages and gates
| Stage | Content | Cost | Gate |
|---|---|---|---|
| S0 | Colour-gap proxy check from §2 | CPU | Sets claim wording only; no stop |
| **S1** | MedMNIST-C inference on existing seed 42–46 checkpoints: 9 methods × blood and path × 2 ratios × 11 corruptions × 5 severities | GPU, inference only | H1: τ_EXP < τ_STR in ≥3/4 cells, and the #1 method changes in ≥2/4 cells |
| S1b | Same inference on organA, organS and tissue | GPU, inference only | Same gate pattern on ≥3/5 datasets |
| S2 | Training with colour augmentation, seeds 42–46, then rerun S1 | GPU training (needs separate approval) | Gaps persist after augmentation, otherwise stop |
| S3 | Add DRO coreset; decide whether a robust-selection method is worth designing | — | Only if S2 passes |

S1 precondition: confirm that final.pt files exist for mv_mean and the 8 methods on seeds 42–46. Any missing checkpoint is a training job, which needs approval and is outside S1.

## 6. Matek-19 (real cross-centre data): parked
- TCIA cannot be reached from the local machine or any of the three luhpc accounts.
- Class sizes from Matek 2019 Table 1: basophil N=79 (precision .48±.16), erythroblast 78, no platelet class.
- Only 4 classes have more than 400 samples and are shared with BloodMNIST: neutrophil, lymphocyte, monocyte, eosinophil. Matek-19 is future work, not the main evidence.
