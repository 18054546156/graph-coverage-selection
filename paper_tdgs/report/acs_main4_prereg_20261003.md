# ACS on the 4 main Table-1 datasets: rule fixed in advance (2026-10-03; stamp on the cluster before any staging or training)

Parent docs: `report/proposal_acs_20261002.md`, `report/acs_e2_prereg_20261002.md` (E2 picked b*=10, q*=0.25 on the 10 high-ambiguity rows).
User approval: 10-03, "1 跑原本 内容：blood/organA/organS/path × 2%/5% × 种子 42–46，A100".
Comparator changed by the user before staging (10-03, "没让你和graph 比 要和sota比"): the headline is ACS vs the per-row SOTA, not vs Graph-A2.

## 1. Question

The E2 rule was chosen on retina/breast/derma/OCT/tissue only. The 4 remaining Table-1 datasets (blood, organA, organS, path)
were **not used** to choose it. This run fills the ACS column of Table 1 for those 8 rows with the frozen rule, and compares it
to the best published method in each row. Nothing is tuned here.

## 2. Arm, rows, seeds, hardware

- Rule b*=10, q*=0.25. bpc: blood 29/74, organS 25/63, organA 62/157, path 199/499, all ≥ 2b* = 20, so τ=0 in every row.
  ACS arm = `acs_q25_t000` (per-class 25% agreement-quantile filter, then the tdgs_cls greedy; no herding picks).
- 4 datasets × {0.02, 0.05} × seeds 42–46 = **40 cells**, A100 only (same GPU type as the bench they are paired with).
- Selection seed 42 (as every staged selector). Protocol as Table 1 / E2: 1000 epochs, 224px, no augmentation,
  DETERMINISTIC_TRAINING=1, final_epoch, `tune_pack.slurm`, `--exclude=hpcgpu108`.
- Staging: `acs_select.py` unchanged (sha256 bb170092…). In `acs_stage.slurm` only the `case` label of the tissue branch is
  extended to `tissuemnist|bloodmnist|organamnist|organsmnist|pathmnist` (same author UNI cache `t1cache/embeddings_img224_smokefull`,
  same bench herding ref `runs/table1_s4246_20260928/sel/seed_42/herding`, same staged refs `sel/<ds>_r<R>_tdgs_cls_s42.npy`).
  CPU FAISS, no `GPUK`, as tissue. acs_select writes all 7 grid arms; only `acs_q25_t000` is linked into `acs/td_sel` and trained.
- **Missing input, written down before staging:** I3 needs `knnf/td_sel/<ds>_knnf_report.json`, which exists for tissue but not
  for the 4 main datasets. It is produced by `knnf_select.py` **unchanged** (sha256 899d47db…), `--seeds 42`, on the same author
  UNI cache and herding ref, exactly as `knnf_stage.slurm` did for tissue. Its own identity check (author herding on the full pool
  == staged bench herding) must pass. Its by-products (`knnf_herding`, `knnf_random42` sets for these datasets) are not trained.
- Identity checks I1–I4 must pass, else the row is not trained and that is reported.

## 3. Analysis (fixed now)

Opponent pool per row = every **published** method with all 5 paired A100 seeds (s42–46) in that row:
the 8 Table-1 methods (graph_a2, herding, facility, random, fps, eva, el2n_top, forgetting) + TypiClust (ICML'22),
ProbCover (NeurIPS'22), MaxHerding (ECCV'24). Our own variants (cls, mv_mean, knnf_*) are not opponents.

Per row, paired over s42–46:
1. **Primary:** ACS − SOTA, SOTA = pool member with the highest seed-mean BA in that row (as `code/acs_vs_table1.py`);
   two-sided paired t, raw and BH over the 8 rows. Same for worst-class recall (SOTA re-chosen on worst recall).
2. ACS rank among (pool + ACS) by seed-mean BA.
3. Secondary: ACS − Graph-A2.

Prediction written in advance (proposal H-main): these rows have low label ambiguity, so ACS ≈ SOTA (|Δ| < 1pp in most rows;
no row significantly behind after BH). A tie here is the predicted outcome, not a failure. The claim this run can support:
"ACS does not lose to the per-row SOTA on the low-ambiguity rows." A significant loss in any row (BH) is reported as such.
No seed is dropped; no row is excluded after seeing results.

## 4. Stamp

(appended on the cluster in `acs/main4_stamp.txt`: sha256 of this file, `acs_stage.slurm` as edited, `acs_select.py`,
`knnf_select.py`, UTC time, before staging)

## 5. Results (appended 2026-10-03 after harvest; nothing above this line changed)

40/40 cells, 0 failures, all A100 (jobs 36315/36316). Harvest `code/acs_main4_eval.py --cluster` → `results/acs/acs_main4_cells.json`;
analysis `code/acs_main4_eval.py` → `results/acs/acs_main4_test.txt`, `acs_main4_analysis.json`. Opponent pool = all 11 published methods in every row.

| row | ACS | SOTA | SOTA BA | ACS − SOTA | p | q (BH) | rank /12 |
|---|---|---|---|---|---|---|---|
| blood 2% | 86.77 | Random | 84.68 | +2.09 | .232 | .464 | 1 |
| blood 5% | 91.72 | Herding | 93.70 | −1.98 | .012 | .073 | 4 |
| organA 2% | 87.70 | Herding | 87.78 | −0.08 | .680 | .806 | 2 |
| organA 5% | 91.91 | Facility | 92.33 | −0.42 | .018 | .073 | 3 |
| organS 2% | 62.08 | Graph-A2 | 63.41 | −1.33 | .219 | .464 | 3 |
| organS 5% | 68.01 | Facility | 68.53 | −0.52 | .449 | .718 | 2 |
| path 2% | 81.15 | Herding | 82.08 | −0.93 | .705 | .806 | 5 |
| path 5% | 87.46 | Facility | 87.64 | −0.18 | .854 | .854 | 3 |

- Prediction (§3): "|Δ| < 1pp in most rows; no row significantly behind after BH" → **holds**: |Δ| < 1pp in 5/8, 0/8 significant after BH.
  Raw p < .05 behind in 2 rows (blood 5% −1.98, organA 5% −0.42, both 0/5 seeds ahead); they do not survive BH, and are reported.
- ACS first 1/8 (blood 2%), mean rank 2.88/12, mean ACS − SOTA −0.42pp. Worst-class recall: 0/8 significantly behind after BH, mean −0.85pp.
- Secondary ACS − Graph-A2: mean +0.61pp; significant only in blood 2% (+5.47, p .013).
- Supported claim: "ACS does not lose to the per-row SOTA on the low-ambiguity rows." Not supported: "ACS beats SOTA on these rows."
- Exploratory, not preregistered: ACS − cls (= F2, since τ=0 here) mean +0.53pp BA, 5/8 rows positive; blood 2% +4.43 (p .012), blood 5% −1.69 (p .040).
