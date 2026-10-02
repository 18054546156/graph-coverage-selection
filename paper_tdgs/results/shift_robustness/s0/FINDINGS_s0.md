# S0 findings: is MedMNIST-C EXP a realistic-size shift? (2026-10-02, CPU job 36219)

Per-image colour statistics (R, G, B means; HSV sat/val; circular hue cos/sin; luminance sd), 3000 images
per set at 224px. Gap = L2 norm of standardized mean differences (scale = train-set sd per feature).
Data: `s0_colour_gap.json`, log `s0colour_36219.log`. Code: `corr/s0_colour_gap.py`.

| | floor train->val | real train->test | EXP sev 1 (clean test -> corrupted) |
|---|---|---|---|
| path  | 0.049 | **0.644** (hue cos/sin -0.50/-0.35; little else) | contrast_up 0.41, saturate 0.60, contrast_down 0.90, brightness_up 1.23, brightness_down 2.19 |
| blood | 0.031 | **0.035** (= floor; same centre) | contrast_up 0.76, contrast_down 0.91, saturate 1.74, brightness_up 3.04, brightness_down 3.55 |

Severity 2-3 gaps are 1.3-10x larger again (blood brightness_down s3 = 10.5).

## Conclusions
1. Path: the real cross-centre shift is ~13x the sampling floor and is almost purely a HUE shift.
   MedMNIST-C EXP matches its MAGNITUDE only at contrast_up s1-2, saturate s1, contrast_down s1, and
   matches its DIRECTION nowhere (EXP barely moves hue). Sev 1-3 averaging overshoots.
2. Blood: there is no real colour shift between train and test. EXP on blood is purely synthetic,
   20-100x anything observed. Blood conclusions must be worded as "under synthetic exposure shift".
3. Claim wording (per PREREG): EXP results = "synthetic exposure shift, mostly larger than the observed
   cross-centre shift"; path's clean test (already cross-centre) is the only realistic-shift evidence.

## Matched-magnitude re-analysis of S1 (path only; subset fixed by S0 rule gap <= 1.5 x 0.644)
Subset: contrast_up s1/s2, saturate s1, contrast_down s1.
- path 2%: tau(clean, matched) .50, #1 mv_mean -> facility, regret 2.9pp; I(mv_mean) +1.1 (t 1.0),
  I(herding) -3.6 (t -1.9), I(fps) +8.4 (t 18.7).
- path 5%: tau .56, #1 facility -> mv_mean, regret 0.5pp; all |I| < 3pp, |t| < 1.4.
- BA still drops 8-24pp under these realistic-size perturbations (models trained without colour aug).
=> Rank reshuffling survives at realistic magnitude at 2%; the mv_mean "negative at 2%" sign does NOT
   (it came from large severities). Herding's 2% sign survives weakly.
