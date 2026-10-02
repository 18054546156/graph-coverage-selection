# S1 findings (2026-10-02, seeds 42-46, 9 methods, blood+path x 2%/5%, all A100 inference)

Full battery: `battery_s1.md`. Data: `s1_corr.json` (9900 rows), `s1_clean.json` (180 rows).

## Gate
H1 PASS: tau_EXP < tau_STR 4/4 cells; #1 method changes clean->EXP 4/4 cells.
Per seed: mean tau_EXP .56 vs tau_STR .81; #1 flips 16/20; clean-pick regret mean 3.4pp, >0.5pp in 16/20.

## What it does and does not mean
1. Noise floor. Among the 6 competitive methods (mv_mean, graph_a2, herding, facility, fps, random),
   tau between the CLEAN rankings of two different seeds is only .20-.49. tau(clean, EXP) within the same
   seed is .15-.52, i.e. EXP reshuffles the ranking about as much as retraining with a new seed.
   STR keeps it (.47-.92). So "clean BA does not predict EXP BA" holds, but partly because clean BA
   does not separate these methods to begin with (known ties).
2. Systematic method x shift interactions (gap to random, EXP minus clean, n=5, uncorrected):
   | | blood 2% | path 2% | blood 5% | path 5% |
   |---|---|---|---|---|
   | mv_mean | -6.3 (p .06) | -2.8 (p .04) | +5.0 (p .06) | +4.2 (p .07) |
   | herding | -6.2 (p .04) | -5.4 (p .01) | +4.5 (p .13) | -0.6 |
   | fps     | -0.3 | +6.4 (p .009) | 0.0 | +3.4 (p .14) |
   Sign pattern is consistent across both datasets: at 2% prototype/coherence selectors (mv_mean, herding)
   lose EXP robustness vs random; at 5% mv_mean gains it. Post hoc, ~4/20 tests at p<.05 (1 expected).
3. mv_mean at 5% is the most EXP-robust method: vs random blood +6.8pp (p .044), path +3.3pp (p .021);
   vs clean winner facility on path +3.0pp (t 11.9). At 2% it is not (blood -6.2pp vs random).
4. Magnitude caveat: EXP severity 1 already costs 15-30pp BA and worst-class recall falls to 5-17% for
   every method. Whether this is a realistic shift size is the S0 question; with no colour augmentation in
   training, S2 may remove most of it.

## Next
S0 (CPU colour-gap check on path), S1b (organA/organS/tissue inference, tests whether the 2%-vs-5% sign
pattern replicates out of sample), S2 colour-aug retraining (needs approval).
