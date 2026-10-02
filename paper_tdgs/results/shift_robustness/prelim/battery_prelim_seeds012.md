# PRELIM shift battery (seeds 0-2, 8 methods, no mv_mean) -- exploratory only

Source: `paper_tdgs/results/shift_robustness/prelim/precheck_all_corr.json` + `paper_tdgs/results/shift_robustness/prelim/precheck_all_clean.json`
EXP = brightness_down, brightness_up, contrast_down, contrast_up, saturate; STR = bubble, defocus_blur, jpeg_compression, motion_blur, pixelate, stain_deposit.
Main metrics average severities 1-3 (4-5 floor out). Only seeds with the full 11x5 grid for every method in a cell are used, so all comparisons are paired.

| dataset | ratio | methods | seeds used |
|---|---|---|---|
| bloodmnist | 0.02 | 8 | [0, 1, 2] |
| bloodmnist | 0.05 | 8 | [0, 1, 2] |
| pathmnist | 0.02 | 8 | [0, 1, 2] |
| pathmnist | 0.05 | 8 | [0, 1, 2] |

## 1. Diagnostics (is there a problem?)

Rankings use the seed-mean BA per method. tau = Kendall tau(clean ranking, shifted ranking). Regret = BA(best under shift) - BA(clean winner) under shift.

| dataset | ratio | tau_EXP | tau_STR | EXP<STR | #1 clean | #1 EXP | #1 STR | regret EXP (pp) | regret STR (pp) |
|---|---|---|---|---|---|---|---|---|---|
| bloodmnist | 0.02 | +0.714 | +0.786 | yes | graph_a2 | herding | graph_a2 | +4.31 | +0.00 |
| bloodmnist | 0.05 | +0.786 | +0.714 | no | herding | graph_a2 | herding | +5.38 | +0.00 |
| pathmnist | 0.02 | +0.786 | +0.929 | yes | graph_a2 | graph_a2 | graph_a2 | +0.00 | +0.00 |
| pathmnist | 0.05 | +0.643 | +0.857 | yes | graph_a2 | random | random | +4.20 | +1.87 |

**H1 gate (proposal S1):** tau_EXP < tau_STR in 3/4 cells (need >=3/4); #1 changes clean->EXP in 3/4 cells (need >=2/4) -> **PASS**

Per-seed (rankings within one seed):

| dataset | ratio | seed | tau_EXP | tau_STR | #1 clean | #1 EXP | regret EXP (pp) |
|---|---|---|---|---|---|---|---|
| bloodmnist | 0.02 | 0 | +0.786 | +0.571 | graph_a2 | graph_a2 | +0.00 |
| bloodmnist | 0.02 | 1 | +0.571 | +0.714 | graph_a2 | graph_a2 | +0.00 |
| bloodmnist | 0.02 | 2 | +0.714 | +1.000 | herding | herding | +0.00 |
| bloodmnist | 0.05 | 0 | +0.500 | +0.714 | herding | graph_a2 | +8.97 |
| bloodmnist | 0.05 | 1 | +0.714 | +0.643 | herding | graph_a2 | +4.83 |
| bloodmnist | 0.05 | 2 | +0.643 | +0.429 | graph_a2 | fps | +2.71 |
| pathmnist | 0.02 | 0 | +0.571 | +0.929 | herding | graph_a2 | +5.73 |
| pathmnist | 0.02 | 1 | +0.929 | +0.929 | graph_a2 | graph_a2 | +0.00 |
| pathmnist | 0.02 | 2 | +0.857 | +0.929 | herding | graph_a2 | +0.33 |
| pathmnist | 0.05 | 0 | +0.286 | +0.714 | herding | fps | +6.50 |
| pathmnist | 0.05 | 1 | +0.786 | +0.929 | graph_a2 | facility | +1.18 |
| pathmnist | 0.05 | 2 | +0.857 | +0.786 | random | random | +0.00 |

Per-seed summary: mean tau_EXP +0.685, mean tau_STR +0.774, EXP<STR 7/12, #1 flips 7/12, mean regret +2.52 pp, regret>0.5pp 6/12.

## 2. Main: BA by family x severity (mean +- sd over seeds, %)

BE = 1 - BA. rBE not computable (no AlexNet reference checkpoint).

### bloodmnist r=0.02

| method | clean | EXP s1 | EXP s2 | EXP s3 | STR s1 | STR s2 | STR s3 | EXP s1-3 BE | STR s1-3 BE | clean_ba_drop EXP |
|---|---|---|---|---|---|---|---|---|---|---|
| graph_a2 | 84.8 | 63.3+-5.5 | 39.4+-3.7 | 27.2+-2.4 | 82.6+-2.1 | 79.3+-1.7 | 76.4+-1.8 | 56.7 | 20.6 | +41.45 |
| herding | 84.3 | 68.8+-4.7 | 45.1+-5.7 | 28.9+-4.3 | 79.5+-4.4 | 77.0+-4.8 | 74.9+-4.7 | 52.4 | 22.9 | +36.68 |
| facility | 81.7 | 58.3+-2.8 | 37.7+-3.0 | 25.2+-4.7 | 80.0+-1.5 | 77.8+-1.9 | 75.5+-2.1 | 59.6 | 22.2 | +41.24 |
| fps | 74.5 | 49.3+-4.8 | 30.4+-4.1 | 22.8+-1.9 | 72.1+-1.7 | 70.4+-2.2 | 68.7+-2.4 | 65.8 | 29.6 | +40.39 |
| random | 82.2 | 55.3+-9.2 | 36.8+-8.0 | 27.7+-7.2 | 77.4+-4.2 | 75.4+-4.3 | 73.2+-4.4 | 60.0 | 24.7 | +42.20 |
| eva | 74.0 | 51.4+-4.0 | 35.1+-0.7 | 26.1+-1.1 | 70.5+-1.6 | 67.6+-1.7 | 65.2+-2.5 | 62.5 | 32.2 | +36.46 |
| el2n_top | 58.7 | 41.0+-4.6 | 25.9+-4.7 | 18.6+-3.0 | 57.2+-0.7 | 54.8+-1.5 | 52.4+-2.1 | 71.5 | 45.2 | +30.17 |
| forgetting | 63.9 | 46.5+-4.6 | 32.0+-4.4 | 24.6+-2.9 | 58.0+-4.2 | 53.8+-5.2 | 50.0+-5.8 | 65.6 | 46.1 | +29.53 |

### bloodmnist r=0.05

| method | clean | EXP s1 | EXP s2 | EXP s3 | STR s1 | STR s2 | STR s3 | EXP s1-3 BE | STR s1-3 BE | clean_ba_drop EXP |
|---|---|---|---|---|---|---|---|---|---|---|
| graph_a2 | 93.0 | 77.7+-3.9 | 51.9+-2.9 | 34.1+-4.0 | 86.8+-1.4 | 81.9+-2.2 | 78.5+-2.6 | 45.4 | 17.6 | +38.39 |
| herding | 94.2 | 74.3+-4.8 | 43.9+-3.0 | 29.4+-1.1 | 91.1+-0.6 | 86.6+-1.6 | 83.3+-1.9 | 50.8 | 13.0 | +45.00 |
| facility | 91.4 | 74.9+-1.8 | 48.2+-2.4 | 31.9+-0.6 | 89.2+-1.4 | 86.4+-0.4 | 83.7+-0.2 | 48.4 | 13.6 | +39.74 |
| fps | 90.0 | 71.7+-0.8 | 44.3+-5.7 | 31.2+-2.1 | 87.2+-1.3 | 84.5+-1.6 | 81.8+-2.2 | 50.9 | 15.5 | +40.96 |
| random | 92.8 | 71.3+-3.1 | 45.7+-3.7 | 30.3+-2.7 | 89.7+-0.6 | 85.2+-1.4 | 82.1+-1.7 | 50.9 | 14.3 | +43.67 |
| eva | 86.7 | 67.0+-2.4 | 45.1+-3.8 | 31.3+-1.7 | 82.4+-1.8 | 78.4+-2.3 | 75.4+-3.0 | 52.2 | 21.3 | +38.90 |
| el2n_top | 71.4 | 55.1+-0.9 | 35.4+-2.8 | 26.0+-3.4 | 68.0+-1.1 | 63.7+-1.7 | 60.7+-2.0 | 61.1 | 35.8 | +32.54 |
| forgetting | 83.3 | 62.0+-3.3 | 41.0+-3.0 | 30.5+-1.5 | 78.0+-1.3 | 72.1+-0.9 | 67.8+-1.5 | 55.5 | 27.4 | +38.81 |

### pathmnist r=0.02

| method | clean | EXP s1 | EXP s2 | EXP s3 | STR s1 | STR s2 | STR s3 | EXP s1-3 BE | STR s1-3 BE | clean_ba_drop EXP |
|---|---|---|---|---|---|---|---|---|---|---|
| graph_a2 | 82.7 | 60.2+-2.5 | 42.7+-1.2 | 32.1+-0.9 | 76.8+-0.8 | 75.4+-0.7 | 72.7+-1.1 | 55.0 | 25.0 | +37.69 |
| herding | 81.8 | 53.3+-3.7 | 36.4+-2.9 | 29.1+-3.5 | 76.2+-2.6 | 74.1+-2.6 | 71.3+-2.8 | 60.4 | 26.1 | +42.24 |
| facility | 80.2 | 54.9+-0.9 | 38.7+-0.9 | 31.1+-1.0 | 76.1+-1.9 | 74.7+-1.8 | 72.1+-2.4 | 58.4 | 25.7 | +38.63 |
| fps | 72.6 | 52.4+-3.4 | 37.8+-2.8 | 28.1+-2.1 | 68.6+-1.0 | 67.2+-0.9 | 64.6+-1.0 | 60.6 | 33.2 | +33.19 |
| random | 78.6 | 56.3+-2.5 | 39.3+-2.0 | 28.9+-1.6 | 72.3+-0.6 | 70.6+-0.6 | 67.5+-1.3 | 58.5 | 29.9 | +37.09 |
| eva | 47.0 | 32.9+-1.8 | 21.6+-2.8 | 16.8+-3.8 | 43.8+-4.7 | 43.6+-4.6 | 42.0+-4.4 | 76.2 | 56.9 | +23.22 |
| el2n_top | 36.6 | 25.9+-1.2 | 19.8+-2.6 | 15.2+-3.7 | 34.2+-1.0 | 34.0+-1.1 | 32.0+-1.3 | 79.7 | 66.6 | +16.27 |
| forgetting | 43.9 | 33.0+-5.6 | 27.6+-6.2 | 23.8+-6.4 | 40.6+-5.3 | 40.7+-5.4 | 39.0+-4.8 | 71.9 | 59.9 | +15.81 |

### pathmnist r=0.05

| method | clean | EXP s1 | EXP s2 | EXP s3 | STR s1 | STR s2 | STR s3 | EXP s1-3 BE | STR s1-3 BE | clean_ba_drop EXP |
|---|---|---|---|---|---|---|---|---|---|---|
| graph_a2 | 86.7 | 67.4+-1.2 | 49.1+-1.1 | 36.3+-2.4 | 80.7+-0.6 | 78.6+-0.8 | 75.1+-0.7 | 49.1 | 21.9 | +35.80 |
| herding | 86.5 | 66.2+-3.7 | 47.2+-4.0 | 33.3+-3.2 | 80.5+-1.5 | 78.1+-1.2 | 74.5+-1.7 | 51.1 | 22.3 | +37.61 |
| facility | 86.6 | 69.1+-1.2 | 49.9+-3.0 | 36.5+-1.2 | 81.4+-1.1 | 79.1+-1.2 | 75.3+-1.2 | 48.2 | 21.4 | +34.72 |
| fps | 83.0 | 68.8+-1.9 | 51.2+-3.9 | 36.1+-2.9 | 76.4+-0.7 | 74.4+-0.9 | 70.9+-0.8 | 48.0 | 26.1 | +30.98 |
| random | 86.7 | 70.6+-3.1 | 54.2+-5.1 | 40.6+-6.1 | 82.1+-1.1 | 80.4+-1.2 | 77.6+-1.5 | 44.9 | 20.0 | +31.52 |
| eva | 60.8 | 46.0+-9.6 | 33.7+-10.0 | 25.1+-8.3 | 53.5+-4.3 | 52.8+-4.2 | 50.3+-4.4 | 65.1 | 47.8 | +25.90 |
| el2n_top | 50.2 | 39.7+-1.7 | 31.8+-0.6 | 25.4+-2.3 | 45.3+-3.4 | 44.4+-3.1 | 42.2+-2.9 | 67.7 | 56.0 | +17.92 |
| forgetting | 61.0 | 48.1+-4.5 | 40.1+-3.8 | 32.1+-3.4 | 54.0+-3.3 | 53.0+-3.4 | 50.3+-3.3 | 59.9 | 47.6 | +20.93 |

## 3. Worst-class recall (severity 1-3, mean over seeds, %)

### bloodmnist r=0.02

| method | worst clean | worst EXP | worst STR | worst drop EXP (pp) | most-hurt class under EXP |
|---|---|---|---|---|---|
| graph_a2 | 59.4 | 8.8 | 45.7 | +50.53 | recall_6 (-67.1pp) |
| herding | 58.8 | 15.2 | 44.3 | +43.60 | recall_6 (-53.0pp) |
| facility | 50.1 | 7.9 | 45.5 | +42.28 | recall_4 (-65.3pp) |
| fps | 34.1 | 5.2 | 30.1 | +28.93 | recall_6 (-69.4pp) |
| random | 56.4 | 8.2 | 47.2 | +48.20 | recall_4 (-64.1pp) |
| eva | 42.3 | 6.5 | 29.0 | +35.71 | recall_6 (-56.3pp) |
| el2n_top | 20.9 | 0.9 | 13.8 | +19.98 | recall_4 (-41.4pp) |
| forgetting | 19.6 | 2.2 | 11.7 | +17.42 | recall_4 (-60.3pp) |

### bloodmnist r=0.05

| method | worst clean | worst EXP | worst STR | worst drop EXP (pp) | most-hurt class under EXP |
|---|---|---|---|---|---|
| graph_a2 | 74.8 | 17.1 | 53.2 | +57.72 | n/a (no clean per-class) |
| herding | 81.8 | 12.0 | 60.8 | +69.83 | n/a (no clean per-class) |
| facility | 71.6 | 14.0 | 59.4 | +57.65 | n/a (no clean per-class) |
| fps | 60.2 | 10.5 | 49.7 | +49.64 | n/a (no clean per-class) |
| random | 77.1 | 10.3 | 55.7 | +66.88 | n/a (no clean per-class) |
| eva | 59.1 | 10.5 | 40.3 | +48.62 | n/a (no clean per-class) |
| el2n_top | 27.7 | 2.5 | 21.4 | +25.28 | n/a (no clean per-class) |
| forgetting | 53.7 | 3.8 | 34.8 | +49.91 | n/a (no clean per-class) |

### pathmnist r=0.02

| method | worst clean | worst EXP | worst STR | worst drop EXP (pp) | most-hurt class under EXP |
|---|---|---|---|---|---|
| graph_a2 | 41.8 | 7.8 | 32.9 | +34.04 | recall_4 (-66.3pp) |
| herding | 35.9 | 4.8 | 30.1 | +31.06 | recall_6 (-77.2pp) |
| facility | 36.0 | 5.0 | 31.0 | +31.04 | recall_4 (-66.0pp) |
| fps | 34.1 | 5.5 | 24.8 | +28.66 | recall_4 (-63.1pp) |
| random | 35.3 | 5.4 | 28.0 | +29.92 | recall_4 (-66.0pp) |
| eva | 5.2 | 0.1 | 3.4 | +5.13 | recall_6 (-61.2pp) |
| el2n_top | 0.0 | 0.0 | 0.0 | -0.02 | recall_3 (-45.1pp) |
| forgetting | 5.6 | 0.9 | 3.5 | +4.66 | recall_0 (-35.7pp) |

### pathmnist r=0.05

| method | worst clean | worst EXP | worst STR | worst drop EXP (pp) | most-hurt class under EXP |
|---|---|---|---|---|---|
| graph_a2 | 44.5 | 12.8 | 35.0 | +31.74 | n/a (no clean per-class) |
| herding | 43.9 | 11.0 | 36.8 | +32.95 | n/a (no clean per-class) |
| facility | 44.3 | 11.0 | 37.4 | +33.24 | n/a (no clean per-class) |
| fps | 39.3 | 13.5 | 30.6 | +25.82 | n/a (no clean per-class) |
| random | 43.5 | 14.2 | 33.9 | +29.39 | n/a (no clean per-class) |
| eva | 6.7 | 1.6 | 4.1 | +5.08 | n/a (no clean per-class) |
| el2n_top | 0.2 | 0.5 | 0.2 | -0.32 | n/a (no clean per-class) |
| forgetting | 13.9 | 4.0 | 6.8 | +9.83 | n/a (no clean per-class) |

## 4. Risk across seeds (H2)

Paired gap = method - random, same seed. sd ratio = sd(gap under EXP) / sd(gap clean); >1 means shift amplifies the seed lottery. p10 = 10% quantile, CVaR20 = mean of worst 20% of seeds. With <=5 seeds these tails are near the minimum; read as format until more seeds exist.

### bloodmnist r=0.02

| method | gap clean mean | gap EXP mean | sd clean | sd EXP | sd ratio | p10 EXP | CVaR20 EXP | gap STR mean |
|---|---|---|---|---|---|---|---|---|
| graph_a2 | +2.61 | +3.36 | 4.3 | 10.9 | 2.57 | -7.63 | -12.06 | +4.06 |
| herding | +2.14 | +7.67 | 1.5 | 3.4 | 2.22 | +4.21 | +2.82 | +1.79 |
| facility | -0.50 | +0.45 | 0.1 | 10.0 | 96.03 | -9.71 | -13.56 | +2.44 |
| fps | -7.61 | -5.80 | 1.6 | 4.6 | 2.97 | -10.53 | -12.15 | -4.91 |
| eva | -8.15 | -2.41 | 3.4 | 9.1 | 2.68 | -11.68 | -15.19 | -7.55 |
| el2n_top | -23.49 | -11.46 | 3.4 | 11.2 | 3.33 | -22.65 | -25.80 | -20.56 |
| forgetting | -18.25 | -5.57 | 2.3 | 4.9 | 2.13 | -10.55 | -12.39 | -21.39 |

tau(EXP mean-BA ranking, EXP CVaR20 ranking) = +0.857; #1 by mean = herding, #1 by CVaR20 = herding

### bloodmnist r=0.05

| method | gap clean mean | gap EXP mean | sd clean | sd EXP | sd ratio | p10 EXP | CVaR20 EXP | gap STR mean |
|---|---|---|---|---|---|---|---|---|
| graph_a2 | +0.17 | +5.44 | 0.5 | 3.7 | 7.14 | +2.71 | +2.63 | -3.27 |
| herding | +1.40 | +0.06 | 0.5 | 5.0 | 10.34 | -4.94 | -6.34 | +1.35 |
| facility | -1.42 | +2.50 | 1.3 | 2.1 | 1.55 | +0.40 | -0.25 | +0.77 |
| fps | -2.76 | -0.06 | 1.6 | 4.6 | 2.84 | -4.42 | -5.40 | -1.19 |
| eva | -6.10 | -1.33 | 0.5 | 0.9 | 1.89 | -2.18 | -2.37 | -6.95 |
| el2n_top | -21.41 | -10.28 | 2.0 | 3.5 | 1.77 | -13.74 | -14.59 | -21.49 |
| forgetting | -9.52 | -4.67 | 1.4 | 5.2 | 3.80 | -9.88 | -11.96 | -13.03 |

tau(EXP mean-BA ranking, EXP CVaR20 ranking) = +1.000; #1 by mean = graph_a2, #1 by CVaR20 = graph_a2

### pathmnist r=0.02

| method | gap clean mean | gap EXP mean | sd clean | sd EXP | sd ratio | p10 EXP | CVaR20 EXP | gap STR mean |
|---|---|---|---|---|---|---|---|---|
| graph_a2 | +4.12 | +3.52 | 1.9 | 2.1 | 1.13 | +1.93 | +1.85 | +4.86 |
| herding | +3.24 | -1.92 | 2.6 | 2.4 | 0.92 | -3.72 | -3.77 | +3.75 |
| facility | +1.63 | +0.09 | 2.7 | 2.0 | 0.73 | -1.43 | -1.51 | +4.18 |
| fps | -5.95 | -2.05 | 1.2 | 1.0 | 0.88 | -3.08 | -3.35 | -3.34 |
| eva | -31.58 | -17.71 | 3.2 | 1.1 | 0.34 | -18.74 | -18.94 | -27.00 |
| el2n_top | -42.01 | -21.19 | 1.8 | 0.5 | 0.28 | -21.68 | -21.80 | -36.73 |
| forgetting | -34.66 | -13.38 | 5.6 | 6.7 | 1.19 | -19.99 | -21.71 | -30.02 |

tau(EXP mean-BA ranking, EXP CVaR20 ranking) = +0.857; #1 by mean = graph_a2, #1 by CVaR20 = graph_a2

### pathmnist r=0.05

| method | gap clean mean | gap EXP mean | sd clean | sd EXP | sd ratio | p10 EXP | CVaR20 EXP | gap STR mean |
|---|---|---|---|---|---|---|---|---|
| graph_a2 | +0.08 | -4.20 | 1.4 | 6.3 | 4.50 | -10.62 | -12.64 | -1.87 |
| herding | -0.19 | -6.27 | 1.5 | 1.4 | 0.97 | -7.68 | -8.27 | -2.32 |
| facility | -0.11 | -3.31 | 1.0 | 6.5 | 6.47 | -9.89 | -11.92 | -1.39 |
| fps | -3.65 | -3.11 | 1.4 | 4.1 | 2.92 | -7.31 | -8.64 | -6.10 |
| eva | -25.86 | -20.23 | 4.1 | 8.9 | 2.16 | -29.21 | -31.83 | -27.83 |
| el2n_top | -36.45 | -22.84 | 4.0 | 4.8 | 1.18 | -27.69 | -29.21 | -36.04 |
| forgetting | -25.66 | -15.07 | 3.5 | 4.6 | 1.30 | -18.67 | -18.93 | -27.56 |

tau(EXP mean-BA ranking, EXP CVaR20 ranking) = +0.786; #1 by mean = random, #1 by CVaR20 = random

## 5. Appendix: calibration under shift (severity 1-3, mean over seeds)

### bloodmnist r=0.02

| method | ece15 clean/EXP/STR | brier clean/EXP/STR | nll clean/EXP/STR | aurc clean/EXP/STR | risk_at_80 clean/EXP/STR |
|---|---|---|---|---|---|
| graph_a2 | 0.088 / 0.475 / 0.125 | 0.240 / 1.013 / 0.325 | 0.745 / 10.129 / 1.154 | 0.039 / 0.416 / 0.072 | 0.069 / 0.524 / 0.120 |
| herding | 0.092 / 0.418 / 0.146 | 0.245 / 0.907 / 0.361 | 0.782 / 13.316 / 2.753 | 0.044 / 0.376 / 0.092 | 0.075 / 0.470 / 0.143 |
| facility | 0.117 / 0.497 / 0.143 | 0.302 / 1.058 / 0.360 | 1.004 / 15.303 / 1.198 | 0.067 / 0.464 / 0.092 | 0.101 / 0.555 / 0.141 |
| fps | 0.158 / 0.589 / 0.196 | 0.407 / 1.237 / 0.482 | 1.218 / 21.610 / 1.967 | 0.107 / 0.603 / 0.158 | 0.170 / 0.655 / 0.218 |
| random | 0.111 / 0.518 / 0.165 | 0.287 / 1.093 / 0.400 | 1.162 / 49.481 / 3.108 | 0.063 / 0.491 / 0.118 | 0.096 / 0.563 / 0.163 |
| eva | 0.169 / 0.519 / 0.211 | 0.400 / 1.087 / 0.489 | 1.813 / 20.931 / 2.300 | 0.095 / 0.452 / 0.128 | 0.161 / 0.551 / 0.215 |
| el2n_top | 0.282 / 0.607 / 0.320 | 0.654 / 1.268 / 0.729 | 2.216 / 19.243 / 2.875 | 0.191 / 0.568 / 0.225 | 0.334 / 0.670 / 0.377 |
| forgetting | 0.261 / 0.589 / 0.336 | 0.587 / 1.230 / 0.748 | 2.392 / 14.640 / 3.247 | 0.169 / 0.569 / 0.252 | 0.272 / 0.644 / 0.376 |

### bloodmnist r=0.05

| method | ece15 clean/EXP/STR | brier clean/EXP/STR | nll clean/EXP/STR | aurc clean/EXP/STR | risk_at_80 clean/EXP/STR |
|---|---|---|---|---|---|
| graph_a2 | - / 0.329 / 0.119 | - / 0.759 / 0.299 | - / 4.437 / 1.157 | - / 0.306 / 0.095 | - / 0.400 / 0.117 |
| herding | - / 0.391 / 0.073 | - / 0.861 / 0.205 | - / 4.109 / 0.534 | - / 0.368 / 0.040 | - / 0.449 / 0.063 |
| facility | - / 0.373 / 0.070 | - / 0.838 / 0.212 | - / 3.813 / 0.534 | - / 0.336 / 0.037 | - / 0.441 / 0.061 |
| fps | - / 0.410 / 0.086 | - / 0.904 / 0.248 | - / 4.432 / 0.637 | - / 0.361 / 0.053 | - / 0.473 / 0.079 |
| random | - / 0.393 / 0.083 | - / 0.866 / 0.232 | - / 3.565 / 0.613 | - / 0.366 / 0.043 | - / 0.461 / 0.073 |
| eva | - / 0.399 / 0.124 | - / 0.874 / 0.323 | - / 4.206 / 0.989 | - / 0.323 / 0.064 | - / 0.451 / 0.117 |
| el2n_top | - / 0.493 / 0.229 | - / 1.056 / 0.551 | - / 6.869 / 1.665 | - / 0.467 / 0.148 | - / 0.559 / 0.261 |
| forgetting | - / 0.418 / 0.170 | - / 0.908 / 0.411 | - / 4.865 / 1.340 | - / 0.359 / 0.094 | - / 0.473 / 0.169 |

### pathmnist r=0.02

| method | ece15 clean/EXP/STR | brier clean/EXP/STR | nll clean/EXP/STR | aurc clean/EXP/STR | risk_at_80 clean/EXP/STR |
|---|---|---|---|---|---|
| graph_a2 | 0.059 / 0.417 / 0.119 | 0.221 / 0.927 / 0.352 | 0.583 / 3.033 / 0.928 | 0.049 / 0.369 / 0.100 | 0.066 / 0.496 / 0.150 |
| herding | 0.070 / 0.487 / 0.139 | 0.215 / 1.048 / 0.358 | 0.716 / 4.240 / 1.131 | 0.068 / 0.458 / 0.122 | 0.068 / 0.550 / 0.151 |
| facility | 0.068 / 0.448 / 0.115 | 0.234 / 0.979 / 0.344 | 0.567 / 3.304 / 0.852 | 0.040 / 0.402 / 0.087 | 0.074 / 0.521 / 0.145 |
| fps | 0.106 / 0.462 / 0.164 | 0.359 / 1.041 / 0.473 | 0.853 / 3.820 / 1.192 | 0.095 / 0.454 / 0.146 | 0.161 / 0.578 / 0.234 |
| random | 0.081 / 0.439 / 0.153 | 0.251 / 0.966 / 0.403 | 0.670 / 3.122 / 1.112 | 0.063 / 0.373 / 0.129 | 0.083 / 0.519 / 0.178 |
| eva | 0.439 / 0.643 / 0.466 | 0.961 / 1.361 / 1.022 | 3.047 / 5.207 / 3.405 | 0.439 / 0.710 / 0.510 | 0.519 / 0.750 / 0.562 |
| el2n_top | 0.464 / 0.609 / 0.489 | 1.041 / 1.329 / 1.088 | 3.517 / 4.637 / 3.655 | 0.553 / 0.761 / 0.564 | 0.586 / 0.775 / 0.612 |
| forgetting | 0.447 / 0.606 / 0.482 | 0.993 / 1.295 / 1.058 | 3.190 / 4.889 / 3.438 | 0.453 / 0.662 / 0.488 | 0.551 / 0.725 / 0.589 |

### pathmnist r=0.05

| method | ece15 clean/EXP/STR | brier clean/EXP/STR | nll clean/EXP/STR | aurc clean/EXP/STR | risk_at_80 clean/EXP/STR |
|---|---|---|---|---|---|
| graph_a2 | - / 0.316 / 0.092 | - / 0.764 / 0.302 | - / 2.092 / 0.740 | - / 0.290 / 0.077 | - / 0.422 / 0.121 |
| herding | - / 0.355 / 0.097 | - / 0.816 / 0.288 | - / 2.411 / 0.746 | - / 0.308 / 0.075 | - / 0.437 / 0.114 |
| facility | - / 0.308 / 0.093 | - / 0.745 / 0.291 | - / 2.130 / 0.715 | - / 0.281 / 0.073 | - / 0.403 / 0.116 |
| fps | - / 0.286 / 0.119 | - / 0.739 / 0.366 | - / 1.979 / 0.860 | - / 0.313 / 0.087 | - / 0.418 / 0.161 |
| random | - / 0.289 / 0.088 | - / 0.701 / 0.271 | - / 1.955 / 0.682 | - / 0.253 / 0.058 | - / 0.376 / 0.100 |
| eva | - / 0.524 / 0.362 | - / 1.151 / 0.847 | - / 3.963 / 2.601 | - / 0.595 / 0.403 | - / 0.644 / 0.467 |
| el2n_top | - / 0.481 / 0.389 | - / 1.099 / 0.915 | - / 3.452 / 2.860 | - / 0.604 / 0.451 | - / 0.633 / 0.511 |
| forgetting | - / 0.446 / 0.339 | - / 1.014 / 0.809 | - / 3.134 / 2.443 | - / 0.486 / 0.388 | - / 0.571 / 0.440 |

