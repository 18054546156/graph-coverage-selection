# Selection robustness pilot

Source: existing selected-index artifacts read from the isolated HPC Table 1 package. No downstream test labels or MedMNIST-C data were used.

The pilot contains 5 datasets x 2 ratios x 2 methods x 3 selection seeds, yielding 60 pairwise comparisons. Graph-A2 selected indices are identical across selection seeds 1, 42, and 2026 for every dataset and ratio: mean/min/max pairwise Jaccard = `1.000000 / 1.000000 / 1.000000`.

Random selection is seed-sensitive. Pairwise Jaccard summaries are:

| Dataset | 2% mean (min-max) | 5% mean (min-max) |
| --- | ---: | ---: |
| OrganSMNIST | 0.022321 (0.016636-0.026119) | 0.037693 (0.033557-0.040541) |
| OrganAMNIST | 0.013632 (0.010370-0.017151) | 0.035072 (0.031045-0.039110) |
| PathMNIST | 0.009393 (0.009298-0.009583) | 0.027533 (0.025811-0.028513) |
| TissueMNIST | 0.016148 (0.014586-0.018025) | 0.044558 (0.041682-0.046029) |
| BloodMNIST | 0.017547 (0.015317-0.019780) | 0.031660 (0.030461-0.033159) |

Across all five datasets, Random mean Jaccard is `0.015808` at 2% and `0.035303` at 5%. These values are expected to be near the sampling fraction for independently drawn subsets; they establish selection instability, not a downstream accuracy claim.

The six other methods currently have only one selection seed in the available isolated artifacts. The audit therefore records 180 missing pairwise cells for them instead of treating a single index set as evidence of robustness. No downstream BA or MedMNIST-C metric is inferred from this pilot.
