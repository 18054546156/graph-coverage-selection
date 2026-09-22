# PathMNIST 5% fixed-selection paired result

Decision: **paper ranking not reproduced**.

Primary endpoint: final official-test balanced accuracy. Both arms use one frozen, canonically sorted subset; only downstream seed changes.

| Seed | Random seed-42 BA (%) | Graph-A2 BA (%) | Graph-A2 - Random (pp) | Winner |
|---:|---:|---:|---:|---:|
| 42 | 87.5343 | 85.4356 | -2.0987 | random_seed42 |
| 43 | 89.1369 | 86.0182 | -3.1187 | random_seed42 |
| 44 | 87.2975 | 83.8195 | -3.4781 | random_seed42 |
| 45 | 86.9717 | 85.5318 | -1.4399 | random_seed42 |
| 46 | 87.5884 | 88.2319 | +0.6435 | graph_a2 |

## Aggregate

- Random: 87.7058+/-0.8362%
- Graph-A2: 85.8074+/-1.5881%
- Mean paired delta: -1.8984 pp
- Descriptive paired 95% t interval: [-3.9288, +0.1320] pp
- Wins: Graph-A2 1, Random 4, ties 0

## Interpretation

The paper ranking is not reproduced under fixed seed-42 subsets; varying Random selection seeds is not a sufficient explanation for the earlier reversal.

This controlled result diagnoses the selection-seed confound. It is not a new method result or a SOTA claim.
