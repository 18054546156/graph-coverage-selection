# PathMNIST 5% fixed-selection paired A/B

This isolated experiment tests whether the public Table 1 runner's changing Random subset across trials explains the observed PathMNIST 5% rank reversal.

The package freezes one Random seed-42 subset and one Graph-A2 subset. Both are sorted by training-set index, validated as 499 samples from each of nine classes, and reused byte-for-byte for downstream seeds 42-46. Every array task trains both arms sequentially on the same RTX 4090 through the same vendored `evaluate_selection` function.

The primary quantity is the paired final-BA difference `Graph-A2 - Random` in percentage points. A positive mean reproduces the paper ordering; a non-positive mean does not.

Authoritative files after completion:

- `artifacts/manifest.json`: source paths, file SHA values, canonical index SHA values, order, and class quotas.
- `results/seed*/{random_seed42,graph_a2}/metrics.json`: final BA for each paired arm.
- `reports/paired_results.csv`: per-seed paired values.
- `reports/summary.json`: machine-readable aggregate and frozen decision.
- `reports/RESULT.md`: human-readable result.
- `submitted_jobs.json` and `logs/`: Slurm provenance.

The vendored source is pinned to upstream commit `8cf757adc4c333dc1427d511f0de2f246d15ebac` and verified at runtime.
