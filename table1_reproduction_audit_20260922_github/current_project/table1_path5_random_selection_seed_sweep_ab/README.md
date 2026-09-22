# PathMNIST 5% Random selection-seed sensitivity

This isolated experiment measures how much the public GraphCov Random selector's seed changes the downstream result. It generates five equal-quota subsets with selection seeds `42-46`, while fixing the downstream ResNet-18 training seed at `42` for every arm.

The selector is the vendored public implementation at commit `8cf757adc4c333dc1427d511f0de2f246d15ebac`. Selection uses training labels only. The primary endpoint is final official-test balanced accuracy; best-test BA is retained as audit-only.

Authoritative outputs after completion:

- `artifacts/manifest.json`: canonical index SHA, class quotas, and pairwise Jaccard overlap.
- `results/random_seed*/metrics.json`: one final and one audit result per selection seed.
- `reports/summary.json`, `reports/selection_seed_results.csv`, `reports/RESULT.md`: aggregate sensitivity result.
- `submitted_jobs.json`, `logs/`: Slurm provenance.
