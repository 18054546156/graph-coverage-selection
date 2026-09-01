# Baseline: official Graph-A2

This is a standalone copy of the official author repository at commit
`8cf757adc4c333dc1427d511f0de2f246d15ebac`.

Selection is the original global Graph-A2 kernel
`K = A_sym + A_sym^2`, followed by the official class-constrained facility
location greedy solver. The code default is `k=10`; `configs/sweep.json` keeps
`k=10` and the paper-reproduction candidate `k=50` as separate cases.

Only experiment artifact persistence was added: selected indices and SHA256 are
saved beside the official metrics. The selection implementation is unchanged.

