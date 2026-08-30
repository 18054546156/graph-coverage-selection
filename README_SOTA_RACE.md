# SOTA Race v1

This worktree starts from official commit `b66ae44` and keeps the official
Graph-A2 implementation as the control. The added methods preserve the total
selection budget and use the same UNI embeddings, global graph, downstream
training, and evaluation code.

## Candidate families

- `graph_a2_dec`: remove cross-class coverage credit.
- `graph_a2_damp25/50/75`: retain 25%, 50%, or 75% cross-class credit.
- `graph_a2_sqrt`: allocate class quotas proportional to `sqrt(n_c)`.
- `graph_a2_dec_sqrt`: combine hard decoupling with sqrt quotas.
- `graph_a2_marginal`: allocate quota units by class-local graph marginal gain.
- `graph_a2_dec_marginal`: combine hard decoupling and graph-marginal quotas.

The first run is a two-seed TissueMNIST 2% screening experiment using 1,000
training iterations. It is not a strict Table 1 reproduction or a five-seed
SOTA claim. Any winner must subsequently pass the strict epoch protocol on all
reported datasets and five seeds.

## Managed paths

- Code: this Git worktree.
- Frozen protocol: `configs/race_v1_screening.json`.
- Remote results: `/root/graphcov_pathmnist_sota/results/sota_race_v1`.
- Remote screens: `sota_race_v1_gpu0`, `sota_race_v1_gpu1`.

Run unit tests with:

```bash
python tests/test_selection_opt.py
```

Launch on the configured two-GPU server with:

```bash
bash scripts/launch_race_v1.sh
```
