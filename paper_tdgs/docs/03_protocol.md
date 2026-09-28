# Protocol, config truth, and the cells the author already occupies

## The protocol we must match

| item | value |
|---|---|
| datasets | pathmnist, organamnist, organsmnist, bloodmnist, tissuemnist |
| budgets | 2% and 5% of the training pool |
| model | ResNet-18, **from scratch**, 224×224 |
| main metric | **balanced accuracy** — does not change |
| epochs | 1000, `checkpoint_rule: final_epoch` |
| batch size | 256 |
| embeddings for selection | frozen **UNI** |
| budget formula | `bpc = int(N·ratio) // C`, `|S| = bpc·C` |
| baselines | the author's 8 methods |
| seeds | 5 in the archive; **3 for new-arm screening** |

Corruption results all use the **same final checkpoint** — no retraining per
corruption, and the corruption test set is never used to pick a checkpoint.

## Config truth — the divergence that had to be settled first

The archive's real selection config is **`k_neighbors=50`, `k_hops=2`,
`graph_scope=global`**, established two independent ways:

1. `protocol.json`'s `selection_config`.
2. **Replay.** k=50 set-matches the archived selections on 7 of 9 cells; k=10
   matches none.

The **repo's CLI default is `k_neighbors=10` and `global_selection=False`.** That
is a divergence between the released default and the paper's configuration, not an
error on our side. Independent confirmation: k=50 matches the author's own Table 2
k=50 ablation row digit-for-digit, and `--global` is required.

Consequences that bit us and are now encoded:
- `graph_a3..a5` **as registered** are off-config (they inherit k=10), so they
  cannot be used as controls without overriding k.
- Registered `graph_a3` computes `K = A + A² + A³`, **not** `A + A³`. It therefore
  cannot isolate "one more hop" against anything. (The whole kernel-variant line is
  scoped out anyway.)

## Reproduction status of the archive

- Table 1: **394/400 cells done**, 91.4% within ±4pp of the paper, mean error
  2.13pp.
- **bloodmnist's archived `graph_a2` selection is not reproducible** under k=10 or
  k=50, at either ratio (tissuemnist 5% also fails). Untested hypothesis:
  `diag(K)` is never zeroed in the author's code, which is known to change 15% of
  bloodmnist's picks. Until resolved, **archive-derived bloodmnist claims are
  un-reproduced** — but note the round-1 bloodmnist numbers here are from
  *our own* regenerated `graph_a2` arm, so they are internally consistent.

## Author-occupied ablation cells — cannot be claimed as a contribution

| script / arm | what it sweeps |
|---|---|
| `compare_k.py` | k = 5 / 10 / 20 (README), plus the k=50 Table 2 row |
| `compare_global.py` | **global vs per-class scope** ← the threat to `tdgs_cls` |
| `compare_sizes.py` | budget sizes |
| `compare_selections.py` | selection-set comparison |
| `graph_a1..a5` | number of hops |

`compare_global.py` is why `a2_perclass` is the first block of the round-2
worklist. Per-class scope also confines coverage credit within class; if it
recovers the gain, our arm is inside the author's own ablation.

## Known defects in the author's paper / code

- The released default is **per-class**, but the paper's Table 1 is **global**.
- `diag(K)` is never zeroed, so a point gets credit for covering itself; this
  changes 15% of bloodmnist's picks.
- The paper's sign test is reported as n=10 but is effectively n≈4, because
  organamnist and organsmnist are the same 201 LiTS volumes and the two budgets
  are not independent draws.

## Harness plumbing and traps

- New selectors are staged as **`precomputed`** selection files:
  `${DATASET}_r${RATIO}_${ARM}_s42.npy`. `graphcov/` is never edited — the source
  guard `assert_graphcov_source_unchanged()` git-diffs against a pinned commit and
  must not be weakened.
- `td_run_one.sh` writes `methods: [precomputed]` into every config, so the
  **arm identity exists only in the output path**. Reading `method` from metadata
  collapses all arms into one.
- `selection_seed` is pinned to 42 while the **training seed** varies. The harvest
  key is `(dataset, ratio, arm, train_seed)`; keying on `selection_seed` silently
  deduplicates every seed down to one row.
- Importing the reliability `pipeline.py` **runs the whole select/train pipeline at
  module level**. Neutralise with `PHASE` / `METHODS` / `RUN_FULL_TRAIN` /
  `CLEAN_ONLY` / `AUTO_CONSOLIDATE` before importing.
- The universal seed set in the older archive is `{0,1,2,3,4,5,6,2026}`, **not**
  `{42..46}`. TDGS rounds use `{42..46}`.
- Table-1 archive defects to work around: duplicate rows, and
  `selection_seed ≡ training_seed`.

## The 3-seed policy

Baselines are **not** re-run: 394/400 already exist and are validated to ±4pp.
The 3-seed policy applies to **new arms only**, as a screen. Per-cell MDE at n=3 is
2.4–17.6pp, so 3 seeds cannot test a 1pp effect — it can only rank arms and
establish sign consistency. The headline contrast is lifted to n=5 in round-2
block P1-c. See [04_instrument.md](04_instrument.md).
