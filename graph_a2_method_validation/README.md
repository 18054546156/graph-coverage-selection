# Graph-A2 isolated method validation

This directory contains four standalone copies of the official Graph-A2 code at
commit `8cf757adc4c333dc1427d511f0de2f246d15ebac`.

| Directory | Registered method | Isolated change |
|---|---|---|
| `baseline/` | `graph_a2` | Official Graph-A2, unchanged selection formula |
| `m1_dynamic_quota/` | `graph_a2_m1` | Safe normalized dynamic class quota |
| `m2_relation_gate/` | `graph_a2_m2` | Scalar or class-pair cross-class coverage credit |
| `m3_trust_weighting/` | `graph_a2_m3` | Purity/local-support client weighting |

Each M directory still registers the unmodified `graph_a2`, so every run can
compare the variant with its local baseline. The disabled setting must reproduce
the original selected-index order exactly:

- M1: `--quota-floor-ratio 1.0`
- M2: `--relation-mode baseline` (or class-pair strength `0`)
- M3: `--purity-exponent 0`

The source default is `k=10, H=2`. A separate `k=50, H=2` candidate is included
because the paper/config evidence is ambiguous; it is never silently substituted
for the source default.

Run from inside one package:

```bash
python -m graphcov.run --list-methods
python scripts/run_sweep.py --dry-run
python scripts/run_sweep.py --datasets pathmnist --ratios 0.02 --seeds 42
```

Every full run stores exact selected indices under
`<output>/runs/<run_id>/selection_artifacts/` and records their SHA256 in
`results.csv`.

