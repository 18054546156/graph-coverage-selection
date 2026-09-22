# Table 1 Ours: Graph-A2 k=50 reproduction

This isolated package fills only the paper's `Ours` column, where Ours is
`graph_a2` from the author repository:

<https://github.com/zahiriddin-rustamov/graph-coverage-selection>

Frozen protocol: UNI 224 embeddings, global graph, `k=50`, `H=2`, equal class
quota, ratios 2% and 5%, seeds 42--46, ResNet-18 from scratch, 1000 epochs,
batch size 256, SGD (`lr=0.1`, momentum `0.9`, weight decay `5e-4`), cosine
schedule, and no augmentation. The primary endpoint is the final epoch
`balanced_accuracy` field from the vendored runner. Best-test fields are audit
only.

The vendor snapshot is checked against commit
`8cf757adc4c333dc1427d511f0de2f246d15ebac` by `official_runtime.py`.

## Run on HPC

```bash
sbatch ours_array.slurm
sbatch --dependency=afterok:<array_job_id> summary.slurm
python build_notebook.py
```

The array writes one independent dataset output under `results/raw/`. The
summary copies selected arrays into `selection/` and writes:

- `reports/ours_table1.csv`
- `reports/ours_raw_rows.csv`
- `reports/summary.json`
- `RESULT.md`
- `reports/table1_ours_reproduction.ipynb`

No official test split is used for selection. The downstream vendor runner
evaluates test during training as in the public implementation and reports
the final field for comparison with Table 1; it does not use the best-test
field in the summary.
