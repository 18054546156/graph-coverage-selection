# Comparing With the Author's GraphCov Code

## Pinned upstream

- Repository: <https://github.com/zahiriddin-rustamov/graph-coverage-selection>
- Exact commit used by the experiment: `8cf757adc4c333dc1427d511f0de2f246d15ebac`
- The experiment pipeline asserts this commit is an ancestor and that the upstream `graphcov/` tree has no local modifications before execution.
- GitHub branch: `codex/table1-v11-execution`
- Compare view: <https://github.com/18054546156/graph-coverage-selection/compare/8cf757adc4c333dc1427d511f0de2f246d15ebac...codex/table1-v11-execution>

## What is upstream and what is experiment code

The author's selector implementation is under the repository's `graphcov/` directory, especially `graphcov/run/selection.py`, `graphcov/run/graph.py`, `graphcov/run/embeddings.py`, `graphcov/run/evaluation.py`, and `graphcov/run/experiment.py`. The Table 1 pipeline imports these modules; it does not fork or edit the upstream selector implementation.

The experiment-specific orchestration is in `reliability_medmnistc/pipeline.py` and `reliability_medmnistc/scripts/`. It supplies the clean MedMNIST data path, per-class 2% budget, method arguments, paired seeds, training/evaluation sequence, completion markers, and artifact hashes. `run_clean_only.slurm` freezes the actual Table 1 run mode and deterministic setting. The 40 files under `../configs/` are the executed configurations; this directory's `configs/` are defaults only.

Protocol choices that should be checked against the paper and author README during review:

- five MedMNIST datasets, eight selection methods, 2% selection budget;
- seeds `0, 1, 2, 3, 4, 5, 6, 2026`, with selection and training seeds paired;
- clean training set for selection and downstream training; official clean test for evaluation;
- ResNet-18, 224-pixel inputs, 1000 epochs, final-epoch checkpoint;
- Graph-A2 global selection with `k=50` and `hops=2`;
- deterministic training enabled in the experiment launcher.

## Reproduce the comparison locally

The branch is based on the pinned upstream commit. From a clone of this repository:

```bash
git diff --stat 8cf757adc4c333dc1427d511f0de2f246d15ebac..codex/table1-v11-execution
git diff 8cf757adc4c333dc1427d511f0de2f246d15ebac..codex/table1-v11-execution -- graphcov/
git diff --no-index /path/to/author/graphcov/run/selection.py graphcov/run/selection.py
```

The `graphcov/` diff should be empty if the checked-out branch preserves the pinned upstream implementation. The first command includes all repository work on the branch since that base commit, not just this experiment package. For the scoped reproduction artifacts, review `code_snapshot/`, `../configs/`, `../logs/`, and `../Table1_clean_only_results_20260922.xlsx`.

The experiment wrapper is not a same-path replacement for upstream `graphcov/run/experiment.py`; comparing those two files as if they were competing versions of one module would be misleading. The wrapper calls upstream APIs and adds the benchmark protocol around them.

## Artifact boundary

The package includes source, runtime/launch scripts, actual configs, Slurm logs, and the result workbook. Dataset archives, pretrained UNI weights, embedding caches, final checkpoints, and per-image prediction tensors are intentionally omitted; they are large binary artifacts and are not needed to inspect the source diff.
