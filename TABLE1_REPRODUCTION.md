# Table 1 forensic reproduction protocol

This is the strictest configuration reconstructable from the paper and released
code. It must not be described as author-confirmed in every detail.

## What is established

| Item | Reconstructed setting | Status |
| --- | --- | --- |
| Data | MedMNIST v2 official train/test splits, 224 x 224 | Paper |
| Embedding | frozen `hf-hub:MahmoodLab/UNI`, 1024 dimensions | Paper/code |
| Graph-A2 | global graph, `A_sym + A_sym^2`, equal class budgets | Paper/code |
| Budget | 2% and 5%, implemented as `floor(floor(N*r)/C) * C` | Code |
| Classifier | torchvision ResNet-18, random initialization | Paper/code |
| Optimizer | SGD, lr 0.1, momentum 0.9, weight decay 5e-4 | Paper/code |
| Training | 1000 epochs, batch 256, cosine schedule, no augmentation | Paper |
| Trials | base seed 42 with trials 0..4 gives seeds 42..46 | Code reconstruction |
| Metric | test balanced accuracy, mean and sample std across five trials | Paper/code |

`k=50` is directly identified for the BloodMNIST and OrganSMNIST Table 1 cells:
their Table 2 `k=50` values match Table 1 digit for digit. Using `k=50` for
OrganAMNIST, PathMNIST, and TissueMNIST is the strongest common-setting
reconstruction, but is still an inference because their Table 1 `k` values are
not separately disclosed.

The paper also does not disclose whether Table 1 uses final epoch weights or a
checkpoint selected from periodic test evaluations. The released main runner
writes final metrics to `balanced_accuracy` and periodic maxima to
`best_balanced_accuracy`. Use the final column as the primary released-code
result and retain both columns for forensic comparison. Do not tune on test.

## Why the commands must be split

The CLI `--global` flag changes both `graph_a2` and `facility`. The released
README defines Facility with its default per-class branch, while Graph-A2's main
claim uses the global graph. Running both under one `--global` command silently
changes the Facility baseline.

## Preflight

```bash
cd /root/graphcov_pathmnist_sota/reproduction-official-code
python -m graphcov.run --list-methods
python -m pytest -q tests/test_artifacts.py
df -h .
nvidia-smi
```

Reserve at least 25 GB for 400 final ResNet-18 checkpoints plus metadata and
logs. The runner uses one GPU per process; it does not distribute one command
over multiple GPUs.

## Group A: Graph-A2 and unaffected geometry methods

`--global` affects Graph-A2; Random, FPS, and Herding ignore it.

```bash
python -u -m graphcov.run \
  --datasets organsmnist organamnist pathmnist tissuemnist bloodmnist \
  --methods random fps herding graph_a2 \
  --embeddings uni \
  --ratios 0.02 0.05 \
  --trials 5 --seed 42 \
  --k-neighbors 50 --global --sparse-cpu \
  --training-paradigm epoch --epochs 1000 \
  --test-every-n-epochs 10 \
  --batch-size 256 --lr 0.1 --momentum 0.9 \
  --weight-decay 0.0005 --size 224 --num-workers 4 \
  --output /root/graphcov_pathmnist_sota/results/table1_artifacts_k50 \
  -v
```

## Group B: Facility and training-dynamics baselines

There is deliberately no `--global` flag in this command.

```bash
python -u -m graphcov.run \
  --datasets organsmnist organamnist pathmnist tissuemnist bloodmnist \
  --methods facility el2n_top forgetting eva \
  --embeddings uni \
  --ratios 0.02 0.05 \
  --trials 5 --seed 42 \
  --k-neighbors 50 \
  --eva-epochs 200 --eva-window-size 10 --eva-size 28 \
  --training-paradigm epoch --epochs 1000 \
  --test-every-n-epochs 10 \
  --batch-size 256 --lr 0.1 --momentum 0.9 \
  --weight-decay 0.0005 --size 224 --num-workers 4 \
  --output /root/graphcov_pathmnist_sota/results/table1_artifacts_k50 \
  -v
```

Together these commands run 5 datasets x 2 budgets x 8 methods x 5 trials =
400 downstream trainings. EVA/EL2N/Forgetting additionally share a cached
200-epoch dynamics run per dataset and scoring resolution.

## Output and acceptance checks

The combined root CSV is:

```text
/root/graphcov_pathmnist_sota/results/table1_artifacts_k50/results.csv
```

Each command creates a separate run directory:

```text
.../runs/<run_id>/config.json
.../runs/<run_id>/training_history.csv
.../runs/<run_id>/per_class_results.csv
.../runs/<run_id>/artifacts/...
```

For every successful CSV row, verify the path in its `artifact_dir` column:

```bash
python - <<'PY'
from pathlib import Path
from graphcov.run.artifacts import verify_artifacts

root = Path('/root/graphcov_pathmnist_sota/results/table1_artifacts_k50')
artifact_dirs = [p.parent for p in root.glob('runs/*/artifacts/**/final_model.pt')]
assert len(artifact_dirs) == 400, len(artifact_dirs)
for path in artifact_dirs:
    verify_artifacts(path)
print(f'verified {len(artifact_dirs)} artifact sets')
PY
```

Formal acceptance requires 400 result rows, 400 verified index files, 400
verified final checkpoints, five seeds per dataset-method-budget cell, and no
failed configuration in each run's `summary.json`.

