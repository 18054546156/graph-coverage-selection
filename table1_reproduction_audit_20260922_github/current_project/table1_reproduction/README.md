# Isolated Table 1 reproduction

For the HCP/Slurm deployment and the ordered Table 1 -> v11 workflow, see
[`../HPC_TABLE1_V11_EXECUTION_CN.md`](../HPC_TABLE1_V11_EXECUTION_CN.md).

This directory is the reproducibility package for the paper's Table 1. It is
separate from `v11/`, which contains the PPR/margin research experiments, and
it leaves room for a future `v12/` new-selector package.

The Table 1 jobs do not import the mutable project-level `graphcov` package.
They first verify and import the author-code snapshot at `vendor/graphcov/`,
pinned to upstream commit
`8cf757adc4c333dc1427d511f0de2f246d15ebac`. The file hashes are checked at
startup and the job fails closed if the snapshot changes. Edits to the root
`graphcov/`, `v11/`, or future `v12/` code therefore cannot silently change a
Table 1 comparison method.

## What is and is not Table 1

The paper's pasted Table 1 has exactly these eight columns:

```text
Random, EL2N, Forgetting, EVA, Facility, FPS, Herding, Graph-A2/Ours
```

`Heat Kernel` and `Graph Coverage` are registered repository diagnostics, but
they are not columns in the pasted Table 1. They are configured separately in
`configs/job1_extra_diagnostics.json` and write below
`outputs/job1_extra_diagnostics/`.

The new PPR/margin method is not called Graph-A2 and must not replace the
`graph_a2` Table 1 column. It remains under `v11/` until a future method is
formally defined and, if needed, placed in `v12/`.

## Protocol encoded here

The config uses the paper's stated protocol:

- five datasets: OrganSMNIST, OrganAMNIST, PathMNIST, TissueMNIST, BloodMNIST;
- UNI 224x224, 1024-dimensional cached embeddings;
- balanced per-class budgets;
- ResNet-18 from scratch at 224x224;
- SGD, learning rate 0.1, momentum 0.9, weight decay 5e-4;
- cosine annealing, batch size 256, no augmentation;
- 1000 epochs;
- training seeds 42, 43, 44, 45, 46;
- test balanced accuracy is the Table 1 metric.

The exact evidence and unresolved assumptions are recorded in
[`CONFIG_EVIDENCE.md`](CONFIG_EVIDENCE.md).

The repository supplies details that the paper text does not fully spell out:

- paper Table 2 identifies the Table 1 Graph-A2 setting as global kNN,
  `k=50`, `H=2`, and `K=A_hat+A_hat^2`;
- Facility uses the public runner's per-class full-cosine implementation. A
  global full-cosine matrix is not a viable Table 1 job for Path/Tissue on a
  24 GiB 3090; making Facility global would be a different engineering
  implementation and is not silently substituted here;
- dynamics scores use the repository's 28x28, 200-epoch, window-10 path;
- the effective budget is
  `floor(floor(n_train * ratio) / n_classes)` per class;
- to match the existing official runner, Random and FPS use a different
  selection seed for each training trial, while deterministic methods reuse a
  seed-42 subset and only vary the downstream training seed.

Facility scope, dynamics details, and seed/budget behavior are implementation
assumptions; the Graph-A2 k=50/global/H=2 setting is paper-supported.

## Two jobs

Job 1 reads only the training split. It loads the existing UNI cache and, for
EL2N/Forgetting/EVA, creates or loads one training-dynamics cache. It never
loads validation or test data and saves frozen indices at:

```text
table1_reproduction/outputs/job1_selection_k50_global/
  DATASET/r0p02/METHOD/seed42/selected_indices.npy
  DATASET/r0p02/METHOD/seed42/selection_metrics.json
```

Random/FPS additionally have `seed43` through `seed46` because their selection
seed follows the training seed. Each directory also contains the exact
selection order and a sorted copy for audit.

Job 2 never regenerates a subset. It verifies the index hash, range, uniqueness,
and class quota, then trains ResNet-18. During training it reads only the
validation split, saves the checkpoint with the highest validation balanced
accuracy, reloads that checkpoint, and evaluates the test split exactly once.
The strict-run outputs are:

```text
table1_reproduction/outputs/job2_table1_k50_global_valckpt/
  DATASET/r0p02/METHOD/seed42/test_result.json
  DATASET/r0p02/METHOD/seed42/best_val_checkpoint.pt
  DATASET/r0p02/METHOD/seed42/history.json
  DATASET/r0p02/METHOD/seed42/per_class.json
```

`test_result.json` reports test balanced accuracy at the validation-selected
checkpoint. It does not report the maximum test score observed during
training. Previous k=10 roots, including `outputs/job1_selection_clean/` and
`outputs/job2_table1_valckpt/`, are historical and must not be merged into the
corrected strict summary.

The complete workload is `5 x 2 x 8 x 5 = 400` downstream runs. Selection is
much smaller: 160 frozen selections under the seed policy above, with dynamics
shared within each dataset.

## Isolation rule

`v11/` and `v12/` may use their own selectors, configs, caches, and output
directories. Their results must not be copied into
`table1_reproduction/outputs/`, and the `vendor/` snapshot must not be edited
for ordinary method development. A deliberate upstream-version change is a
new reproduction-package revision with a new pinned commit and regenerated
manifest; it is not a v11/v12 modification.

## Server commands

On the 3090 server:

```bash
cd /root/graphcov_pathmnist_sota
export GRAPHCOV_PYTHON=/root/miniconda3/envs/graphcov_pathmnist/bin/python

# Preflight, then inspect the workload
$GRAPHCOV_PYTHON table1_reproduction/experiments/preflight.py \
  --config table1_reproduction/configs/job1_table1.json
$GRAPHCOV_PYTHON table1_reproduction/experiments/job1_select.py \
  --config table1_reproduction/configs/job1_table1.json --dry-run
$GRAPHCOV_PYTHON table1_reproduction/experiments/job2_downstream.py \
  --config table1_reproduction/configs/job2_table1.json --dry-run

# Job 1: selection only
bash table1_reproduction/scripts/run_job1.sh \
  table1_reproduction/configs/job1_table1.json

# Job 2: check both GPUs first, then launch two resumable shards
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
screen -dmS table1_gpu0 bash -lc \
  'cd /root/graphcov_pathmnist_sota && bash table1_reproduction/scripts/run_job2_gpu0.sh table1_reproduction/configs/job2_table1.json'
screen -dmS table1_gpu1 bash -lc \
  'cd /root/graphcov_pathmnist_sota && bash table1_reproduction/scripts/run_job2_gpu1.sh table1_reproduction/configs/job2_table1.json'

# After jobs finish
bash table1_reproduction/scripts/summarize.sh
```

Start with a smoke run before the full workload:

```bash
$GRAPHCOV_PYTHON table1_reproduction/experiments/job1_select.py \
  --config table1_reproduction/configs/job1_table1.json \
  --only-dataset pathmnist --only-method graph_a2
$GRAPHCOV_PYTHON table1_reproduction/experiments/job2_downstream.py \
  --config table1_reproduction/configs/job2_table1.json \
  --only-dataset pathmnist --only-method graph_a2 --only-ratio 0.02 \
  --dry-run
```

Do not launch the full Job 2 until Job 1 has finished and its selection
manifest shows the expected quotas. The summarizer writes
`outputs/job2_table1_k50_global_valckpt/summary/table1_reproduction.md` and a machine-readable
`per_condition_mean_std.csv`.

## Local verification

```bash
python -m pytest -q table1_reproduction/tests
python -m compileall -q table1_reproduction/experiments
```
