# GraphCov v11: atomic calibration and frozen downstream evaluation

For the HCP/Slurm deployment, resource limits, and the complete frozen
calibration workflow, see
[`../HPC_TABLE1_V11_EXECUTION_CN.md`](../HPC_TABLE1_V11_EXECUTION_CN.md).

v11 is an experiment package, not a claim that PPR is already better than
Graph-A2. It separates selection, validation calibration, and final test access
so that dataset-specific parameters can be tuned without using the test split.

## Non-negotiable protocol

- Job 1 reads only training UNI embeddings and `train_labels`.
- Job 1 builds global graphs and enforces equal per-class budgets.
- Job 2 loads frozen `selected_indices.npy`; it cannot regenerate a subset.
- Validation chooses one parameterization per dataset at 5%.
- The chosen parameterization is then frozen and reused at both 2% and 5%.
- Test is read only by the generated five-seed confirmation config.
- No image files are copied. All variants share the same MedMNIST archives and
  embedding caches.

## Methods under test

The baseline is the official global Graph-A2 configuration:

```text
k=50, H=2, uniform weights, no truncation, equal class quotas
K = A_hat + A_hat^2
```

The bounded diffusion candidate is:

```text
K = sum_{h=1..H} gamma^(h-1) A_hat^h
```

`gamma=0.85` is the continuation factor corresponding to restart probability
`r=0.15`. The common factor `r` is omitted because multiplying every kernel
entry by the same positive constant does not change facility-location greedy
selection. `per_hop` truncation keeps only the strongest `max_degree` entries
in each row after every sparse multiplication and accumulation.

Margin safety is defined on the full training pool:

```text
gap_i = nearest_different_cosine_distance - nearest_same_cosine_distance
unsafe_i = gap_i <= 0
```

The hard-cap variants constrain each class's unsafe count relative to the
original Graph-A2 subset. The post-hoc variant starts from the exact same
uncapped subset and performs only the number of swaps required to meet that
cap. This distinguishes diffusion from safety instead of changing both at once.

## Experiment map

| Config | Purpose | Downstream runs |
|---|---|---:|
| `job1_derma_atomic.json` | Generate 12 Derma 5% atomic subsets | 0 |
| `job2_derma_atomic_seed42.json` | Cheap first screen | 12 |
| `job2_derma_atomic_3seeds.json` | Full Derma mechanism ablation | 36 |
| `job1_table1_calibration.json` | Generate five-dataset 2%/5% subsets | 0 |
| `job2_table1_validation_seed42.json` | Cheap 5% validation screen | 25 |
| `job2_table1_validation_3seeds.json` | Stable 5% validation calibration | 75 |
| generated test config | Baseline vs frozen winner, 2%/5%, 5 seeds | at most 100 |

The five-dataset target parameters are deliberately different:

| Dataset | Target k | Target H | Max row degree | Reason for bounded search |
|---|---:|---:|---:|---|
| OrganS | 30 | 4 | 70 | medium graph, moderate diffusion |
| OrganA | 20 | 4 | 70 | v10 k=50/H=6 changed the set heavily but gave only a small mean gain |
| Path | 20 | 3 | 70 | UNI graph is already very pure; avoid unnecessary long diffusion |
| Tissue | 15 | 3 | 50 | largest dataset and weaker UNI domain match; control fill-in |
| Blood | 30 | 4 | 70 | small graph permits moderate expansion |

These values are candidates, not reported winners. They were designed when the
baseline was incorrectly assumed to use k=10. Against the corrected k=50 A0,
the legacy ID `a4_larger_k` and the k=10/15/20/30 variants are lower-k
exploratory candidates, not a valid progressive ablation. Existing v11
selection and validation outputs therefore require rerunning in the new
`*_k50baseline` roots, followed by candidate redesign before a novelty claim.

## Server setup

```bash
ssh lab
cd /root/graphcov_pathmnist_sota

# Safe for the existing server worktree: fetch the fork and update only v11/.
git fetch https://github.com/18054546156/graph-coverage-selection.git \
  codex/sync-remote-reproduction
git checkout FETCH_HEAD -- v11

export GRAPHCOV_PYTHON=/root/miniconda3/envs/graphcov_pathmnist/bin/python

$GRAPHCOV_PYTHON v11/experiments/preflight_server.py
$GRAPHCOV_PYTHON -m pytest -q v11/tests
```

For a clean clone, use
`https://github.com/18054546156/graph-coverage-selection.git` and switch to
branch `codex/sync-remote-reproduction`. The original author's `origin` does
not contain this branch.

The preflight checks all six UNI caches, MedMNIST archives, dimensions,
dependencies, and CUDA visibility before any selection starts.

## Job 1: selection only

Derma atomic selection:

```bash
bash v11/scripts/run_job1.sh v11/configs/job1_derma_atomic.json
```

Five Table-1 datasets:

```bash
bash v11/scripts/run_job1.sh v11/configs/job1_table1_calibration.json
```

To run only one dataset or to inspect a config without loading data:

```bash
$GRAPHCOV_PYTHON v11/experiments/job1_select.py \
  --config v11/configs/job1_table1_calibration.json \
  --only-dataset pathmnist

$GRAPHCOV_PYTHON v11/experiments/job1_select.py \
  --config v11/configs/job1_table1_calibration.json --dry-run
```

Each subset is written as:

```text
v11/outputs/<job1>/DATASET/r0p05/VARIANT/
  selected_indices.npy
  selection_order.npy
  selection_metrics.json
  variant_config.json
```

`selection_metrics.json` records index hashes, Jaccard with Graph-A2, exact
margin statistics, self-kernel and Graph-A2 coverage, kNN purity, graph
sparsity, cross-class mass, and asymmetry.

Derma 5% and OrganA 5% also have required real-data reproduction checks against
the frozen official Graph-A2 indices produced by the isolated Table 1 pipeline.
For Table 1 calibration, `reference_mode: frozen` makes Job 1 copy those arrays
directly into `a0_original`; v11 generates only the modified candidate subsets.
The source path and array hashes are recorded in every A0 selection audit.

## Job 2: frozen ResNet-18 evaluation

Check GPU availability before launch:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
```

Launch deterministic shards in two screen sessions. Replace the config path as
needed:

```bash
screen -dmS v11_val_gpu0 bash -lc \
  'cd /root/graphcov_pathmnist_sota && bash v11/scripts/run_job2_gpu0.sh v11/configs/job2_table1_validation_3seeds.json'

screen -dmS v11_val_gpu1 bash -lc \
  'cd /root/graphcov_pathmnist_sota && bash v11/scripts/run_job2_gpu1.sh v11/configs/job2_table1_validation_3seeds.json'

screen -ls
tail -f v11/logs/job2_table1_validation_3seeds_gpu0.log
```

The three-seed config uses the same output root as the seed-42 config, so
completed seed-42 runs are skipped rather than repeated.

Job 2 defaults to validation regardless of the selected config. Reading the
test split requires both a test config and the explicit CLI authorization
`--evaluation-split test`; otherwise the process stops before loading data.

On the Lingnan Slurm cluster, the auditable end-to-end chain is:

```bash
VALIDATION_JOB=$(sbatch --parsable v11/slurm/validation_3seeds_array.slurm)
A2_JOB=$(sbatch --parsable table1_reproduction/slurm/graph_a2_3seeds_downstream.slurm)
FREEZE_JOB=$(sbatch --parsable --dependency=afterok:${VALIDATION_JOB} v11/slurm/freeze_table1_winners.slurm)
sbatch --dependency=afterok:${FREEZE_JOB} v11/slurm/frozen_test_array.slurm
```

The freeze job requires a mean best-validation-BA gain of at least 0.5
percentage points and permits at most a 2 percentage point loss in the
mean worst-class recall at the validation-selected checkpoint. It
writes the immutable decision manifest and generated test config below
`v11/outputs/frozen_protocol/`. The final test uses seeds 42--46 for both the
frozen A0 subset and the selected v11 winner at 2% and 5%.

Summarize completed runs:

```bash
bash v11/scripts/summarize_job2.sh \
  v11/outputs/job2_table1_validation_k50baseline_valckpt
```

The primary output is the balanced accuracy at the validation-selected
checkpoint. Outputs also retain validation BA, accuracy, worst-class recall,
class-CVaR20, per-class recall, training history, elapsed time, checkpoint path,
and frozen selection hashes.
The strict v11 runner persists `best_val_checkpoint.pt`. It trains while
checking validation only, then loads that checkpoint. Validation calibration
does not read test; frozen confirmation reads test once after training. All
reported per-class diagnostics are from the validation-selected checkpoint.

## Freeze validation winners, then read test

Run this only after all 75 validation runs exist:

```bash
$GRAPHCOV_PYTHON v11/experiments/freeze_validation_winners.py \
  --validation-root v11/outputs/job2_table1_validation_k50baseline_valckpt \
  --validation-config v11/configs/job2_table1_validation_3seeds.json \
  --selection-root v11/outputs/job1_table1_calibration_k50baseline \
  --output-config v11/configs/generated_job2_table1_test.json \
  --test-output-root v11/outputs/job2_table1_test_valckpt
```

The default freeze rule selects the highest mean best validation BA candidate
only when its mean worst-class recall at the validation-selected checkpoint is no
more than 2 percentage points below the Graph-A2 baseline. If no candidate
passes, Graph-A2 remains the winner. The
script writes both the test config and a `freeze_manifest.json` decision audit.

Then launch the generated config on both GPUs using the same Job-2 scripts.
Do not modify its dataset-specific winners after seeing test results.

## Code layout

```text
v11/
  configs/                 explicit parameter combinations
  methods/selection.py     graph, diffusion, margins, constrained greedy
  experiments/job1_select.py
  experiments/job2_downstream.py
  experiments/freeze_validation_winners.py
  experiments/summarize_downstream.py
  scripts/                 server launchers
  tests/                   synthetic and config tests
```

`--force` intentionally recomputes and overwrites a completed stage. Without
it, Job 2 is resumable and skips result JSON files that already exist.
