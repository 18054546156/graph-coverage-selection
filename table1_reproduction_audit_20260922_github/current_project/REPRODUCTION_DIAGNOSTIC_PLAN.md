# PathMNIST 5% Reproduction Diagnostic

Date: 2026-09-18  
Scope: PathMNIST 5%, Random and Graph-A2, clean training first  
Purpose: explain the difference between the paper Table 1 and the current reproduction before extending the robustness study.

## Current Status

- `HPC_USER_A` and `HPC_USER_B` have no active training jobs.
- The existing formal selection root contains `5 datasets x 8 methods x 2 ratios = 80` selection artifacts.
- The offline selection audit passed `80/80` artifacts.
- The audit checked selected-index hashes, class counts, source index and label hashes, UNI embedding file hashes, and Graph-A2 configuration (`global=true`, `k=50`, `H=2`).
- No new diagnostic training job has been submitted yet.
- Existing formal results remain untouched.

Audit output:

```text
/project/PROJECT_ROOT/USER/reliability_medmnistc_ab/results/selection_audit_20260918/selection_audit.json
/project/PROJECT_ROOT/USER/reliability_medmnistc_ab/results/selection_audit_20260918/selection_audit.csv
```

## Fixed References

Current reliability pipeline:

```text
/project/PROJECT_ROOT/USER/reliability_medmnistc_ab/sources/graph-coverage-selection-medmnistc-reliability-clean-d74
HEAD=d74e2d6fb6b348ad0cbb410b4931d90540afd754
GraphCov base=8cf757adc4c333dc1427d511f0de2f246d15ebac
MedMNIST-C=8acfd2710c6e0e8b2745be8b1fa1c17b94f8a7
```

Author-code checkout used for the code-path audit:

```text
/project/PROJECT_ROOT/USER/reliability_medmnistc_ab/sources/graph-coverage-selection-180545-b66
HEAD=b66ae4406e2d8cfa1f1d0eac22a6b3f65040ef17
```

The author README entry point is `python -m graphcov.run`. The current formal entry point is `reliability_medmnistc/scripts/train.py` and `evaluate.py`, both routed through `run_pipeline.py` and `reliability/pipeline.py`.

## Step 1: Code-Path Audit

### Data, split, and preprocessing

Both paths load MedMNIST train/test splits at size 224 and use the standard no-augmentation transform based on `ToTensor()` and per-channel mean/std `0.5`. The current pipeline records the source train indices and labels in each selection manifest. The 80-artifact audit found no source-index or class-count mismatch.

The remaining data provenance checks are:

1. Record SHA256 for every official train/test NPZ used by both entry points.
2. Verify that the author entry and current pipeline resolve the same MedMNIST data directory.
3. Verify that labels and image ordering are identical before applying transforms.
4. Verify that the same UNI embedding file is used and that its row order matches the train labels.

### Selected indices and order

The existing formal selection artifacts are internally consistent. For every artifact:

- `selected_indices.npy` exists;
- `selected_local_indices.npy` exists;
- the stored order hash equals the recomputed array hash;
- the class count file equals a recomputation from the official train labels;
- the source-label and source-index hashes match the formal pipeline convention.

This proves internal selection integrity. It does not yet prove that the author run produced the same Random subset, because the author run does not expose a matching selection manifest for every formal artifact. That comparison is a separate selection-seed audit.

### RNG and deterministic execution

This is the first confirmed path discrepancy.

Author entry:

```text
graphcov/run/evaluation.py
set_seed(seed, deterministic=deterministic)
```

The author Table 1 Slurm script passes:

```text
--deterministic
```

The author `set_seed(..., deterministic=True)` sets cuDNN deterministic mode and disables cuDNN benchmark mode.

Current formal pipeline:

```text
reliability_medmnistc/reliability/pipeline.py
train_model(...) -> set_seed(seed, deterministic=False)
```

The Slurm script exports `DETERMINISTIC=1`, but the current pipeline does not read that variable at the training call. Therefore the export alone has no effect. The current path leaves cuDNN benchmark enabled and does not enable cuDNN deterministic mode.

This is a real reproducibility difference and must be fixed in the diagnostic checkout before comparing the two paths.

Neither path currently calls `torch.use_deterministic_algorithms(True)`. That stronger setting should not be introduced silently into the first comparison; it should be recorded as a separate environment factor if needed.

### Model initialization and DataLoader

Both paths set Python, NumPy, and Torch seeds before constructing the training DataLoader and model. Both use:

```text
batch_size=256
num_workers=4
shuffle=True for training
shuffle=False for test
drop_last=False
```

The diagnostic run must additionally record:

- the initial model state tensor hash immediately after initialization;
- the first epoch batch-index order;
- the worker seed configuration and Torch initial seed;
- the actual number of batches per epoch.

### Update count, optimizer, and scheduler

For PathMNIST 5%:

```text
selected samples = 4491
batch size = 256
batches per epoch = ceil(4491 / 256) = 18
epochs = 1000
updates = 18000 per task
```

Both paths use SGD with learning rate `0.1`, momentum `0.9`, weight decay `0.0005`, and cosine annealing stepped once per epoch when using the epoch paradigm.

The author path also supports an iteration paradigm. The diagnostic comparison must explicitly use `--training-paradigm epoch --epochs 1000`; the README default of 1000 iterations is a different protocol and must not be mixed into the Table 1 comparison.

### Evaluation during training

The author epoch path evaluates the test set every 10 epochs and records the best test BA, while continuing training. The current formal pipeline trains through all 1000 epochs without intermediate test evaluation and evaluates the final checkpoint afterward.

The author intermediate evaluation uses a non-shuffled test DataLoader. It should not change model weights, but its worker creation and RNG behavior must be checked empirically in the four-task comparison. The current path's lack of intermediate evaluation is a confirmed execution-path difference in logging and possible RNG consumption.

### Checkpoint and clean metrics

The current formal result is explicitly `final_epoch`. The author path returns both final metrics and best metrics. The comparison must save both values when available but use final BA as the primary matched metric unless the paper protocol proves otherwise.

Checkpoint file SHA256 verifies file integrity only. To test whether two checkpoints represent the same model, compare the ordered state-dict tensor hashes and the saved configuration hash.

## Step 1 Result

The selection artifacts are internally valid, but the training paths are not yet equivalent. The deterministic flag is exported in the current Slurm script but ignored by the current training call. The author path also performs intermediate test evaluation while the current path does not.

Therefore the existing Random-versus-Graph-A2 ranking cannot yet be used as an exact author-entry reproduction. The first correction is to make deterministic behavior an explicit, recorded current-pipeline setting in an isolated diagnostic checkout.

## Step 2: Four-Task Clean Comparison

After the deterministic setting is corrected and recorded, run only:

| Entry | Method | Selection | Training |
|---|---|---|---|
| author | Random | seed 42 | seed 42 |
| current pipeline | Random | seed 42 | seed 42 |
| author | Graph-A2 | seed 42 | seed 42 |
| current pipeline | Graph-A2 | seed 42 | seed 42 |

All four tasks use the same PathMNIST 5% selected index arrays, same index order, same data/cache, same GPU type, and the same epoch protocol. Outputs go to a new diagnostic root and do not overwrite formal runs.

Each task must save:

```text
effective software/config versions
selected index SHA256
initial model state tensor hash
first epoch batch-order hash
training loss and learning-rate history
number of optimizer updates
final checkpoint and state-dict hashes
clean predictions
clean ACC and BA
```

Interpretation:

| Observation | First check |
|---|---|
| initial model hash differs | seed placement and model initialization |
| initial model matches, batch order differs | index order, DataLoader generator, worker seeds |
| model and batch order match, first update differs | input transform, model, optimizer, or numerical settings |
| checkpoint matches, metrics differ | evaluation transform, label order, or metric code |

If a concrete discrepancy is found, correct it and rerun the affected comparison before adding more seeds. If the four tasks are still unresolved, add training seed 43 for the same four paths, for eight tasks total.

## Step 3: Selection-Randomness Check

Only after the entry-point comparison is consistent:

```text
PathMNIST 5%
x {Random, Graph-A2}
x selection_seed {42, 43}
x training_seed {42, 43, 44, 45, 46}
= 20 cells
```

Existing cells are reused only when their configuration hash, selection hash, checkpoint rule, and training protocol match this document. Otherwise they remain historical evidence and are not used to fill the matrix.

Report per selection seed:

- mean and standard deviation across training seeds;
- paired Graph-A2 minus Random differences;
- clean BA and corruption BA separately;
- corruption severity breakdown;
- worst-class recall.

Two selection seeds provide an initial sensitivity check, not a stable estimate of the full selection distribution.

## Step 4: Environment A/B and Robustness

Environment comparison is conditional on a discrepancy that remains after the entry-point audit. Fix code, selected indices, training seed, data/cache, GPU type, and epoch protocol. Change one software factor at a time, such as the PyTorch/CUDA environment.

Start with two additional clean tasks, one per method. Do not change multiple environment variables in one comparison.

Once a checkpoint is accepted, robustness requires no new training:

```text
same checkpoint
-> clean prediction
-> every corruption
-> severities 1..5
-> validate labels and sample ordering
-> recompute ACC, BA, BA drop, and worst-class recall
```

The report must state the exact corruption aggregation rule. A small BA drop is not sufficient evidence of higher robustness; absolute corruption BA and clean BA must be reported together.

## Compute Budget

| Stage | Full training tasks | Expected GPU use |
|---|---:|---|
| code/data/selection audit | 0 | CPU/read-only analysis |
| initial author/current comparison | 4 | 4 GPUs, 32 CPUs, 192G RAM |
| optional second training seed | +4 | same |
| selection 42/43 matrix | up to 20 cells, reuse compatible cells | up to 5 GPUs in batches |
| one-factor environment check | +2 initially | 2 GPUs |
| robustness audit | 0 training tasks | reuse checkpoints; inference only if needed |

## Current Decision

Do not interpret the current ranking as a clean author-entry reproduction until the deterministic mismatch and intermediate-evaluation difference are addressed. Do not select seeds because they make Random look closer to the paper. Use the predeclared seed matrix and report every completed cell.

## Execution Log (2026-09-18)

The shared record for this plan is:

```text
/project/PROJECT_ROOT/USER/reliability_medmnistc_ab/results/reproduction_diagnostic_20260918
```

The isolated current checkout is:

```text
sources/graph-coverage-selection-medmnistc-reliability-diagnostic-d74-20260918
HEAD=d74e2d6fb6b348ad0cbb410b4931d90540afd754
```

The isolated author checkout is:

```text
sources/graph-coverage-selection-author-diagnostic-b66-20260918
HEAD=b66ae4406e2d8cfa1f1d0eac22a6b3f65040ef17
```

The four-way Slurm script is saved at:

```text
results/reproduction_diagnostic_20260918/reproduction_diagnostic_4way.slurm
```

Job `32368` was submitted with four tasks. Its current-entry tasks initially failed before training because the diagnostic script omitted the existing ImageMagick initialization; the error was `MagickWand shared library not found`. This is retained in `logs/fourway_32368_1.err` and `logs/fourway_32368_3.err` as execution evidence. It was a harness environment error, not a model or selection result.

The script was corrected to source `common_env.sh` and call `configure_imagemagick`. Only the failed current-entry array elements were resubmitted as job `32372` (`--array=1,3`); the author tasks `32368_0` and `32368_2` were still running and were not duplicated.

At the last check, `32368_0`, `32368_2`, `32372_1`, and `32372_3` were running. No clean metric or checkpoint from this diagnostic comparison is accepted yet. The next record update must include each task's status, trace hashes, final ACC/BA, and any first-divergence classification.

### First trace evidence

The restarted current-entry tasks have reached model initialization and the first epoch. Both report:

```text
n_selected=4491
batches_per_epoch=18
first_epoch_order_length=4491
deterministic=true
cudnn_deterministic=true
cudnn_benchmark=false
```

The initial model and first-epoch order hashes are identical across the two current-entry tasks:

```text
initial_model_state_sha256 = d82ed31ce5592366e909a7fba784551c6764efbbfabd657d7ca16d86669a74b8
first_epoch_batch_order_sha256 = ba5b551ab4ba3ab62b4c7439b4c9abaa8bee6acd1360d145140e53bb069dfbcc
```

This is an internal control result: current Random and current Graph-A2 use the same training initialization and loader order. Their selected-index hashes differ as expected:

```text
Random   8e82b8d02b5daa8cde32bb701668b76db23b29ea6b80c063641c11014dbcf4e4
Graph-A2 c0f91b1658fab2334202cdfe2537137f097fa882010ba77146239e93fdaa4b90
```

This does not yet compare author versus current. That comparison is valid only after the two author tasks write the same trace fields and all four tasks finish.

The first author attempt (`32368_0` and `32368_2`) loaded the cached PathMNIST file but stopped before training because the driver used `train_ds=` instead of the author's `train_dataset=` parameter. This was corrected in the isolated driver. Only author elements `0,2` were resubmitted as job `32374`; no model output from the failed attempt is treated as data.

## Isolated `HPC_USER_B` Environment A/B

The first `HPC_USER_B` venv attempt (`32377`) was cancelled and is invalid: it inherited the baseline package versions and also stopped on shared-checkout Git ownership protection. Its output is not used.

The valid environment A/B is a separate Conda prefix:

```text
environment:
/home/HPC_USER_B/prj_HPC_USER_B/conda_envs/medmnistc-torch241-cu121

results/logs/manifests:
/home/HPC_USER_B/prj_HPC_USER_B/reproduction_diagnostic_torch241_cu121_20260918
```

The valid environment is pinned to:

```text
Python 3.11
PyTorch 2.4.1
torchvision 0.19.1
pytorch-cuda 12.1
MedMNIST 3.0.1
NumPy 1.26.4
Pandas 3.0.5
Pillow 12.3.0
```

The `HPC_USER_A` baseline is `PyTorch 2.5.1`, `torchvision 0.20.1`, CUDA 12.1, and the same listed non-Torch package versions. Thus this A/B changes the PyTorch/torchvision stack while keeping code commit, GraphCov/MedMNIST-C commits, selection arrays, selection seed 42, training seed 42, GPU partition, batch size, workers, optimizer, scheduler, deterministic setting, and 1000 epochs fixed.

The valid isolated Slurm script is:

```text
/home/HPC_USER_B/prj_HPC_USER_B/reproduction_diagnostic_torch241_cu121_20260918/reproduction_diagnostic_qiang_torch241.slurm
```

Job `32381` was submitted with array throttle `%1`. Its four tasks are the same author/current x Random/Graph-A2 comparison as the baseline, but all outputs are private to `HPC_USER_B`. The earlier invalid venv job and results remain separate and are excluded from analysis.
