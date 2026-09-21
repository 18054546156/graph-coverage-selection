# PathMNIST ratio sweep

Unified index for the existing PathMNIST results at 1%, 2%, 5%, and 10%.

The HPC directory `results/pathmnist_ratio_sweep/` contains links to the
original result directories. The original directories are preserved.

## Protocol status

| Ratio | Methods | Clean test | PathMNIST-C | Provenance |
|---|---|---|---|---|
| 1% | Random, Graph-A2, Herding, FPS | Yes | 11 x 5 | Current reliability protocol |
| 2% | Random, Graph-A2, EL2N, Forgetting | Yes | No | Legacy Table 1 run; final and best fields are both present |
| 5% | Random, Graph-A2, Herding, FPS | Yes | 11 x 5 | Current reliability protocol |
| 10% | Random, Graph-A2, Herding, FPS | Yes | 11 x 5 | Current reliability protocol |

The 2% directory is intentionally isolated because it used the older Table 1
runner and is not directly comparable with the current corruption protocol.

## Code snapshot

The unified HPC directory also contains `code/`, a copy of the files used to
launch and audit this sweep:

- `code/slurm/v29_path_full_benchmark.slurm`
- `code/slurm/v29_integration_smoke.slurm`
- `code/scripts/download_medmnist_224.py`
- `code/scripts/generate_medmnistc.py`
- `code/scripts/generate_medmnistc_224.slurm`
- `code/notebooks/reliability_full_experiment.ipynb`
- `code/protocol/` configuration and manifests
- `code/source_snapshots/graphcov_8cf757a/`
- `code/source_snapshots/medmnistc_8acfd271/`

For easier browsing, the same source snapshots are also grouped by function:

- `code/selection_algorithms/graphcov_8cf757a/` contains the selector
  implementation used by Random, EL2N, Forgetting, EVA, Facility, FPS,
  Herding, and Graph-A2.
- `code/robust_data/medmnistc_8acfd271/` contains the original MedMNIST-C
  corruption registry, corruption implementations, dataset manager, and
  corrupted-dataset reader.

The source snapshots are copies for audit convenience. The Slurm benchmark
continues to execute from the pinned source checkout under `sources/`.

## Current reliability protocol

- Dataset: PathMNIST, 224 x 224.
- Selection seed: 42.
- Training seed: 42.
- Checkpoint: final epoch `final.pt`.
- Corruption evaluation: same checkpoint, 11 corruption types, severities 1-5.
- `metrics.csv`: one clean row plus 55 corruption rows.
