# PathMNIST Random/Graph-A2 Cross-Seed Pilot

This pilot uses the frozen reliability checkout at GraphCov commit
`d74e2d6fb6b348ad0cbb410b4931d90540afd754` and MedMNIST-C commit
`8acfd2710c6e0e8b2745be8b1fa1c17b94f8a7`.

## Design

- Dataset: `PathMNIST`
- Methods: `random`, `graph_a2`
- Ratios: `0.02`, `0.05`
- Selection seeds: `42`, `43`
- Training seeds: `42`, `43`
- Full cross: every selection seed is paired with every training seed
- Total: `1 x 2 x 2 x 2 x 2 = 16` train/evaluate cells
- GPU allocation: one `gpu-a100`, 8 CPUs, 48G RAM per cell
- Maximum concurrency: 5 cells

The selection artifacts are reused, not regenerated. Seed 42 artifacts come
from the formal single-seed selection directory; seed 43 artifacts come from
the existing variance-pilot selection directory.

## Entrypoint and Outputs

- Slurm wrapper: `reliability_medmnistc/slurm/cross_path_random_grapha2.slurm`
- Core config: `reliability_medmnistc/configs/full_5datasets_8methods.yaml`
- Formal output root:
  `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/formal_runs/cross_path_random_grapha2_sel42_43_train42_43_20260917`
- Complete results will contain `final.pt`, training history, clean and
  corruption predictions/metrics, and `run_complete.json` for each cell.

The wrapper is only an experiment-control script. It does not change the
Python selection, training, evaluation, or metric implementation.

## GitHub Policy

The repository stores source, configuration, Slurm scripts, manifests, and
compact CSV/JSON summaries. Large checkpoints, prediction arrays, caches, and
raw Slurm logs remain on HPC and are referenced by the output root. They are
not committed to GitHub as experiment source files.
