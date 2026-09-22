# Table 1 Clean-Only Execution Snapshot

This folder is the code and launcher snapshot used for the completed clean-only Table 1 benchmark. It is an experiment wrapper around the pinned upstream GraphCov implementation.

## Protocol actually run

- Five datasets: OrganSMNIST, OrganAMNIST, PathMNIST, TissueMNIST, BloodMNIST.
- Eight methods: Random, EL2N-top, Forgetting, EVA, Facility, FPS, Herding, Graph-A2.
- Eight paired seeds: `0, 1, 2, 3, 4, 5, 6, 2026`; selection seed equals training seed.
- A 2% subset of clean training data is selected, then used for clean training; evaluation uses the official clean test split.
- ResNet-18, 224-pixel input, 1000 training epochs; final-epoch checkpoint.
- Deterministic training is enabled by the launcher.
- Graph-A2 uses global selection, `k=50`, `hops=2`.
- Total: 5 x 8 x 8 = 320 model cells, all complete.

No corruption or validation-selected checkpoint enters this Table 1 result set.

## Files

- `reliability_medmnistc/pipeline.py`: experiment orchestration and audit records; imports selection, embeddings, transforms, and training primitives from upstream GraphCov.
- `reliability_medmnistc/scripts/`: phase entry points for selection, training, evaluation, and summarization.
- `reliability_medmnistc/slurm/`: shared runtime environment and helper jobs.
- `run_clean_only.slurm`: exact formal/smoke Slurm launcher. In formal mode it writes the actual per-run config and runs all eight methods for one dataset and one seed.
- `submit_formal_queue.sh`: queue worker used to submit dataset/seed cells.
- `configs/`: source defaults; the authoritative 40 executed configurations are in the sibling `../configs/` directory.
- `manifests/`: upstream commit, protocol metadata, and runtime package versions.
- `PROVENANCE.md`, `UPSTREAM_COMPARISON.md`, and `RUNTIME_ENVIRONMENT.md`: source origins/hashes, comparison instructions, software versions, resource allocations, and GPU measurement limits.

## Reproduce or inspect

Start with `../HANDOFF.md`, then inspect the exact executed configuration and logs in `../configs/` and `../logs/`. The formal output workbook is `../Table1_clean_only_results_20260922.xlsx`.

The run scripts contain cluster-specific paths and are provided as provenance; they are not directly portable without adapting paths, Slurm account/QOS, environment, and dataset/cache locations. The dataset files, pretrained UNI weights, embedding caches, model checkpoints, and prediction tensors are not part of this code snapshot.
