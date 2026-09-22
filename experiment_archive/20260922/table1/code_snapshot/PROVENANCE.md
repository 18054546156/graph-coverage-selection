# Table 1 execution provenance

Snapshot date: 2026-09-22 18:20 HKT

## Executed source and launcher

- HPC source root: `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/sources/table1_clean_only_20260922/`
- Python package: `reliability_medmnistc/`
- Formal launcher: `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_20260922/run_clean_only.slurm`
- Queue worker: `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_20260922/submit_formal_queue.sh`
- Result root: `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_20260922/formal/`
- Log root: `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_20260922/logs/`
- Per-run configuration snapshots: `../configs/<dataset>/seed_<seed>/config.yaml`

The files in this directory are copied from the HPC execution tree. `reliability_medmnistc/pipeline.py` and `reliability_medmnistc/scripts/` contain the experiment orchestration and phase entry points. `reliability_medmnistc/slurm/` contains shared environment helpers. The formal launcher and queue worker are included at this directory's root. The 40 YAML files in the sibling `../configs/` directory are the actual dataset/seed run configurations; the YAMLs here are source defaults and are not substitutes for those run configurations.

## Table 1 protocol represented by this snapshot

- Five clean MedMNIST datasets x eight selection methods x eight paired seeds = 320 completed model cells.
- Ratio 2%; selection seed equals training seed; seeds `0, 1, 2, 3, 4, 5, 6, 2026`.
- Clean training data for selection and downstream training; official clean test for final evaluation.
- ResNet-18, input size 224, 1000 epochs; final-epoch checkpoint.
- Deterministic training enabled by `run_clean_only.slurm` through `DETERMINISTIC_TRAINING=1`.
- Graph-A2: global selection, `k=50`, `hops=2`.
- No corruption data or validation-selected checkpoint is used in this benchmark.

## Source versions and SHA256

Source manifest records GraphCov commit `8cf757adc4c333dc1427d511f0de2f246d15ebac` and MedMNIST-C commit `8acfd2710c6e0e8b2745be8b1fa1c17b94b8f8a7`.

| File | SHA256 |
|---|---|
| `reliability_medmnistc/scripts/run_pipeline.py` | `9c6f00ecfa7bdc4398ec28f093de3fdc549cba8c18d151161684d6dc0c6c6077` |
| `reliability_medmnistc/scripts/select.py` | `514705ccd2e14485a1656f8cf77d6ccaaf757feacceb02d8c6424c27a2c528c4` |
| `reliability_medmnistc/scripts/train.py` | `31dcd6613464295919439e29159f4895ec3377273659a860b1bab82238fb4c29` |
| `run_clean_only.slurm` | `6ce0b6456b419576bed7a4b13ce8f99a5d706e34f47233d05aa9becaedf8f046` |
| `submit_formal_queue.sh` | `2f59c83b47f857a07068f3ee01e2e56a65ae49a919c1a8c097071ba2ca8b5bc5` |
| TissueMNIST seed 2026 `config.yaml` | `eacb56000e69142d191a08fac1c6b0e2ea0b2601a35d6ead21f958c23d4e6a56` |

## Result and artifact inventory

The workbook includes only clean-test metric rows whose run has `run_complete.json` and whose `metrics.jsonl` SHA256 matches the completion marker. Final audit: Table 1 320/320 and 0 metric-hash mismatches. This GitHub package includes 40 actual run configurations and 160 Slurm `.out`/`.err` logs. It does not include dataset files, pretrained UNI weights, embedding caches, model checkpoints, or prediction tensors.

The pinned upstream GraphCov tree was not modified. The experiment pipeline checks the upstream commit and clean `graphcov/` worktree before running. See `UPSTREAM_COMPARISON.md` for the exact commit and scoped comparison instructions.
