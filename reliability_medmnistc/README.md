# MedMNIST-C Reliability Pipeline

This directory contains the reproducible experiment wrapper around the
official GraphCov selector and the pinned MedMNIST-C API. The upstream source
implementations are not modified.

## Frozen protocol

- Clean MedMNIST train is used for selection and downstream training.
- Clean official test is used for the clean evaluation.
- MedMNIST-C is test-only and is never used for selection or checkpoint choice.
- Selection seed is 42.
- ResNet-18 is trained from scratch at 224 pixels for 1000 epochs.
- The final epoch checkpoint is evaluated on clean and corrupted test data.
- Graph-A2 uses a global graph, 50 neighbors, and 2 hops.
- Facility uses per-class selection (`global_selection: false`).

## Run order

```bash
python reliability_medmnistc/scripts/prepare_medmnist_224.py --root data/medmnist
python reliability_medmnistc/scripts/generate_medmnistc.py \
  --medmnist-root data/medmnist \
  --output-root data/medmnistc \
  --source-root third_party/medmnistc
python reliability_medmnistc/scripts/select.py \
  --config reliability_medmnistc/configs/full_5datasets_8methods.yaml
python reliability_medmnistc/scripts/train.py \
  --config reliability_medmnistc/configs/full_5datasets_8methods.yaml
python reliability_medmnistc/scripts/evaluate.py \
  --config reliability_medmnistc/configs/full_5datasets_8methods.yaml
python reliability_medmnistc/scripts/summarize.py \
  --config reliability_medmnistc/configs/full_5datasets_8methods.yaml
```

`run_pipeline_full.py` combines selection, training, and evaluation for a
single resumable invocation. The Slurm files provide the HPC equivalents.

## Configurations

`full_5datasets_8methods.yaml` is the paper-scale configuration. It uses five
datasets, eight methods, 2% and 5%, selection seed 42, and training seeds
42-46. `pathmnist_ratio_sweep.yaml` is the current single-seed PathMNIST
sweep at 1%, 2%, 5%, and 10%.

## Output contract

Selections are saved with the selected indices, class counts, seed, config,
and SHA256. Each completed run saves `final.pt`, predictions, metrics, and a
completion marker. Summary CSV files are produced only from completed runs.
Datasets, checkpoints, predictions, caches, and logs are intentionally not
tracked by Git.
