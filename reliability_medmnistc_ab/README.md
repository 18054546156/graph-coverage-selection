# Reliability and Selection-Robustness Experiment

This package is intentionally separate from the Table 1 reproduction.

## Source isolation

- Official GraphCov source: `/project/prj-sis01/xuxiaoyu/graph_select_pristine_origin_main` at commit `8cf757adc4c333dc1427d511f0de2f246d15ebac`.
- MedMNIST-C source: `sources/medmnistc-api` at commit `8acfd2710c6e0e8b2745be8b1fa1c17b94b8f8a7`.
- No file in either source checkout is modified by this package.
- Table 1 outputs remain in their existing directories and are not copied here.

## Directory contract

`protocol/` contains frozen protocol and source/environment manifests.
`notebooks/` contains the executable audit and API smoke-test notebooks.
`data/medmnist/` contains clean MedMNIST files only.
`data/medmnistc/` contains generated or downloaded corrupted test files only.
`artifacts/selections/` contains selector indices and selection metadata.
`artifacts/checkpoints/` and `artifacts/predictions/` are reserved for downstream robustness evaluation.
`results/raw_metrics/`, `results/summary_tables/`, and `results/figures/` contain derived outputs.
`logs/` contains Slurm, API, and synchronization evidence.

## Full experiment snapshot

The forked runnable Notebook is preserved unchanged at `sources/graph-coverage-selection-codex-reliability-audit-v29/reliability_full_experiment.ipynb`. Its parent commit is `be53fd357ed8ceffd1ed536b11ed0dab06f89519`; its GraphCov and MedMNIST-C submodules are pinned to the official commits above. The Notebook defaults are a smoke/first-pass configuration (`pathmnist,organsmnist`, seeds `0,1,2`), so the strict five-dataset run must explicitly override `DATASETS`, `RATIOS`, and `SEEDS` as recorded in `protocol/protocol.json`.

Do not run it on the login node. Execute it inside a GPU Slurm job after the official 224-pixel clean files are present and verified. The Notebook's output belongs in this package's `artifacts/` and `results/` trees, never in the Table 1 reproduction directories.

## Official data download

Run `scripts/download_medmnist_224.py` from this package. By default it writes to `data/medmnist/`, downloads all five datasets at 224 pixels, verifies each official MD5, and records `protocol/medmnist_download_manifest.json`. A failed Zenodo transfer leaves only a `.part` file and never marks the archive as valid:

```bash
python scripts/download_medmnist_224.py
```

For a single dataset:

```bash
python scripts/download_medmnist_224.py --datasets pathmnist
```

## Experimental separation

The clean-train selector is the primary object. For each selector, only clean training data may be used to create the selected indices. MedMNIST-C is test-only for downstream robustness and must never affect selection, checkpoint choice, or tuning.

The selection-robustness audit is a separate stress test. It compares selected-index overlap across frozen selection seeds and, when explicitly enabled, evaluates how a corrupted-view stress input changes the selected set. It must not be reported as the paper's Table 1 result.

## Current status

- The MedMNIST-C repository is cloned and clean on HPC.
- Its Python dependencies are isolated in `envs/medmnistc-py311`; ImageMagick is isolated in `envs/imagemagick`.
- API import and the synthetic smoke pipeline pass when `LD_LIBRARY_PATH` includes the isolated ImageMagick prefix; the debug notebook records this requirement.
- The official MedMNIST-C generation job `31176` completed with exit code 0 and generated 56/56 corruption files for the five target datasets.
- The generation manifest is recorded at `protocol/medmnistc_generation_manifest.json`; the generated files remain on HPC under `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/data/medmnistc/` and are intentionally not committed to Git.
- The formal five-dataset, eight-method downstream experiment has not started. Its current single-seed planned size is 80 model runs: 5 datasets x 2 ratios x 8 methods x selection seed 42 x training seed 42. Multi-seed variance is a separate diagnostic experiment.

## Reproducibility contract

The strict single-seed run must use selection seed 42 and downstream training seed 42 for each method. The v29 protocol uses the final epoch checkpoint and does not read validation or corrupted test data for checkpoint choice. The same checkpoint is evaluated on the official clean test and every MedMNIST-C corruption at severities 1-5.

The frozen configuration is in `protocol/protocol.json`. The detailed execution order, Slurm resources, expected output paths, and acceptance checklist are in `handoff.md`.
