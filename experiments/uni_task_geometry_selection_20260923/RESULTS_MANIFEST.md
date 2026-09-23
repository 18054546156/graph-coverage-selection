# Results Manifest

## Included compact artifacts

- `results/gate_training.json`: final 360-cell weighting interaction gate.
- `results/probe_vs_training.json`: probe versus real-training ranking check.
- `results/ladder.json`: budget/cap ladder for the three-dataset probe study.
- `results/dataset_priors.json`: original capped-prior output, retained for audit but superseded for full-scale claims.
- `results/screen_smoke.jsonl`: smoke-test rows only; not a confirmatory result.

## Omitted artifacts

The following remain on HPC and are intentionally not committed:

- UNI embedding caches and raw MedMNIST archives;
- `results/training_33697/` and `results/training_33702/` shard files;
- selection manifests, intermediate logs, CUDA caches, Python caches, and model checkpoints;
- 200-epoch dynamics caches (`*_train_dynamics_28_e200_s42.npz`).

The source HPC project is:

```text
/home/xiaoyuxu2/uni_task_geometry_selection_20260923/
```

## Status at export

- The 360 real-training cells were complete.
- `gate_training.py` returned `REFUTED` for the Voronoi-within interaction.
- The old functional screen job `33707` produced zero rows after a shell-variable path bug.
- The repaired dynamic screen code exists, but a full confirmatory screen output was not included because it was not complete at export.
- Full-scale priors are represented by `measure_priors_fullscale.py`; its output must be generated before using pathmnist headroom or nearest-neighbor claims.
