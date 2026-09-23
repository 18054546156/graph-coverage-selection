# Table 1 Clean-Only 5% Run

Independent 5% counterpart to `table1/`. The scientific protocol matches the completed 2% clean-only benchmark; the selected-sample ratio is the only protocol parameter changed. All 5% configs, outputs, and Slurm logs are isolated from the 2% experiment.

## Frozen protocol

- Datasets: `organsmnist`, `organamnist`, `pathmnist`, `tissuemnist`, `bloodmnist`.
- Methods: `random`, `el2n_top`, `forgetting`, `eva`, `facility`, `fps`, `herding`, `graph_a2`.
- Ratio: `0.05`.
- Seeds: `0, 1, 2, 3, 4, 5, 6, 2026`; selection and training seeds paired.
- Clean selection, selected-clean training, official clean-test evaluation.
- 1000 epochs, final-epoch checkpoint, deterministic training, 224-pixel input.
- Graph-A2: global, `k=50`, two hops.

## HPC layout

- Experiment root: `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_5pct_20260922/`
- Smoke outputs: `smoke/<dataset>/seed_0/`
- Formal outputs: `formal/<dataset>/seed_<seed>/`
- Slurm and dispatcher logs: `logs/`
- Each task generates its actual 5% `config.yaml` under its own smoke/formal directory.

The run reuses the audited source snapshot and precomputed embeddings/dynamics from `table1_clean_only_20260922`. Cache contents are reused when present; the pipeline may touch cache lock files. All 5% configs, selections, training metrics, completion markers, and Slurm logs are stored under the new 5% experiment root. The cache is keyed by dataset, seed, image size, and dynamics settings, not by selection ratio. The launcher emits `ratios: [0.05]` and the pipeline records the ratio in each run's config and output path.

Each smoke task uses 8192 source examples, one training epoch, and runs all eight methods. Smoke outputs are never merged into formal results. The formal plan has 40 Slurm tasks (dataset x seed), with eight methods run sequentially in each task, yielding 320 model cells.

The intended account split is `xiaoyuxu2`: seeds `0,2,4,6`; `qiangzeng`: seeds `1,3,5,2026`. Dataset memory requests are 12G / 16G / 32G / 48G / 12G for OrganSMNIST / OrganAMNIST / PathMNIST / TissueMNIST / BloodMNIST. Each account dispatcher keeps at most 15 submitted tasks; QoS limits concurrent jobs to five per account.

This directory is populated with smoke/formal configurations, logs, a verified results workbook, and completion audit as the run progresses.
