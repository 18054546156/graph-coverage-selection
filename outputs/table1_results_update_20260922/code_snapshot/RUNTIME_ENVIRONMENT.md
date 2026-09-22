# Runtime Environment and GPU Allocation Record

Snapshot taken 2026-09-22 18:29 HKT. This records the completed Table 1 execution and a live scheduler check; it distinguishes requested resources from measured utilization.

## Software and source versions

| Component | Recorded value | Source |
|---|---|---|
| Python | 3.11.9 | HPC environment package lock |
| PyTorch | 2.5.1 | HPC environment package lock |
| torchvision | 0.20.1 | HPC environment package lock |
| PyTorch CUDA build | 12.1 | `torch.version.cuda` environment lock |
| MedMNIST | 3.0.1 | HPC environment package lock |
| NumPy | 1.26.4 | HPC environment package lock |
| GraphCov upstream | `8cf757adc4c333dc1427d511f0de2f246d15ebac` | upstream commit pinned and checked by pipeline |
| Experiment pipeline SHA256 | `4a06c4a69ca30254f643cc4995d008b2188057e18007e4c7967b35742bccc693` | copied execution source |
| Formal Slurm launcher SHA256 | `6ce0b6456b419576bed7a4b13ce8f99a5d706e34f47233d05aa9becaedf8f046` | copied execution source |
| Queue worker SHA256 | `2f59c83b47f857a07068f3ee01e2e56a65ae49a919c1a8c097071ba2ca8b5bc5` | copied execution source |

The exact package lock is `manifests/environment-lock.txt`; all source hashes are also listed in `PROVENANCE.md`. The CUDA build version is not the host NVIDIA driver version.

## Experiment GPU allocation

Each formal Slurm job represents one dataset plus one paired seed and sequentially runs all eight methods.

| Dataset | GPU requested per job | CPU requested per job | Memory requested per job |
|---|---:|---:|---:|
| OrganSMNIST | 1 | 8 cores | 12 GB |
| OrganAMNIST | 1 | 8 cores | 16 GB |
| PathMNIST | 1 | 8 cores | 32 GB |
| TissueMNIST | 1 | 8 cores | 48 GB |
| BloodMNIST | 1 | 8 cores | 12 GB |

Slurm accounting confirms these allocations for the completed jobs. Run configuration records `device=cuda`. Representative completed jobs: TissueMNIST seed 6 (`33385`) used one allocated GPU for 6:21:07; TissueMNIST seed 2026 (`33378`) used one allocated GPU for 6:23:46. Across the benchmark there are 40 completed dataset-seed jobs, each with one allocated GPU. This note does not aggregate their GPU-hours.

## Live scheduler snapshot

At 18:29 HKT, `squeue -u xiaoyuxu2` returned no jobs. Both visible `gpu-a100` nodes were `IDLE`:

| Node | Scheduler GRES | CPUs | State |
|---|---|---:|---|
| `hpcgpu107` | `gpu:nvidia:4` | 128 | IDLE |
| `hpcgpu109` | `gpu:nvidia:8` | 224 | IDLE |

Thus the two queried nodes showed 12 unallocated GPUs at that snapshot. The partition is named `gpu-a100`, but Slurm's node record exposed only generic `gpu:nvidia` GRES, not the physical GPU model or memory size. No job was running from which to query `nvidia-smi` at snapshot time.

## Measurements not retained

The representative Slurm accounting rows inspected had empty `TRESUsageInAve`, `TRESUsageInMax`, `MaxRSS`, and `AveCPU` fields. The archived job logs also do not record per-process GPU utilization or peak VRAM. Therefore this package can report allocated GPU/CPU/memory and elapsed wall time, but cannot claim an average or peak GPU utilization, actual CPU use, peak host RAM, or per-job peak VRAM. A prior live `nvidia-smi` sample is not a historical measurement for all 40 jobs.
