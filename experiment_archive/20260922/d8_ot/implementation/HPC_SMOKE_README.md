# D8 HPC selection smoke

This is a single, isolated CPU selection smoke on the full OrganSMNIST clean
training embedding cache. It selects 25 samples per class (2% benchmark quota)
with seed 0. It does not train a classifier, read validation/test labels, or
write into the active Table 1 result tree. The purpose is to validate data
alignment, quota enforcement, runtime, memory, and output provenance before any
downstream D8 training is considered.

The Slurm request is 4 CPU cores, 4 GiB RAM, and 2 hours on `qos-normal`; this
algorithm is CPU/NumPy/SciPy code, so requesting an A100 would waste a GPU.

Expected outputs:

- `smoke/organsmnist_seed0_20260922/selected_indices.npz`
- `smoke/organsmnist_seed0_20260922/run_metadata.json`
- `logs/d8-smoke-JOBID.out` and `.err`

The metadata records input and selector SHA256 values, environment, Slurm
allocation, class counts, output-index hash, validation checks, wall time, and
peak RSS. A successful selection smoke does not establish downstream accuracy,
scientific novelty, or superiority to Graph-A2.

Submission after the account owner authorizes the public key and the files are
copied to the HPC project directory:

```bash
cd /project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/d8_class_conditional_ot_20260922
mkdir -p logs
sbatch d8_smoke.slurm
```
