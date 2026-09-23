# 5% Table 1 Run Status

Last verified: 2026-09-22 21:50 HKT

## Smoke gate

- Smoke jobs `33554`-`33558` completed successfully; all five datasets ran all eight methods.
- Audit result: `SMOKE_AUDIT_PASS cells=40 ratio=0.05`.
- Smoke used one training epoch and up to 8192 clean-test examples. It verifies the pipeline, not model performance.
- Selection counts were balanced and matched the per-class 5% budget: OrganSMNIST 407, OrganAMNIST 407, PathMNIST 405, TissueMNIST 408, BloodMNIST 408.
- Clean-test rows covered every class. Metrics hashes matched completion markers. No smoke task failed for OOM. Slurm did not report `MaxRSS`, so measured peak host RAM is not available.
- Initial smoke attempts `33549`-`33553` failed before running because `REPO_ROOT` was unset. The launcher was fixed and retry jobs passed.

## Formal run

- Formal plan: 5 datasets x 8 paired seeds (`0,1,2,3,4,5,6,2026`) = 40 Slurm tasks.
- Each task runs eight methods sequentially, for 320 result cells total.
- Selection and training seeds are paired. Each task uses the dataset-specific memory request: OrganSMNIST 12G, OrganAMNIST 16G, PathMNIST 32G, TissueMNIST 48G, BloodMNIST 12G.
- Submitted jobs for `xiaoyuxu2`: `33561`-`33575` (seeds 0, 2, 4).
- Submitted jobs for `qiangzeng`: `33576`-`33590` (seeds 1, 3, 5).
- At last verification, both accounts had 5 RUNNING and 10 PENDING jobs; 5 tasks per account remained in the active queue for the detached dispatcher to submit as slots open.
- The running allocation totals 10 GPUs, 80 CPUs, and 240G requested RAM across both accounts (120G per account).
- Live logs show clean UNI embeddings and seed-matched dynamics caches loaded, then 5% selection and training starting. Sampled job state was RUNNING on both accounts.
- The Slurm QoS limits each account to 5 running and 15 submitted jobs. The dispatchers maintain up to 15 submitted jobs per account and run on HPC, so an intermittent local VPN disconnect does not terminate them or the submitted jobs.

## Operational note

The first dispatcher attempt exposed CRLF line endings in the TSV queues, which appended a carriage return to `--mem` and caused Slurm to reject the request. No formal jobs were accepted in that attempt; tasks were returned to their queues. The dispatcher now strips CRLF before parsing. A non-submitting `sbatch --test-only` check accepted the 12G request, and the subsequent formal submission succeeded.

Do not treat pending jobs as failed. Check current work with:

```bash
ssh luhpc 'squeue -u xiaoyuxu2'
ssh luhpc-qiangzeng 'squeue -u qiangzeng'
```

HPC experiment root:

```text
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_5pct_20260922/
```

Formal results are isolated under `formal/`, logs under `logs/`, and the immutable original queue plans under `queues/initial/`. The live queues are under `queues/active/`.
