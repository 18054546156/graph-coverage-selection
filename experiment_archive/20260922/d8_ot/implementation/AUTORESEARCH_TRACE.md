# D8 Auto-Research Execution Trace

This is an execution trace of actions actually performed. It is not a simulated multi-agent debate or a claim that literature novelty has been established.

## 2026-09-22

1. **Account and resource check.** Verified `danranwang` SSH public-key login, `hpc-sis` association, `qos-high-gpu` limits (12 GPUs, 128 CPUs, 256G RAM, five running jobs, 15 submitted jobs), and `qos-normal` limits. Confirmed the account had no jobs. The CPU partition had idle nodes; the A100 partition had 12 devices with six allocated at that observation.

2. **Selection-only smoke, job 33459.** Ran full OrganSMNIST source-train selection with 25 points per each of 11 classes. The output had 275 unique indices and exact quotas; selection completed in 3.146 seconds with peak RSS 324368 KiB. Input and selector hashes are in the HPC `smoke/organsmnist_seed0_20260922/run_metadata.json`. This did not train a classifier.

3. **Isolated training integration.** Copied the Table 1 source into the D8 experiment directory. In the copy only, registered `d8_ot`, marked it as requiring embeddings, recorded the D8 solver parameters in selection metadata, and narrowed the upstream-integrity guard to allow only the explicit registry file change. Copied the five existing full UNI caches into the D8-owned cache and byte-compared them with their benchmark sources. The active Table 1 source, results, and cache were not patched or overwritten.

4. **Integration smoke attempts.** Jobs 33462 and 33463 stopped before training because the launcher omitted `PYTHON` and then `REPO_ROOT`; both variables were added. Jobs 33464 and 33465 confirmed the upstream-integrity guard was active, then exposed an overly strict diff-count check after line-ending normalization. The guard was changed to require exactly one changed tracked file (`graphcov/run/selection.py`) and exactly one D8 registration import/call, while all other tracked GraphCov files must remain unchanged.

5. **Passing integration smoke, job 33466.** One OrganSMNIST seed-0 cell completed one epoch on one A100 in 24 seconds. It produced `final.pt`, clean predictions, metrics, and a validated `run_complete.json`. This verifies the D8 registry-to-training-to-clean-test path, but is not a performance result.

6. **Formal paired benchmark, array 33468.** Submitted 10 worker elements throttled to five concurrent A100 jobs. The 40 planned cells are five datasets by eight paired seeds, with selection seed equal to training seed, 2% class quota, deterministic 1000-epoch clean training, and clean-test evaluation. Each worker processes four cells sequentially. At the initial progress snapshot, workers 0-4 were running, workers 5-9 were pending on the array throttle, five cells had started, and no formal cell had completed. No D8-versus-Graph-A2 conclusion is available yet.
7. **Live progress check, 13:47 HKT.** Array 33468 has five workers running (one GPU each) and five pending on the array concurrency throttle. Five of 40 completion markers are present, all for OrganSMNIST seeds 0-4. All five workers have advanced to OrganAMNIST seeds 2-6; observed training histories were at epochs 202-280/1000. The five formal worker stderr files were empty. The matching Graph-A2 comparator still had 34/40 completion markers. No method-performance conclusion is available from this progress snapshot.

## Research caveats

- The D8 solver is an entropic-OT/geometric-median/projection heuristic, not the exact discrete Wasserstein optimum.
- The integrated smoke establishes code execution only. The 40-cell formal run is required to estimate downstream behavior.
- Graph-A2 had 34/40 complete paired cells at D8 launch; final analysis must wait for all matching baseline cells.
- FDMat and other OT representative-subsampling literature remain direct prior-art threats. Novelty is not confirmed.
- The primary comparison is the complete paired clean-BA table by dataset and seed. Do not tune against clean test or report only favorable seeds.
