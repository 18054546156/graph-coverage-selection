# D8 Clean Benchmark Experiment Plan

**Status:** Formal paired benchmark submitted as Slurm array `33468`; five workers run concurrently. See `RUN_METADATA.json` and `AUTORESEARCH_TRACE.md` for the execution record.

## Question and claim

At a fixed 2% per-class budget, does selecting equal-mass class-conditional feature representatives improve clean balanced accuracy over the frozen Graph-A2 selector when the downstream training recipe is unchanged? Improvement over Graph-A2 is a hypothesis, not an expected or established result.

## Prerequisites

1. Keep the current clean-only benchmark outputs immutable. D8 may run while its last Graph-A2 cells finish because D8 code and parameters are frozen; paired analysis remains blocked until all matching baseline cells pass provenance validation.
2. Freeze the D8 code, feature-cache identity, L2 normalization, epsilon, iterations, initialization seed rule, class quotas, and downstream config before any clean-test result is opened.
3. Confirm selected indices use only source-train embeddings and labels. Clean test and MedMNIST-C must not be used for selector, solver, or hyperparameter tuning.
4. Profile D8 selection time and peak RAM from source-train embeddings. Full OrganSMNIST selection and one-epoch pipeline integration smokes are complete; formal-run resource use is still being measured.

## Confirmatory matrix

- Datasets: OrganSMNIST, OrganAMNIST, PathMNIST, TissueMNIST, BloodMNIST.
- Budget: ratio 0.02 with exactly the same class quotas/rounding as Graph-A2.
- Seeds: 0, 1, 2, 3, 4, 5, 6, 2026; selection seed equals training seed.
- Pipeline: clean source-train selection -> clean selected training -> one clean-test evaluation after the selector is frozen.
- Candidate: D8 only; pair each cell against the matching Graph-A2 cell in the frozen benchmark. Random and Herding are secondary context controls from that benchmark.
- Total: 5 × 8 = 40 D8 downstream training cells, in a directory separate from both the benchmark and legacy outputs.

## Outcomes and analysis

Primary outcome is per-dataset paired clean-BA difference versus Graph-A2, with an overall dataset-aware summary and confidence intervals. Also report accuracy, every class recall, worst-class recall, selection overlap, source-train entropic transport cost, selection runtime, peak RAM/VRAM, and failures. Seeds on one dataset are paired runs, not independent datasets; do not claim universal superiority from a pooled mean alone.

## Controls and kill criteria

Before a method claim, compare against the frozen Graph-A2 results, balanced Random, Herding, and a disclosed FDMat-UNI adaptation. Add a subtype-proportional random control only if subtype strata are defined from source data before outcomes are inspected. Abandon the proposed mechanism if FDMat/Herding matches within a preregistered practical margin, if equal-mass OT does not reduce an independently measured source distribution discrepancy, if the result depends on test-driven tuning, or if it cannot meet the fixed budget reproducibly.

## Execution boundary

The D8 array uses ten worker elements throttled to five concurrent jobs, each requesting one A100, 8 CPUs, and 48G RAM. Each worker processes four cells sequentially. The formal D8 output root is separate from the Table 1 root. Graph-A2 had 34/40 complete cells at D8 launch; do not make a paired-performance claim until all 40 matching Graph-A2 cells and D8 cells are complete and validated.
