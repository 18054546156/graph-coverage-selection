# Build Note — D8 selector prototype

| Rung | Feature | Acceptance check | Tier | Status |
|------|---------|-----------------|------|--------|
| F0 | CLI entry point and output contract | `python d8_class_conditional_ot_20260922/select_d8.py --help` | MUST | ✅ |
| F1 | Equal-mass classwise Sinkhorn selection core | `python -m unittest discover -s d8_class_conditional_ot_20260922/tests -v` | MUST | ✅ |
| F2 | Determinism, quota, uniqueness, finite diagnostics, and registry adapter tests | Same test command | MUST | ✅ |
| F3 | Integration with clean-only training and 5 datasets × 8 seeds | Deferred: requires resolved source protocol, runtime profiling, and explicit HPC submission confirmation | DEFERRED | ⬜ |

## Run record

1. `python -m unittest discover -s d8_class_conditional_ot_20260922/tests -v` — exit 0; 6 tests passed.
2. `python d8_class_conditional_ot_20260922/select_d8.py --help` — exit 0.
3. `python -m py_compile d8_class_conditional_ot_20260922/d8_ot.py d8_class_conditional_ot_20260922/select_d8.py` — exit 0.

## Deferred

- Pipeline registration and full UNI feature extraction integration.
- Live HPC runtime/RAM profile and 40 clean train/test runs.
- Independent cross-model silent-assumption sweep: `SWEEP_UNAVAILABLE`; no reviewer MCP is exposed in this Codex session.
- Any claim of improvement over Graph-A2.

## Blockers

- At 2026-09-22 11:08 HKT, the active benchmark queue had ten RUNNING and four PENDING tasks; do not add D8 jobs to it.
- At the same check, HPC had 242 `run_complete.json` markers overall and 26 under Graph-A2. A marker count is not a full result audit.
- The local CSV snapshot is time-stamped 2026-09-22 10:48 and contains 236 COMPLETE / 84 INCOMPLETE rows; it is stale relative to the live marker count.
