# code_asrun — cluster snapshot (2026-09-28 18:25 HKT)

Byte-for-byte copy of `tdgs_shared/code/` as it ran on the cluster; `MANIFEST.md5` holds the checksums.
The version in `../code/` is a superset. Differences:

- `tdgs_select.py`: the snapshot here is the **frozen Round 1 version**; all round 1–3 selections came from it.
  - `../code/` adds `build_graph_kernel(return_base=...)` (default behaviour unchanged).
  - It also adds environment-variable path overrides.
- `tdgs_select_r2.py`: `../code/` adds the diagnostic `build_mask_before_prop_kernel`.
- `td_run_one.sh`, `td_pack.slurm`, `harvest_tdgs.py`: exist only here. These are the actual training runner for rounds 1–3 (`TD_SEL_DIR` + harness `precomputed`) and the Round 1 harvester.
