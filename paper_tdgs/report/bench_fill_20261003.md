# Bench fill: complete the published-method columns on retina / breast / derma / OCT (2026-10-03; stamp before staging)

User request 10-03: "表 补充完整" (complete the BA table in `report/baselines_and_results_20261003.md`).

## Why

On the 4 added datasets the table had only 4 Table-1 columns, and on derma/OCT two of those were **not the author pipeline**:
`a2_uni` is our re-implementation of the Graph-A2 objective (`mv_select.py`, CPU FAISS) and `rand_cls` is our class-balanced random
with selection seed fixed at 42 (`w1_stage.slurm`). The author's random uses selection seed = training seed.

## What is run

All selections come from the **author pipeline** (`run_pipeline.py --phase select`, graphcov unchanged, source guard on), with the
Table-1 config of `t1_select.slurm` unchanged (only the method list differs; see `code/bench_fill_select.slurm`, a diffable copy).
Selection seed = training seed (42–46), as in the original bench. graph_a2 is seed-independent and is staged once (seed 42).

| datasets | methods | cells |
|---|---|---|
| retina, breast | fps, eva, el2n_top, forgetting | 2 × 4 × 2 × 5 = 80 |
| derma, OCT | fps, eva, el2n_top, forgetting, random, graph_a2 | 2 × 6 × 2 × 5 = 120 |

- Training: `tune_pack.slurm` + `td_run_one.sh` (precomputed selections), as every other cell on these datasets; 1000 epochs,
  224px, no augmentation, `DETERMINISTIC_TRAINING=1`, final_epoch, **H100** (same GPU type as these datasets' existing cells),
  `--exclude=hpcgpu108`. Per-seed selections get arm names `<method><seed>` trained only at that seed (as `random42..51` in ambig).
- Trees: `runs/bench_fill_20261003` (selections + training), td_sel `benchfill/td_sel`, worklist `work/bench_fill_h100.txt`.
- No tuning. No seed or row is dropped after results are seen.

## How the table changes

- retina/breast: FPS/EVA/EL2N/Forgetting columns filled.
- derma/OCT: FPS/EVA/EL2N/Forgetting filled; the Graph-A2 and Random columns are **replaced** by the author-pipeline cells.
  The old `a2_uni` / `rand_cls` numbers stay in the record (appendix), labelled as our re-implementations.
- The ACS − per-row SOTA comparison (`code/acs_vs_table1.py`) is re-run on the completed pool, whatever it shows.

## Stamp

(appended on the cluster in `benchfill/stamp.txt`: sha256 of this file and `bench_fill_select.slurm`, UTC, before staging)
