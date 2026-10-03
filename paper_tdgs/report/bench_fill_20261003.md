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

## Results (appended 2026-10-03 evening after harvest; nothing above this line changed)

200/200 cells, 0 failures, all H100 (jobs 36326 xiaoyuxu2 96 cells, 36327 danranwang 104 cells).
Harvest: `code/bench_fill_eval.py --cluster` → `results/acs/bench_fill_cells.json`. Each cell writes metrics.jsonl twice, so cells are deduplicated per (ds, ratio, method, seed).
Merged in `acs_eval.load()`: derma/OCT `graph_a2` and `random` are now the AUTHOR cells. Our re-implementations stay as `a2_uni` / `rand_cls`, for the appendix only.

- **ACS vs per-row best published method** (all 11 published methods in every row, `results/acs/acs_vs_published_benchfill.txt`): still first in 9/10 rows; OCT 2% −1.12 vs MaxHerding.
  - derma 5%: best = author Random 51.40, ACS +1.81 (p .058; it was +3.43 vs our rand_cls).
  - OCT 5%: best = FPS 89.10, ACS +1.36 (p .012).
  - Raw p < .05 in 5 rows.
- **Graph-A2 (author) vs ACS:** ACS ahead 10/10; derma 2% +2.51, OCT 2% +0.50.
- **Graph-A2 < Random:** 8/10 high-ambiguity rows (was 7/10 with the re-implementations).
- **Worst-class recall:** FPS is far ahead on OCT (2%: 74.32 vs ACS 62.16, −12.16, p .003; 5%: −3.76).
- **Ablation re-read with author A2** (`results/acs/e2_ablation_benchfill.txt`): F1 (cls vs A2) +3.74pp, 10/10; ACS vs A2 +7.14pp, 10/10.
