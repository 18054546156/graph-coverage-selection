# paper_tdgs — clean package for the GraphCov / TDGS paper

> **2026-09-28** — To sync onto another server, see [REPRODUCE.md](REPRODUCE.md) (environment / data / models / paths / pipeline).
> Current status: [HANDOFF_20260927.md](HANDOFF_20260927.md) §0 and §8.7.
> Audit ledger: [report/evidence_ledger_20260928.md](report/evidence_ledger_20260928.md).
> Round 3 preregistration and amendments: `report/r3_prereg_*.md`.
> The "TD" in TDGS means Task-Demand (train-pool OOF proxy), **not test/target**. No arm reads the test set. See HANDOFF §0.1.

Target of comparison: **https://github.com/zahiriddin-rustamov/graph-coverage-selection**
(Rustamov et al. 2026, MICCAI, arXiv:2606.22002), Table 1 protocol — 5 MedMNIST
datasets, 2% / 5% budgets, ResNet-18 from scratch, **main metric balanced
accuracy (the metric does not change)**.

Everything needed to finish the paper is in this folder or reachable from the
paths in [docs/07_paths.md](docs/07_paths.md). Nothing here depends on the
working state of the other checkout.

---

## Status in one paragraph

Round 1 is done and read: **75 real training cells** (job 34970, all COMPLETED).
The designed novelty — class-pair *direction* demand — is **falsified by its own
permutation control**. The bottom rung of the ablation ladder, `tdgs_cls`
(within-class coverage credit + class-balanced demand weighting), is the **first
5/5-datasets-same-sign positive this project has produced**: BA **+1.76pp**,
worst-class recall **+3.54pp**, with the effect size predicted a priori by
cross-class edge share. Round 2 is fully staged and exists to answer the one
question that can still kill it: whether the author's own per-class scope already
gets that gain.

**Live as of 2026-09-28 11:25 — see HANDOFF §0.** Round 2 is 145/145 final:
tdgs_cls − graph_a2 at 2% +1.57pp 5/5 (excl. tissue +0.71); G1 passes; G2 pooled
+1.44 but tissue-only; **G4 fails at 5% (2/5 datasets) → claim is 2%-specific**.
Round 2b (40 cells, 4 control arms to n=5) running as jobs 35119/35124/35126.
Table 1 baselines: **re-run on author seeds 42–46** (user decision 12:00, reverses the
0–4 + footnote plan): 350 cells, jobs 35134/35136/35137 (staging) → 35138/35139/35140
(training), see HANDOFF §8.4. Plain-language status page: `report/explainer_20260928.html`.
Table-1 archive (seeds 0–4) is 400/400.
The paragraphs below are older snapshots.

**Snapshot 2026-09-27 23:55.** Round-2 staging done (job 35062, 59/60 selections;
only `tissuemnist_r0.05_tdgs_cls` still building). Round-2 training **submitted and
running**: jobs **35067 / 35068 / 35074**, 5 array tasks on each of the three
accounts, 145 cells into the shared tree. Per-class cross-class share queued as job
**35076** (waits behind training — `qos-high-gpu` allows 5 running jobs per user and
this account is not associated with any usable CPU qos).

Two corrections landed today, both against earlier claims in these docs:
- the cross-class edge share prior was measuring **1-hop `A`**, not `K = A+A²`.
  Correct values 2.49 / 26.52 / 42.41 / 53.77 / 66.06 %. **Order-preserving**, so
  every rank statistic survives; absolute values are not interchangeable.
- the class-level mechanism regression (N 5→47) is **supporting, not decisive**:
  `R²` ceiling 0.37, seed-noise sd 2.91pp exceeds real class-heterogeneity sd
  2.22pp. See [results/round1/tables.md](results/round1/tables.md) §5b.

---

## Read in this order

| # | file | what it is |
|---|---|---|
| 0 | [HANDOFF_20260927.md](HANDOFF_20260927.md) | the full narrative handoff, single source of truth for "where are we" |
| 1 | [docs/01_story.md](docs/01_story.md) | the paper: 5 acts, what each claims, what supports it |
| 2 | [docs/02_method.md](docs/02_method.md) | formulas and all 10 arms, with what each isolates |
| 3 | [docs/03_protocol.md](docs/03_protocol.md) | author protocol, the config-truth finding, cells the author already occupies |
| 4 | [docs/04_instrument.md](docs/04_instrument.md) | noise floor, MDE, dataset priors — why 3 seeds is a screen and not a test |
| 5 | [docs/05_future_experiments.md](docs/05_future_experiments.md) | round 2 (staged, 145 cells) and round 3, with pre-registered gates |
| 6 | [docs/06_closed_threads.md](docs/06_closed_threads.md) | the 20 closed negatives and the single mechanism that explains them |
| 7 | [docs/07_paths.md](docs/07_paths.md) | every cluster path, account, and limit |
| 8 | [results/round1/tables.md](results/round1/tables.md) | round-1 numbers as tables |
| 9 | [docs/08_paper_design.md](docs/08_paper_design.md) | principles, sufficiency/necessity network, Tables 1–6, Figs 1–6 |
| 10 | [docs/09_plan_20260928.md](docs/09_plan_20260928.md) | **current plan**: round-2 interim read, claim decision tree, round 2b, table/figure evidence map, ETA |

---

## Layout

```
paper_tdgs/
  HANDOFF_20260927.md          narrative handoff
  docs/                        07 documents, above
  code/
    tdgs_select.py             ROUND 1 SELECTOR — FROZEN, do not edit
    tdgs_select_r2.py          round 2: 5 new arms (probe-free, CPU-cheap)
    tdgs_pack.slurm            3-account packed trainer, umask 002, shared tree
    make_worklist.py           round-2 worklist, ordered by decisiveness
    launch_3accounts.sh        stage / train / status / harvest / cancel
    harvest.py                 run tree -> one row per cell (BA, worst, recalls)
    ladder.py                  paired stats, 2x2 factorial, printed gates
    tdgs_stage.slurm           round-1 staging
    tdgs_smoke.slurm           round-1 validity smoke (reproduces the archive)
    proxy_regime.py/.slurm     RETIRED diagnostic, kept for the record only
  results/round1/
    tdgs_round1_harvest.json   75 cells, BA + worst + per-class recalls
    tables.md                  the same numbers as tables
    selection_reports/*.json   per-dataset staging reports (probe validity, Jaccard)
  logs/round1/                 slurm logs of the 4 array tasks that produced round 1
```

`tdgs_select.py` is **frozen**: it produced the 25 round-1 selections and its
`graph_a2` arm reproduces the archived selection order-exact. Round-2 arms are in
a separate file that imports from it, so round 1 stays bit-reproducible.

---

## Continue the experiments (3 accounts, 15 jobs, 30 concurrent trainings)

```bash
cd code
./launch_3accounts.sh stage     # 5 array tasks, CPU-cheap: round 2 needs no probe
./launch_3accounts.sh train     # 3 accounts x 5 jobs x 2 GPUs
./launch_3accounts.sh status    # progress, per-arm completion, failures
./launch_3accounts.sh harvest   # -> round2_harvest.json + round2_ladder.txt
```

Two design decisions make this safe to re-run and safe to leave unattended:

**One shared result tree.** All three accounts write
`/mnt/prj01/hgrp-1502-5TB/tdgs_shared/runs/round2_20260927` (group `hgrp-1502`,
setgid, launched with `umask 002`). `td_run_one.sh` already skips any cell that
has `predictions_clean.npz`, and that path is now shared — so **cross-account
deduplication is automatic and there is no merge step**. The earlier
three-account rounds wrote to three separate homes and had to merge done-keys by
hand before every launch; forgetting that re-ran finished work.

**The worklist is ordered by decisiveness.** The 15 jobs tile it by a global
stride (`SHARD*NTASKS + TASK`, step `NSHARD*NTASKS`), so all of them start near
the front. Whatever is finished when you stop is a prefix, and the prefix is
always the most decisive experiment available — P0 first.

Limits honoured: `qos-high-gpu` is 12 GPUs / 256G / **5 running jobs** per user,
so 5 jobs × 2 GPUs = 10 GPUs per account, 15 jobs total. Want GPUs over job
count? `NJOB=4 NGPU=3 ./launch_3accounts.sh train` → 12 GPUs/account, 36 slots.
All GPU jobs carry `--exclude=hpcgpu108`.

---

## Hard constraints (do not relax)

- **Never weaken** `assert_graphcov_source_unchanged()`. New selectors are staged
  as `precomputed` selections; `graphcov/` is never edited.
- Push only to `fork18`. The upstream author repo is the **comparison target**,
  never a push target.
- Do not touch the other checkout's `graph_select` working tree, its
  `fix-cuda-dtype` branch, or its untracked files.
- All GPU jobs must `--exclude=hpcgpu108`.
- Heavy analysis dependencies go in an isolated venv, never the shared training
  env `medmnistc-py311`.
- Private keys are never read or displayed; cluster access only via the
  configured aliases `luhpc`, `luhpc-qiangzeng`, `luhpc-danranwang`.

## Unresolved, flagged repeatedly, never touched

Jobs `34873 lam0stage`, `34883 corrsmoke`, `34884 bloodrep` and several files
under `~/tdfix/` on the shared `xiaoyuxu2` account were not created by this line
of work. They have been left alone. Everything here is isolated under
`~/tdgs/`, `/mnt/prj01/hgrp-1502-5TB/tdgs_shared/`, and
`$PROJECT/tdgs_round1_20260927`.
