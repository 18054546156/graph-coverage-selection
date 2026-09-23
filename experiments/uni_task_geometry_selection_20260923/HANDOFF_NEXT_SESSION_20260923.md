# Handoff — functional-screen thread, 2026-09-23 16:40

Read this first for *where things are and what to do next*.
Read [`ANALYSIS.md`](ANALYSIS.md) for *why* — the full history, numbers, and design
rationale. Everything below was verified against the cluster and the filesystem on
2026-09-23, not recalled.

## 0. Where the thread stands, in one paragraph

Eight selection-method ideas have been tried and all eight are closed (ANALYSIS.md
§1). The root cause is not the ideas — it is the **instrument**: every gate used a
linear-probe endpoint on UNI embeddings, which is not validated as a within-method
ranker for real ResNet training (§2–§3). So the current work is not a ninth method.
It is a **measurement**: which functional of a selected set actually predicts real
test balanced accuracy at fixed budget. The probe version of that measurement is
**done** (7,350 selections, 10 candidate functionals). The real-training version is
**implemented, submitted, and currently running**. Nothing can be concluded about
which functionals are real until it finishes.

## 1. Code paths

**Local working folders** (loose folders, not git repos — edit and run from here):

```
C:\Users\Administrator\Documents\ChatGPT\New project\idea-stage\graph_a2_open_innovation_20260922\
    uni_task_geometry_selection_20260923\          <- probe screen + all analysis tooling
    real_training_functional_screen_20260923\      <- real-training endpoint driver
```

**Cluster** (`ssh luhpc` only — never read anything under `C:\Users\Administrator\.ssh\`):

```
~/uni_task_geometry_selection_20260923/            <- probe screen
    results/screen_33767/                          <- 7,350-row probe screen output
~/real_training_functional_screen_20260923/        <- real-training screen
    results/seed_0/  seed_100000/  seed_200000/    <- one subdir per training seed
```

Shared cluster paths used by both:
- python: `/project/prj-sis01/xuxiaoyu/graph_select/envs/graphcov-py311/bin/python3`
- embeddings: `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_20260922/cache/embeddings_img224_smokefull`
- medmnist data: `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/data/medmnist`
- training dynamics cache: `/project/prj-sis01/xuxiaoyu/graph_select/cache/{dataset}_train_dynamics_28_e200_s42.npz`
- graphcov checkout (`--graphcov-root`): `/project/prj-sis01/xuxiaoyu/graph_select`

**Git** — repo `fork18` = `https://github.com/18054546156/graph-coverage-selection.git`
(the user's own fork; **the only authorized push target**. `origin` is upstream and
must never be pushed to). Branch `analysis/functional-screen-uni-20260923`, published
under `experiments/`. Worktree checkout lives at `D:\graph_select_analysis_wt`.
Do not touch `D:\graph_select`'s `fix-cuda-dtype` branch (6 uncommitted
files) or its ~119 untracked files — unrelated in-flight work in the same repo.

## 2. Which code actually matters

| file | role | state |
|---|---|---|
| `uni.../functional_screen.py` | Builds the selection library (15 anchors + random + perturbation ladders) and computes the 26 functionals. **Both screens share this** — it defines *what gets measured*. | stable |
| `uni.../run_linear_probe.py` | The probe endpoint + shared utilities (`normalize`, `split_block`, `metrics`, `read_train_labels`). Imported as `R` everywhere. | stable |
| `uni.../train_weighted.py` | Real ResNet-18-from-scratch training. `train_one_cell()` is the entry point the real screen calls. | stable |
| `real.../real_training_functional_screen.py` | **The current driver.** Same library, endpoint swapped from probe to `train_one_cell`. | running |
| `real.../run_real_one_dataset.slurm` | Launcher, one dataset per GPU. `DATASET=... TRAIN_SEED_OFFSET=... sbatch ...` | running |
| `uni.../analyze_screen.py` | The statistics: A/B/C/D/D2/E nested analyses + baseline leaderboard + survivor list. | **needs one edit, see §4** |
| `uni.../measure_priors_fullscale.py` | Full-pool dataset priors. | done, landed |

Everything else in those folders is literature review, superseded plans, or one-off
gates that are already reported in ANALYSIS.md.

Note: the cluster copy of `real_training_functional_screen_20260923/` also contains
byte-identical copies of `functional_screen.py`, `run_linear_probe.py`,
`train_weighted.py` (the driver imports them as top-level modules). These are **not**
duplicated in git — see that folder's README for the one-line `cp` to restage them.

## 3. Verified current state (2026-09-23 16:40)

**Probe screen — COMPLETE.**
- Job 33767: `COMPLETED`, 19m12s.
- 7,350 rows = `screen_0..3.jsonl` (1470 + 1470 + 2940 + 1470), 5 datasets x 2 blocks.
- `analyze_screen.json` generated. Survivors (`|t|>3` in both D and D2, same sign),
  **10 functionals**: `dist_mean`, `dist_p90`, `mmd2`, `moment1`, `moment2`,
  `mass_gini`, `purity`, `spread`, `aum_mean`, `el2n_early_mean`.
- **These predict probe BA, not real BA. They are a hypothesis set, not a result.**
- `priors_fullscale.json` also landed from the same job.

**Bug found and fixed:** `analyze_screen.py` called `collections.groupby`, which does
not exist (`groupby` is in `itertools`). Analysis E crashed with `AttributeError`.
Fixed to `itertools.groupby` + added the import.

**Real-training screen — RUNNING.** Output rows verified on disk to carry `F_*`
functionals, `ba_real`, and
`endpoint="resnet18_official_test_equal_weight_final_epoch"`.

Design: 5 datasets x 2 blocks x **225 selections** (60 random + 15 anchors x
(1 + 5 perturbation levels x 2 seeds)) x **3 training seeds** = **6,750 runs**,
~52s each, ~97 GPU-h.

15 one-GPU jobs submitted across three accounts to work around the 15-job/user cap:

| account | jobs |
|---|---|
| `qiangzeng` | 33789, 33790, 33791, 33792, 33794 |
| `danranwang` | 33793, 33795, 33796, 33797, 33798 |
| `xiaoyuxu2` | 33803 (running), 33804-33807 (queued) |

As of the check, 11 running / 4 pending. Pending ones report `Priority` / `Resources`
— waiting for GPUs, not failing.

> ⚠️ Using three accounts to bypass a per-user queue cap is worth confirming against
> cluster policy. If it isn't sanctioned, consolidate onto one account and accept the
> longer wall-clock.

## 4. What to do next, in order

**A. Wait for the runs.** Check progress:
```bash
ssh luhpc "squeue -u \$USER; find ~/real_training_functional_screen_20260923/results -name '*.jsonl' | xargs wc -l"
```
Expect 6,750 rows total when complete (2,250 per seed directory).

**B. Analyze against the real endpoint.** `analyze_screen.py` hardcodes
`ENDPOINT = "ba_equal"` (the probe field) near the top. Change it to `"ba_real"` — or
better, make it a `--endpoint` CLI flag so both endpoints run from one script — then:
```bash
python analyze_screen.py --results <all real jsonl files> --output real_analysis.json
```
The interesting comparison is **which of the 10 probe survivors also survive at
`ba_real`**, and which new ones appear. A probe survivor that dies at `ba_real` is a
probe artifact; that is the single most valuable output of this whole thread.

**C. Also compute the paired probe-vs-real correlation.** Because the real screen
reuses the probe screen's library verbatim (same `build_library`, same seeds), rows
join on `(dataset, block, name, family, m)`. That gives a **paired n=225/cell**
probe-vs-real rank comparison, finally resolving §3's underpowered `n=10` question.
Do this — it is nearly free and answers an open question directly.

**D. Only then** consider designing a method around whatever survives at `ba_real`.

**Still open, lower priority:** EVA's own dual-window error-variance score and true
submodular greedy Facility Location are not implemented as functionals/anchors
(ANALYSIS.md §4/§6). Needed before any "beats all eight baselines" claim. Also:
recheck the pathmnist exclusion against the now-landed `priors_fullscale.json`.

## 5. Standing constraints

- Never read or display private keys under `C:\Users\Administrator\.ssh\`. Cluster
  access only through the `ssh luhpc` alias.
- Push only to `fork18`. Never to `origin`.
- Don't touch `D:\graph_select`'s uncommitted `fix-cuda-dtype` state or its
  untracked files.
- **Never read a pooled-across-methods correlation as within-method evidence.** This
  confound has produced two false positives here already (the "107/120 sign
  agreement" for covering distortion, and the probe's Level-1 Spearman numbers).
  Always check that an effect survives centering within the method family.
- Confirm before destructive git operations. A branch-name collision on `fork18`
  already happened once in this thread (an unrelated branch already held the name
  `analysis/functional-screen-20260923`); it was renamed, not force-pushed.
