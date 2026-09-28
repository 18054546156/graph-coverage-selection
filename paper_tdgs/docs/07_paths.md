# Paths, accounts, limits

## Accounts

| alias | user | `$PROJECT` write | home |
|---|---|---|---|
| `luhpc` | `xiaoyuxu2` | **YES** (the only one) | `/home/xiaoyuxu2` |
| `luhpc-qiangzeng` | `qiangzeng` | no (read-only) | `/home/qiangzeng` |
| `luhpc-danranwang` | `danranwang` | no (read-only) | `/home/danranwang` |

All three are in group **`hgrp-1502`**. All three can read `$PROJECT` (env, data,
embeddings, source). Only `xiaoyuxu2` can write it — which is why the shared tree
lives elsewhere.

Per user, `qos-high-gpu`: **12 GPUs, 256G, 5 running jobs, 15 submitted jobs**.
`MaxRSS` accounting is disabled, so no historical peak memory exists.
**All GPU jobs must carry `--exclude=hpcgpu108`.**

> Git Bash rewrites POSIX paths in SSH output, so a remote `/home/xiaoyuxu2/x.log`
> may print as `C:\Users\...\x.log`. Verify with `od -c`, and send remote commands
> via a **quoted heredoc** (`ssh host 'bash -s' <<'REMOTE'`) rather than `-c`.

## The shared tree — this is the important one

```
/mnt/prj01/hgrp-1502-5TB/tdgs_shared/        root:hgrp-1502  2770 (setgid)
  code/        all scripts, synced from paper_tdgs/code
  sel/         every selection .npy, read by all three accounts
  runs/
    round2_20260927/        TD_EXP_ROOT — ALL THREE ACCOUNTS WRITE HERE
  work/        round2.txt worklist
  logs/{xiaoyuxu2,qiangzeng,danranwang,staging}/
  archive/
    round1/
      tdgs_round1_harvest.json            75 cells
      selection_reports/*.json            5 staging reports
      cells/ratio_0.02/<ds>/<arm>/seed_<s>/   metrics.jsonl, predictions_clean.npz,
                                              run_config.json, run_complete.json,
                                              selected_indices.npy  (89 MB total)
      checkpoints_MANIFEST.txt            paths of the 75 final.pt left in place
      logs/                               slurm logs of jobs 34966–34970
    round2/                               round-2 reports + harvest
```

`/mnt/prj01/hgrp-1502-5TB` is group-writable with the setgid bit, so new
subdirectories inherit `hgrp-1502`. Jobs are launched with **`umask 002`** so the
group also gets write permission — without it, the first account to create a
directory locks the other two out halfway through the matrix.

**Why this removes a whole class of bug.** `td_run_one.sh` skips any cell that
already has `predictions_clean.npz`, and that path is now under the *shared*
`TD_EXP_ROOT`. So a cell finished by `qiangzeng` is skipped by `danranwang`
automatically — no coordinator, no lock file, **no post-hoc merge**. The earlier
three-account rounds wrote to three separate homes and had to merge done-keys by
hand before every launch; forgetting that re-ran finished work.

## Read-only project paths

```
$PROJECT = /project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab
  envs/medmnistc-py311/bin/python                  the shared training env
  sources/table1_clean_only_20260922/              the harness source tree
  data/medmnist/<ds>.npz                           pool images + labels
  data/medmnistc/                                  corruption data
  table1_clean_only_20260922/
    cache/embeddings_img224_smokefull/<ds>_train_uni_224.npz    frozen UNI
    cache/                                          graph cache
    hf_cache/                                       HF_HOME (offline)
  table1_clean_only_20260922/formal/               the Table-1 archive (394/400)
  tdgs_round1_20260927/formal/ratio_0.02/<ds>/<arm>/seed_<s>/results/.../aug_0/
                                                   round 1, 75 cells + 3.3G of final.pt

/project/prj-sis01/xuxiaoyu/graph_select           graphcov source (sys.path)
```

Mirror root for embeddings if the first is missing:
`/mnt/prj01/hgrp-1502-5TB/xuxiaoyu/reliability_medmnistc_ab/table1_clean_only_20260922/cache/embeddings_img224_smokefull`

## Per-account writable graph cache

`TD_CACHE_ROOT` defaults to the project cache, which the two non-owner accounts
cannot write. `launch_3accounts.sh prep_caches` creates `$HOME/tdgs_cache` on each
account, seeded with **symlinks** to the real cache entries: reads hit the
originals, and any write the pipeline attempts lands in the account's own home
instead of failing on a read-only mount.

## Harness env-var contract (`td_run_one.sh`)

| var | meaning |
|---|---|
| `TD_EXP_ROOT` | result tree root; `RUN_ROOT = $TD_EXP_ROOT/formal/ratio_$RATIO/$DATASET/$ARM/seed_$SEED` |
| `TD_SEL_DIR` | where `${DS}_r${RATIO}_${ARM}_s42.npy` is read from |
| `TD_CACHE_ROOT` | `GRAPH_CACHE` |
| `TD_HF_ROOT` | `HF_HOME`; `HF_HUB_OFFLINE=1` |
| `SEL_WAIT` | seconds to wait for a not-yet-staged selection before erroring |

Fixed by the harness: `methods: [precomputed]`, `selection_seed: 42`,
`epochs: 1000`, `image_size: 224`, `batch_size: 256`, `embedding_source: uni`,
`checkpoint_rule: final_epoch`, `DETERMINISTIC_TRAINING=1`, `CLEAN_ONLY=1`.

## Job history for this line of work

| job | what | state |
|---|---|---|
| 34966/34967 | `tdgssmoke` — organsmnist validity smoke | COMPLETED; `graph_a2` arm reproduces the archive **order-exact** |
| 34968/34969 | `tdgssel` — staging, 5 datasets × 5 arms @2% | COMPLETED; 25 selections |
| **34970_0..3** | `tdpack` — round 1, **75 cells** | **all COMPLETED** (2:32–2:58 each) |
| 34974 | `proxyreg` — demand-proxy regime diagnostic | **CANCELLED** (not by this session); **retired, do not resubmit** |

Not ours, never touched, flagged repeatedly: `34873 lam0stage`, `34883 corrsmoke`,
`34884 bloodrep`, and several files under `~/tdfix/` on the shared `xiaoyuxu2`
account.
