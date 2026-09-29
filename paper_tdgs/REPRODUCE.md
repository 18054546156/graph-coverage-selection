# REPRODUCE — syncing code, data, models and experiments on another server

This directory is a self-contained **research package**. It holds the code, preregistrations, handoff, raw results, logs and selection sets.
All training happens in the harness below; this package only generates *selections* and does the *analysis*.

> Naming note (2026-09-28): the "TD" in TDGS stands for **Task-Demand**. It refers to the out-of-fold proxy "demand" of round 1.
> It does **not** mean target/test: **no TDGS arm reads test-set images or embeddings** (source-only; only the train pool plus train labels are used).
> Rounds 1–2 showed that the demand term does not help (difficulty −2.51pp, direction inert).
> The surviving arm `tdgs_cls` is really "class-scoped coverage + class-flat weighting", so the paper will be renamed (see HANDOFF §0.1).
> Code and file names keep `tdgs_*` so results stay traceable.

## 0. What is in this directory

| Path | Contents |
|---|---|
| `HANDOFF_20260927.md` | **Current status, single source of truth** (read §0 first) |
| `report/evidence_ledger_20260928.md` | Full audit ledger (verified / unsupported / post-hoc interpretation) |
| `report/r3_prereg_20260928.md` + `report/r3_prereg_amendment_20260928.md` | Round 3 preregistration and its dated amendments |
| `report/research_plan_20260928.md` | Route comparison, maths, phased plan, paper plan |
| `docs/` | Method, protocol, instruments, closed routes, paper design |
| `code/` | **Development version** (paths overridable via environment variables) |
| `code_asrun/` | The **exact version that ran** on the cluster (snapshot 2026-09-28; `MANIFEST.md5`) |
| `results/round1`, `results/round2` | Harvested per-cell BA / worst-recall JSON and ladder tables |
| `results/selections/r1r2`, `results/selections/r3` | **Selected-index `.npy` files** (train-pool indices, `{ds}_r{ratio}_{arm}_s42.npy`); lets you train directly without redoing selection |
| `results/r3_report/` | R3 selection-space diagnostics (val kNN BA, xshare, cross-view coverage, Jaccard, identity checks) |
| `results/archive_round2/` | Round 2 selection reports |
| `results/work/` | Training worklists (`<ds> <ratio> <arm> <seed>`) |
| `results/prereg_cluster/` | Read-only preregistration copies from the cluster, with timestamps (`STAMP*.txt`) |
| `logs/cluster/` | Slurm logs |
| `env/` | Python version, `pip freeze`, harness commit, and harness patch `harness_pipeline_tdhook.patch` |
| `scripts/download_medmnist_224.py` | Downloads MedMNIST 224 and checks md5 |

## 1. Environment

- Python 3.11.9. Key packages: torch 2.5.1 (CUDA 12.1), torchvision (bundled with torch), timm 1.0.28, numpy 1.26.4, scipy 1.17.1, scikit-learn 1.9.0, faiss 1.9.0, medmnist 3.0.1. Full list: `env/requirements_medmnistc-py311_freeze.txt`.
- GPU: A100 / H100 were used. **Training must set `DETERMINISTIC_TRAINING=1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `PYTHONHASHSEED=0`** to be bit-reproducible across repeat runs on the same hardware.
  - Across hardware, inference logits differ at about 1e-2 while argmax is identical (`val_eval.py` self-check).

```bash
python3.11 -m venv envs/medmnistc-py311
envs/medmnistc-py311/bin/pip install -r paper_tdgs/env/requirements_medmnistc-py311_freeze.txt
```

## 2. Code

```bash
# The training harness = this repo (fork18) at branch codex/medmnistc-reliability-clean, commit 565ea59
git clone https://github.com/18054546156/graph-coverage-selection.git harness
cd harness && git checkout 565ea591b6145f6dbad54ef6f31eda8ef0b06298
git apply /path/to/paper_tdgs/env/harness_pipeline_tdhook.patch   # adds the precomputed/TD_SEL_DIR hook actually used on the cluster
git submodule update --init third_party/medmnistc                  # medmnistc-api @ 8acfd27 (only for corruption evaluation; not needed for clean-only)
cd ..
# This research package = this repo at branch tdgs-research
git clone -b tdgs-research https://github.com/18054546156/graph-coverage-selection.git graph_select
```

`graphcov/` is the unmodified upstream Graph-A2 code (commit 8cf757a). **Do not edit it**; the harness checks its hash.

## 3. Data

```bash
python paper_tdgs/scripts/download_medmnist_224.py --root /data/medmnist   # {ds}_224.npz, md5 checked
# also needed: 28px {ds}.npz (labels are read from it; same order as the 224 version). Download with the medmnist package:
python -c "import medmnist; [getattr(medmnist, medmnist.INFO[d]['python_class'])(split='train', download=True, root='/data/medmnist') for d in ['bloodmnist','organamnist','organsmnist','pathmnist','tissuemnist']]"
```

Five datasets: blood, organA, organS, path, tissue. **Test labels are used only for the final evaluation.** Selection and every configuration choice read the train split or the val split only.

## 4. Models (frozen encoders)

| View | Source | Notes |
|---|---|---|
| UNI | `hf-hub:MahmoodLab/UNI` | **Gated**: accept the licence on the HF page and `huggingface-cli login` first. Build exactly as graphcov `get_uni_model` does (init_values=1e-5, dynamic_img_size=True) |
| DINOv2-L | timm `vit_large_patch14_dinov2.lvd142m` | Public; timm downloads it automatically |
| CLIP-L | timm `vit_large_patch14_clip_224.openai` | Public |

After downloading, set `HF_HOME=<cache>` and `HF_HUB_OFFLINE=1` for offline use on compute nodes.

## 5. Path environment variables (defaults are the original cluster paths)

| Variable | Used by | Meaning |
|---|---|---|
| `TDGS_PROJECT` | tdgs_select.py, val_eval.py | Project root; `$TDGS_PROJECT/data/medmnist/{ds}.npz` must exist |
| `MEDMNIST_ROOT` | extract_emb.py, val_eval.py | Directory holding `{ds}_224.npz` |
| `TDGS_UNI_EMB_DIRS` | tdgs_select.py | Directories holding `{ds}_train_uni_224.npz` (colon-separated) |
| `TDGS_SHARED` | mv_select.py | Shared tree (`r3/emb`, `r3/sel`, `sel`) |
| `GRAPH_SELECT_ROOT` | tdgs_select*.py, mv_select.py | This repo's root (to import `graphcov`) |

The Slurm scripts (`*.slurm`, `t1_run_one.sh`, `code_asrun/td_run_one.sh`) contain absolute cluster paths and the `qos-high-gpu` / `--exclude=hpcgpu108` settings. **Edit the variables at the top of each script** on another cluster.

## 6. Reproduction pipeline

```bash
cd graph_select/paper_tdgs/code
# (a) UNI train embeddings. The archive uses the graphcov extraction; the re-extraction below was checked bit-identical on blood/organA with a 512-sample smoke (maxabs 0)
python extract_emb.py --datasets bloodmnist --encoders uni --splits train val --out $EMB
python extract_emb.py --datasets bloodmnist --encoders dinov2 clip --out $TDGS_SHARED/r3/emb
export TDGS_UNI_EMB_DIRS=$EMB
# (b) Selection (seconds to tens of minutes; tissue needs ~200 GB RAM)
python tdgs_select.py    --dataset bloodmnist --ratio 0.02 ...     # round 1 arms, see --help
python tdgs_select_r2.py --dataset bloodmnist --ratio 0.02 ...     # round 2 arms
python mv_select.py      --dataset bloodmnist --ratio 0.02 --out-dir $TDGS_SHARED/r3/sel --report $TDGS_SHARED/r3/report
#     or skip selection and use results/selections/*.npy directly
# (c) Training: one cell = (ds, ratio, arm, seed), ResNet-18 from scratch, 1000 epochs, final epoch
TD_SEL_DIR=<selection dir> TD_EXP_ROOT=<output tree> bash ../code_asrun/td_run_one.sh bloodmnist 0.02 tdgs_cls 42
#     batch mode: tdgs_pack.slurm + results/work/*.txt worklists (cells already holding predictions_clean.npz are skipped automatically)
# (d) Val evaluation (Amendment 2; no retraining)
python val_eval.py --roots <output tree>/formal --out <val_eval dir>
# (e) Harvest and statistics
python harvest.py ... ; python ladder.py ... ; python audit_contrasts.py ...
```

Seed convention: training seed ∈ {42..46} (the author's convention). The selection itself is deterministic (`_s42` only marks it).

## 7. Status labels (read before quoting anything; updated 2026-09-29)

- **Verified (training results)**: rounds 1–2 (260 cells), Round 3 P4 (130 cells) + hardware control P4-e (25), Table 1 baselines on seeds 42–46 (350). Gates read 09-29: **G-MV STOP, G-R2 FAIL, extended vsel FAIL** — see `report/r3_validation_report_20260928.md` §2, §5–§7.
- **Harness identity**: 40/40 overlapping A100 cells are bit-identical to the Table-1 archive; H100 cells are not (same selection sha). Measured GPU-type noise on BA: per-cell sd 2.38pp, unbiased (P4-e). Reproduce on A100 to match bit for bit.
- **Not executed**: R3 at 5%, DermaMNIST, CAMELYON17 (not triggered by the gates).

## 8. Re-running the analysis from the shipped files (no GPU)

```bash
cd paper_tdgs/code
python r3_gates.py --r3 ../results/round3/r3_harvest.json --r3hw ../results/round3/r3hw_harvest.json     --round1 ../results/round1/tdgs_round1_harvest.json --round2 ../results/archive_round2/round2_harvest.json     --r3-report ../results/r3_report --out /tmp/r3_gates.json
python make_table1.py --baselines ../results/table1/t1_s4246_harvest.json     --round2 ../results/archive_round2/round2_harvest.json --round1 ../results/round1/tdgs_round1_harvest.json
```
`vsel_analysis.py` additionally needs each cell's `metrics.jsonl`, which the full repository ships under `cluster_outputs/runs/` (paths inside `val_metrics.jsonl` are cluster paths; see its docstring).
