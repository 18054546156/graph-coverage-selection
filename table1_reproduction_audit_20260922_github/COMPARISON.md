# 当前项目与作者实现的差异摘要

## 结论

Table 1 严格复现路径将作者仓库固定在 commit `8cf757adc4c333dc1427d511f0de2f246d15ebac`，并通过 `official_manifest.json` 校验。作者 `graphcov/` 算法源码与 vendored 副本相同；本地增加的是配置、分阶段选择/训练 orchestration、Slurm 部署和防泄漏评估流程。

clean-only 多种子实验是另一条独立实现。它复用作者方法/选择代码，但有自己的训练、输出和 seed 协议，不能与 strict runner 混称为完全相同的复现。

## 重要差异

| 维度 | 作者论文/公开代码 | 当前项目 | 影响 |
|---|---|---|---|
| 复现配置入口 | 上游 README/CLI 示例与默认值；论文 Table 1 细节在论文设置中 | `table1_reproduction/configs/` 显式配置 | 用项目 config 才能还原论文设定；README demo 不是 Table 1 全配置 |
| Graph-A2 | Table 2 支持 global kNN、k=50、H=2 | strict config 显式配置 k=50、H=2、global | 对齐论文证据；不要使用 CLI 默认 k=10 |
| Facility scope | 公开 runner 默认 per-class；论文对 global/per-class 语义不够明确 | strict 和 clean-only 均 per-class | 属于实现假设，不是论文完全证明的设置 |
| 评估 | 作者公开 epoch runner 每 10 epoch 在 test 上评估并跟踪 best test BA；最后仍评估 final 模型 | strict runner 用 validation BA 选 checkpoint，再 test 一次；clean-only 用 final epoch，最后 test | strict 是防 test 泄漏的本地修正；clean-only 的最终指标口径更接近作者返回的 final BA，但没有周期性 test 评估 |
| 选样随机性 | 作者 orchestration 对确定性方法复用冻结子集；Random/FPS 按 trial 改 selection seed | clean-only 每个 seed 都使用 selection_seed=training_seed | clean-only 同时改变选集与训练随机性；不能只解释为 training-seed 方差 |
| trial 数 | 论文报告 5 trials | clean-only 设计 8 paired seeds | 是重复数扩展，不是逐字复刻论文 trial 设置 |
| 比例 | Table 1 报 2% 和 5% | clean-only 目前配置 2% | 当前 clean-only 只覆盖 Table 1 的 2% 条件 |
| 可复现性设置 | 作者 CLI 默认不启用 `--deterministic`；论文未明确 deterministic kernel | clean-only 启用 `DETERMINISTIC_TRAINING=1` | 协议有意不同；会改变运行时间/复现噪声，不应与旧非确定性结果混池 |
| 数据/任务边界 | 作者 Table 1 使用 clean MedMNIST | clean-only 是 clean select/train/test；其他可靠性项目另跑 MedMNIST-C | 不要把 corruption 三臂结果并入 Table 1 clean-clean 指标 |

## 两条本地 pipeline 的定位

1. `table1_reproduction/`：严格对齐论文主要超参数，Job 1 冻结选集，Job 2 只看 validation 选 checkpoint、test 评估一次。这是防泄漏的本地协议修正，不是原作者公开 runtime 原样行为。
2. `table1_clean_only_20260922/`：隔离的 clean-only paired-seed 扩展，2%、8 个 seed、final-epoch test。它的实验问题是跨选样与训练随机性的联合敏感度，不是固定选集下单独估计训练 seed 方差。

## 生成的 diff 文件

- `diffs/PROJECT_VS_AUTHOR.patch`：新增的项目 runner/config/脚本相对于作者仓库的 Git diff。
- `diffs/VENDOR_VS_AUTHOR.patch`：作者源码与 Table 1 vendored 源码的差异；预期无差异。
- `diffs/SHARED_GRAPHCOV_VS_AUTHOR.patch`：项目共享 `graphcov/` 相对于作者源码的差异。该代码不代表 strict Table 1 runner 实际导入的源码。
- `diffs/FILE_MANIFEST.csv`：审计副本文件、来源类别和 SHA256。

## 文件对应关系

- 作者算法实现：`original_author/graphcov/run/selection.py`、`graph.py`、`evaluation.py`、`experiment.py`。Table 1 vendored 源码位于 `current_project/table1_reproduction/vendor/graphcov/`，与作者源码一致。
- 严格 runner：作者单体调用路径主要是 `graphcov/run/experiment.py` + `evaluation.py`；本地拆成 `current_project/table1_reproduction/experiments/job1_select.py`、`job2_downstream.py`、`official_runtime.py`，并由 `configs/` 和 `slurm/` 驱动。
- clean-only runner：`current_project/table1_clean_only_20260922/pipeline.clean_only.patch.py` 与作者的 `evaluation.py` 不是逐行同文件对照；它是独立 pipeline，差异细节见上表，参数和 checkpoint 写入点见该脚本及 `run_clean_only.slurm`。
- 论文未规定或本地新增的设置会体现在本地 JSON/YAML/Slurm 配置及 orchestration 中，而不是作者源码改动。

`git diff --check` 发现快照里有少量原有格式提示（文件末尾空行及一处行尾空格）。源文件为保持可审计性没有被格式化或修改；这些不是运行逻辑差异。

审计包还带有 Table 1 组会目录里的历史配置/证据、派生汇总表和相关生成 notebook。没有纳入其图表 PNG、Slurm `.out/.err` 日志或其他实验数据。clean-only 的 CSV 快照只用于说明当时的结果状态，不会被视作源代码或原作者结果。
