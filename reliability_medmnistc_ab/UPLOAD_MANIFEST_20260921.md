# GitHub Upload Manifest

更新时间：2026-09-21（本地归档快照）

这份清单说明本次提交上传的文件是什么、来自哪里、能用于什么。HPC 数据集、embedding、checkpoint、prediction `.npz` 和 Python/Conda 环境不上传。

## 来源分层

| 来源 | 含义 |
| --- | --- |
| `official GraphCov` | 作者仓库的 GraphCov 选样实现。当前实验记录的基准 commit 是 `8cf757adc4c333dc1427d511f0de2f246d15ebac`。 |
| `MedMNIST-C API` | 官方 corruption 生成和注册表。当前实验记录的 API commit 是 `8acfd2710c6e0e8b2745be8b1fa1c17b94b8f8a7`。 |
| `reliability wrapper` | 本项目为固定数据、训练、评估、哈希和 Slurm 执行增加的封装，不改变 GraphCov 选择公式。 |
| `diagnostic` | 为定位 selection/training/test 流程问题新增的诊断脚本，不是论文主结果。 |
| `HPC snapshot` | 从 `xiaoyuxu2` HPC 项目目录取回的日志、smoke 结果和汇总表。 |

## 代码与配置

| 路径 | 功能 | 来源 |
| --- | --- | --- |
| `graphcov/` | 作者 GraphCov 的主要选样、图构建、embedding、训练评估实现 | 作者仓库固定版本 |
| `reliability_medmnistc_ab/pipeline_remote_snapshot.py` | HPC 上实际运行的审计版 pipeline 快照，包含数据加载、选样调用、训练、评估、指标和完成标记 | HPC `reliability_medmnistc` 运行仓库 |
| `reliability_medmnistc_ab/scripts/` | 下载/预处理、MedMNIST-C 生成、方差诊断和汇总脚本 | 本项目 reliability wrapper |
| `reliability_medmnistc_ab/configs/` | 正式实验、qiangzeng 多种子和修复任务的 YAML 配置 | 本项目实验协议 |
| `reliability_medmnistc_ab/protocol/` | 数据下载、MedMNIST-C 生成和固定协议 manifest | HPC 运行时生成/核验 |
| `reliability_medmnistc_ab/slurm/` | HPC 作业入口、资源申请、环境变量和任务编排 | 本项目 Slurm 封装 |
| `reliability_medmnistc_ab/tests/` | 配置、指标和诊断完整性测试 | 本项目测试 |
| `reliability_medmnistc_ab/research/` | 鲁棒选择相关文献和实验设计记录 | 本项目研究记录 |

## 本次新增的修复验证代码

| 路径 | 功能 | 来源/状态 |
| --- | --- | --- |
| `reliability_medmnistc_ab/scripts/diagnostics/eval_test_corruption.py` | 只加载已有 ABC `final.pt`，在 clean test 和 MedMNIST-C test 上推理；不重新选样、不训练 | 从 HPC 修复版 `tmp/remote_patch/eval_test_corruption.py` 归档；SHA256 与远端文件一致 |
| `reliability_medmnistc_ab/slurm/phase1_diag_stage_b_factorial36.slurm` | 运行 Random/Graph-A2 × Path/Blood × selection seed 42/43/44 × training seed 42/43/44 的 36 格析因实验 | 从 HPC 修复版归档；修复了 Bash 数组与环境变量同名导致的默认配置回退 |

## 结果与日志

| 路径 | 内容 | 是否正式论文结果 |
| --- | --- | --- |
| `reliability_medmnistc_ab/summaries/` | 单种子三臂逐运行、逐 corruption、Graph-A2 胜出位置和多种子已有汇总 CSV | 单种子是描述性结果；多种子须按完成标记复核 |
| `reliability_medmnistc_ab/results/summary_tables/` | selection robustness 和其他诊断汇总 | 诊断结果 |
| `reliability_medmnistc_ab/results/validation_smoke_20260921/` | Job `32949` 的单元 smoke：Random/PathMNIST/selection 42/training 42/10 epochs；包含配置、history、metrics、完成标记和选择哈希 | 仅验证代码，不计入正式 benchmark |
| `reliability_medmnistc_ab/logs/diagnostics/` | Job `32945` item1 日志和 Job `32949` item3 smoke 日志 | 运行证据；item1 日志快照时仍在运行 |
| `reliability_medmnistc_ab/logs/multiseed_32712/` | 原三臂多种子作业的 Slurm stdout/stderr | 历史运行日志 |
| `outputs/medmnistc_final_handoff_20260921/MedMNISTC_experiment_snapshot.xlsx` | 最新实验状态、协议、单种子结果、selection 诊断和 benchmark 计划的 Excel 快照 | 汇总快照，不替代原始 HPC 产物 |
| `outputs/medmnistc_analysis_20260920/MedMNISTC_selection_analysis.xlsx` | selection 分析 Excel | 汇总快照 |
| `outputs/experiment_results_20260918/qiangzeng_experiment_results.xlsx` | qiangzeng 环境对照结果 Excel | 环境 A/B 快照 |
| `outputs/pathmnist_formal_results_20260918/pathmnist_formal_results.xlsx` | PathMNIST 形式化结果 Excel | 历史正式结果快照 |

## 2026-09-21 上传时状态

- `32949`：item3 smoke 已完成；日志确认只使用 `datasets=['pathmnist']`、`methods=['random']`、`epochs=10`，生成 `run_complete.json`。
- `32945`：item1 BloodMNIST 推理任务仍在运行；日志已有 checkpoint 推理完成行，没有实际训练 epoch 行，不能在完成前当作 80 个 checkpoint 的最终结果。
- smoke 和诊断结果不会混入正式 Excel 主结果，也不会用于宣称 Graph-A2 SOTA。

## 明确不上传的内容

以下内容留在 HPC 或本地缓存，不进入 Git：

- MedMNIST/MedMNIST-C 原始数据和生成缓存；
- UNI embedding 和 graph cache；
- `final.pt`、预测 `.npz`、训练动态大文件；
- Conda/venv、`node_modules`、Python bytecode；
- 临时的失败输出和未验证的 partial checkpoint。

GitHub 上的 CSV/Excel/日志是可读的实验审计摘要；要重建完整结果，仍需使用文档中的 HPC 路径和固定 commit，并重新生成未上传的大文件。
