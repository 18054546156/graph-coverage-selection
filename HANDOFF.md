# MedMNIST-C Reliability Handoff

核查日期：2026-09-16。本文记录当前 GitHub clean 脚本版和 HPC 正式单种子实验状态。它不是论文结果报告；没有 `run_complete.json` 和正式汇总前，不得宣称鲁棒性实验完成。

## 研究目标

比较 5 个 MedMNIST 数据集上 8 种 equal-class-quota 数据选择方法，在 2% 和 5% 训练预算下训练同一 ResNet-18，并在相同 clean test 和 MedMNIST-C test 上比较：

- Accuracy、balanced accuracy；
- 每类 recall、worst-class recall；
- NLL、Brier、ECE；
- Risk@80%、AURC、高置信错误数；
- clean 到 corruption 的 BA drop。

当前协议是单 selection seed 和单 training seed：

```text
selection_seed = 42
training_seed  = 42
5 datasets × 8 methods × 2 ratios = 80 subset runs
```

这不是“只标注 2%”实验，因为选择器使用完整训练标签和等类配额；准确表述是减少下游训练集规模。

## 唯一代码来源

GitHub 分支：

```text
https://github.com/18054546156/graph-coverage-selection/tree/codex/medmnistc-reliability-clean
```

当前远端 HEAD：

```text
565ea591b6145f6dbad54ef6f31eda8ef0b06298
```

正式新提交使用固定执行快照：

```text
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/sources/graph-coverage-selection-medmnistc-reliability-clean-d74/
HEAD = d74e2d6fb6b348ad0cbb410b4931d90540afd754
MedMNIST-C = 8acfd2710c6e0e8b2745be8b1fa1c17b94f8a7
GraphCov = 8cf757adc4c333dc1427d511f0de2f246d15ebac
```

当前运行中的旧任务 `31830_4` 是在固定提交要求之前从 `565ea591...` 启动的既有任务。它必须保留，不取消、不重复提交；该例外已记录在正式 manifest。后续新任务必须使用 `d74e2d6...`。

## 正式代码入口

```text
reliability_medmnistc/scripts/prepare_medmnist_224.py
reliability_medmnistc/scripts/generate_medmnistc.py
reliability_medmnistc/scripts/select.py
reliability_medmnistc/scripts/validate_selections.py
reliability_medmnistc/scripts/train.py
reliability_medmnistc/scripts/evaluate.py
reliability_medmnistc/scripts/summarize.py
reliability_medmnistc/reliability/pipeline.py
```

Slurm 入口：

```text
reliability_medmnistc/slurm/00_preflight.slurm
reliability_medmnistc/slurm/01_generate_corruptions.slurm
reliability_medmnistc/slurm/02_select_array.slurm
reliability_medmnistc/slurm/03_train_eval_array.slurm
reliability_medmnistc/slurm/04_summarize.slurm
reliability_medmnistc/slurm/05_integration_smoke.slurm
```

配置：

```text
reliability_medmnistc/configs/full_5datasets_8methods.yaml
```

关键配置是 224 下游图像、28 dynamics 图像、dynamics 200 epochs、ResNet-18 1000 epochs、batch size 256、4 workers、无增强、final epoch checkpoint。

## HPC 路径

正式根目录：

```text
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/formal_runs/single_seed_sel42_train42_20260915/
```

数据：

```text
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/data/medmnist/
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/data/medmnistc/
```

正式产物：

```text
selections/   # 80 个 selection artifacts
runs/         # final.pt、训练历史、预测、metrics、完成标记
logs/         # Slurm stdout/stderr
cache/        # UNI embedding 和 dynamics cache
submission_manifest.txt
```

Python：

```text
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/envs/medmnistc-py311/bin/python
```

## 当前完成度

### 已完成

- clean GitHub 脚本版已上传；
- 5 个数据集的 28/224 MedMNIST 数据已准备；
- MedMNIST-C 数据已生成并缓存；
- TissueMNIST 5% selection 修复作业 `31838_3` 已成功，`ExitCode=0`；
- selection 文件数量已达到 `80/80`；
- 旧作业 `31757_3` 已确认 `CANCELLED`，不计入结果；
- 旧 pending 链 `31832、31833、31834、31835` 已取消并保留日志；
- `PathMNIST 2%` 的 `31830_4` 正在继续运行，未重复提交。

### 尚未完成

- `80/80` selection 的严格验证尚未成功；
- 训练测试尚未完成；
- 当前已有部分 `final.pt` 和 `training_history.json`，但不是最终结果；
- `metrics.jsonl`、prediction 文件和 `run_complete.json` 尚未全部生成；
- `04_summarize.slurm` 尚未提交；
- 没有正式 clean/corruption BA、ECE、AURC 或 BA drop 结果。

## 最近失败与修复

验证作业 `31843` 失败，原因不是 selection artifact 内容错误，而是验证作业申请了 `0 GPU`，没有执行 Slurm 环境中的 ImageMagick 初始化，并触发了 Wand/MagickWand 与 Faiss CUDA 环境错误。

失败作业和日志必须保留：

```text
selection_validate_31843.out
selection_validate_31843.err
```

`31844_[0-3]` 因依赖 `31843` 失败而进入 `DependencyNeverSatisfied`，已取消，不作为训练结果。

修复后的提交：

```text
31857       GPU + ImageMagick 初始化的 selection 验证
31858_[0-3]  afterok:31857，OrganSMNIST/OrganAMNIST 2%/5% train/eval
```

`31858` 每个任务申请：

```text
1 GPU / 8 CPU / 48G RAM
```

## 正确执行顺序

```text
31857 严格验证 80/80 selection
        ↓ afterok
31858_[0-3] 先跑 OrganSMNIST、OrganAMNIST 2%/5%
        ↓
保留运行中的 31830_4 PathMNIST 2%
        ↓
继续提交尚未运行的 task 5、6、7、8、9
        ↓
验证 80 个 final.pt、training_history、clean/corruption predictions、metrics、run_complete
        ↓
提交 04_summarize.slurm
        ↓
核验 metrics.csv、clean_metrics.csv、corruption_metrics_by_severity.csv、summary_rejections.json
```

任务映射：

```text
0 OrganSMNIST 2%     1 OrganSMNIST 5%
2 OrganAMNIST 2%     3 OrganAMNIST 5%
4 PathMNIST 2%       5 PathMNIST 5%
6 TissueMNIST 2%     7 TissueMNIST 5%
8 BloodMNIST 2%      9 BloodMNIST 5%
```

当前 QoS 是 `5 running jobs / 12 GPU / 128 CPU / 256G RAM`。在 `31830_4` 仍申请 64G 时，最多同时运行 4 个新的 48G train/eval 任务：

```text
64G + 4 × 48G = 256G
```

不要重复提交已有运行或 pending 的 task，不要删除有效 selection、cache、checkpoint、prediction 或结果目录。

## 结果判定规则

只有某个 run 同时具备 checkpoint、训练历史、clean prediction、全部支持的 corruption/severity prediction、metrics 和 `run_complete.json`，才算该 run 完成。

只有 80 个 run 全部完成并通过 hash/config 检查，且 summary 没有拒绝项，才算单种子正式实验完成。旧 v29 notebook、smoke 结果、旧 Table 1 结果不能混入正式汇总。

Step 1 表征漂移/coverage 诊断和 Step 2 selection/training variance 诊断是后续解释实验，不属于当前 80-run 主实验完成度。

## 当前结论

当前状态是：

```text
代码：完成并上传
数据和 corruption：准备完成
selection：80/80 文件已生成，严格验证待修复作业 31857
train/test：进行中
正式指标：尚未完成
汇总：尚未开始
```

因此目前不能报告八方法的最终鲁棒性排名，也不能把已有 checkpoint 当作论文结果。
