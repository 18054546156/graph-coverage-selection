# Reliability MedMNIST-C Experiment Handoff

更新时间：2026-09-20（以 HPC 实际文件和 Slurm 记录为准）

本文档是 `reliability_medmnistc_ab` 的执行交接记录。该实验独立于 Table 1 复现，不修改官方 GraphCov 或 MedMNIST-C 核心源码。

## 0. 最新交接摘要：2026-09-20

### 单种子正式三臂实验：已完成 `120/120`

固定配置：

```text
datasets: OrganSMNIST, OrganAMNIST, PathMNIST, TissueMNIST, BloodMNIST
methods: Random, EL2N, Forgetting, EVA, Facility, FPS, Herding, Graph-A2
ratio: 2%
selection_seed: 42
training_seed: 42
epochs: 1000
corrupted train view: pixelate, severity=3
checkpoint: final epoch
```

三个实验臂均已完成：

```text
clean selection -> clean training -> clean/corrupted test: 40/40
corrupted selection -> clean training -> clean/corrupted test: 40/40
corrupted selection -> corrupted training -> clean/corrupted test: 40/40
```

本轮合并来源：clean baseline `40` 个，加上 `xiaoyuxu2` 已完成的 `55` 个，再加上 `qiangzeng` 修复的 `25` 个。修复 Job `32707` 的五个子任务全部 `COMPLETED, ExitCode=0`。

### 单种子汇总结果

下表是五个数据集的平均值。`corruption BA` 是该数据集全部 MedMNIST-C corruption/severity 条件的平均；`BA drop = clean BA - corruption BA`。百分比均已乘以 100。

#### Clean selection -> clean training

| 方法 | Clean ACC | Clean BA | Corruption BA | BA drop | Clean worst recall | Corruption worst recall |
|---|---:|---:|---:|---:|---:|---:|
| Random | 71.50 | 70.07 | 53.67 | 16.40 | 44.93 | 23.18 |
| EL2N | 41.74 | 43.70 | 33.60 | 10.11 | 12.20 | 6.30 |
| Forgetting | 55.19 | 55.76 | 41.59 | 14.17 | 23.49 | 10.49 |
| EVA | 58.37 | 58.74 | 44.57 | 14.18 | 20.80 | 10.21 |
| Facility | 72.94 | 71.48 | 54.04 | 17.44 | 47.53 | 24.24 |
| FPS | 68.55 | 67.93 | 50.97 | 16.95 | 37.42 | 18.36 |
| Herding | 73.83 | 72.29 | 53.19 | 19.10 | 46.53 | 22.32 |
| Graph-A2 | 73.30 | 71.83 | 52.14 | 19.69 | 47.43 | 21.95 |

#### Corrupted selection -> clean training

| 方法 | Clean ACC | Clean BA | Corruption BA | BA drop | Clean worst recall | Corruption worst recall |
|---|---:|---:|---:|---:|---:|---:|
| Random | 72.44 | 70.93 | 53.12 | 17.80 | 46.89 | 23.91 |
| EL2N | 42.45 | 43.19 | 32.22 | 10.98 | 7.69 | 3.80 |
| Forgetting | 54.42 | 53.54 | 40.16 | 13.38 | 20.61 | 9.28 |
| EVA | 59.32 | 57.89 | 43.10 | 14.79 | 22.90 | 10.91 |
| Facility | 72.17 | 70.48 | 53.96 | 16.52 | 45.10 | 23.55 |
| FPS | 66.09 | 65.76 | 48.64 | 17.11 | 34.25 | 16.41 |
| Herding | 72.24 | 70.89 | 52.80 | 18.09 | 45.35 | 22.69 |
| Graph-A2 | 72.06 | 70.46 | 50.76 | 19.70 | 43.90 | 19.48 |

#### Corrupted selection -> corrupted training

| 方法 | Clean ACC | Clean BA | Corruption BA | BA drop | Clean worst recall | Corruption worst recall |
|---|---:|---:|---:|---:|---:|---:|
| Random | 68.60 | 67.18 | 50.33 | 16.85 | 41.47 | 21.81 |
| EL2N | 45.62 | 46.81 | 34.39 | 12.42 | 9.14 | 4.14 |
| Forgetting | 53.86 | 52.89 | 39.51 | 13.37 | 16.71 | 8.52 |
| EVA | 56.14 | 55.02 | 40.86 | 14.16 | 21.30 | 10.23 |
| Facility | 70.81 | 69.84 | 52.98 | 16.87 | 43.90 | 23.55 |
| FPS | 64.25 | 63.76 | 48.91 | 14.86 | 28.83 | 16.73 |
| Herding | 72.40 | 70.75 | 52.98 | 17.76 | 43.53 | 23.53 |
| Graph-A2 | 71.81 | 70.58 | 54.05 | 16.53 | 43.68 | 25.18 |

这些是单种子描述性结果，不是跨 training seed 的置信区间。NLL、Brier、ECE15、Risk@80、AURC、逐 corruption/severity 指标和每个 selected index SHA256 已保存在逐运行 CSV 与原始 `metrics.jsonl` 中。

### 单种子结果路径

```text
# Clean baseline, 40/40
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/formal_runs/single_seed_5datasets_8methods_ratio2_clean_20260918_xiaoyuxu2/

# Corrupted selection -> clean training, original 30 + qiangzeng repair 10
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/formal_runs/three_arm_single_seed_20260919_xiaoyuxu2/selection_corrupted_clean_train_pixelate_s3/
/home/qiangzeng/prj_qiangzeng/three_arm_single_seed_20260920_qiangzeng_repair/selection_corrupted_clean_train_pixelate_s3/

# Corrupted selection -> corrupted training, original 25 + qiangzeng repair 15
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/formal_runs/three_arm_single_seed_20260919_xiaoyuxu2/selection_corrupted_corrupted_train_pixelate_s3/
/home/qiangzeng/prj_qiangzeng/three_arm_single_seed_20260920_qiangzeng_repair/selection_corrupted_corrupted_train_pixelate_s3/

# Generated complete summaries
/home/qiangzeng/prj_qiangzeng/summaries/three_arm_single_seed_20260920_summary.csv
/home/qiangzeng/prj_qiangzeng/summaries/three_arm_single_seed_20260920_per_run.csv
```

`three_arm_single_seed_20260920_per_run.csv` has 120 rows, one per completed dataset-method-arm cell, and includes clean/corruption ACC and BA, BA drop, worst recall, NLL, ECE15, AURC, selection SHA256 and checkpoint SHA256. The summary CSV has 24 rows, one per arm-method combination.

### 主比较口径：不跨数据集平均

平均表只用于快速总览，不用于主要结论。正式比较单位应是：

```text
one row = arm × dataset × selection method × training seed
```

每一行直接比较：

```text
clean ACC/BA
corruption ACC/BA
BA drop
clean/corruption worst-class recall
NLL, ECE15, AURC
selection index SHA256
checkpoint SHA256
```

因此，主分析必须按以下方式展开，而不是先把五个数据集求平均：

```text
固定 dataset：比较 8 个 selection methods
固定 method：比较 5 个 training seeds
固定 arm：分别比较 clean selection、corrupted selection->clean、corrupted selection->corrupted
```

当前可直接逐行查看的文件：

```text
单种子完整矩阵，120 行：
/home/qiangzeng/prj_qiangzeng/summaries/three_arm_single_seed_20260920_per_run.csv

本地副本：
reliability_medmnistc_ab/summaries/three_arm_single_seed_20260920_per_run.csv

多种子当前已完成子集，不代表完整矩阵：
/home/qiangzeng/prj_qiangzeng/summaries/three_arm_multiseed_20260920_completed_per_run.csv

不对 corruption 做任何汇总的原始条件表，6840 行：
/home/qiangzeng/prj_qiangzeng/summaries/three_arm_single_seed_20260920_per_condition.csv

本地副本：
reliability_medmnistc_ab/summaries/three_arm_single_seed_20260920_per_condition.csv
```

其中 `per_condition.csv` 每行是一个具体 `corruption × severity` 条件；如果要比较某个数据集、某个 training seed、某个扰动下的八种 selection method，应以此文件为主，而不是使用前面的跨 corruption 平均列。

当前多种子逐行文件有 `40` 行，全部是 clean selection -> clean training，覆盖 `Random、EL2N、Forgetting、EVA`、两个数据集和 training seeds `42–46`。它还不能回答八算法全数据集结论；必须等待其余方法、数据集和两个 corrupted-training arms 完成。

单种子 `training_seed=42` 的逐数据集主比较如下。每个单元格为 `clean BA / corruption BA`，不是平均排名。

#### clean selection -> clean training

| Dataset | Random | EL2N | Forgetting | EVA | Facility | FPS | Herding | Graph-A2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BloodMNIST | 81.35/60.83 | 53.38/37.27 | 71.55/42.56 | 76.97/51.93 | 79.97/55.48 | 82.50/58.92 | 82.46/57.34 | 84.31/52.33 |
| OrganAMNIST | 86.10/65.15 | 70.05/54.45 | 74.48/57.87 | 73.82/54.33 | 86.16/64.38 | 84.77/61.34 | 87.37/64.23 | 87.64/64.60 |
| OrganSMNIST | 59.79/49.94 | 37.47/32.01 | 42.54/35.73 | 45.70/38.29 | 62.12/52.86 | 60.47/49.64 | 63.30/51.90 | 61.88/51.61 |
| PathMNIST | 78.08/51.34 | 44.01/30.67 | 51.92/37.23 | 58.08/43.53 | 82.63/56.42 | 75.56/51.70 | 83.69/54.82 | 81.62/55.03 |
| TissueMNIST | 45.02/41.10 | 13.61/13.57 | 38.29/34.55 | 39.15/34.74 | 46.52/41.08 | 36.32/33.26 | 44.66/37.67 | 43.72/37.12 |

#### corrupted selection -> clean training

| Dataset | Random | EL2N | Forgetting | EVA | Facility | FPS | Herding | Graph-A2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BloodMNIST | 82.50/60.49 | 49.44/32.20 | 63.44/41.74 | 71.26/46.52 | 82.53/63.39 | 79.62/52.47 | 80.54/55.51 | 80.18/51.75 |
| OrganAMNIST | 86.48/63.00 | 68.40/50.92 | 71.24/52.32 | 73.96/55.22 | 88.35/67.12 | 83.77/62.50 | 86.89/64.84 | 88.25/61.85 |
| OrganSMNIST | 60.41/52.06 | 43.02/35.56 | 42.24/35.53 | 47.68/40.23 | 62.03/51.22 | 56.92/47.82 | 62.23/52.01 | 61.06/49.86 |
| PathMNIST | 82.77/52.10 | 35.13/23.64 | 50.33/36.32 | 58.42/39.78 | 75.40/48.73 | 72.19/48.88 | 81.45/51.99 | 78.68/53.09 |
| TissueMNIST | 42.48/37.96 | 19.97/18.75 | 40.44/34.88 | 38.11/33.75 | 44.11/39.36 | 36.30/31.54 | 43.36/39.65 | 44.14/37.26 |

#### corrupted selection -> corrupted training

| Dataset | Random | EL2N | Forgetting | EVA | Facility | FPS | Herding | Graph-A2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BloodMNIST | 81.01/55.21 | 65.01/41.40 | 68.63/42.47 | 66.41/40.62 | 84.48/59.56 | 75.49/56.25 | 83.66/55.26 | 83.15/60.31 |
| OrganAMNIST | 86.20/63.41 | 69.26/53.05 | 70.56/54.20 | 70.20/52.31 | 87.79/67.24 | 86.95/64.42 | 88.58/68.11 | 88.02/66.98 |
| OrganSMNIST | 60.93/49.95 | 39.81/32.50 | 43.10/35.47 | 48.84/40.48 | 58.89/48.22 | 56.66/47.23 | 61.76/51.21 | 60.01/50.93 |
| PathMNIST | 65.50/45.06 | 37.33/25.35 | 42.16/29.97 | 54.39/39.60 | 73.83/51.89 | 67.31/47.18 | 74.21/50.60 | 78.92/52.62 |
| TissueMNIST | 42.25/38.01 | 22.64/19.64 | 39.99/35.46 | 35.24/31.27 | 44.23/37.96 | 32.40/29.45 | 45.53/39.74 | 42.80/39.41 |

### 当前多种子进度

修正后的多种子 Job：

```text
Job: 32712
Script: /home/qiangzeng/prj_qiangzeng/three_arm_multiseed_qiangzeng_single_method_20260920.slurm
Output: /home/qiangzeng/prj_qiangzeng/three_arm_multiseed_20260920_qiangzeng/
Configuration: selection_seed=42, training_seed=42,43,44,45,46
Methods: 8
Arms: 3
Target: 8 × 5 datasets × 5 training seeds × 3 arms = 600 model cells
```

当前 Slurm 状态：

```text
32712_0..4: RUNNING, five method tasks using five GPUs
32712_5..7: PENDING, JobArrayTaskLimit
```

当前已生成 `run_complete.json` 的多种子单元为 `40`，全部来自 clean selection -> clean training 的前四个方法（Random、EL2N、Forgetting、EVA），覆盖两个数据集和 training seeds `42–46`；不能将这 40 个单元当作完整多种子结果。corrupted 两个 arm、另外四个方法和另外三个数据集仍未完成。

旧多种子 Job `32679` 已取消。它使用了 8 方法配置并导致不同 task 争用相同 run 目录，出现 `existing run has a different configuration`；其 `final.pt`、日志和部分输出均不纳入正式结果。

### 修复历史

```text
32687: ImageMagick 不可用
32692: Git ownership/configuration 锁冲突
32697: qiangzeng 无 UNI gated model 权限，HTTP 401
32702: METHODS 注入未真正约束 pipeline，实际每个 task 跑了全部 8 方法，已取消
32707: 修复后五个方法各自单独 YAML，5/5 完成，25/25 单元有效
```

修复方案使用共享的已验证 corrupted embedding/dynamics cache，只读复用，不重新下载 UNI：

```text
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/formal_runs/three_arm_single_seed_20260919_xiaoyuxu2/selection_corrupted_clean_train_pixelate_s3/cache
```

### 当前分析结论

1. 单种子三臂实验的文件验收已经完成，当前可以开始分析，不需要再等待单种子训练。
2. Clean baseline 中 Herding、Graph-A2、Facility 的 clean BA 最高；Random 不是 clean BA 的最高方法。
3. Corrupted selection -> clean training 中 Random 的 clean BA 略高于 Graph-A2，但 Facility 的 corruption BA 更高；Graph-A2 的 BA drop 较大。不能只按 BA drop 排名。
4. Corrupted selection -> corrupted training 中 Graph-A2 的 corruption BA 最高（54.05%），高于 Random（50.33%），说明选样和训练同时面对 pixelate 时，Graph-A2 的扰动绝对性能更好。
5. EL2N、Forgetting、EVA 的绝对 clean BA 明显低于几何方法和 Random；它们的 BA drop 较小不能解释为更鲁棒，因为 clean BA 本身较低。
6. 目前仍不能用单个 training seed 判断 selection variance，也不能给出 Graph-A2 相对 Random 的统计显著性。必须等待 Job `32712` 的五个 training seeds，并按 arm 分开计算均值、标准差和配对差值。
7. `corruption BA` 是所有 corruption/severity 的平均，必须和 clean BA、逐 corruption/severity、worst-class recall、ECE/NLL/AURC 一起报告；不能单独用 BA drop 宣称鲁棒性。

### 尚未完成

```text
[ ] Job 32712 的 600 个多种子模型全部完成
[ ] 其余三个 array task（FPS、Herding、Graph-A2）运行
[ ] corrupted selection -> clean training 的多种子结果
[ ] corrupted selection -> corrupted training 的多种子结果
[ ] 选择方差：selection_seed=43 或更多 selection seeds
[ ] 逐 corruption/severity 的 bootstrap CI 和方法配对 CI
[ ] coverage、coverage drop、CKA、directed/mutual-kNN、difficulty spectrum 机制诊断
[ ] 多种子最终汇总表、置信区间和论文图表
```

## 1. 当前状态

### MedMNIST-C 生成

MedMNIST-C 生成任务：

```text
Job ID:    31176
Job name:  medmnistc-generate
状态:      COMPLETED
ExitCode:  0
作业类型:  只生成 corruption，不是模型训练
```

最终验收：

```text
已生成：56 / 56 个 corruption 文件
磁盘占用：约 118G
manifest：已生成
```

文件数量：OrganSMNIST=13、OrganAMNIST=13、PathMNIST=11、TissueMNIST=8、BloodMNIST=11。

MedMNIST-C 已经准备好用于测试，但八算法正式选样、下游训练和 corruption 指标测试仍未完成。

当前验收结论：可以进入 Stage 1 选样，但仍须先完成 clean 224 数据的 MD5 验收，以及对 56 个 `.npz` 的 labels/severity 对齐检查。

## 2. 固定源码与环境

### 官方 GraphCov

```text
HPC:    /project/prj-sis01/xuxiaoyu/graph_select_pristine_origin_main/
Commit: 8cf757adc4c333dc1427d511f0de2f246d15ebac
```

用于 Table 1 选样和下游训练的 Python：

```text
/project/prj-sis01/xuxiaoyu/graph_select/envs/graphcov-py311/bin/python
```

已核对的主要版本：Python 3.11.9、PyTorch 2.5.1、TorchVision 0.20.1、CUDA build 12.1、MedMNIST 3.0.2、NumPy 1.26.4、Pandas 3.0.5、SciPy 1.17.1、scikit-learn 1.9.0、timm 1.0.28。

### 官方 MedMNIST-C API

```text
HPC:    /project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/sources/medmnistc-api/
Commit: 8acfd2710c6e0e8b2745be8b1fa1c17b94b8f8a7
```

用于生成 corruption 的 Python：

```text
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/envs/medmnistc-py311/bin/python
```

主要版本：Python 3.11.9、PyTorch 2.5.1、TorchVision 0.20.1、MedMNIST 3.0.1、NumPy 1.26.4、scikit-image 0.23.2、SciPy 1.17.1、OpenCV 4.10.0、Wand 0.7.2。

ImageMagick：

```text
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/envs/imagemagick/
ImageMagick 7.1.2-31 Q16-HDRI
```

运行 MedMNIST-C 时需要 `MAGICK_HOME`、`LD_LIBRARY_PATH` 和 `PYTHONPATH`，具体见 Slurm 脚本。运行过程中可能生成 `__pycache__`；最终归档前应清理这些运行时文件并再次确认核心 `.py` 文件没有变化。

## 3. 目录与文件约定

本地实验包：

```text
C:\Users\Administrator\Documents\ChatGPT\New project\reliability_medmnistc_ab\
```

HPC 实验包：

```text
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/
```

重要路径：

```text
protocol/protocol.json                         # 冻结协议
protocol/environment_manifest.json             # MedMNIST-C 环境记录
protocol/medmnist_download_manifest.json       # clean 224 MD5 记录
protocol/medmnistc_generation_manifest.json    # 全部生成后才出现

scripts/download_medmnist_224.py               # 下载 clean MedMNIST 224
scripts/generate_medmnistc.py                  # 调用官方 DatasetManager
scripts/generate_medmnistc_224.slurm           # 当前生成任务入口

data/medmnist/                                 # 官方 clean 224 数据
data/medmnistc/                                # 官方 API 生成的 corruption

artifacts/selections/                          # 选中的 train indices
artifacts/checkpoints/                         # 下游 checkpoint
artifacts/predictions/                         # 可选预测/概率文件

results/raw_metrics/                           # 逐运行指标
results/summary_tables/                        # 汇总表
results/figures/                               # 图表
logs/                                          # Slurm 和运行日志
```

当前 MedMNIST-C 日志：

```text
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/logs/medmnistc_generate_31176.out
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/logs/medmnistc_generate_31176.err
```

Table 1 的输出不得复制或混入上述目录；Table 1 仍在原来的 `table1_reproduction` 和 HPC `graph_select` 目录中。

## 4. 实验冻结配置

数据集：

```text
organsmnist, organamnist, pathmnist, tissuemnist, bloodmnist
```

选择比例：

```text
0.02, 0.05
```

八个方法：

```text
random, el2n_top, forgetting, eva,
facility, fps, herding, graph_a2
```

几何方法使用 UNI 224x224 frozen embedding。动态方法只使用 clean train，并使用 28x28、200 epoch 的 dynamics scoring。下游训练使用 224x224、ResNet-18，从头训练 1000 epochs，batch size 256，SGD learning rate 0.1、momentum 0.9、weight decay 5e-4、cosine schedule、无 augmentation。

方法特定配置：

| 方法 | 选样配置 | 选择随机性 |
|---|---|---|
| Random | 每类均衡随机抽样 | 主鲁棒性实验固定 selection seed=42 |
| EL2N | 前 20 dynamics epochs 的 L2 error 平均值，每类取最高 | dynamics seed=42 |
| Forgetting | correct-to-incorrect 转换次数，每类取最高 | dynamics seed=42 |
| EVA | early variance + late variance，每类取最高，window=10 | dynamics seed=42 |
| Facility | 每类 cosine facility location，`global=false` | 固定 42 |
| FPS | 每类 farthest point sampling | 主鲁棒性实验固定 selection seed=42 |
| Herding | 每类 normalized embedding mean matching | 固定 42 |
| Graph-A2 | global graph，`k=50`，`H=2`，kernel `A_sym + A_sym^2` | 固定 42 |

注意：`k=50,H=2,global` 只属于 Graph-A2。Facility 不使用 Graph-A2 的 k 或 hop 参数。

EVA 窗口：

| Dataset | 2% | 5% |
|---|---|---|
| OrganSMNIST | early=90, late=100 | early=150, late=190 |
| OrganAMNIST | early=1, late=190 | early=1, late=190 |
| Path/Tissue/Blood | early=1, late=100 | early=1, late=100 |

## 5. 推荐实验顺序

### Stage 0：完成数据准备

1. Slurm `31176` 已完成，确认 `ExitCode=0`。
2. 检查五个数据集的 corruption 文件数量为 `13+13+11+8+11=56`。
3. 检查每个 `.npz` 含 `test_images` 和 `test_labels`，severity 数量为 5。
4. 检查标签与官方 clean test 对齐。
5. 确认 `protocol/medmnistc_generation_manifest.json` 已生成。
6. 不使用 smoke 数据作为正式结果。

### Stage 1：clean train 选样

MedMNIST-C 不参与这一阶段。建议每个数据集提交一个 GPU Slurm 任务，任务内完成八方法和两个比例；动态训练和 UNI embedding 只计算一次并缓存。

资源建议：

```text
OrganS/OrganA/Path/Blood: 8 CPU, 1 GPU, 64G RAM
Tissue:                    8 CPU, 1 GPU, 96G RAM
```

选样输出：

```text
artifacts/selections/{method}/{dataset}/ratio_{ratio}/selection_seed_42/
```

每个运行必须保存：

```text
selected_indices.npy
selection_config.json
selection_seed.txt
index_order_sha256.txt
class_counts.json
```

### Stage 2：clean downstream training

本轮主鲁棒性比较固定同一份 selection indices，并固定 training seed=42：

```text
training seed = 42
```

规模：

```text
5 datasets × 2 ratios × 8 methods × 1 selection seed × 1 training seed = 80 model runs
```

为减少 Slurm 调度开销，可以组织为 10 个作业单元：每个作业单元固定一个 dataset 和 ratio，依次处理八个方法。每个模型必须有独立的 checkpoint 子目录，不能共享 run ID 或输出文件。

资源建议：

```text
OrganS/OrganA/Path/Blood: 8 CPU, 1 GPU, 64G RAM
Tissue:                    8 CPU, 1 GPU, 96G RAM
```

训练只读 selected clean train。v29 固定使用 final epoch checkpoint，不读取 validation 或 corrupted test 来选择 checkpoint。

### Stage 3：clean/corrupted evaluation

同一个 checkpoint 按顺序评估：

```text
clean official test
MedMNIST-C corruption severity 1-5
```

不得针对不同 corruption 重新训练。建议在训练作业完成后直接对该 checkpoint 做推理，或单独提交 evaluation array。推理应逐 corruption 流式计算指标，避免保存所有 logits 导致存储膨胀。

输出：

```text
artifacts/checkpoints/{method}/{dataset}/ratio_{ratio}/train_seed_{seed}/
results/raw_metrics/{method}/{dataset}/ratio_{ratio}/train_seed_{seed}/
```

每个运行至少保存：

```text
clean metrics
corruption name
severity
accuracy
balanced_accuracy
per_class_recall
worst_class_recall
nll
brier
ece
aurc
checkpoint path and hash
```

### Stage 4：汇总与分析

本轮主汇总使用单个 training seed=42，不计算均值和标准差。多 training seed 方差属于后续 Step 2 诊断：

```text
mean ± standard deviation
clean BA
corrupted BA
BA drop = clean BA - corrupted BA
NLL/Brier/ECE/AURC
per-class and worst-class recall
```

selection robustness 单独报告：

```text
pairwise Jaccard
Jaccard to seed42
class-count deviation
selection size
index-order SHA256
```

不能把 selection robustness 结果写成 Table 1 结果，也不能把 MedMNIST-C 结果混入 Table 1 汇总表。

## 6. Slurm 资源与时间估计

当前可用的账户路径配置：

```bash
# GPU 训练/选样
#SBATCH --account=hpc-sis
#SBATCH --qos=qos-high-gpu
#SBATCH --partition=gpu-rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# CPU/ImageMagick 生成任务当前不申请 GPU，使用同一 GPU partition
#SBATCH --mem=24G
```

资源说明：

- 当前 MedMNIST-C 生成：8 CPU、24G、0 GPU；Slurm `31176` 最终耗时约 5:32:29，56/56 文件完成。
- 历史 GraphCov Tissue 任务曾申请 96G；不要用 24G 作为 Tissue Graph-A2 的默认资源。
- 历史普通数据集 GraphCov 任务通常申请 64G、8 CPU、1 GPU。
- `sacct` 的历史 MaxRSS 没有记录，因此 64G/96G 是申请上限，不应写成实际峰值。
- 当前 MedMNIST-C 数据已经约 118G；后续应预留至少 150-200G 工作空间。

粗略墙钟时间，仅用于排队计划：

```text
MedMNIST-C 生成：已完成；后续只需做完整性验收
Stage 1 选样：约 2-8 小时，取决于 cache 和排队
Stage 2 80 个模型：约 0.5-2 天，取决于 GPU 并发和 Tissue 作业
Stage 3 评估：约 0.5-2 天
汇总和 Notebook：约 2-6 小时

从当前状态估计，若训练阵列保持最多 4 个 GPU 任务并发，完整鲁棒性结果大约还需 3-8 天；这是排队和 1000 epoch 训练的计划估计，不是完成承诺。
```

这些不是完成承诺；每个阶段必须以文件、manifest、Slurm `sacct` 和指标数量验收。

## 7. 验收清单

- [x] `31176` 完成且 exit code 为 0
- [x] 56 个 MedMNIST-C `.npz` 存在
- [x] generation manifest 存在
- [ ] 官方源码 commit 与协议一致
- [ ] 核心官方 `.py` 没有本地修改
- [ ] 五数据集八方法的 selected indices 数量和类别配额正确
- [ ] 所有 selection 记录包含 seed、config、SHA256
- [ ] 80 个训练运行的 checkpoint/日志/指标路径唯一
- [ ] checkpoint 使用 final epoch，且没有用 validation/corruption test 选择
- [ ] 每个 checkpoint 的 clean test 和 corrupted test 使用同一模型
- [ ] 没有按 corruption 重新训练
- [ ] summary 表只从 raw metrics 生成
- [ ] smoke、Table 1、MedMNIST-C robustness 三类结果没有混目录

## 8. 关键配置文件索引

本实验：

```text
reliability_medmnistc_ab/protocol/protocol.json
reliability_medmnistc_ab/protocol/environment_manifest.json
reliability_medmnistc_ab/scripts/download_medmnist_224.py
reliability_medmnistc_ab/scripts/generate_medmnistc.py
reliability_medmnistc_ab/scripts/generate_medmnistc_224.slurm
reliability_medmnistc_ab/notebooks/medmnistc_api_debug.ipynb
reliability_medmnistc_ab/notebooks/selection_robustness_audit.ipynb
sources/graph-coverage-selection-codex-reliability-audit-v29/reliability_full_experiment.ipynb
```

Table 1 复现，保持独立：

```text
table1_reproduction/configs/job1_table1.json
table1_reproduction/configs/job2_table1.json
table1_reproduction/CONFIG_EVIDENCE.md
```

## 9. 2026-09-19 single-seed three-arm execution

The requested pilot is now defined as:

```text
datasets: organsmnist, organamnist, pathmnist, tissuemnist, bloodmnist
methods: random, el2n_top, forgetting, eva, facility, fps, herding, graph_a2
ratio: 0.02
selection_seed: 42
training_seed: 42
epochs: 1000
Graph-A2: global=true, k=50, hops=2
```

The three arms are separated by output root and selection cache:

```text
clean selection -> clean training -> clean + MedMNIST-C test
corrupted selection -> clean training -> clean + MedMNIST-C test
corrupted selection -> corrupted training -> clean + MedMNIST-C test
```

For the first mechanism pilot, `corrupted` is explicitly registered as:

```text
corruption: pixelate
severity: 3
view: deterministic on-the-fly transform from the clean train split
```

This is a pilot condition, not an average over all MedMNIST-C corruptions. The official MedMNIST-C files remain test-only and are not reused as train data. Expanding to all corruption views is a separate factorial experiment and would multiply the training cost by the number of corruption/severity conditions.

The runnable entry point is:

```text
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/three_arm_single_seed.slurm
```

The patched pipeline used by this pilot is currently a dirty working-tree change (the pinned GraphCov and MedMNIST-C sources remain unchanged):

```text
pipeline SHA256: 3a421f4d5e833fa1fe7c9ba44ee61767f84806cd567d002e70539974bad25c13
backup: reliability_medmnistc/reliability/pipeline.py.pre_three_arm_20260919
```

This hash must be retained with the results; the Git HEAD alone identifies the original d74 checkout and does not include the uncommitted three-arm adapter.

Its controls are `ARM`, `ACCOUNT_TAG`, and `TASK_METHODS`. Each array element is one method over all five datasets; `TRAIN_SOURCE=corrupted` changes only the selection view, while `TRAIN_ON_CORRUPTED=1` changes the downstream training view as well. The pipeline writes `selection_train_source`, `selection_train_corruption`, `selection_train_severity`, and `training_source` into every new manifest/config.

### Current execution and errors

The GPU preflight `32462` completed with exit code 0 and created a PathMNIST corrupted-train selection artifact. The initial production arrays exposed only a Slurm export mistake: comma-separated `TASK_METHODS` was parsed as multiple environment exports, so array elements above index 0 failed with `METHODS_LIST[...] unbound variable`. Those failed elements are invalid and excluded. The script now uses colon-separated method lists.

Current relevant jobs:

```text
32463_0: clean/FPS, running
32474_0: clean/Herding, running
32474_1: clean/Graph-A2, running
32466_0: corrupted-selection -> clean-training/Random, running
32476: remaining corrupted-selection -> clean-training methods, queued/running
32478: corrupted-selection -> corrupted-training, submitted under qiangzeng, queued by QOS
```

Existing clean baseline remains at:

```text
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/formal_runs/single_seed_5datasets_8methods_ratio2_clean_20260918_xiaoyuxu2
/home/qiangzeng/prj_qiangzeng/single_seed_5datasets_8methods_ratio2_clean_20260918
```

At the start of this pilot, 25/40 clean dataset-method cells were complete. The 3 missing methods are being filled without retraining the completed 25. The two new corrupted-view arms are isolated under:

```text
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/formal_runs/three_arm_single_seed_20260919_xiaoyuxu2/
/home/qiangzeng/prj_qiangzeng/three_arm_single_seed_20260919_qiangzeng/
```

### Status snapshot: 2026-09-19 14:15 HKT

```text
clean baseline:                         40/40 complete
corrupted selection -> clean training:  20/40 complete
corrupted selection -> corrupted train:   0/40 complete; job 32516 submitted
```

The 20 completed corrupted-selection cells are Random, EL2N, Forgetting, and EVA across all five datasets. Facility, FPS, Herding, and Graph-A2 initially failed during corrupted-view selection because the adapter returned scalar integer labels to a GraphCov embedding worker that requires MedMNIST's one-element label shape. The adapter now preserves the `(1,)` label shape; repair array `32512` is running.

The previous `32478` corrupted-training submission failed before execution because qiangzeng's old q260 jobs exhausted the QoS job limit. It produced no result and is excluded. The replacement is `32516` under xiaoyuxu2.

### What was missing before this repair

1. The current formal run only implemented clean selection and clean training; the two corrupted-train arms had no selection input or training-view switch.
2. Generated MedMNIST-C data contained corrupted test files only. It did not provide a corrupted train pool.
3. The corrupted-train condition had no fixed corruption/severity, so it was not a uniquely reproducible experiment.
4. The first array launcher passed a comma-delimited method list through Slurm `--export`, causing method-index failures.
5. Login-node imports of `faiss-gpu` and Wand/ImageMagick were misleading preflight failures; all preflight and production commands must run inside a GPU Slurm allocation with `configure_imagemagick`.

### Acceptance for each cell

Do not count a cell from Slurm state alone. Require `run_complete.json`, `final.pt`, `training_history.json`, clean and all configured corruption predictions, `metrics.jsonl`, selection hashes, class counts, and a config showing the correct arm. Summaries must keep the three arms separate and report clean BA, corruption BA, BA drop, worst-class recall, ECE/NLL/AURC, selection hash, corruption, and severity.

### Status snapshot: 2026-09-20 01:22 HKT

The single-seed array jobs have finished. Verified output counts are:

```text
clean selection -> clean training:              40/40
corrupted selection -> clean training:          30/40
corrupted selection -> corrupted training:      25/40
total verified three-arm cells:                 95/120
```

The remaining single-seed gaps are Facility and Herding in the corrupted-selection -> clean-training arm (10 cells), plus EL2N, Forgetting, and EVA in the corrupted-selection -> corrupted-training arm (15 cells). These are not counted as complete.

The multi-seed follow-up was submitted as Slurm job `32679` after the single-seed jobs ended:

```text
script: /project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/three_arm_multiseed_by_method.slurm
output: /project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/formal_runs/three_arm_multiseed_20260919_xiaoyuxu2/
selection_seed: 42
training_seeds: 42,43,44,45,46
datasets: organsmnist, organamnist, pathmnist, tissuemnist, bloodmnist
methods: random, el2n_top, forgetting, eva, facility, fps, herding, graph_a2
```

Each array element runs one method over all three arms sequentially. The array is `0-7%5`; at submission, methods Random through Facility were running and FPS, Herding, and Graph-A2 were pending on the array task limit. Each method has an isolated writable GraphCov cache to prevent concurrent selection artifacts from overwriting one another.

## 10. qiangzeng repair rerun: 2026-09-20

The previous qiangzeng repair attempts are excluded from formal results:

```text
32687: ImageMagick unavailable
32692: Git ownership/configuration failure
32697: UNI gated-model HTTP 401 because the method cache was empty
32702: cancelled; the launcher did not constrain the method list and logs showed all 8 methods per task
```

The invalid partial output from `32702` is not reused. A fresh isolated rerun was submitted:

```text
Job: 32707, array 0-4%5, account=qiangzeng
Script: /home/qiangzeng/prj_qiangzeng/three_arm_repair_qiangzeng_single_method_20260920.slurm
Output root: /home/qiangzeng/prj_qiangzeng/three_arm_single_seed_20260920_qiangzeng_repair/
```

The five array elements are deliberately one-method YAML configurations, so the YAML is the only source of `methods`:

```text
32707_0 facility   corrupted selection -> clean training
32707_1 herding    corrupted selection -> clean training
32707_2 el2n_top   corrupted selection -> corrupted training
32707_3 forgetting corrupted selection -> corrupted training
32707_4 eva       corrupted selection -> corrupted training
```

All five tasks passed the GPU/PyTorch preflight (`torch 2.5.1+cu121`, CUDA available) and are running. The shared verified corrupted-view cache is reused read-only:

```text
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/formal_runs/three_arm_single_seed_20260919_xiaoyuxu2/selection_corrupted_clean_train_pixelate_s3/cache
```

Early progress after launch: Facility 2/5 selections, Herding 2/5, EL2N 3/5; no repair errors, checkpoints, or final metrics yet. Do not count a cell until `run_complete.json`, `final.pt`, predictions, metrics, selection hashes, and the correct arm/method configuration are present.

The first multi-seed attempt `32679` was cancelled because each task loaded the eight-method YAML and collided on shared run directories, producing `existing run has a different configuration`. It is invalid and excluded.

A corrected multi-seed array is scheduled only after successful repair:

```text
Job: 32712, dependency=afterok:32707, array 0-7%5
Script: /home/qiangzeng/prj_qiangzeng/three_arm_multiseed_qiangzeng_single_method_20260920.slurm
Output root: /home/qiangzeng/prj_qiangzeng/three_arm_multiseed_20260920_qiangzeng/
Methods: random, el2n_top, forgetting, eva, facility, fps, herding, graph_a2
Training seeds: 42,43,44,45,46
Arms: clean->clean; corrupted selection->clean; corrupted selection->corrupted
```

Each multi-seed array element has one method-specific YAML and one isolated method output tree. It will not start until all five `32707` repair elements finish successfully.

## 11. Authoritative latest snapshot: 2026-09-20

This section supersedes earlier status snapshots in this file. Historical job IDs and partial counts above are retained as provenance only.

### Single-seed three-arm matrix: complete

The repaired single-seed matrix is now verified at `120/120` cells:

```text
clean selection -> clean training:              40/40
corrupted selection -> clean training:          40/40
corrupted selection -> corrupted training:      40/40
```

The completed sources are:

```text
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/formal_runs/single_seed_5datasets_8methods_ratio2_clean_20260918_xiaoyuxu2/
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/formal_runs/three_arm_single_seed_20260919_xiaoyuxu2/
/home/qiangzeng/prj_qiangzeng/three_arm_single_seed_20260920_qiangzeng_repair/
```

All single-seed cells use `selection_seed=42` and `training_seed=42`. They are descriptive evidence only and must not be presented as multi-seed confidence intervals.

### Multi-seed three-arm matrix: in progress

The target is:

```text
3 arms x 5 datasets x 8 methods x 5 training seeds = 600 model cells
```

Current verified completion is `68/600`. All 68 completed cells are in the `clean selection -> clean training` arm:

```text
Random:     17/25
EL2N:       17/25
Forgetting: 17/25
EVA:        17/25
Facility:    0/25 (running)
FPS:         0/25 (pending)
Herding:     0/25 (pending)
Graph-A2:    0/25 (pending)
```

For Random, EL2N, Forgetting, and EVA, the completed clean-training cells currently cover:

```text
OrganSMNIST:  training seeds 42-46
OrganAMNIST:  training seeds 42-46
PathMNIST:    training seeds 42-46
TissueMNIST:  training seeds 42-43
BloodMNIST:   not started
```

The remaining two arms have no verified multi-seed cells yet:

```text
corrupted selection -> clean training:      0/200
corrupted selection -> corrupted training:  0/200
```

Active Slurm array:

```text
job: 32712
output: /home/qiangzeng/prj_qiangzeng/three_arm_multiseed_20260920_qiangzeng/
32712_0 Random      RUNNING
32712_1 EL2N        RUNNING
32712_2 Forgetting  RUNNING
32712_3 EVA         RUNNING
32712_4 Facility    RUNNING
32712_5 FPS         PENDING (JobArrayTaskLimit)
32712_6 Herding     PENDING (JobArrayTaskLimit)
32712_7 Graph-A2   PENDING (JobArrayTaskLimit)
```

`xiaoyuxu2` currently has no running task. A model cell is counted only when its `run_complete.json`, checkpoint, predictions, metrics, selection hash, and arm configuration pass validation.

### Current Graph-A2 SOTA evidence

The complete single-seed per-run comparison uses BA as the primary ranking metric. Graph-A2 is first in 7 aggregate slots:

```text
clean -> clean, BloodMNIST:          clean BA = 0.8431
clean -> clean, OrganAMNIST:         clean BA = 0.8764
corrupted select -> clean, PathMNIST: mean corruption BA = 0.5309
corrupted select -> clean, TissueMNIST: clean BA = 0.4414
corrupted select -> corrupted, BloodMNIST: mean corruption BA = 0.6031
corrupted select -> corrupted, PathMNIST: clean BA = 0.7892
corrupted select -> corrupted, PathMNIST: mean corruption BA = 0.5262
```

At individual corruption/severity level, Graph-A2 is first in 170 single-seed rows. These counts are descriptive and do not establish a multi-seed SOTA claim because Graph-A2 has not yet completed the multi-seed matrix.

Generated analysis files:

```text
reliability_medmnistc_ab/summaries/graph_a2_sota_single_seed_per_run.csv
reliability_medmnistc_ab/summaries/graph_a2_sota_single_seed_per_condition.csv
```

### Paper-faithful replication benchmark

The current 600-cell three-arm matrix is a robustness audit. It must remain separate from the benchmark intended to reproduce the paper's Table 1 trend.

The paper-faithful benchmark must first resolve the protocol difference between the current `1000 epochs` setting and the author README/code wording `1000 iterations`. Do not assume that 1000 epochs is equivalent to 1000 optimizer updates. The author entry point and training loop must determine the actual update budget.

After that audit, create an isolated benchmark with:

```text
author repository and pinned commit
same dataset version, split, preprocessing, image size, backbone, optimizer, LR, batch size
Graph-A2 k=50, hops=2
clean selection -> clean training
5 datasets x 8 methods x 5 pre-specified trial seeds = 200 model cells
```

Each trial must preserve the selection seed and training seed policy used by the paper. Do not select favorable seeds or datasets after inspecting results. Save selected-index SHA256, class counts, effective update count, initialization hash, final checkpoint, predictions, and clean metrics.

The primary report for this benchmark is per-dataset mean BA, standard deviation, confidence interval, and Graph-A2 minus each baseline. A Graph-A2 SOTA claim is valid only if it is supported by the complete pre-specified matrix. If the strict protocol does not reproduce the paper ranking, record the result as partial or failed replication rather than altering seeds or conditions to force a win.

Corruption evaluation starts only after the clean benchmark selections and checkpoints are frozen. It is a separate robustness analysis and must not replace the clean Table 1 reproduction.
