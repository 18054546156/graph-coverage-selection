# Reliability MedMNIST-C Experiment Handoff

更新时间：2026-09-14（以 HPC 实际文件和 Slurm 记录为准）

本文档是 `reliability_medmnistc_ab` 的执行交接记录。该实验独立于 Table 1 复现，不修改官方 GraphCov 或 MedMNIST-C 核心源码。

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

主鲁棒性比较固定同一份 selection indices，只改变 training seed：

```text
training seeds = 42, 43, 44, 45, 46
```

规模：

```text
5 datasets × 2 ratios × 8 methods × 5 training seeds = 400 model runs
```

为减少 Slurm 调度开销，可以组织为 50 个作业单元：每个作业单元固定 dataset、ratio、training seed，依次处理八个方法。每个模型必须有独立的 checkpoint 子目录，不能共享 run ID 或输出文件。

资源建议：

```text
OrganS/OrganA/Path/Blood: 8 CPU, 1 GPU, 64G RAM
Tissue:                    8 CPU, 1 GPU, 96G RAM
```

训练只读 selected clean train 和 clean validation。选择 validation BA 最佳 checkpoint；不得读取 corrupted test 来选 checkpoint。

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

主汇总按五个 training seeds 计算：

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
Stage 2 400 个模型：约 2-5 天，取决于 GPU 并发和 Tissue 作业
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
- [ ] 400 个训练运行的 checkpoint/日志/指标路径唯一
- [ ] checkpoint 只由 clean validation 选择
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
