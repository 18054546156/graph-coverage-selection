# MedMNIST-C 当前实验实现清单

核对时间：2026-09-20 约 17:00（北京时间）。这份清单描述已完成的单种子三臂实验和正在运行的 `qiangzeng` 多种子 Job `32712`；不代表另外提出的 200-run 论文复现 benchmark 已启动。远端绝对路径均是 HPC 路径，本机没有对应的大型数据和 checkpoint。

## 1. 哪份代码在运行

| 对象 | 实际路径或版本 | 角色 |
| --- | --- | --- |
| 当前 `qiangzeng` 运行仓库 | `/home/qiangzeng/prj_qiangzeng/graph-coverage-selection-medmnistc-reliability-clean-repair-20260920`，HEAD `d74e2d6fb6b348ad0cbb410b4931d90540afd754` | 编排和审计代码、GraphCov 导入入口 |
| GraphCov 方法基准 | 作者 commit `8cf757adc4c333dc1427d511f0de2f246d15ebac` | `pipeline.py` 在启动时检查该提交是祖先，且 `graphcov/` 相对该提交无差异 |
| MedMNIST-C API | 生成 manifest 记录 `8acfd2710c6e0e8b2745be8b1fa1c17b94b8f8a7` | 官方 `DatasetManager` 生成 test corruption，注册表还用于 train 的 on-the-fly pixelate |
| `xiaoyuxu2` 单种子编排 | `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/sources/graph-coverage-selection-medmnistc-reliability-clean-d74` | 早期 clean/三臂任务，结果在该用户的 project 树 |
| 当前审计代码 | 远端仓库 `reliability_medmnistc/reliability/pipeline.py` | 数据、选样、训练、评估及逐条件验证 |

不要把仓库 HEAD `d74e2d6` 与作者 GraphCov 基准 commit `8cf757a` 混为一谈。远端 MedMNIST-C 子模块的 Git 元数据目前不能正常解析；运行日志的 `MedMNIST-C commit: unknown`，不能把 generation manifest 里的 pin 当成对本次运行时 checkout 的独立 Git 核验。后续归档应记录实际源码树或修复元数据后再核对 commit。

## 2. 配置、作业和执行顺序

当前多种子主脚本是本地镜像 [`slurm/three_arm_multiseed_qiangzeng_single_method.slurm`](slurm/three_arm_multiseed_qiangzeng_single_method.slurm)，提交时远端副本为：

```text
/home/qiangzeng/prj_qiangzeng/three_arm_multiseed_qiangzeng_single_method_20260920.slurm
```

Job `32712` 是 `--array=0-7%5`：数组下标依次为 `random, el2n_top, forgetting, eva, facility, fps, herding, graph_a2`，最多同时运行 5 个方法任务。每个任务请求 `gpu-a100`、`qos-high-gpu`、`hpc-sis`、1 GPU、8 CPU、48 GB、48 小时。一个方法任务顺序完成三个臂；每臂依次处理五个数据集、五个 training seeds，总计每方法 75 个模型单元，八方法共 600 个。一个 Slurm task 不等于一个模型单元。

每个方法有一份单方法 YAML，目录为远端仓库 `reliability_medmnistc/configs/multiseed_qiangzeng/`；本地镜像见 [`configs/multiseed_qiangzeng`](configs/multiseed_qiangzeng)。例如 [`graph_a2.yaml`](configs/multiseed_qiangzeng/graph_a2.yaml) 的专有配置为 `global_selection=true, k_neighbors=50, k_hops=2, kernel=A_sym_plus_A_sym_squared`。Facility 是 `global_selection=false`，其执行设备策略是 `auto`，最大类别规模达到 40000 时改用 CPU。其他方法的 `method_config` 为空。`run_pipeline.py` 从 YAML 写入方法、数据集和训练参数；作业脚本设置三臂来源、输入输出目录及系统环境。因此只看 YAML 不足以复原一次运行，还要保存对应 Slurm 脚本和每个 run 的 `run_config.json`。

共同参数：

```ini
datasets = organsmnist,organamnist,pathmnist,tissuemnist,bloodmnist
methods = 每个 array task 对应的一个方法
ratio = 0.02
selection_seed = 42
training_seeds = 42,43,44,45,46
epochs = 1000
dynamics_epochs = 200
image_size = 224
dynamics_image_size = 28
batch_size = 256
num_workers = 4
embedding_source = uni
augmentation = false
checkpoint_rule = final_epoch
train_corruption = pixelate
train_severity = 3
```

三个臂的调用规则：

| 臂 | 选样看到的 train | 下游训练看到的 train | `select.py` | 选集来源 |
| --- | --- | --- | --- | --- |
| `clean_selection_clean_train` | clean | clean | 不重跑 | `/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/formal_runs/single_seed_sel42_train42_20260915/selections/` |
| `selection_corrupted_clean_train_pixelate_s3` | pixelate severity 3 | 相同 ID 的 clean 样本 | 运行 | 该臂自身的 `selections/` |
| `selection_corrupted_corrupted_train_pixelate_s3` | pixelate severity 3 | 相同 ID 的 pixelate severity 3 样本 | 运行 | 该臂自身的 `selections/` |

每臂随后执行 `train.py --config <method.yaml>` 和 `evaluate.py --config <method.yaml>`。三个入口都是薄封装，最终由远端 `reliability_medmnistc/scripts/run_pipeline.py` 载入 YAML、设置 `PHASE`，再运行 `reliability/pipeline.py`。作业 `unset METHODS`，避免旧版环境变量让单方法数组任务误跑八个方法。corrupted 两臂复用同一只读 embedding/dynamics 缓存：

```text
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/formal_runs/three_arm_single_seed_20260919_xiaoyuxu2/selection_corrupted_clean_train_pixelate_s3/cache
```

已完成单种子 `120/120` 来自 clean baseline、`xiaoyuxu2` 三臂输出及 `qiangzeng` 修复输出；不能只检查 `32712` 目录来代表这 120 个历史单元。单种子修复脚本见 [`slurm/three_arm_repair_qiangzeng_single_method.slurm`](slurm/three_arm_repair_qiangzeng_single_method.slurm)。早期 `xiaoyuxu2` 脚本见 [`../three_arm_single_seed.slurm`](../three_arm_single_seed.slurm)。各次作业的差异必须按自身脚本和 `run_config.json` 判断。

## 3. 数据下载、生成与预处理

当前训练作业设置 `DOWNLOAD_MEDMNIST=0`：它们**不下载数据**。数据已在共享 project：

```text
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/data/medmnist/
  <dataset>.npz        # 28x28，供训练动态分数
  <dataset>_224.npz    # 224x224，供 UNI 选样和下游训练/clean test
/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/data/medmnistc/
  <dataset>/<corruption>.npz
```

实际远端下载器是 `reliability_medmnistc/scripts/prepare_medmnist_224.py`：读取 `medmnist.INFO` 的官方 URL 和 MD5，下载并验证 28、224 两种档案，检查 train/test 图像尺寸，写 `data/medmnist/download_manifest.json`。224 数据的另一个本地封装为 [`scripts/download_medmnist_224.py`](scripts/download_medmnist_224.py)，只处理 224；不要误以为它单独准备好了 dynamics 所需的 28 档案。远端 protocol 下还保存了 `medmnist_download_manifest.json` 和 `data/medmnist/download_manifest_28.json`。目前五个 224 档案的 manifest 均有 MD5 记录。

MedMNIST-C 由远端 `reliability_medmnistc/scripts/generate_medmnistc.py` 调官方 `DatasetManager`，`random_seed=0`，生成 56 个 corruption `.npz`（OrganS 13、OrganA 13、Path 11、Tissue 8、Blood 11），每个文件含 5 个 severity 的 test 图像。生成器逐文件检查 `uint8`、数量、标签顺序及五个 severity 与 clean test 标签对齐，记录 `protocol/medmnistc_generation_manifest.json`。远端 Slurm 入口为 `reliability_medmnistc/slurm/01_generate_corruptions.slurm`；本地早期生成封装 [`scripts/generate_medmnistc.py`](scripts/generate_medmnistc.py) 和 [`scripts/generate_medmnistc_224.slurm`](scripts/generate_medmnistc_224.slurm) 不是正在运行的 `32712` 入口。生成需要可用的 ImageMagick/Wand；实际作业 source 远端仓库的 `reliability_medmnistc/slurm/common_env.sh`。

运行时 `pipeline.py` 通过 MedMNIST 的 `train`/`test` split 加载档案。`graphcov/run/data.py:get_transform` 做 `ToTensor` 后以每通道 `mean=0.5, std=0.5` 归一化；档案本身已是 224 或 28，函数**没有**额外 `Resize`。灰度数据保留原通道数。corrupted train 是由原 clean train 的同一索引即时应用固定 `pixelate`、severity 3，再还原灰度通道并应用同一归一化；不是事先下载的“污染 train 数据集”。clean test 取官方 test；corrupted test 从 MedMNIST-C 文件按 severity 1–5 读取，核验标签与 clean ID 对齐。五个数据集所有臂均评估完整可用 corruption 集，不只评估 pixelate。

## 4. 选样、训练与产物

`select.py` 为每类分配 `floor(floor(N_train * 0.02) / C)` 个索引，并核对无重复、索引在界内、类别配额恰好满足。Random 用类别平衡随机选样；Facility、FPS、Herding、Graph-A2 用缓存的 UNI embedding；EL2N、Forgetting、EVA 用 28x28、200 epoch 的 dynamics。Graph-A2 不是额外训练一个图网络；它在 frozen embedding 的全局 kNN 图上用 `A_sym + A_sym^2` 做覆盖选样。`selection_seed=42` 固定，所以当前 600 单元只估计固定选集下的 training seed 波动，不能估计选集随机性。

选集目录模式：

```text
<SELECTION_OUT>/<method>/<dataset>/ratio_0.02/selection_seed_42/
  selected_indices.npy             # 原 train split ID，顺序保留
  selected_local_indices.npy
  class_counts.json
  index_order_sha256.txt
  selection_config.json             # source path、文件/数组 SHA256、选样参数
```

训练：`set_seed(training_seed, deterministic=False)`，ResNet-18 从头初始化；选中样本由 DataLoader `shuffle=True, drop_last=False` 读取；SGD `lr=0.1, momentum=0.9, weight_decay=5e-4`、交叉熵、`CosineAnnealingLR` 每 epoch 更新，共 1000 epochs。没有训练增强，也没有 validation-best 选择；保存 final epoch。评估时加载同一 checkpoint，以 `eval()` 和 `no_grad()` 分别计算 clean 和所有 corruption/severity。主指标 BA，另有 ACC、每类 recall、worst recall、NLL、Brier、ECE15、Risk@80、AURC 和 clean-to-corruption BA drop。BA drop 必须与绝对 corruption BA 一起看。

多种子输出根：

```text
/home/qiangzeng/prj_qiangzeng/three_arm_multiseed_20260920_qiangzeng/
<root>/<arm>/<method>/<dataset>/<method>/ratio_0.02/selection_seed_42/train_seed_<42..46>/aug_0/
```

每个完成单元含 `run_config.json`、`selected_indices.npy`、`final.pt`、`training_history.json`、`predictions_clean.npz`、各 corruption 的 prediction `.npz`、`metrics.jsonl`、`run_complete.json`。`run_complete.json` 存配置、选集、checkpoint、训练历史、指标和预测文件哈希。只有文件齐全、clean 一行加该数据集所有 corruption×5 行且完成标记有效的单元，才应计入进度。

单种子结果的原始根目录及三臂来源见 [`handoff.md`](handoff.md)；本地逐运行表 [`summaries/three_arm_single_seed_20260920_per_run.csv`](summaries/three_arm_single_seed_20260920_per_run.csv) 有 120 行，逐条件表 [`summaries/three_arm_single_seed_20260920_per_condition.csv`](summaries/three_arm_single_seed_20260920_per_condition.csv) 有 6840 行。最新 Excel 是 `../outputs/medmnistc_final_handoff_20260920/MedMNISTC_experiment_snapshot.xlsx`，其多种子页是时间戳快照，不会随 HPC 自动刷新。

## 5. 当前必须保留的口径和风险

1. 作业导出 `PYTHONHASHSEED=0`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`DETERMINISTIC=1`，但已核对的训练函数显式调用 `set_seed(seed, deterministic=False)`；该分支令 `torch.backends.cudnn.benchmark=True`。在当前 `pipeline.py`、入口脚本和 `graphcov/run/evaluation.py` 中未找到对 `DETERMINISTIC` 环境变量的读取或 `torch.use_deterministic_algorithms(True)`。因此不能写“PyTorch 全确定性已生效”；多 seed 可以统计波动，但不能保证 bitwise 重复。若未来修复，应使用新输出根、记录实际运行时设置，不能与现有结果无标记混合。
2. 多种子三臂固定一份 selection seed 42；Random 的五个 training seeds 也复用同一选集。这与作者 runner 对 Random/FPS 每 trial 可重新选集的口径不同。不能用当前 `32712` 直接声称严格复现论文 Table 1。
3. `32712` 实际 Python 为 3.11.9、PyTorch 2.5.1 + CUDA 12.1、torchvision 0.20.1、medmnist 3.0.1（来自 job log）。`qiangzeng` 的旧 PyTorch 2.6 环境对照不属于这次三臂作业。
4. `2026-09-20 16:57` 快照：单种子 120/120；`32712` 已验证 100/600，Random/EL2N/Forgetting/EVA 各 25、Facility 正运行且 0、FPS/Herding/Graph-A2 排队。此处状态会过期；以 Slurm 和完成标记为准。Graph-A2 单种子有 7 个主指标第一名和 170 个逐条件第一名，但多种子 SOTA 结论尚不存在。
5. `table1_reproduction/configs/job1_table1.json` 与 `job2_table1.json` 是另一条工作流，目前同时列有 2% 与 5%，且 Job 2 使用 validation-best checkpoint，不是当前三臂的 final-epoch 协议。另议的 **5 数据集 × 8 方法 × 5 seeds = 200** 个 2% clean benchmark 尚未提交；其首个点可设 selection/training 42/42，其余 trial 必须按预注册的选择种子规则执行，并隔离结果。参见 [`../table1_reproduction/CONFIG_EVIDENCE.md`](../table1_reproduction/CONFIG_EVIDENCE.md)。
