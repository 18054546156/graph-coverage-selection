# 对比算法、代码来源、参数、已有结果、ACS 的文献来源（2026-10-03）

所有数字都由 `code/acs_full_table.py` 生成，原表在 `results/acs/full_table.md`。BA 和最差类召回都是种子 42–46 的均值，每格 5 个配对种子。

---

## 1. 对比了哪些算法，论文和官方 GitHub 在哪

| 方法 | 论文 | 官方代码 | 我们实际用的代码 |
|---|---|---|---|
| **Graph-A2** | Rustamov et al., MICCAI 2026, arXiv:2606.22002 | https://github.com/zahiriddin-rustamov/graph-coverage-selection | 作者仓库 `graphcov/`（commit 8cf757a），**未改**；derma/OCT 例外，见 2.3 节 |
| Herding | Welling, ICML 2009 | 无独立官方实现，用 Graph-A2 作者仓库的实现 | `graphcov/run/selection.py`（作者），未改 |
| Facility Location | Wei et al., ICML 2015 | 同上 | 同上，未改 |
| FPS / k-Center | Sener & Savarese, ICLR 2018 | 同上 | 同上，未改 |
| Random | — | 同上 | 同上，未改 |
| EL2N | Paul et al., NeurIPS 2021 | 同上 | 同上，未改 |
| Forgetting | Toneva et al., ICLR 2019 | 同上 | 同上，未改 |
| EVA | Graph-A2 论文 Table 1 自带的训练动态基线 | 同上 | 同上，未改 |
| **TypiClust** | Hacohen et al., ICML 2022 | https://github.com/avihu111/TypiClust | 官方 `pycls.al.typiclust.TypiClust` 类，经 `code/al_baselines_select.py` 调用 |
| **ProbCover** | Yehuda et al., NeurIPS 2022 | https://github.com/avihu111/TypiClust（同一仓库） | 官方贪心逻辑在 GPU 上重写，见第 2.2 节 |
| **MaxHerding** | Bae et al., ECCV 2024 | https://github.com/BorealisAI/uherding | 官方 `pycls.al.herding.Herding` 类，经 `code/al_baselines_select.py` 调用 |

- 前 8 个是 Graph-A2 原论文 Table 1 的全部方法。
- 后 3 个是近年低预算选择的 SOTA，原论文没有比较，是我们补的。
- 还查过 HyperCore（Moser 2025）和 Uncertainty Herding（ICLR 2025）。前者没有公开代码，后者没跑。

## 2. 改了吗？参数和原作者一样吗？

### 2.1 Table-1 的 8 个方法：算法代码一行没改

- **代码：** `graphcov/` 是作者原仓库 commit 8cf757a。每次运行前，`assert_graphcov_source_unchanged()` 会用 git diff 对比固定 commit，有改动就拒绝运行。
- **我们唯一加的东西：** 训练框架里的 `precomputed` 钩子（`env/harness_pipeline_tdhook.patch`）。它只让框架读入我们自己选好的索引文件，不碰任何基线。
- **参数**（按作者论文 §3.1 和作者仓库配置）：

| 参数 | 作者 | 我们 |
|---|---|---|
| 分类网络 | ResNet-18 从零训练 | 同 |
| 训练轮数 / batch | 1000 epochs / 256 | 同 |
| 优化器 | SGD，lr 0.1，momentum 0.9，wd 5e-4 | 同 |
| 输入 / 增广 | 224px / 无 | 同 |
| 选择用的特征 | UNI 冻结特征 | 同（作者缓存的同一份 UNI 特征） |
| Graph-A2 | k=50，2 跳，全局选择 | 同 |
| Facility | 类内 | 同 |
| EVA / EL2N / Forgetting 的代理训练 | 28px，200 epochs，窗口 10 | 同 |
| 训练种子 | 42–46 | 同 |
| 取哪个 checkpoint | 最后一轮 | 同 |

- 全部基线都按作者配置运行，**没有调参**。
- 唯一和作者不同的是 `DETERMINISTIC_TRAINING=1`（确定性训练，结果可以逐位复现）。

**复现质量**（A100，`results/table1/table1_s4246_a100.txt`）：80 个格子对比论文数值，平均绝对误差 **1.82pp**，中位数 1.39pp，72/80 在 ±4pp 以内。

### 2.2 TypiClust / ProbCover / MaxHerding：用官方核心代码，有 5 处适配（都要在论文里写）

1. **在每个类里分别运行，每类配额相等。** 这 3 个原本是不看标签的主动学习方法；Table 1 的所有方法都是"每类等配额"，为了公平比较也这样做。原版无标签、全池运行的版本也已经选好（后缀 `_nolab`），只作附录备选，不进主表。
2. **特征换成与 bench 相同的 UNI 特征**，并做 L2 归一化。原论文用的是自监督的 SimCLR/DINO 特征。
3. **ProbCover 的 δ 按原论文规则选：** 取最大的 δ，使得用 k-means 伪标签（k=类别数）时至少 95% 的 δ 球是纯的。贪心部分在 GPU 上重写（δ 球用稀疏矩阵存储），并在一个小样本池上和官方 `ProbCover` 类做了逐项一致性单元测试，通过后才运行。
4. **TypiClust 加了容错：** 官方轮询遇到已经选空的簇会崩溃，我们让它跳过空簇。聚类方式、典型性定义（K_NN=20）和其他参数不变。
5. **MaxHerding 用官方 `run_budget.sh` 的默认值：** kernel=rbf，δ=1.0，候选上限 35000。

这 3 个方法也**没有调参**，选择种子固定为 42。

### 2.3 更正（10-03）：derma/OCT 上的 Graph-A2 和 Random 两列不是作者代码

本文第一版说这一列"是同一份作者实现"，**这是错的**：

- **Graph-A2 列（`a2_uni`）是我们自己复写的 Graph-A2 目标**（`mv_select.py`，λ=0，CPU FAISS），不是作者管线跑出来的。参数相同（k=50、2 跳、全局），但实现不同；Graph-A2 的选择对微小扰动很敏感，两者的结果不能视为等同。
- **Random 列（`rand_cls`）也是我们自己写的**类平衡随机，选择种子固定为 42；作者的 random 是选择种子等于训练种子。

**处理：** 已在 derma/OCT 上用作者管线补跑 graph_a2 和 random（见第 3 节"补跑"）。结果出来后，这两列换成作者代码的数字，旧数字放附录并标明是复写版本。retina/breast 的 4 列是作者管线跑的，没有这个问题。

## 3. 已有结果（全表）

### 3.1 平衡准确率 BA（%），种子 42–46 均值；加粗是该行最高

| 数据集 | 比例 | Graph-A2 | Herding | Facility | Random | FPS | EVA | EL2N | Forgetting | TypiClust | ProbCover | MaxHerding | **ACS（我们）** | ACS 名次 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blood | 2% | 81.30 | 83.70 | 83.18 | **84.68** | 79.38 | 75.36 | 51.78 | 65.20 | 83.39 | 78.81 | 81.15 | 训练中 | — |
| blood | 5% | 92.73 | **93.70** | 90.99 | 91.49 | 89.73 | 86.61 | 75.93 | 84.96 | 92.30 | 91.55 | 91.55 | 训练中 | — |
| organA | 2% | 86.96 | **87.78** | 86.29 | 86.88 | 86.51 | 69.92 | 66.23 | 73.10 | 87.10 | 84.66 | 87.01 | 训练中 | — |
| organA | 5% | 91.76 | 91.25 | **92.33** | 91.16 | 90.62 | 82.33 | 81.48 | 86.05 | 91.52 | 90.32 | 91.98 | 训练中 | — |
| organS | 2% | **63.41** | 62.05 | 61.81 | 60.38 | 60.38 | 47.76 | 38.22 | 43.83 | 62.51 | 57.67 | 61.66 | 训练中 | — |
| organS | 5% | 67.45 | 67.70 | **68.53** | 65.89 | 67.61 | 64.06 | 50.58 | 54.54 | 66.77 | 65.37 | 67.19 | 训练中 | — |
| path | 2% | 81.60 | **82.08** | 81.48 | 79.26 | 73.12 | 51.13 | 41.97 | 48.50 | 80.98 | 81.88 | 80.18 | 训练中 | — |
| path | 5% | 86.73 | 87.43 | **87.64** | 87.62 | 82.21 | 58.71 | 53.73 | 57.63 | 86.62 | 87.01 | 87.16 | 训练中 | — |
| tissue | 2% | 42.78 | 42.85 | 43.09 | 44.12 | 34.68 | 39.03 | 12.15 | 38.56 | 44.00 | 44.55 | 44.85 | **48.08** | 1/12 |
| tissue | 5% | 44.40 | 45.81 | 44.67 | 45.89 | 36.77 | 43.01 | 16.81 | 42.15 | 47.01 | 46.58 | 44.18 | **48.97** | 1/12 |
| retina | 2% | 22.01 | 31.32 | 29.66 | 29.22 | — | — | — | — | 31.54 | 26.80 | 27.54 | **36.42** | 1/8 |
| retina | 5% | 24.07 | 33.38 | 24.12 | 31.46 | — | — | — | — | 30.33 | 34.08 | 27.64 | **35.24** | 1/8 |
| breast | 2% | 54.39 | 63.91 | 55.20 | 55.35 | — | — | — | — | 63.37 | 63.45 | 56.59 | **65.40** | 1/8 |
| breast | 5% | 65.08 | 62.73 | 65.09 | 61.93 | — | — | — | — | 67.36 | 65.76 | 58.43 | **77.41** | 1/8 |
| derma | 2% | 39.74 | 41.50 | 41.65 | 37.59 | — | — | — | — | 38.97 | 40.45 | 43.07 | **43.81** | 1/8 |
| derma | 5% | 47.62 | 48.33 | 48.27 | 49.78 | — | — | — | — | 49.67 | 47.58 | 49.00 | **53.21** | 1/8 |
| OCT | 2% | 82.14 | 84.32 | 85.62 | 85.30 | — | — | — | — | 83.62 | 82.78 | **85.96** | 84.84 | 4/8 |
| OCT | 5% | 88.12 | 87.04 | 88.74 | 86.42 | — | — | — | — | 87.98 | 88.36 | 88.80 | **90.46** | 1/8 |

- **原论文的 5 个数据集**（blood、organA、organS、path、tissue）：8 个原方法加 3 个新 SOTA，共 11 个对手，全部在 A100 上复现完成。
- **ACS 在 blood/organA/organS/path 上的 40 格**：10-03 已提交到 A100，正在训练。按 rule 本身算，这 4 个数据集是样本外的，因为它们没有参与选 rule。
- **我们加的 4 个数据集**（retina、breast、derma、OCT）：原论文没有这几个。原来只跑了 4 个 Table-1 方法加 3 个新 SOTA；FPS、EVA、EL2N、Forgetting 是"—"。
- **derma/OCT 的 Graph-A2 和 Random 两列是我们的复写版本**，见 2.3 节。
- **补跑（10-03，用户要求"表补充完整"）：** 共 200 格，计划和盖章在 `report/bench_fill_20261003.md`。
  - retina/breast：FPS、EVA、EL2N、Forgetting，80 格；
  - derma/OCT：上面 4 个，加作者版 Graph-A2 和 Random，120 格。
  - 全部用作者管线，配置与 Table 1 相同，选择种子等于训练种子，H100。
  - 已提交选择阶段（作业 36317、36321），选完即训练。

### 3.2 最差类召回（%），种子 42–46 均值

完整表格见 `results/acs/full_table.md`。ACS 在 10 行里有 6 行第一。明显落后的两行：

- OCT 2%：62.16，MaxHerding 是 68.24，差 −6.08，原始 p=.049；
- derma 2%：28.85，MaxHerding 是 33.00。

两行都是 F2（kNN 过滤）在每类预算很小时的代价，论文里要如实写。

### 3.3 ACS 对每行 SOTA（BA，配对 t 检验；`code/acs_vs_table1.py` → `results/acs/e2_vs_table1.txt`）

| 行 | SOTA | 差（ACS−SOTA） | p |
|---|---|---|---|
| retina 2% | TypiClust | +4.87 | .006 |
| retina 5% | ProbCover | +1.16 | .563 |
| breast 2% | Herding | +1.49 | .748 |
| breast 5% | TypiClust | +10.05 | .034 |
| derma 2% | MaxHerding | +0.75 | .689 |
| derma 5% | Random | +3.43 | .023 |
| OCT 2% | MaxHerding | −1.12 | .354 |
| OCT 5% | MaxHerding | +1.66 | .005 |
| tissue 2% | MaxHerding | +3.23 | .044 |
| tissue 5% | TypiClust | +1.96 | .003 |

**9/10 行第一，6 行显著，没有一行显著落后。**

## 4. 新算法 ACS 是什么，参考了哪些论文

ACS（ambiguity-aware class-conditional selection）由三部分组成。每部分都是把一个已发表的原理迁移进 Graph-A2 的框架：

| 组件 | 做什么 | 来源论文 | 我们和原文的不同 |
|---|---|---|---|
| F1 类条件覆盖 | 覆盖信用只在同类内部流动 | Graph-A2（Rustamov 2026）的覆盖目标；Wei et al. ICML 2015 的类条件 Facility Location | 在全局 kNN 图上按类做掩码，而不是每类重新建图；已验证这和作者的类内选择是不同的算子 |
| F2 逐类分位数过滤 | 去掉每类里 kNN 同类占比最低的 q 比例样本 | Confident Learning（Northcutt et al., JAIR 2021）：逐类阈值；HyperCore（Moser 2025）；反例是 Brodley & Friedl（JAIR 1999）的全局过滤，会把少数类整个删掉 | 没有训练好的分类器，用 UNI kNN 的同类占比代替置信度；阈值取类内分位数，保证每类都至少留下 (1−q) |
| F3 典型 ↔ 覆盖切换 | 每类预算小时选典型点（herding），大时选覆盖 | TypiClust（Hacohen ICML 2022）、ProbCover（Yehuda NeurIPS 2022）：低预算选典型、高预算选多样，中间有相变；Herding（Welling 2009） | 原文按**总预算**、不看标签切换；我们按**每类预算** bpc 切换，并写成确定的规则 |

**规则：** τ(bpc) = 1（bpc ≤ b\*）、0.5（b\* < bpc < 2b\*）、0（bpc ≥ 2b\*）。E2 选出 **b\*=10, q\*=0.25**，之后冻结，不再调。

**代码：**

- `code/acs_select.py`、`code/acs_stage.slurm`：选择；
- `code/acs_eval.py`、`code/acs_vs_table1.py`、`code/acs_ablation.py`、`code/acs_full_table.py`：分析；
- 训练用作者框架加 `tune_pack.slurm`。

**消融**（不需要新训练，直接从 E2 的析因实验读出，BA）：

| 组件 | 贡献 | 有收益的行 | 配对 p |
|---|---|---|---|
| F1（cls 对 Graph-A2） | +3.95pp | 10/10 | 6e-9 |
| F2 | +1.89pp | 8/10 | .015 |
| F3 | +2.29pp | 只在它启用的 4 行起作用，4 行全涨 | 2e-4 |
| 三者合计（ACS 对 Graph-A2） | +7.35pp | 10/10 | 9e-14 |

## 5. 结论，以及必须如实写的限制

1. **已有的 10 行**（tissue 加我们新增的 4 个数据集）：BA 上 ACS 对每行 SOTA 赢 9/10，6 行显著，没有一行显著落后。
2. **最大的限制：** b\* 和 q\* 是在**这 10 行、这 5 个种子**上选的，所以这 10 行的领先幅度偏乐观。要干净地声称"超过 SOTA"，还需要：
   - **E4**：用新种子在这 10 行上复测（尚未批准）；
   - **main-4 的 40 格**：rule 没见过的数据集，正在训练。
3. **预注册的预期：** main-4 上 ACS ≈ SOTA（打平）。这 4 个数据集的标签歧义低，所有好方法本来就挤在一起，没有拉开差距的空间。如果确实打平，论文的说法应是：高歧义数据集上 ACS 明显领先，低歧义数据集上 ACS 不输。
4. **最差类召回的代价：** OCT 2% 和 derma 2% 上 ACS 不如 MaxHerding。论文里要写，并解释为 F2 在小预算下的代价。
