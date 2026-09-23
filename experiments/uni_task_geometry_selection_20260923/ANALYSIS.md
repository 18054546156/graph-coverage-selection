# UNI 医学图像选样实验：结果、失败原因与下一步

本目录记录 `uni_task_geometry_selection_20260923` 主线。它的目的不是声称某个方法已经胜出，而是回答一个更基础的问题：在固定每类预算下，哪些选择集合的性质真正能预测从头训练模型的表现，哪些只是在离线 probe 上好看。

## 1. 研究范围

输入表示分成两类：

1. **UNI 表示**：`*_train_uni_224.npz` 中的 1024 维 embedding，用于距离、邻域、覆盖、边界、动态特征筛选和廉价 probe。
2. **原始图像训练端点**：MedMNIST 224px 图像，由 `graphcov.run.data.load_dataset` 载入，使用独立的 `train_weighted.py` 从头训练 ResNet18。

这两个空间不能混为一谈。UNI 空间中的距离小，不自动意味着从像素训练的 ResNet 会受益；因此真实训练是最终验证端点。

官方 validation 没有用于选择方法。真实训练阶段按预注册协议读取 official test，并且每个 cell 只在最终 epoch 评估一次；probe 阶段只使用 train-only audit split。

## 2. 八个已有算法与隐含假设

Table-1 reproduction 中真正需要超过的基线是：

| 方法 | 选择规则 | 隐含假设 |
|---|---|---|
| Random | 每类随机选样 | 样本差异的平均收益足够好 |
| Herding | 逐步逼近类别均值/一阶矩 | 保持类别内部总体分布最重要 |
| Forgetting | 选择训练中经常被遗忘的样本 | 难学或不稳定样本信息量更高 |
| EL2N | 选择早期误差/梯度范数大的样本 | 早期训练困难度可以预测价值 |
| FPS | farthest-point sampling | 覆盖最远点、控制最坏距离最重要 |
| Facility Location | 最大化平均覆盖收益 | 平均代表距离最重要 |
| EVA | 跨 epoch 的误差方差 | 动态不稳定性包含选择信号 |
| GraphCov / Graph-A2 | 图覆盖和邻域结构 | 图上的多尺度覆盖能代表训练需求 |

这些方法是“直接产生选择集的算法”。如果只比较最终 BA，就无法区分“算法有效”和“算法附带改变了很多别的性质”。当前 functional screen 的角色不同：它先把 `dist_max`、`moment1`、`forget_mean`、`EL2N` 等性质单独测量，再决定第九个选择器应该优化什么。

两个需要补齐的基线：

- `forgetting`、`AUM`、`EL2N`、`CCS`、`Moderate` 是动态锚点或组成部分，不等于完整复现 EVA 的双窗口误差方差规则。
- `facility_from_features` 是覆盖式实现；最终对照仍应明确使用真正的 facility-location objective，不能把 k-medoids 或普通 k-center 近似误称为 Facility Location。

## 3. 已完成实验链

### 3.1 五数据集 UNI probe

脚本：`run_linear_probe.py`。

- bloodmnist、organamnist、organsmnist、pathmnist、tissuemnist，均为 224 版本；
- 读取完整 train embedding 文件和标签，但主要候选池为每类最多 1,000 张；
- 候选池按类拆成约 80% selection、20% train-only audit；
- Round 1 每类选 50 张，比较 Random、Herding、Graph-A2、geometry；
- Round 2 每类先固定 10 张 pilot，最终仍是 50 张；
- Graph-A2 的 `k_neighbors=10` 与每类预算 50 是不同参数。

主 task-geometry 对照是：

```text
task_geometry_plus_pilot - permuted_task_geometry_plus_pilot
```

五个数据集平均为 `+0.191pp`，预注册门槛为平均至少 `+1.0pp` 且至少 3/5 数据集达到门槛，因此 `gate_decision.py` 输出 `NO_GO`。这关闭的是 task-geometry 分支，不是所有流形表示。

### 3.2 预算阶梯

脚本：`ladder_locate_operating_point.py`。

它只使用 bloodmnist、organsmnist、tissuemnist，测试每类预算 10、25、50、100、250，并同时比较每类 cap=1,000 和无 cap 的 full train split。每个 block 仍然把候选拆成 80% selection 和 20% audit。

在 full pool、预算 25 时，probe 预测的 weighting interaction 为：

| 数据集 | Random weighting gain | Herding weighting gain | probe interaction |
|---|---:|---:|---:|
| bloodmnist | +1.034pp | -0.016pp | +1.05pp |
| organsmnist | +2.038pp | -0.886pp | +2.92pp |
| tissuemnist | +1.289pp | -0.316pp | +1.60pp |

四个 block 在三个数据集上均为正，且超过当时的 2SE 规则。因此预算 25 被预注册为真实训练 operating point。但这是 probe 证据，不是最终训练证据。

### 3.3 真实训练 interaction

脚本：`build_training_cells.py`、`train_weighted.py`、`gate_training.py`。

设计为：

```text
3 datasets × 2 selection arms × 3 weightings × 10 replicates × 2 train seeds = 360 cells
```

真实训练使用完整 train split，不设每类 1,000 cap；每个 replicate 按类抽 80% selection pool，每类选择 25 张。weighting 为 `equal`、`voronoi_within` 和 `permuted_within`。runner 固定 Python、NumPy、PyTorch、CUDA、DataLoader 和 cuDNN；不同 train seed 的差异是正常 seed 方差。

正式输出来自 `results/gate_training.json`：

| 数据集 | 真实训练 DiD | SE | 正向单位 |
|---|---:|---:|---:|
| bloodmnist | -0.090pp | 1.082pp | 8/20 |
| organsmnist | +0.382pp | 0.700pp | 11/20 |
| tissuemnist | -0.484pp | 0.330pp | 9/20 |
| 三数据集 mean | **-0.064pp** | 按 dataset mean 汇总 | 0/3 达到 +1pp |

permuted control 的 pooled DiD 为 `+0.057pp`。代码要求 pooled ≥ +1pp、至少 2/3 数据集达到 +1pp，并且置换控制接近零；实际 pooled 为负，输出为：

```text
REFUTED
```

结论：Voronoi weighting × random/herding interaction 不支持继续投入。这是该预注册机制的真实训练负结果，不是“所有几何选样都失败”。

## 4. Probe 能否排序真实训练选择？

脚本：`probe_vs_training.py`；结果：`results/probe_vs_training.json`。

360 个训练 cell 背后有 60 个不同选择集。固定 dataset 后，合并 random 与 herding 时的 Spearman 为 blood `+0.212`、organs `+0.798`、tissue `+0.481`。但合并会把方法族差异误当成选择集内部排序。

按同一 dataset、同一 arm 分开后：

```text
blood/random   -0.067
blood/herding  -0.261
organs/random  +0.321
organs/herding +0.176
tissue/random  +0.564
tissue/herding -0.018
mean within-cell Spearman = +0.119
```

每个 cell 只有 10 个选择，因此不能把 `0.119` 当作“真实相关必然为零”的证明。更准确的决策语言是：当前样本量不足以让 probe 担任最终裁判；它仍可作为便宜的前置粗筛和协变量。新实验需要在同一方法族内增加约 200 个受控选择集，并直接用真实训练端点测量。

## 5. 重要的测量修正

旧的 `measure_dataset_priors.py` 有两个范围问题：

1. 所谓 `probe_ba_full_pool` 实际是 capped candidate pool 的 BA。对 pathmnist，它只代表约 7,200/89,996 的池，不能称为完整训练池的上限。
2. 近重复率是在 capped pool 中再抽样得到的；它没有足够能力否定全量数据上的近邻结构。

`measure_priors_fullscale.py` 已写好，使用无 cap 的完整 train split，并用 blockwise exact k-NN 计算全池最近邻和 KNN label entropy，还会计算 cross-class Voronoi cell purity/entropy。正式输出尚未进入本目录的 `results/`，因此 pathmnist 是否应重新纳入真实训练仍是待验证问题。

## 6. 动态泛函筛选：不是第九个算法

`functional_screen.py` 当前加入了几何泛函、forgetting、AUM、EL2N、CCS、Moderate 以及 herding、k-center、k-medoids 等锚点。每个 anchor 再做同类点替换扰动 `m∈{1,2,5,10,25}`，保持每类预算不变。

旧作业 `33707` 在读取数据前因 shell 保留数组名 `GROUPS` 产生错误路径，`results/screen_33707/` 为空，不能作为正式结果。随后代码已改用 `DS_GROUPS` 并加入 `--dynamics-root`。五个 dynamics cache 存在且样本数与 train split 对齐，但完整 screen 仍需以新作业输出为准。

`results/screen_smoke.jsonl` 只是单数据集、单 block 的 smoke test，不能用来宣布 Forgetting、EL2N、CCS 或 Moderate 的最终排名。

## 7. 与原来八个算法的根本区别

原来的八个方法直接回答：

> 给定预算，我应该选择哪些图？

当前 functional-screen 路线先回答：

> 在预算固定、方法身份控制后，选择集合的哪些性质仍然和真实训练收益有关？

因此它们处在不同层次：

```text
八个算法：规则 → 选择集合 → 训练模型
当前测量：选择集合 → 多个性质 → 真实训练 endpoint
第九个算法：只使用跨 block、跨 dataset 复现的性质
```

这不是把第九个方法换成更复杂的分数，而是先避免把“方法身份”误认为“分数有效”。如果某个动态特征没有在固定方法内部复现，它就不应进入新算法。

## 8. 下一步实验设计

### 阶段 A：全尺度先验和 screen

在五个数据集、两个独立 pool/audit block 上运行 full-scale prior、几何与动态泛函，并预先定义 competitive families。不要按当前 endpoint 选择竞争区；按结果筛选会造成 range restriction/collider bias。

### 阶段 B：直接用真实训练筛选性质

建议每个 block 约 214 个选择集：约 60 个 random、14 个 anchor 家族及其 10 级扰动。每个选择集使用 3 个 train seeds：

```text
5 datasets × 2 blocks × 214 selections × 3 seeds = 6,420 training cells
```

每次训练当前约 48–52 秒，按中位/均值估算约 86–93 GPU-hours。12 卡并发时墙钟时间约 7.5–8 小时；这是资源计划，不是已完成结果。它同时提供真实 endpoint 泛函筛、n≈214/cell 的 probe-vs-training 验证，以及八个基线的真实 leaderboard。

### 阶段 C：构造候选第九个方法

只有阶段 B 产生复现的性质后，才构造新选择器。候选可以是动态难度、类内原型和稀有模式保护的联合贪心：

```text
gain(i|S) = λ_dynamic · dynamic_value(i)
          + λ_boundary · boundary_value(i)
          + λ_proto · prototype_gain(i|S)
          + λ_rare · rare_mode_value(i)
          - λ_redundancy · similarity(i,S)
```

`λ` 必须在 calibration/source-validation 上冻结，不能用 official test 调整。不能把“高维”直接解释成“应该多选”。

### 阶段 D：五数据集确认

冻结方法、预算和 seed schedule 后，比较 Random、Herding、FPS、真正 Facility Location、EVA、Forgetting、EL2N、GraphCov/Graph-A2 和候选新方法。BA 为主要端点，NLL、Brier、ECE、worst-class recall 和每类 recall 为辅助端点。

目标可以是超过当前最强基线，但不能事先承诺一定超过所有方法；不能超过时也要保留清晰的负结果。

## 9. 决策原则

1. Probe 可用于粗筛，不能独立放行最终算法。
2. 固定预算、固定方法族内的关系优先于跨方法整体相关。
3. 真实训练 endpoint 优先于离线代理 endpoint。
4. pathmnist 的旧 capped headroom 不再用于排除它，等 full-scale prior 后重判。
5. 不把 smoke test、单次运行或 outcome-conditioned subset 当成正式证据。
6. 动态基线和完整 EVA/Facility Location 必须进入对照，否则不能声称超过八个算法。
7. 任何新方法都必须保留 random、herding、permutation 和 oracle/upper-bound 对照。

## 10. 文件索引

- `protocol.json`：UNI probe 协议。
- `protocol_training.json`：真实训练 interaction 协议和 gate。
- `run_linear_probe.py`：UNI probe、选择和权重计算。
- `ladder_locate_operating_point.py`：预算阶梯。
- `build_training_cells.py`：生成真实训练 cells。
- `train_weighted.py`：确定性的 ResNet18 runner。
- `gate_training.py`：预注册 DiD 裁决。
- `probe_vs_training.py`：probe 与真实训练排序检查。
- `functional_screen.py`：几何/动态泛函与受控扰动库。
- `analyze_screen.py`：whole-library、competitive、within-family 分析。
- `measure_priors_fullscale.py`：无 cap 全尺度先验。
- `results/gate_training.json`：360 cell 的真实训练 gate 汇总。
- `results/probe_vs_training.json`：60 个选择集的排序汇总。
- `results/ladder.json`：预算阶梯汇总。
- `docs/`：面向非专业读者的 HTML 解释页。

embedding、原始数据、dynamics cache、selection manifest、训练 shard、HPC 日志和模型权重没有提交；其排除理由见 `RESULTS_MANIFEST.md`。
