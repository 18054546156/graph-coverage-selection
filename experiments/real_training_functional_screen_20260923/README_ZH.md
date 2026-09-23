# 真实训练 Functional Screen：中文说明

详细的面向外行人的公式、代码流程和例子见：

- [`REAL_TRAINING_FUNCTIONAL_SCREEN_ZH.tex`](REAL_TRAINING_FUNCTIONAL_SCREEN_ZH.tex)

这条实验线回答的问题是：在每类固定 25 张图像的预算下，选择集合的哪些性质与真实 ResNet-18 下游效果有关。它不是直接把某个特征宣布成新算法，而是先做测量。

## 当前配置

```text
数据集：bloodmnist、organamnist、organsmnist、pathmnist、tissuemnist
图像：224px
候选池：每个 block 的训练集 80%
block：2 个独立 pool/audit 划分
预算：每类 25 张
每 block：300 个选择集合
训练 offset：0、100000、200000
ResNet-18：从零训练 200 epochs，batch size 128
总目标：5 × 2 × 300 × 3 = 9000 条真实训练记录
```

每个 block 的 300 个选择集合为：

```text
60 个 Random
+ 15 个锚点 × (1 个原始集合 + 5 个扰动等级 × 3 个扰动种子)
= 300
```

15 个锚点包括 8 个几何锚点和 7 个训练动态锚点：

```text
几何：herding, kcenter, kmedoids, dense, sparse, hard, easy, impure
动态：forget_high, forget_low, el2n_high, aum_low, aum_high, ccs, moderate
```

## 三层代码流程

1. `functional_screen.py`
   - 读取 UNI embedding；
   - 建立 80% candidate pool；
   - 生成 300 个选择集合；
   - 计算 15 个 embedding 几何特征和 11 个训练动态特征。

2. `real_training_functional_screen.py`
   - 把选择集合索引映射回原始 224px 图像；
   - 调用 `train_weighted.train_one_cell()`；
   - 每个选择集合训练一个独立 ResNet-18；
   - 写出一行 JSONL。

3. `train_weighted.py`
   - 固定训练 seed；
   - SGD 训练 200 epochs；
   - 在官方 test split 上计算 `ba_real`、accuracy、NLL、Brier、ECE、per-class recall。

4. `merge_real_screen.py` 和 `analyze_screen.py`
   - 先跨账号合并、去重、检查协议和 GPU 协变量；
   - 再使用 `analyze_screen.py --endpoint ba_real` 做组内中心化、相关性、两个 block 复现和基线排行榜。

## 26 个特征

### 15 个 UNI 几何特征

```text
覆盖：dist_mean, dist_p90, dist_max
分布：moment1, moment2, mmd2, mass_gini
多样性：spread, centrality, logdet
局部/边界：purity, margin, density, sel_margin_min, xclass_sep
```

### 11 个训练动态特征

```text
forget_mean, forget_sd, forget_frac_zero
aum_mean, aum_sd
el2n_final_mean, el2n_early_mean
difficulty_ks, difficulty_w1
frac_hardest_decile, frac_easiest_decile
```

这些量的数学定义和直观例子都在 TeX 文档中。重要的是：它们是选择集合的描述，不是最终准确率；`ba_real` 才是实际训练端点。

## 结果目录

HPC 结果分别位于：

```text
/home/xiaoyuxu2/real_training_functional_screen_20260923/results/
/home/qiangzeng/real_training_functional_screen_20260923/results/
/home/danranwang/real_training_functional_screen_20260923/results/
```

本仓库不包含 embedding、MedMNIST 原始数据、动态缓存、模型权重、HPC 日志和 JSONL 结果。

## 当前状态快照

2026-09-23 的实时检查中，三账号合并去重后约有 `3453 / 9000` 条逻辑记录；27 张 GPU 正在运行，9 张 GPU 对应的 job 排队。单次训练中位数约 50 秒。一个 worker 出现过 CUDA busy/unavailable，需要补跑其缺失分片。当前尚未形成最终 feature survivor 结论。

旧结果中有一批记录缺少 `epochs`、`batch_size` 等协议字段，合并时不能只按文件行数相加，也不能把 smoke 结果与正式 200 epoch 结果混在一起。

## 重要限制

- 当前 `ba_real` 是 200 epoch 屏幕实验端点，不等于论文 Table 1 的 1000 epoch 数值。
- 26 个特征之间可能高度相关，单个相关系数不能证明独立作用。
- 方法族之间的 BA 差异不能直接解释为某一个特征的因果效果。
- 只有真实训练端点、方法族内扰动和两个 block 复现都支持时，才应据此设计新方法。
