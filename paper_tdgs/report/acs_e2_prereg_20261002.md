# ACS E2 开发筛选：规则先写（2026-10-02，草稿；用户批准 D4 后拷到集群盖章，再提交任何 E2 训练）

上级文档：`report/proposal_acs_20261002.md`（§3 方法、§4 预测、§5 E2）。E0/E1 结果见 `report/acs_e0_20261002.md`。

## 1. 要回答的问题

在开发种子 42–46 上，从 12 条候选规则里选出 1 条（2 个全局参数），之后冻结、在新种子上确认（E4）。
E2 不判方法成败；它的唯一产出是冻结的 (b*, q*)，以及描述性的析因表（F2 × F3 × 行类型）。

## 2. 行、臂、硬件、种子

| 行 | 新训练的臂 | 复用的臂（已有，同硬件） | GPU | 格数 |
|---|---|---|---|---|
| retina、breast、derma、OCT × 2%/5%（8 行） | acs_q00_t050、acs_q25_t000、acs_q25_t050、acs_q25_t100、acs_q50_t000、acs_q50_t050、acs_q50_t100（7）+ typiclust、probcover、maxherding（3） | cls_uni（= q0,τ0）、herding（= q0,τ1）、facility、random、graph_a2、knnf_herding、knnf_random、mv_mean：均为 H100 | H100 | 8 × 10 × 5 = 400 |
| tissue × 2%/5%（2 行） | 同上 10 臂 | tdgs_cls、herding、knnf 两臂、bench：A100 | A100 | 2 × 10 × 5 = 100 |
| blood、organA、organS、path × 2%/5%（8 行） | typiclust、probcover、maxherding | bench 8 列：A100 | A100 | 8 × 3 × 5 = 120 |

- 种子 42–46（retina/breast 也只用 42–46，与其余行一致）；选择种子固定 42（与所有已 staging 的选择器相同）。
- 主 4 数据集的 ACS 臂在 E2 不训练：按 §3 的任何候选规则，它们的 bpc ≥ 25，τ=0；只有 q* > 0 时 ACS 才与 cls 不同。若选出的 q* > 0，主行 ACS 在 E4 直接按冻结规则跑。
- 复用臂的来源（10-02 核对 worklist）：retina/breast → `runs/ambig_20261002`（cls_uni、herding、facility、graph_a2、random、mv_mean，s42–51，E2 只取 s42–46）与 `runs/knnf_20261002`（knnf 两臂）；derma/OCT → `runs/w1_20261001`（cls_uni、a2_uni = graph_a2 的同实现、rand_cls = 每类配额随机、mv_mean）与 `runs/knnf_20261002`（herding、facility、knnf 两臂）；tissue → A100 Table-1 树与 `runs/knnf_20261002`。
- **例外（10-02 盖章前核对日志发现）**：tissue 2% 已训练的 tdgs_cls 中 s42–44 只在 H100 上（`runs/r3hw_20260928`，作业 35306），A100 上只有 s45、s46（`runs/round2_20260927`）。因此 E2 在 A100 上重训 tissue 2% `tdgs_cls` s42–46 共 5 格（选择集即 `$S/sel/tissuemnist_r0.02_tdgs_cls_s42.npy`，软链进合并目录），作为该行的 (0,0) 格；tissue 5% tdgs_cls s42–46 已全部在 A100（round2），直接复用。A100 总格数 225。
- 同一行内所有比较都在同一 GPU 型号上（表中"复用"列已核对：retina/breast/derma/OCT 的 cls、herding、facility、knnf 全部是 H100；tissue 全部 A100）。

## 3. 候选规则（12 条）与选择准则

每行的配置只由该行的每类预算 b = bpc 决定：

- τ(b; b*) = 1 若 b ≤ b*；0.5 若 b* < b < 2b*；0 若 b ≥ 2b*。b* ∈ {0（F3 关闭）, 5, 10, 20}。
- q = q* ∈ {0（F2 关闭）, 0.25, 0.5}，所有行相同。
- (b*, q*) = (0, 0) 就是 cls。

10 个高歧义行的 bpc：retina 4 / 10，breast 5 / 13，derma 20 / 50，OCT 487 / 1218，tissue 413 / 1034。各 b* 下的 τ：

| b* | τ=1 的行 | τ=0.5 的行 | 其余 τ=0 |
|---|---|---|---|
| 0 | 无 | 无 | 全部 |
| 5 | retina 2%、breast 2% | 无 | |
| 10 | retina 2%、retina 5%、breast 2% | breast 5% | |
| 20 | retina 2%、retina 5%、breast 2%、breast 5%、derma 2% | 无 | |

**主准则**：对每条规则，取它在 10 行上分配的配置，计算"10 行的种子均值 BA 的平均"，选最大者。
**并列**（与最大者差 < 0.25pp）：依次比 ① 10 行最差类召回均值，② 更简单（q* 小、b* 小）。
**继续到 E4 的条件**：选出的规则 ≠ (0, 0)。若选出 (0, 0)，ACS 没有超过 cls，E4 不跑 ACS，论文方法部分退回 cls + 诊断。

## 4. 同时报告（描述，不改变选择）

1. 完整析因表：9 个 (q, τ) × 10 行的 BA / 最差类召回，配对 s42–46，附 se。
2. §4 预测的开发期读数：H-F2a（纯度、零删除，E0 已测）；H-F2b（ACS vs knnf 尾部）；H-F3a/b（τ>0 在小 bpc 行涨、在大 bpc 行不涨）。
3. 每行事后最优配置（oracle）与规则配置之差，作为"规则损失了多少"的估计。
4. 新基线的位置：TypiClust、ProbCover、MaxHerding 在 18 行上相对行最好的差。
5. 选择偏差的提醒：从 12 条规则中按同一批 test 挑 1 条；E4 的新种子用来估计这部分偏差。
6. 逐行对照全部 bench（用户 10-02 的问题"有没有能超过所有 bench 的新算法"）：选出的规则配置在 10 个高歧义行上相对"该行最好的 bench 方法"的配对差（BA 与最差类召回），bench = Table-1 8 方法（或该行已有的同硬件对照：herding、facility、random、graph_a2）+ knnf 两臂 + TypiClust/ProbCover/MaxHerding + mv_mean；报告第一的行数、显著落后的行数（配对 t，n=5，原始与 BH 两种）。这是描述；"超过所有 bench"在 E2 上只作开发期读数，结论以 E4 新种子为准。

## 2b. 执行细节（与 §2 一起冻结）

- 合并选择目录 `$S/acs/td_sel`：软链到 `$S/acs/sel/*.npy`（70）、`$S/acs/al_sel` 中非 `_nolab` 的 54 个文件、以及 `$S/sel/tissuemnist_r0.02_tdgs_cls_s42.npy`（1）；不复制、不改名。
- 结果树 `$S/runs/acs_20261002`；worklist `$S/work/acs_e2_h100.txt`（400 行）、`$S/work/acs_e2_a100.txt`（225 行），格式 `ds ratio arm seed`，按种子外层、行内层排序（先出完整的 s42 一轮）。
- 执行器 `tune_pack.slurm` + `td_run_one.sh`（协议同 Table 1：1000 epochs、224px、无增广、DETERMINISTIC_TRAINING=1、final_epoch），`STOP_ON_FAIL=1`，`--exclude=hpcgpu108`。
- 收割时逐格核对 GPU 型号；任何一格不在规定型号上，该格从比较中剔除并在结果里列出。

## 5. 已知问题，事先写明

- ProbCover 按论文规则（k-means 伪标签纯度 α=.95）在已检查的行上 δ 很小，类内球的中位大小为 1（球里只有自己），贪心退化为索引顺序。这是作者规则在 UNI 特征上的行为，照实跑、照实报，不改 δ。
- MaxHerding 用作者仓库默认：RBF 核、δ=1.0、L2 归一化特征、候选上限 35000（超过时按 np.random.seed(42) 随机抽候选）。
- HyperCore：未找到公开代码（2026-10-02 检索），不作为对照；论文里引用并讨论。
- TypiClust 的 k-means 有随机性，固定 np.random.seed(42)。

## 6. 盖章

用户 10-02 批准 D4（"你都跑 看看有没有能超过所有bench 的新算法"）。盖章于 **2026-10-02 13:58:53 UTC**，早于 worklist 生成与任何 E2 训练提交。集群拷贝 `$S/code/acs_e2_prereg_20261002.md`。

| 文件 | sha256 |
|---|---|
| 本文件（盖章前版本，即本节以上全部内容） | a96c2017f0f038baa7afe90edd23101f7d58a0ff3774028ffefefd7165a1cfb1 |
| `acs_select.py` | bb170092db4da55b0c4b0f196c545f27ffdaef16fd186a5407dd0f6000c29b2d |
| `al_baselines_select.py` | 6d658a51393f2e46fb241eaa2c8ee6c45c1057a09214a06dbd19b58391c9a01e |
| `td_run_one.sh` | 5787a0400e07fa1ae0a5b4b302256a914fde163719f3306d0468e9aeb4967d4c |
| `tune_pack.slurm` | 0f239e83a7c24c5242bfe776d31b62fdf4c3661f5a4b4102ece94b28fba03509 |
| 选择集清单 `$S/acs/e2_sel_manifest.sha256`（125 个 npy 的 sha256） | 6ad62a787954d483e61c1c38d0e76cfb3c2fd83e236e32d13567ffdc8d1b0500 |

## 7. 执行记录

- 10-02 21:59 HKT 提交（盖章后）。合并目录 `$S/acs/td_sel` 125 个软链；worklist H100 400 行、A100 225 行，逐行核对选择集都存在、无重复。
- H100：36206、36207（xiaoyuxu2，hpcgpu111）、36208、36209（qiangzeng，hpcgpu112），各 4 卡。A100：36210（danranwang，hpcgpu107）、36211（danranwang，hpcgpu109）、36212（xiaoyuxu2，hpcgpu109），各 4 卡。共 16 H100 + 12 A100。
- 收割：集群 `cd $S/code && python acs_eval.py --cluster` → 拷回 `results/acs/acs_cells.json` → 本地 `python code/acs_eval.py test`（规则选择、析因表、对全部 bench、新基线位置）。`acs_eval.py` 在盖章后编写，只实现本文件 §3/§4 已写明的计算。
