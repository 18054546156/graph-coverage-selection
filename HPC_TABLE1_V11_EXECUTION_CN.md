# HCP 上的 Table 1 与 v11 执行说明

本文档固定用于以下代码与服务器：

```text
GitHub: https://github.com/18054546156/graph-coverage-selection.git
branch: codex/table1-v11-execution
baseline commit: 666e3839cbe3d1e3a12ec66b282547233a844317
HCP root: /project/prj-sis01/xuxiaoyu/graph_bench
Slurm account: hpc-sis
partition: gpu-rtx4090
```

后续文档或 Slurm 修订可以产生新提交，但 `table1_reproduction/vendor/graphcov`
必须继续通过 manifest 校验，并保持来源为作者提交 `8cf757a`。根目录算法、`v11/`
和未来 `v12/` 的修改不得进入 Table 1 vendor。

## 1. 现有资产

五个数据集的 28 和 224 归档位于：

```text
/home/xiaoyuxu2/.medmnist/{organsmnist,organamnist,pathmnist,tissuemnist,bloodmnist}_224.npz
```

UNI 模型已在用户 Hugging Face cache。Path/Blood 的 UNI embedding 和 dynamics 已存在于
旧 benchmark cache；OrganS/OrganA/Tissue 的 embedding 与 dynamics 会由 Table 1 Job 1
首次生成。模型已下载不等于每个数据集的 embedding 都已生成。

新 benchmark 只复用不可变数据、模型和 Python 环境；代码、选择结果、训练结果和日志均写入
`graph_bench` 自己的目录。

## 2. 一次性服务器设置

```bash
ssh luhpc
cd /project/prj-sis01/xuxiaoyu
git clone --branch codex/table1-v11-execution --single-branch \
  https://github.com/18054546156/graph-coverage-selection.git graph_bench
cd graph_bench

mkdir -p cache envs table1_reproduction/logs v11/logs
ln -s /project/prj-sis01/xuxiaoyu/graph_select/cache/embeddings cache/embeddings
ln -s /project/prj-sis01/xuxiaoyu/graph_select/envs/graphcov-py311 envs/graphcov-py311

export GRAPHCOV_PYTHON=/project/prj-sis01/xuxiaoyu/graph_bench/envs/graphcov-py311/bin/python
```

不要复制 12.6 GB PathMNIST。MedMNIST 归档和 Hugging Face cache 本来就在用户目录，VPN
断开不会影响已由 `sbatch` 提交的任务。

## 3. 启动前检查

FAISS GPU 1.9.0 在登录节点导入时会访问 CUDA driver，并可能直接 abort。因此不要在登录节点
运行 Table 1 dry-run 或 `faiss.get_num_gpus()`。先做不导入 FAISS 的 shell/Slurm 静态检查：

```bash
cd /project/prj-sis01/xuxiaoyu/graph_bench

bash -n table1_reproduction/slurm/*.slurm v11/slurm/*.slurm
sbatch --test-only table1_reproduction/slurm/job1_select_array.slurm
sbatch --test-only table1_reproduction/slurm/job2_downstream_array.slurm
sbatch --test-only v11/slurm/job1_select_array.slurm
sbatch --test-only v11/slurm/job2_array.slurm
```

再提交一次短 GPU preflight。它不做选择和训练，只验证 CUDA/FAISS、vendor hash、数据与 UNI
cache、编译和 dry-run 数量：

```bash
PREFLIGHT_JOB=$(sbatch --parsable table1_reproduction/slurm/preflight_gpu.slurm)
echo "$PREFLIGHT_JOB"
squeue -j "$PREFLIGHT_JOB"
sacct -j "$PREFLIGHT_JOB" --format=JobID,State,ExitCode,Elapsed
cat "table1_reproduction/logs/preflight_${PREFLIGHT_JOB}.out"
```

预期日志以 `PREFLIGHT PASSED` 结束，并报告 Table 1 `160/400`，v11 `100/25/75`。
当前复用环境没有安装 pytest，因此正式启动门槛使用上述可执行 preflight；不为测试工具改动共享
训练环境。

## 4. Table 1

Table 1 固定八个方法：Random、EL2N、Forgetting、EVA、Facility、FPS、Herding、
Graph-A2。Graph-A2 固定 global、`k=10`、`H=2`、`K=A_hat+A_hat^2`。

### 4.1 Job 1：五数据集并行选择

```bash
cd /project/prj-sis01/xuxiaoyu/graph_bench
mkdir -p table1_reproduction/logs
TABLE1_JOB1=$(sbatch --parsable table1_reproduction/slurm/job1_select_array.slurm)
echo "$TABLE1_JOB1"
squeue -j "$TABLE1_JOB1"
```

数组 `0-4%5`：每个数据集一个任务，各申请 1 GPU、16 CPU、48 GiB；总计最多
5 GPU、80 CPU、240 GiB。每个任务在同一进程中共享该数据集的 embedding 和 dynamics，
避免八个方法重复加载。

只有五个数组元素全部 `COMPLETED 0:0` 后才能启动 Job 2：

```bash
sacct -j "$TABLE1_JOB1" --format=JobID,State,ExitCode,Elapsed
find table1_reproduction/outputs/job1_selection \
  -name 'selection_manifest_*.json' -maxdepth 1 -type f
```

### 4.2 Job 2：400 次下游训练

```bash
TABLE1_JOB2=$(sbatch --parsable table1_reproduction/slurm/job2_downstream_array.slurm)
echo "$TABLE1_JOB2"
squeue -j "$TABLE1_JOB2"
```

数组 `0-9%5` 是十个 resumable shard worker，每个顺序处理约 40 次训练，最多五个 worker
并发。资源峰值为 5 GPU、40 CPU、240 GiB。之所以仍给 48 GiB，是当前官方 Job 2 会加载
完整 224 train/test NPZ；它不是仅加载 selected images 的轻量数据管线。

任务重提时，已有 `test_result.json` 会自动跳过。完成后：

```bash
$GRAPHCOV_PYTHON table1_reproduction/experiments/summarize.py \
  --input-root table1_reproduction/outputs/job2_table1 \
  --output-dir table1_reproduction/outputs/job2_table1/summary
```

最终主指标是五个训练 seed 的 test balanced accuracy mean +/- std。不要用
`best_balanced_accuracy` 替代最终 1000-epoch test BA。

## 5. v11 校准与冻结

v11 不是 Table 1 的替代实现。所有 `a0_original` 必须与 Table 1 Job 1 的五数据集 x
两预算共十个 Graph-A2 indices 完全一致；缺失或 hash/index 不一致会停止 Job 1。

### 5.1 生成 v11 subsets

```bash
mkdir -p v11/logs
V11_JOB1=$(sbatch --parsable v11/slurm/job1_select_array.slurm)
echo "$V11_JOB1"
```

必须先完成 Table 1 Job 1，因为 v11 的十个 frozen references 来自该阶段。

### 5.2 validation-only 校准

先跑 seed 42 的 25 个低成本筛查：

```bash
V11_CONFIG=/project/prj-sis01/xuxiaoyu/graph_bench/v11/configs/job2_table1_validation_seed42.json
V11_VAL1=$(sbatch --parsable --export=ALL,V11_CONFIG="$V11_CONFIG" v11/slurm/job2_array.slurm)
```

确认无系统性失败后跑三种子 config。它与 seed-42 config 共用输出路径，因此会跳过已完成
seed 42，只补 seed 43/44：

```bash
V11_CONFIG=/project/prj-sis01/xuxiaoyu/graph_bench/v11/configs/job2_table1_validation_3seeds.json
V11_VAL3=$(sbatch --parsable --export=ALL,V11_CONFIG="$V11_CONFIG" v11/slurm/job2_array.slurm)
```

不要同时提交 seed-42 与三种子数组；共享输出可能发生竞争。HCP 最多提交 15 个任务，
每个数组包含 10 个元素，因此应分阶段提交。

### 5.3 冻结 winner

冻结程序只接受 validation config 明确列出的 dataset/ratio/variant/seed。额外、重复或缺失
结果都会拒绝，防止旧输出污染。推荐门槛为 validation BA 至少提高 0.5 percentage point，
且 mean worst-class recall 不下降：

```bash
$GRAPHCOV_PYTHON v11/experiments/freeze_validation_winners.py \
  --validation-root v11/outputs/job2_table1_validation \
  --validation-config v11/configs/job2_table1_validation_3seeds.json \
  --selection-root v11/outputs/job1_table1_calibration \
  --output-config v11/configs/generated_job2_table1_test.json \
  --test-output-root v11/outputs/job2_table1_test \
  --minimum-ba-gain 0.005 \
  --worst-recall-tolerance 0.0
```

校准只看 5% validation。每个数据集冻结一个 winner 后，同一个 winner 必须同时用于 2% 和
5% test，不能根据 test 或 ratio 再改参数。

### 5.4 五种子 test confirmation

```bash
V11_CONFIG=/project/prj-sis01/xuxiaoyu/graph_bench/v11/configs/generated_job2_table1_test.json
V11_TEST=$(sbatch --parsable --export=ALL,V11_CONFIG="$V11_CONFIG" v11/slurm/job2_array.slurm)
```

generated config 同时评估 Graph-A2 baseline 与 frozen winner，最多 100 次训练。完成后：

```bash
$GRAPHCOV_PYTHON v11/experiments/summarize_downstream.py \
  --input-root v11/outputs/job2_table1_test \
  --output-dir v11/outputs/job2_table1_test/summary
```

## 6. v11 参数与论文结论 gate

当前候选不是“已证明 winner”：

| Dataset | k | H | max degree |
|---|---:|---:|---:|
| OrganS | 30 | 4 | 70 |
| OrganA | 20 | 4 | 70 |
| Path | 20 | 3 | 70 |
| Tissue | 15 | 3 | 50 |
| Blood | 30 | 4 | 70 |

合法结论必须满足：

1. Table 1 的 vendor hash 与实际 import path 检查通过。
2. v11 的十个 `a0_original` 与 Table 1 frozen indices 完全一致。
3. winner 仅由 5% validation 三种子选择，随后冻结。
4. 2%/5% test 均使用同一 dataset-level winner，并报告五种子 mean +/- std。
5. 结果应报告 paired seed difference、worst-class recall、运行时间和峰值内存，而不是只挑 BA。
6. 不能以“把五个数据集都调赢”为参数选择原则；未通过 validation gate 的数据集保持
   Graph-A2 baseline。

五数据集或十个条件均胜出只能证明经验性能，不能单独证明 novelty 或新理论。当前 v11 的
bounded PPR-style diffusion 与 margin safety 是可检验的方法组合，但 PPR、稀疏截断和 margin
约束本身已有相关思想。若要主张新理论，需要独立定义目标、性质或保证，例如预算校准的 shell
diffusion、单调/次模性条件、近似界或复杂度界，并在代码和实验中实现；文档中的设想不能写成
已经完成的理论贡献。

## 7. Slurm 运维

```bash
squeue -u xiaoyuxu2
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
scancel JOBID
```

HCP 限制为同时运行最多 5 个任务、最多 12 GPU、128 CPU、256 GiB 内存、最多提交 15 个
任务。本方案始终限制为 5 个单 GPU worker；不要给单个 Python 进程申请多张 GPU，因为当前
FAISS/ResNet runner 不会自动使用多卡。
