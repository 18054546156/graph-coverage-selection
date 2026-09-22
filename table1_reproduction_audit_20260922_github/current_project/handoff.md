# Research Handoff

> 最新核验交接（2026-09-21 18:04）：[FINAL_HANDOFF_20260921.md](<C:/Users/Administrator/Documents/ChatGPT/New project/reliability_medmnistc_ab/FINAL_HANDOFF_20260921.md>)。已对照两账号实际代码、Slurm 和产物；下文保留为历史。多种子已到 336/600，32945 已结束但只完成 BloodMNIST 20 个模型重评估；新 Phase 0 仍有入口冲突。

更新时间：2026-09-21  
项目目录：`C:\Users\USER\Documents\ChatGPT\New project`  
ARIS 仓库：`C:\Users\USER\aris_repo`

## 1. 研究问题

固定 2%-5% 样本预算，只使用源域 train/validation 信息，研究如何提高未见域 MedMNIST-C 的 balanced accuracy，同时控制：

- clean BA 下降；
- corruption BA；
- 最差类别 recall；
- selection seed 和 training seed 带来的不确定性。

硬约束：不能使用目标域或 MedMNIST-C 官方 test 来挑选方法；单数据集或单 seed 只能作为探索性证据；没有人工确认时不运行新实验、不提交 HPC。

## 2. 当前实验进度

### 多种子 Job 32712

当前有效完成数：`325/600` 个 `run_complete.json`。

| 方法 | 完成数 | 状态 |
|---|---:|---|
| Random | 75/75 | 三个 arm 已完成 |
| EL2N | 75/75 | 三个 arm 已完成 |
| Forgetting | 75/75 | 三个 arm 已完成 |
| EVA | 75/75 | 三个 arm 已完成 |
| Facility | 25/75 | clean selection -> clean training 已完成，其余运行中 |
| FPS | 0/75 | 当前没有有效执行记录 |
| Herding | 0/75 | 当前没有有效执行记录 |
| Graph-A2 | 0/75 | task 正在运行，尚无完成 cell |

三个 arm：

```text
clean selection -> clean training                  125/200
corrupted selection(pixelate,s=3) -> clean training 100/200
corrupted selection(pixelate,s=3) -> corrupted training 100/200
```

当前 Slurm 状态：

```text
HPC_USER_A: 32945_0 RUNNING  # ABC test-only evaluation，不训练模型
HPC_USER_A: 32949_0 COMPLETED # PathMNIST, Random, 10 epochs smoke test，不是正式结果
HPC_USER_B: 32712_4 RUNNING  # Facility
HPC_USER_B: 32712_7 RUNNING  # Graph-A2
```

FPS/Herding 不能直接按“缺失”推断为失败；提交补任务前应先核对 `squeue`、`sacct` 和日志，并等待人工确认。

### 可以使用的结果

- 已验证单 seed 三 arm 矩阵：`120/120`，可做逐数据集、逐方法、逐 arm 的描述性统计。
- Job `32712_0-3`：Random、EL2N、Forgetting、EVA 各 75 个 cell，可做旧流程下的描述性多 seed 比较。
- Facility 已完成的 25 个 cell：只能用于已完成的 clean arm 描述。
- Job `32805`：只能用于 selected-index hash、类别计数、selection-seed 等选样诊断，不能证明下游模型性能。

### 不能作为最终结论的结果

- Job 32712 使用 `deterministic=False`，且 selection seed 固定为 42，不能当作纯 training variance 或 selection variance 的严格估计。
- 单 seed 结果不是置信区间，不能支持 Graph-A2 的 SOTA 声明。
- 旧流程和确定性流程在 selected-index hash 相同的情况下出现 `5.712` 个 clean-BA 百分点差异，因此不能把该差异解释为纯随机训练波动。
- `32949_0` 只是 10 epoch smoke test，不是正式 benchmark。

详细审计：`reliability_medmnistc_ab/research/20260921_repro_noise_audit.md`。

## 3. 研究方向状态

已有方案文件：

```text
idea-stage/CLAUDE_PROPOSAL.md
idea-stage/CODEX_ADVERSARIAL_REVIEW.md
idea-stage/CLAUDE_REBUTTAL.md
idea-stage/FINAL_PROPOSAL.md
idea-stage/EXPERIMENT_PLAN.md
```

旧方案曾把以下方向裁为主备方向：

- D2 DSS：特征漂移稳定性选集；主方向。
- D3 CWM：类条件 Wasserstein 匹配；备用方向。

但是，旧的 `CODEX_ADVERSARIAL_REVIEW.md` 是结构化反方草稿，文件自身注明为同一模型执行，不能当作真实 GPT/Codex 交叉审稿证据。

### 真实 Codex 审查

真实记录：`idea-stage/CODEX_REAL_REVIEW.md`。

本机 Codex CLI 的调用信息：

```text
CLI:       0.154.0-alpha.6.2
requested: gpt-6-astra
provider:  custom
sandbox:   read-only
```

它读取了 `CLAUDE_PROPOSAL.md`、`reliability_medmnistc_ab/handoff.md` 和 `RESEARCH_REVIEW_CN.md`，没有运行实验、提交 HPC、访问 official test 或修改核心代码。

真实审查结论：

- D2 DSS：`REVISE`，不能直接宣布为已验证主方向。
- D3 CWM：`REVISE`，不能直接进入正式实验。
- 必须先隔离旧流程与确定性流程的执行差异。
- 必须补充 `clean selection -> corrupted training` 对照。
- 必须核对 Random 在不同 selection view 下是否真正配对。
- 必须保存 data/cache hash、初始模型 hash、首 epoch batch-order hash、GPU/driver 版本和 pipeline commit。

### Claude 状态

本次任务尚未完成 Claude Opus 的真实 rebuttal。当前 PowerShell 环境中没有可执行的 `claude` 命令，因此不能把 `CLAUDE_REBUTTAL.md`、`FINAL_PROPOSAL.md` 或 `EXPERIMENT_PLAN.md` 描述为真实双模型交叉辩论结果。

## 4. 模型和调用策略

目标策略是：

```text
普通任务：GPT-5.6-Luna
复杂研究/严格反方：请求 gpt-6-astra，但必须记录 provider 和实际返回
普通 Claude 任务：Claude Haiku
困难 Claude 任务：Claude Opus
```

注意：模型名是调用层配置，不等于供应商身份证明。每次需要真实交叉审稿时，必须保存 CLI/API 的调用元数据，并把两边的输出分开保存。

不要把 API token 写入本文件、仓库、CLAUDE.md、Slurm 脚本或聊天记录。此前 token 已经在聊天中暴露，应在测试后轮换，并只通过本机安全配置更新。

## 5. 当前推荐使用方式

### A. 先读取状态

在 Codex 中直接发送：

```text
读取项目根目录 handoff.md 和 reliability_medmnistc_ab/handoff.md。
只汇报当前状态、可信结果和阻塞项，不运行实验、不提交 HPC。
```

### B. 做真实独立 Codex 审查

PowerShell：

```powershell
Set-Location 'C:\Users\USER\Documents\ChatGPT\New project'
codex exec --model gpt-6-astra --sandbox read-only --skip-git-repo-check `
  '读取 idea-stage/CLAUDE_PROPOSAL.md、reliability_medmnistc_ab/handoff.md 和 reliability_medmnistc_ab/research/20260915_robust_selection/RESEARCH_REVIEW_CN.md。只做独立反方审查，不运行实验、不提交 HPC、不访问官方 test。明确记录 requested model、provider 和哪些文件被读取。' `
  -o 'idea-stage/CODEX_REAL_REVIEW_NEW.md'
```

如果只做普通任务，可以使用配置的默认 Luna；复杂审稿才使用高成本模型。

### C. Claude/ARIS 研究命令

如果 Claude Code 和 ARIS skill 已正确安装，可在项目目录启动：

```powershell
Set-Location 'C:\Users\USER\Documents\ChatGPT\New project'
claude
```

然后使用：

```text
/research-wiki init
/research-lit "跨域鲁棒数据选择 MedMNIST-C"
/idea-discovery "读取 handoff.md，先生成候选方向，再等待独立 Codex 反方审查"
```

但是，在当前环境中 `claude` 命令尚不可执行；因此这些命令是目标工作流，不代表本轮已成功调用 Claude。

### D. 正确的双模型辩论顺序

1. Claude 生成候选方案，保存为 `CLAUDE_PROPOSAL.md`。
2. 独立 Codex 只读读取候选方案和研究 handoff，保存为 `CODEX_REAL_REVIEW.md`。
3. Claude 读取真实 Codex 审查并生成 `CLAUDE_REBUTTAL.md`。
4. 独立 Codex 再做最终裁决。
5. 只有人工确认后，才生成或执行 `EXPERIMENT_PLAN.md`。

禁止把同一个模型生成的“反方段落”称为交叉审稿。

## 6. HPC 使用方式

前提：先连接机构 VPN 并完成 Duo。VPN 每小时可能断开，但已通过 `sbatch` 提交的任务不会因为本地 VPN 断开而停止。

SSH 别名：

```powershell
ssh HPC_HOST_USER_A             # HPC_USER_A
ssh HPC_HOST_USER_B   # HPC_USER_B
```

不要复制或打印私钥，也不要把私钥路径写入实验脚本以外的共享文件。登录后常用命令：

```bash
squeue -u HPC_USER_A
squeue -u HPC_USER_B
sbatch script.slurm
sacct -j JOBID
scancel JOBID
```

推荐 Slurm 资源：

```slurm
#SBATCH --partition=YOUR_SLURM_ACCOUNT
#SBATCH --qos=YOUR_SLURM_QOS
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
```

资源上限：每账号最多运行 5 个、提交 15 个；共享额度 12 GPU、128 CPU、256G RAM；最长 7 天。长任务必须 `sbatch`，不要依赖 SSH 前台进程；交互监控可用 `tmux`。

`HPC_USER_B` 的项目目录访问需要登录后再次检查；当前 CLAUDE.md 记录其已加入 `YOUR_LINUX_GROUP`，优先检查 `/project/PROJECT_ROOT`，不可用时使用 `/home/HPC_USER_B`。

## 7. 下一步顺序

在人工确认前：

1. 不提交 FPS/Herding 补任务。
2. 不运行 D2/D3 pilot。
3. 不把旧 `FINAL_PROPOSAL.md` 当作真实交叉审稿结论。
4. 先完成同代码、同 selected indices、同 seed、同 GPU/cache 的 deterministic A/B 审计。
5. 重新判断 D2/D3 是 `PROCEED`、`REVISE` 还是 `ABANDON`。
6. 如果需要 Claude rebuttal，先修复或安装本机 Claude Code CLI，再保存真实调用元数据。

人工确认后，才进入 Phase 0。Phase 0 只能使用 train-side augmentation，不得使用MedMNIST-C official corruption 文件来选方法。
