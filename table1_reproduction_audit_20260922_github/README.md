# Table 1 复现代码审计包

本目录是独立审计快照，不修改原项目。审计范围是 Table 1 复现相关代码、配置和运行脚本，以及作者公开仓库；不包含数据集、模型权重、运行缓存、Slurm 日志和大型实验结果。

## 目录

- `original_author/`：作者仓库固定版本，commit `8cf757adc4c333dc1427d511f0de2f246d15ebac`。
- `current_project/`：项目当前 Table 1 runner、配置、Slurm、相关变体和共享研究代码快照。
- `diffs/`：作者实现与当前项目之间的 Git diff 和摘要。
- `tools/`：复跑差异审计的只读脚本。

当前项目目录覆盖：`table1_reproduction/`、`table1_clean_only_20260922/`、`table1_ours_k50_reproduction/`、两个 PathMNIST 5% 配对/种子扫描目录、共享 `graphcov/`、`v11/`、Table 1 组会证据/汇总表、根目录相关运行脚本和项目说明。`table1_reproduction/vendor/graphcov/` 保留在快照中，用于证明正式复现 runner 导入的是固定作者代码。

## Git diff 怎么看

打开 `diffs/PROJECT_VS_AUTHOR.patch` 可以看到在作者仓库基础上，项目增加了哪些复现 runner、配置和脚本。上游算法源码不应在这个补丁中被误认为已修改；它作为 `original_author/graphcov/` 单独保存。

`diffs/VENDOR_VS_AUTHOR.patch` 比较作者原始 `graphcov/` 与项目 vendored `graphcov/`。预期为空，表示 Table 1 严格 runner 使用未改动的作者源码。

`diffs/SHARED_GRAPHCOV_VS_AUTHOR.patch` 比较项目根目录的可变 `graphcov/` 与作者代码。该目录服务于项目其他实验，不等同于严格 Table 1 runner 实际导入的 vendor 副本。

## 边界

这是围绕 Table 1 复现整理的代码审计发布包，不是工作区中 MedMNIST-C 鲁棒性实验、Feishu worker、报告、数据和结果文件的全量备份。为了避免公开个人/HPC 标识，本发布副本将 HPC 用户名、绝对项目路径、账号、QoS、分区和 Linux 组替换为占位符；因此 Slurm 脚本仅用于比较协议，不可直接提交运行。`node_modules/` 和嵌套 Git 元数据已排除。没有包含 API/SSH 凭证。原始未脱敏本地快照另存于 `table1_reproduction_audit_20260922/`。
