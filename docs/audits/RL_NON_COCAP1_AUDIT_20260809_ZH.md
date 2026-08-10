# `/home/yjq/rl` 非 `CoCap1` 顶层目录只读审计（2026-08-09）

## 结论先行

- 本次范围内有 **17 个顶层目录**，合计 `114,347,589,632 B`（**106.494 GiB**）；`CoCap1` 明确排除，根目录两份奖励函数 Markdown 也不在“项目/目录”范围内。
- 仅有 4 个顶层目录是干净的独立 Git 工作树：`coverage_control_lpac_reference`、`rl_static_formation_maddpg_reference`、`rl_formation_fcca_reference`，以及“源代码部分可由 Git 还原、但含未跟踪 PDF/TXT”的 `MTRL`。其中前三者才是低风险的 `DELETE_AFTER_CONFIRM` 候选。
- “有 GitHub remote”不等于“整个目录已备份”。`TERL`、`multi_uav_encirclement`、`CoCap/TERL` 和 `save` 中的嵌套 TERL 副本均有未提交或未跟踪内容；这些内容不在 remote 可还原范围内。
- **`DualHead_Capture2Cover` 不是 Git 仓库**，且 82.444 GiB 中包括 39.771 GiB checkpoint、21.217 GiB W&B 离线包、18.116 GiB JSONL 指标。它不应因“GitHub 已备份”的假设被删除，结论为 `MANUAL_REVIEW`。
- 立即可低风险释放的 `DELETE_AFTER_CONFIRM` 项仅约 **148.047 MiB**；经“外部介质归档 + 校验 + 还原测试”后，`ARCHIVE_TAR` 组可额外释放约 **5.389 GiB**。其余 **100.929 GiB** 必须先人工复核。

## 范围、方法与限制

### 范围

- 审计根：`/home/yjq/rl`。
- 排除：`/home/yjq/rl/CoCap1`（没有枚举、比较或分类其内容）。
- 纳入：除 `CoCap1` 外所有顶层目录，包括隐藏目录 `.claude`。

### 只读方法

- 所有目录遍历、`du`、Git 元数据和进程命令行检查均以 `nice -n 19`、`ionice -c3` 执行；Git 检查使用 `GIT_OPTIONAL_LOCKS=0`，避免可选锁/索引刷新写入。
- 对模型、W&B 包、GIF、JSONL、rollout 文件只读取路径、大小和时间元数据；**未读取 checkpoint、模型权重或大训练日志内容**。
- 仅阅读了小型 README/manifest 和少量源代码树差异，以识别项目用途与独有脚本；未执行训练、评估、安装、`fetch`、`pull`、压缩、移动或删除。
- `git remote` 是本机配置证据，不是远端可用性的证明。本审计没有联网执行 `git ls-remote`/`fetch`，也没有证明远端仍保留相应提交或 LFS 对象。
- `ps` 命令行检查未发现任何活动进程参数引用本报告中的非 `CoCap1` 路径；这不能替代在实际归档/删除窗口再次检查 CWD、打开文件和调度器任务。

### 分类定义

| 分类 | 含义 |
|---|---|
| `KEEP` | 当前证据显示仍有基线/独有补丁价值，且空间收益很小或删除风险不成比例。 |
| `ARCHIVE_TAR` | 可在**不同文件系统/外部存储**创建归档、记录 SHA-256、执行抽样还原后再删原目录。归档写在同一文件系统不会释放空间。 |
| `DELETE_AFTER_CONFIRM` | Git 工作树干净、所有已追踪内容有明确 GitHub remote；仍需用户确认远端/依赖可用。 |
| `MANUAL_REVIEW` | 含未提交源码、未跟踪模型/数据、嵌套仓库、外部/交叉依赖或无 Git 来源；不得直接清理。 |

## 总览与空间估算

| 顶层目录 | 最近目录修改时间（+08） | 大小 | Git 总结 | 建议 | 原目录移除后可释放 |
|---|---:|---:|---|---|---:|
| `.claude` | 2026-06-02 22:55:22 | 8 KiB | 无 | `DELETE_AFTER_CONFIRM` | 8 KiB |
| `MTRL` | 2026-03-26 21:10:45 | 8.84 MiB | Git；干净追踪树，但 5 个未跟踪文件 | `ARCHIVE_TAR` | 8.84 MiB |
| `mtrl_sg_for_terl` | 2026-04-08 15:31:41 | 376 KiB | 无；MTRL 的定制化子树 | `ARCHIVE_TAR` | 376 KiB |
| `TERL` | 2026-04-08 15:54:19 | 91.55 MiB | Git；4 个跟踪修改、7 个未跟踪配置 | `ARCHIVE_TAR` | 91.55 MiB |
| `runtime_patch` | 2026-04-08 16:56:53 | 28 KiB | 无；独有补丁文件 | `KEEP` | 0 |
| `container_exports` | 2026-05-15 14:28:58 | 0.958 GiB | 无 | `ARCHIVE_TAR` | 0.958 GiB |
| `coverage_control_lpac_reference` | 2026-05-20 16:17:10 | 23.24 MiB | Git；干净 | `DELETE_AFTER_CONFIRM` | 23.24 MiB |
| `rl_static_formation_maddpg_reference` | 2026-05-20 16:46:46 | 31.59 MiB | Git；干净 | `DELETE_AFTER_CONFIRM` | 31.59 MiB |
| `rl_formation_fcca_reference` | 2026-05-20 16:46:46 | 93.21 MiB | Git；干净 | `DELETE_AFTER_CONFIRM` | 93.21 MiB |
| `multi_uav_encirclement` | 2026-05-20 19:03:18 | 5.548 GiB | Git；29 个跟踪改动、3,464 个未跟踪文件 | `MANUAL_REVIEW` | 5.548 GiB（暂不可计入） |
| `swarm_guard_mappo` | 2026-05-29 16:15:53 | 1.867 GiB | 无；为 IQN 目录提供软链接目标 | `ARCHIVE_TAR`（与 IQN 成组） | 1.867 GiB |
| `swarm_guard_iqn` | 2026-06-03 15:16:50 | 0.332 GiB | 无；依赖 `swarm_guard_mappo` 软链接 | `ARCHIVE_TAR`（与 MAPPO 成组） | 0.332 GiB |
| `swarm_guard_iqn_v2` | 2026-06-04 17:51:32 | 2.134 GiB | 无 | `ARCHIVE_TAR` | 2.134 GiB |
| `save` | 2026-06-10 13:37:40 | 7.300 GiB | 顶层无 Git；含 2 个有本地状态的嵌套 Git 副本 | `MANUAL_REVIEW` | 7.300 GiB（暂不可计入） |
| `CoCap0` | 2026-06-26 15:02:07 | 32.37 MiB | 无；文档称“current solid CoCap baseline” | `KEEP` | 0 |
| `DualHead_Capture2Cover` | 2026-06-27 14:33:21 | 82.444 GiB | 无 Git；大量不可重建训练证据 | `MANUAL_REVIEW` | 82.444 GiB（暂不可计入） |
| `CoCap` | 2026-06-29 17:52:22 | 5.638 GiB | 顶层无 Git；嵌套 `TERL` 有 1,052 个未跟踪文件 | `MANUAL_REVIEW` | 5.638 GiB（暂不可计入） |

分类空间合计：

| 分类 | 总量 | 说明 |
|---|---:|---|
| `KEEP` | 32.395 MiB | `CoCap0`、`runtime_patch`。 |
| `DELETE_AFTER_CONFIRM` | 148.047 MiB | `.claude` 与 3 个干净参考仓库。 |
| `ARCHIVE_TAR` | 5.389 GiB | 归档必须落在外部/不同文件系统，且验证成功后才可能释放。 |
| `MANUAL_REVIEW` | 100.929 GiB | `CoCap`、`DualHead_Capture2Cover`、`multi_uav_encirclement`、`save`。 |
| **合计** | **106.494 GiB** | 不含 `CoCap1`。 |

## Git 证据汇总

### 顶层独立 Git 工作树

| 目录 | branch / HEAD | origin | 跟踪数 | dirty（跟踪） | untracked | GitHub 可恢复性判断 |
|---|---|---|---:|---|---:|---|
| `MTRL` | `main` / `016fea74ca0f08c522f6ab7d7d8f229bb72d6a8d` | `https://github.com/Guobin-Zhu/MTRL-SG` | 156 | 0 | 5 | 156 个已追踪文件可按该提交从 remote 重建；PDF/TXT 与 pycache 不在 Git。 |
| `TERL` | `main` / `143359b2722d49c29b4fecc0ad1fd8d46326e45a` | `https://github.com/ApricityZ/TERL` | 68 | 4 修改 | 7 | 仅官方基线可重建；本地代码修改和 reward-probe 配置不在 remote。 |
| `coverage_control_lpac_reference` | `main` / `b8e77dcc5a0061109abb0e14a51bb22d882dd071` | `https://github.com/KumarRobotics/CoverageControl.git` | 228 | 0 | 0 | 工作树干净；`pyproject.toml`、`setup.sh`、CMake 源码均已追踪，条件性可重建。 |
| `multi_uav_encirclement` | `main` / `143359b2722d49c29b4fecc0ad1fd8d46326e45a` | `https://github.com/ApricityZ/TERL` | 68 | 23 修改 + 6 删除 | 3,464 | remote 只对应 TERL 基线；本地训练实现、配置、模型和结果不可由 GitHub 重建。 |
| `rl_formation_fcca_reference` | `main` / `e6dcaadc9267a28e804baaeffcb39121b44392d3` | `https://github.com/MACSCLAB/LLM_MARL_FCCA.git` | 458 | 0 | 0 | 工作树干净；源码可重建，但 ROS/仿真部署依赖仍需另行配置。 |
| `rl_static_formation_maddpg_reference` | `main` / `0c9e37a95727b1d52aff762cbcac4625da3dcd03` | `https://github.com/legalaspro/maddpg-zoo-torch.git` | 62 | 0 | 0 | 工作树干净；`environment.yml` 在仓库中，条件性可重建。 |

### 非顶层但影响顶层分类的嵌套 Git 工作树

| 所在顶层目录 | 嵌套仓库 | branch / HEAD | origin | 跟踪状态 | 可恢复性判断 |
|---|---|---|---|---|---|
| `CoCap` | `CoCap/TERL` | `main` / `143359b2722d49c29b4fecc0ad1fd8d46326e45a` | `https://github.com/ApricityZ/TERL.git` | 68 tracked；0 跟踪修改；**1,052 untracked** | 官方 TERL 基线可恢复；`cocap/`、配置、运行记录、milestone/模型均不在 remote。 |
| `save` | `save/terl_native_supervised_20260607/source_official_143359b` | `main` / `143359b2722d49c29b4fecc0ad1fd8d46326e45a` | `https://github.com/ApricityZ/TERL` | 68 tracked；`Dockerfile` 修改；49 untracked | 基线可恢复；Dockerfile 与一整套原生监督训练产物不在 remote。目录属主为 `dnsmasq:systemd-journal`，归档/清理另有权限风险。 |
| `save` | `save/terl_native_supervised_20260607/source_smoke_143359b` | `main` / `143359b2722d49c29b4fecc0ad1fd8d46326e45a` | `https://github.com/ApricityZ/TERL` | 68 tracked；`Dockerfile` 修改；11 untracked | 同上；未跟踪内容为 smoke 模型、配置、评估与摘要。 |

## 逐目录审计

### `.claude`

- 大小/时间：8 KiB；2026-06-02 22:55:22。
- 用途：单个 `settings.local.json` 的本地 Claude/Codex 类工作区配置，不是训练源码、模型或实验记录。
- Git：无 `.git`、无 remote、无 tracked/dirty/untracked 概念。
- 大文件：无 checkpoint、GIF、rollout 或日志。
- 可重建性：可手工重新建立；不依赖 Git。
- 结论：`DELETE_AFTER_CONFIRM`。风险低、空间收益可忽略；若保留本地工具权限偏好则改为 `KEEP`。

### `MTRL`

- 大小/时间：8.84 MiB；2026-03-26 21:10:45。
- 用途：上游 MTRL-SG/skill graph 参考实现；顶层含 `cus_gym`、`mt_marl_sg`、README，以及未跟踪论文 PDF/TXT。
- Git：独立 Git；`main`，HEAD `016fea74ca0f08c522f6ab7d7d8f229bb72d6a8d`，origin `Guobin-Zhu/MTRL-SG`；156 tracked，跟踪树干净，无 staged 项。
- 未提交/独有内容：5 个 untracked：`Multi-Task_Multi-Agent_Reinforcement_Learning_via_Skill_Graphs.pdf`、对应 TXT、3 个 Python 3.12 `__pycache__` 文件。源码本体无本地修改。
- 大文件：论文 PDF 约 6.82 MiB；无模型、GIF、rollout。
- 可重建性：已追踪源码可从 remote 的精确 HEAD 重建；依赖仍须按仓库 README/custom gym 构建，未跟踪论文文件不由 Git 恢复。
- 结论：`ARCHIVE_TAR`。先保留 PDF/TXT（或确认可再下载）后可移除原目录；空间收益仅 8.84 MiB。

### `mtrl_sg_for_terl`

- 大小/时间：376 KiB；2026-04-08 15:31:41。
- 用途：面向 TERL 的 skill-graph 定制子树，只有 `skill_graph/`，不含 Git 元数据。
- Git：无 Git/remote；因此源码不能由 GitHub 直接还原。
- 独有代码：与 `MTRL/mt_marl_sg/skill_graph` 做了仅源代码树比较（排除 `__pycache__`）。该副本额外含 `benchmark_terl_skills.py`、`list_terl_skills.py`、`terl_bridge.py` 和顶层 `__init__.py`；并修改了 `algorithm/__init__.py`、`envs_tasks_desc.py`、`reporter.py`、`skill_graph_base.py`、`skill_graph_wrapper.py`、`inference.py`、`train.py`。
- 大文件：仅 28 个 pycache（约 202 KiB）；无模型/GIF/rollout。
- 可重建性：需要同时保留该定制源码与其 MTRL/TERL 依赖；不可只从 MTRL 的 Git commit 恢复。
- 结论：`ARCHIVE_TAR`。小但有明确定制差异，不能直接删除。

### `TERL`

- 大小/时间：91.55 MiB；2026-04-08 15:54:19。
- 用途：官方 TERL 基线的本地研究副本，含 `Dockerfile`、`requirements.txt`、`train_rl_with_configs.py`、ablation/baseline 入口和六套已追踪演示模型。
- Git：独立 Git；`main`，HEAD `143359b2722d49c29b4fecc0ad1fd8d46326e45a`，origin `https://github.com/ApricityZ/TERL`；68 tracked。
- Dirty/untracked：4 个未暂存跟踪修改：`Dockerfile`、`environment/env.py`、`policy/trainer.py`、`train_rl_with_configs.py`；无 staged 变更。另有 7 个未跟踪 reward-probe YAML：`terl_reward_probe_{base,add_front,add_ms}_*.yaml`（2026-06-08 系列）。
- 大文件：6 个 checkpoint 合计 46.5 MiB；官方 `TrainedModels` 与 Git pack 各约 43.7 MiB 级；无实质 rollout 资产。
- 可重建性：官方 HEAD + `requirements.txt`/Docker 可重建基线；本地四项源码修改、7 个实验配置及任何本地运行结果不能由 GitHub 重建。
- 结论：`ARCHIVE_TAR`。也可先导出 Git patch 与 untracked 配置清单；未完成前不应 `DELETE_AFTER_CONFIRM`。

### `runtime_patch`

- 大小/时间：28 KiB；2026-04-08 16:56:53。
- 用途：只含 `robots/pursuer.py` 的运行时补丁，文件大小 18,377 B。
- Git：无 Git/remote；没有依赖清单，且不能独立运行。
- 独有代码证据：其 SHA-256 为 `90785461991f48c4489f839f4c393862fda7ca800bfac5280cc16ffb60686d5c`；在本次排除 `CoCap1` 的 `robots/pursuer.py` 候选中没有相同哈希，因此不能证明已被其他历史目录吸收。
- 大文件：无。
- 可重建性：需要已知目标基线和补丁应用语义，Git/依赖本身无法重建此文件。
- 结论：`KEEP`。空间无关紧要，但潜在补丁价值高；待确认已合入某个受版本控制的项目后再删除。

### `container_exports`

- 大小/时间：0.958 GiB；2026-05-15 14:28:58。
- 用途：两份容器/训练输出导出：`inner_patrol_1m_status_20260515` 与 `terl_skill_graph_2026-04-08`；没有项目源码根或 Git 元数据。
- Git：无 Git/remote。
- 大文件：700 个 `.pth/.pt` checkpoint 合计 227.4 MB；544 个 GIF 合计 530.0 MB；包括 7 个约 14.37 MB 的 `network_params_v1...v7.pth`，以及多个 4 MB 级 GIF。
- 独有内容：训练 checkpoint、容器导出状态、skill-graph benchmark GIF；未发现可替代的 Git 工作树。
- 可重建性：没有源码/依赖锁定文件，且训练产物和渲染 GIF 非确定性；不能仅靠 GitHub 重建。
- 结论：`ARCHIVE_TAR`。需连同生成元数据归档到外部存储后才可释放约 0.958 GiB。

### `coverage_control_lpac_reference`

- 大小/时间：23.24 MiB；2026-05-20 16:17:10。
- 用途：KumarRobotics CoverageControl/LPAC 上游参考工程，含 C++、Python、CMake、`pyproject.toml` 和 `setup.sh`。
- Git：独立 Git；`main`，HEAD `b8e77dcc5a0061109abb0e14a51bb22d882dd071`，origin `https://github.com/KumarRobotics/CoverageControl.git`；228 tracked、0 dirty、0 untracked。
- 大文件：文档 `LPAC.gif` 6.60 MB；无训练 checkpoint/rollout。
- 可重建性：所有当前文件被 Git 追踪；可 clone 指定 commit 后按 `setup.sh`/CMake/pyproject 恢复。编译器、Torch 和系统库仍由环境决定。
- 结论：`DELETE_AFTER_CONFIRM`。这是本审计中最清晰的远端可恢复候选之一，潜在释放 23.24 MiB。

### `rl_static_formation_maddpg_reference`

- 大小/时间：31.59 MiB；2026-05-20 16:46:46。
- 用途：MADDPG Zoo Torch 静态编队参考实现；含 `environment.yml`、训练入口、数据和绘图输出。
- Git：独立 Git；`main`，HEAD `0c9e37a95727b1d52aff762cbcac4625da3dcd03`，origin `https://github.com/legalaspro/maddpg-zoo-torch.git`；62 tracked、0 dirty、0 untracked。
- 大文件：6 个模型/数据 checkpoint 合计 1.77 MB，3 个 GIF 合计 1.48 MB；均为当前 Git 工作树所追踪内容。
- 可重建性：可从 remote 精确 HEAD 恢复当前追踪状态，依赖通过 `environment.yml` 重建；训练再次运行不保证产出位级相同结果，但现有文件在 Git 中。
- 结论：`DELETE_AFTER_CONFIRM`，潜在释放 31.59 MiB。

### `rl_formation_fcca_reference`

- 大小/时间：93.21 MiB；2026-05-20 16:46:46。
- 用途：LLM_MARL_FCCA 编队控制参考项目，含 ROS/仿真部署、网格与策略代码。
- Git：独立 Git；`main`，HEAD `e6dcaadc9267a28e804baaeffcb39121b44392d3`，origin `https://github.com/MACSCLAB/LLM_MARL_FCCA.git`；458 tracked、0 dirty、0 untracked。
- 大文件：模型/mesh 为项目追踪资源；最大为 Turtlebot STL/DAE 网格（约 3–16 MB）。没有本地 checkpoint、GIF 或 rollout 风险。
- 可重建性：源码与网格都由 Git 支撑；运行仍需 ROS/noetic、仿真和设备相关依赖。
- 结论：`DELETE_AFTER_CONFIRM`，潜在释放 93.21 MiB。

### `multi_uav_encirclement`

- 大小/时间：5.548 GiB；2026-05-20 19:03:18。
- 用途：TERL 基线演化出的无人机围捕/coverage 研究树，含 `train_rl_with_configs.py`、`train_coverage_imitation.py`、容器快照、run/visualization、模型、W&B 缓存和文档。
- Git：独立 Git；`main`，HEAD `143359b2722d49c29b4fecc0ad1fd8d46326e45a`，origin `https://github.com/ApricityZ/TERL`；68 tracked。
- Dirty：29 个跟踪差异，其中 23 修改、6 删除。改动涉及 `Dockerfile`、README、主配置、环境、`policy/{DQN,IQN,MEAN,agent,replay_buffer,trainer}`、机器人/感知、训练与 rollout 可视化脚本；删除包括 `policy/TERL_model.py` 与部分已追踪 TERL 模型/配置。
- Untracked：3,464 个。第一层构成为：`visualization` 1,881、`host_wandb_upload_cache_20260516` 734、`container_snapshot_20260519` 425、`TrainedModels` 205、`container_snapshot_20260506` 160，另有 19 个 config、6 个 scripts、2 个 policy、2 个 experts、训练入口和实验说明。
- 独有代码/配置示例：`train_coverage_imitation.py`；`policy/coverage_imitation_model.py`、`policy/encirclement_model.py`；Voronoi/DAgger buffer 生成脚本；19 份 `config_inner_patrol_*`、`config_coverage_local4_*` 等 curriculum 配置；`EXPERIMENT_NOTES.md` 与中英文下一轮实验 README。
- 大文件：309 个 checkpoint 合计 4.127 GB；1,573 个 GIF 合计 1.154 GB；47 个 W&B 包 316 MB；rollout 119 MB；日志 107 MB。另含容器快照和有绝对外部路径的 W&B 软链接，单独恢复不一定自包含。
- 可重建性：remote 只能还原官方 TERL 基线。上述本地源码、配置、container snapshot、模型、指标和 GIF 不可由 Git/依赖重建；重训也不能复原历史权重/随机轨迹。
- 结论：`MANUAL_REVIEW`。先保存 Git diff、untracked manifest、容器快照依赖说明和模型校验清单；在此之前不应归档后删除，更不能直接删除。

### `swarm_guard_mappo`

- 大小/时间：1.867 GiB；2026-05-29 16:15:53。
- 用途：MAPPO/coverage imitation 研究副本，含 `train_mappo.py`、`pretrain_mappo_coverage.py`、`train_coverage_imitation.py`、`train_mappo_coverage_head_imitation.py`、models、rollouts 与 W&B。
- Git：无 Git/remote，故源码和结果没有 GitHub 还原证据。
- 大文件：111 个 checkpoint 合计 1.361 GB；195 GIF 172 MB；28 个 W&B 包 153 MB；5,353 项 rollout 153 MB；日志 90 MB。
- 交叉依赖：`swarm_guard_iqn` 顶层的 `environment`、`config_manager.py`、`thirdparty`、`utils`、`robots`、`experts`、`mappo` 都是指向本目录的相对软链接。
- 可重建性：有 `requirements.txt` 但没有源码 Git；依赖可安装不代表训练权重/rollout 可再现。
- 结论：`ARCHIVE_TAR`，但必须与 `swarm_guard_iqn` 作为一个保留软链接的归档单元处理；单独删除会破坏 IQN 树。

### `swarm_guard_iqn`

- 大小/时间：0.332 GiB；2026-06-03 15:16:50。
- 用途：IQN 训练/渲染副本，含 `train_iqn.py`、`iqn/`、脚本、模型、日志、W&B；基础 environment/robots/config 等由相对软链接复用 MAPPO 目录。
- Git：无 Git/remote。
- 大文件：14 个 checkpoint 合计 208.8 MB；3 个 W&B 包 71.8 MB；2 份 JSONL 64.6 MB；5 个 GIF 与 rollout 约 1.55 MB。
- 独有内容：IQN 实现、运行配置、3M v27-style 训练 checkpoint/metrics；但其基础模块依赖 `swarm_guard_mappo`。
- 可重建性：没有 Git；单独解压/恢复会因相对软链接失效而不能完整运行。
- 结论：`ARCHIVE_TAR`，和 `swarm_guard_mappo` 成组归档、成组删除，潜在释放 0.332 GiB。

### `swarm_guard_iqn_v2`

- 大小/时间：2.134 GiB；2026-06-04 17:51:32。
- 用途：后续完整 IQN/coverage IL-DAgger 副本，含环境、机器人、MAPPO、IQN、测试、训练入口 `train_iqn.py` 与 `train_iqn_coverage_il_dagger.py`。
- Git：无 Git/remote。
- 大文件：91 个 checkpoint 合计 1.392 GB；6 个 W&B 包 446 MB；12 个 JSONL 313 MB；日志 74 MB；22 GIF 20 MB。包含 `CoverageILDAgger` final 模型。
- 独有内容：scratch/self7/Voronoi curriculum、coverage DAgger 配置和模型；无 Git 证据表明已被推送。
- 可重建性：有 `requirements.txt`，可重建环境；源码、权重、指标和 GIF 不能从 GitHub/依赖恢复。
- 结论：`ARCHIVE_TAR`。归档验证后可释放约 2.134 GiB。

### `save`

- 大小/时间：7.300 GiB；2026-06-10 13:37:40。
- 用途：历史保存集合，而非单一工作树；包含 2026-01/02 训练、3 轮 ablation、demo、`multi_uav_encirclement` 备份、TERL native/supervised/local 保存和 rollout/GIF。
- Git：顶层无 Git/remote；见上方两个嵌套 TERL Git 副本。它们的安全检查最初因所有权被 Git 拒绝，后以单次 `safe.directory` 只读覆盖取证；不应靠修改 global Git config 绕过此问题。
- 主要空间：`ablation_exp_data_2_2026-03-05-12-11-38` 2.492 GiB；`terl_native_supervised_20260607` 1.056 GiB；`ablation_exp_data_3_*` 0.746 GiB；其 copy 0.597 GiB；`（terl）training_*` 0.483 GiB。
- 大文件：194 个 checkpoint 合计 2.032 GB；350 GIF 合计 1.363 GB；2 个 W&B 包约 399 MB；103 个日志约 229 MB；JSONL 约 51 MB。最大项目之一是 399 MB W&B 包，另有 114 MB 级原生训练 stdout/log。
- 嵌套源码本地状态：official 快照的 `Dockerfile` 修改且 49 个 `TrainedModels` untracked（17 个 `network_params_v*`、checkpoint manifest、eval/summary/JSONL 等）；smoke 快照的 `Dockerfile` 修改且 11 个同类 untracked。两个目录都有非当前用户属主，归档/删除要额外核对权限与保留 owner 策略。
- 可重建性：官方 TERL base 可 clone；本地 Dockerfile、原生监督模型/评估结果、ablation GIF 和训练日志不能由 Git/依赖重新得到。
- 结论：`MANUAL_REVIEW`。它已经是“保存集”，但不能因此假设可安全删除。应先按子实验生成 manifest、确认无唯一模型，然后做外部归档和恢复演练。

### `CoCap0`

- 大小/时间：32.37 MiB；2026-06-26 15:02:07。
- 用途：自描述的最小纯 RL CoCap 快照。README 首行明确为“Minimal pure-RL snapshot for the current solid CoCap baseline”；有 `MANIFEST.json`、`COPIED_FILES*.md`、模型 lineage/trim audit 文档、`train_iqn.py` 与 rollout 渲染脚本。
- Git：无 Git/remote；不能以 GitHub commit 证明完整还原。
- 独有代码/配置：M1NEW/cross-init、C3 local-map 与 Cover1 纯 RL 配置；`iqn/trainer.py`、双头网络、action shield、same-episode、纯 RL cleanup；文档逐项记录了来源/依赖理由。
- 大文件：`models/encirclement/iqn_step_003000000.pth` 与 `models/coverage/iqn_step_001000000.pth` 各约 15.31 MB；2 个 rollout GIF 合计 2.41 MB。
- 可重建性：`requirements.txt` 可重建依赖，源文件可由 snapshot 复现；两份专家模型和其具体生成过程不由 Git/依赖重建。
- 结论：`KEEP`。它是小型、带 provenance 的“current solid baseline”，保留成本 32.37 MiB，当前没有空间理由清理。

### `DualHead_Capture2Cover`（重点）

- 大小/时间：**82.444 GiB**；2026-06-27 14:33:21。
- 用途：双头 capture-to-cover 主研究树，含 `train_iqn.py`、`train_iqn_coverage_il_dagger.py`、`iqn/`、`mappo/`、teacher buffers、100+ 分析/生成/监督/评估脚本、c/o/r/multitask 实验配置、训练模型、milestones、W&B 与结果。
- Git：顶层没有 `.git`，也没有 remote、branch、HEAD、tracked/dirty/untracked 证据。`backups/` 目录不是 Git 历史的替代证据。
- 独有源码/配置证据：存在 `build_active_recovery_audit.py`、`build_c12_recovery_comparison.py`、`restore_active_cover_multitask_pipeline.sh`、`materialize_m1_initialization_checkpoint.py`、多套 checkpoint-grid evaluator、C3/Cover2/M1/M2/M3/O/R 配置生成器与 supervisor。没有 Git 就不能证明这些已被推送到任何 GitHub 仓库。
- 一层空间构成：`TrainedModels` 61,669,904,384 B（57.44 GiB）、`wandb_logs` 22,787,715,072 B（21.22 GiB）、`logs` 1.237 GiB、`milestones` 1.688 GiB、`results` 0.647 GiB、`teacher_buffers` 85.6 MiB；源码/config 合计相对很小，但无法由 Git 回收。

#### W&B、metrics 和 checkpoint 证据

| 类别 | 数量 | 字节数 | 约计 | 代表性证据/风险 |
|---|---:|---:|---:|---|
| checkpoint（`.pth/.pt/...`） | 2,027 | 42,703,859,796 B | 39.771 GiB | 大量每步/阶段模型；无 Git/LFS 证据。 |
| W&B `.wandb` 离线包 | 71 | 22,781,333,962 B | 21.217 GiB | 最大包为 `offline-run-20260627_145352-o8vnpw3m/run-o8vnpw3m.wandb`（2.46 GiB）；随后约 2.34 GiB、1.66 GiB 等。 |
| JSONL 指标 | 245 | 19,452,323,113 B | 18.116 GiB | 最大 `training_metrics.jsonl` 单文件约 1.12 GiB、1.08 GiB、1.05 GiB；训练曲线与恢复判断依赖这些历史记录。 |
| GIF | 864 | 1,692,367,997 B | 1.576 GiB | 可视化证据，非确定性重渲染。 |
| 日志 | 38,149 | 1,195,087,697 B | 1.113 GiB | 含训练/监督上下文，数量巨大。 |
| rollout 其他文件 | 708 | 145,836,252 B | 139.08 MiB | 应与配置/模型配套保存。 |

- `TrainedModels` 中多个单训练目录为 2–3 GiB：例如 2026-06-12 M1 early-stage、2026-06-27 M1new boundarydeath、2026-06-15 global-evader/coverage-then-encirclement。它们混有 checkpoint、配置、metrics 与 resume 证据，不能只按目录名断定已过期。
- 名称匹配 `best|final|resume|recovery|latest` 的恢复候选共 84 项、428,826,429 B（约 408.96 MiB），大多为 15.3 MB 级 best/final 模型和小型 `warm_resume_report.json`。这不是全部可恢复状态：其余中间 checkpoint 也可能是 resume 祖先；没有读取 manifest/训练语义，不能安全按文件名批量删。
- W&B 目录还包含指向外部 cache 的日志软链接；归档时若保留软链接，恢复环境可能缺少该外部日志；若解引用，可能意外扩大归档。必须在实际归档计划中明确策略。
- `teacher_buffers` 内有内部软链接复用样本，归档必须保留链接语义或在外部仓库做可验证的去重。

- 可重建性：`requirements.txt`/脚本可帮助重建 Python 环境，但无 Git 源码版本、无 remote、无模型远端证据。重训不能可靠复得 checkpoint、JSONL、W&B、GIF 或 recovery chain。
- 结论：`MANUAL_REVIEW`。优先级是 **保全而非清理**：先确定需保留的 run lineage、模型/配置/manifest 对应关系、W&B 链接策略和外部目标空间；在外部归档完成并抽样恢复前，预计可释放的 82.444 GiB 不能计为可用空间。

### `CoCap`

- 大小/时间：5.638 GiB；2026-06-29 17:52:22。
- 用途：顶层只有嵌套 `TERL/`，但该子树扩展为 CoCap 训练线：`cocap/`、`train_cocap.py`、`config/cocap/`、`CoCapRuns/`、`milestones/`、results 和文档。
- Git：顶层无 Git；嵌套 `TERL` 为 Git `main@143359b2722d49c29b4fecc0ad1fd8d46326e45a`，origin `ApricityZ/TERL.git`，68 tracked、0 tracked dirty、**1,052 untracked**。
- Untracked 分布：`CoCapRuns` 686、`milestones` 316、`config` 18、`results` 16、`docs` 6、`cocap` 5、`scripts` 4、`train_cocap.py` 1。独有入口/工具包括 `train_cocap.py`、two-specialist same-episode rollout、coverage/encirclement renderer，以及 M0/M1 hardgate 专家组合配置。
- 大文件：573 个 checkpoint 合计 5.436 GB（5.063 GiB）；65 GIF 合计 261.6 MB；78 个 JSONL 合计 199.1 MB；最大 `episodes.jsonl` 约 36.9 MB，单 checkpoint 约 15.77 MB。
- 可重建性：官方 TERL 基线可从 Git 恢复，但 CoCap extension、所有 configs、models、runs、milestones 和指标均未跟踪，无法从 GitHub/依赖重建。
- 结论：`MANUAL_REVIEW`。其时间紧邻 CoCap1，且有明显的专家/milestone lineage；不要把它当普通旧 clone 删除。建议先做 run/模型/配置三元组清单，再决定外部归档。

## 建议的后续顺序（不在本审计中执行）

1. 再次确认活动训练、tmux/调度器、进程 CWD 与打开文件；本审计的 `ps` 命令行无命中不足以替代这一检查。
2. 先处理 148.047 MiB 的 `DELETE_AFTER_CONFIRM` 组：确认四个 GitHub remote/提交与依赖安装仍可用；用户明确确认后再操作。
3. 对 `ARCHIVE_TAR` 组在**不同文件系统**制作归档，同时生成文件清单和 SHA-256；抽样解压/读取 Git 状态后才删除源目录。`swarm_guard_mappo` 与 `swarm_guard_iqn` 必须打包为一个单元，保留相对软链接。
4. 对 `MANUAL_REVIEW` 组先导出/保存：Git diff、untracked path manifest、checkpoint-to-config/metrics 关系、W&B 链接策略、权限/owner 信息。`DualHead_Capture2Cover`、`CoCap`、`multi_uav_encirclement`、`save` 都不能以“远端基线存在”作为删除理由。
5. 不要在 `/home/yjq/rl` 的同一文件系统创建大 tar 后期待空间下降；大型 tar 会先占用额外空间。只有经校验移交到外部存储并删除原目录后才会释放对应空间。

## 审计完成声明

- 本报告是本次任务唯一写入内容。
- 未删除、移动、重命名、压缩、上传、修改任何被审计目录或其 Git 状态。
