# CoCap1 副本 / worktree / 大产物只读审计（2026-08-09）

## 结论先行

- 本审计**没有删除、移动、压缩、修改**任何既有项目、产物、Git 元数据或训练输出；仅新增本报告。
- `/home/yjq/rl/CoCap1` 的已分配空间约为 **163 GiB**。两个活动 worktree 必须整体保护：主 worktree 为 **88.322 GiB**，speedopt worktree 为 **2.952 GiB**。
- 旧 `TERL` 为 **71.185 GiB**，不是可直接删除的普通 clone：除旧远端可重建的基线外，含本地未提交代码、未跟踪配置/脚本/文档、47.408 GiB checkpoint 运行记录、22.791 GiB milestone rollout/GIF，以及被活动文档直接引用的材料。
- 历史 `cocap_github_stage1_current_20260727` 仅 **0.252 GiB**，但它是无提交、无 remote 的 Stage1 发布快照；其选中的 A3 checkpoint/GIF、上传范围说明和少量脚本差异不能仅靠 Git 重建。应先做可恢复归档，不能直接删除。
- 当前没有发现 `.tar`、`.tar.gz`、`.zip`、`.7z`、`.rar` 等压缩包；因而不存在“删除已验证压缩包副本”这一即时回收项。
- 唯一相对低风险的删除候选是旧 `TERL/CoCapRuns/` 中名称明确为 `smoke_`、`preflight_`、`tmp_`、`aborted_`，以及 `unified_smoke_200`、`reward3x_smoke`、`step_balanced_smoke_300` 的 **28** 个已完成目录，合计约 **1.920 GiB**。仍须人工确认后才可删除，且本次没有执行删除。

## 范围、口径与分类

- 范围：`/home/yjq/rl/CoCap1`；活动 worktree 为
  `/home/yjq/rl/CoCap1/cocap-voradj` 与
  `/home/yjq/rl/CoCap1/cocap-voradj-speedopt`。
- 大小为 `du -B1` 实测的已分配空间（GiB 以 2^30 B 换算）；目录 mtime 为本机 `+0800`。
- 分类含义：`ACTIVE_PROTECT` = 不做任何清理；`KEEP` = 必须原位保留；`ARCHIVE_TAR` = 先在 CoCap1 之外生成/校验/演练恢复归档；`DELETE_AFTER_CONFIRM` = 仅在明确确认后可删；`MANUAL_REVIEW` = 不能自动判断为冗余。
- 所有训练配置均按用户要求保留；任何带 checkpoint、rollout、GIF、episode 流、manifest 的目录均不因“历史”而自动删除。

## 顶层盘点

| 项目 | 实测大小 | 根目录 mtime | Git / 可重建性 | 独有风险与处置 |
|---|---:|---|---|---|
| `cocap-voradj` | 94,835,535,872 B / **88.322 GiB** | 2026-08-07 15:49 | `ladder/implementation-20260807`，`bd635499a690…`，跟踪 `origin/ladder/implementation-20260807`，remote 为 `https://github.com/Jayyeah/cocap-voradj.git` | 存在大量未跟踪 artifacts/runs 和 `.orig`/`.bak`；活动训练正在向其中写入。`ACTIVE_PROTECT`。|
| `cocap-voradj-speedopt` | 3,170,062,336 B / **2.952 GiB** | 2026-08-09 11:18 | linked worktree：`perf/focal-replay-20260809`，`54c7488fd473…`；共用上项的 Git common dir/remote，本分支未显示 upstream | 相对主分支已有 11 个已提交文件差异，且有当日 speed-validation 产物。`ACTIVE_PROTECT`。|
| `TERL` | 76,434,321,408 B / **71.185 GiB** | 2026-07-28 21:37 | `main`，`143359b2722d…`（2025-11-06），remote `https://github.com/ApricityZ/TERL.git` | 仅该旧提交可由远端重建；本地脏改、未跟踪新主线、实验产物及活动文档引用均不能。整体 `MANUAL_REVIEW`，见下表分组。|
| `cocap_github_stage1_current_20260727` | 270,503,936 B / **0.252 GiB** | 2026-07-27 17:09 | `main` 为 unborn branch，**无 commit、无 remote、无 Git object** | 有意制作的 Stage1 发布快照；`ARCHIVE_TAR`，恢复校验后才可进入 `DELETE_AFTER_CONFIRM`。|
| 压缩包 | **0 B / 未发现** | — | — | 未找到常见 tar/zip/7z/rar/gz/bz2/xz 归档。|

## 活动进程与绝对保护范围

| PID | 观察结果 | 输出绝对保护路径 | 当前大小 | 目录 mtime | 分类 |
|---:|---|---|---:|---|---|
| 527834 | 运行中；`run_continuous_ctde_training.py`，stage4a capture，CUDA:0 | `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-08-08_200k_reference/stage4a` | 2,868,416,512 B / **2.671 GiB** | 2026-08-09 01:59 | `ACTIVE_PROTECT` |
| 528005 | 运行中；同训练入口，stage4c capture，CUDA:1 | `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-08-08_200k_reference/stage4c` | 2,868,391,936 B / **2.671 GiB** | 2026-08-09 03:23 | `ACTIVE_PROTECT` |
| 769401 | 审计时不在进程表 / 无可读 cwd | 不据此回收任何关联产物 | — | — | 仍按保守原则保护可能关联的输出 |

两个运行 PID 的 cwd 均为 `/home/yjq/rl/CoCap1/cocap-voradj`。其父目录 `artifacts/2026-08-08_200k_reference` 合计 **5.343 GiB**，严禁触碰。

## 两个活动 worktree：Git 共享与独立占用

### Git 对象没有重复占盘

`cocap-voradj-speedopt/.git` 是 78 B 的 gitdir 指针：

```text
speedopt git-dir: /home/yjq/rl/CoCap1/cocap-voradj/.git/worktrees/cocap-voradj-speedopt
common dir:        /home/yjq/rl/CoCap1/cocap-voradj/.git
```

共同 object store 位于主 worktree `.git`，有 1,429 个 loose objects、**288.98 MiB**，speedopt 没有第二份 object database。因此不要将主 `.git` 的约 291 MiB 再计入 speedopt；删除或迁移主 `.git` 会直接破坏 speedopt。

speedopt 相对主分支提交差异为 **11 个文件、3,139 行新增 / 117 行删除**，包括：

- `src/cocap_voradj/training/continuous/joint_replay.py`
- `tools/run_continuous_ctde_training.py`、`tools/benchmark_focal_replay_sampler.py`
- focal-replay 测试、周期 checkpoint 测试
- speed-validation 报告与项目 handoff/tracker 文档

故它不是可移除的工作树复制品。

### 独立 artifacts / runs

| 路径 | 大小 | mtime / 状态 | 判断 |
|---|---:|---|---|
| 主 worktree `artifacts/` | 82,575,802,368 B / **76.905 GiB** | 多条 VCT-LS、CE、MS、CTDE、ladder 证据线 | 活动主资产，`ACTIVE_PROTECT`。|
| 主 worktree `runs/` | 11,921,707,008 B / **11.103 GiB** | 含 VCT-LS、CE、CR-MS 历史训练运行 | 在活动根中，`ACTIVE_PROTECT`；speedopt 没有对应 `runs/` 目录。|
| speedopt `artifacts/2026-08-09_focal_replay_speed_validation/` | 2,850,263,040 B / **2.655 GiB** | 2026-08-09 新 speed-validation；含 25k、smoke、finalsave | 分支独有当日结果，`ACTIVE_PROTECT`。|
| speedopt `artifacts/2026-07-23_a3_apfnew_sqrtn_curriculum/` | 256,487,424 B / **0.239 GiB** | 与主根同名且总大小相同 | 74 个同相对路径文件均同大小、**0** 个同 inode（独立占盘）；尚未做内容 hash，故仅标 `MANUAL_REVIEW`。|
| speedopt `artifacts/2026-08-04_crms_vctls_ce_final/` | 53,739,520 B / **0.050 GiB** | 与主根同名且总大小相同 | 5 个同相对路径文件均同大小、**0** 个同 inode；同样为潜在重复但 speedopt 仍活动，`MANUAL_REVIEW`。|

上述两个 metadata-identical 副本最多对应约 **0.289 GiB** 的潜在未来去重空间；因 speedopt 是活动 worktree，且未做逐文件 hash 校验，**不计入当前可回收空间**。

主 worktree 中 5 个 `.orig` / `.bak` 仅约 46.9 KiB（涉及 CE/MS 配置、handoff、训练脚本），均保留在 `ACTIVE_PROTECT` 根中，不建议以微小收益冒险处理。

## 旧 TERL：目录分组、原始模型风险与建议

### Git / 工作树状态

- 基线 Git remote：`https://github.com/ApricityZ/TERL.git`；仅 `143359b2722d49c29b4fecc0ad1fd8d46326e45a` 可按远端基线重建。
- 已跟踪本地修改：`robots/evader.py`、`thirdparty/APF.py`。
- 未跟踪且不可由该 Git commit 复原：`CoCapRuns/`、`cocap/`、`config/cocap/`、`docs/`、`milestones/`、`reference/`、`results/`、`scripts/`、`train_cocap.py`、本地 PDF 等。
- 主活动项目文档仍直接指向 `TERL/docs/CE_COVERAGE_CENTROID_ENERGY_TODO_20260729_ZH.md` 与 `TERL/reference/...`；因此在未迁移这些引用前，不可把 TERL 当成独立冷备删除。

### 大目录明细

| TERL 分组 | 大小 / mtime | 关键内容与独有风险 | 分类 |
|---|---:|---|---|
| `CoCapRuns/` | 50,904,371,200 B / **47.408 GiB**；2026-07-28 15:57 | 2,980 个 `.pt/.pth/.ckpt`，共 **44.296 GiB**；是原 IQN specialist/coverage 与 VorAdj 训练 checkpoint 主体。无 GIF。 | 总体 `MANUAL_REVIEW`；可分层归档，不可整体删。|
| `CoCapRuns/` 原 IQN specialist/coverage（`capture_*`、`cover_*`、`coverage_*`、`m0*`、`m1*`） | 62 个目录，13,104,885,760 B / **12.205 GiB** | 含早期 capture/cover expert、M0/M1、IQN 相关 checkpoint；可能是原模型或其复现实验来源。 | `KEEP` / `MANUAL_REVIEW`。|
| `CoCapRuns/` VorAdj 正式训练（`voradj_*`） | 103 个目录，35,734,421,504 B / **33.280 GiB** | 课程、APF/sqrtN、safety、longwin、A2/A3 训练链；对应配置与监督脚本在本地未跟踪目录。 | `ARCHIVE_TAR`（外置且可恢复）前为 `MANUAL_REVIEW`。|
| `CoCapRuns/` 明确测试/中断输出 | 28 个目录，约 2,062,409,728 B / **1.920 GiB**；最后目录 mtime 不晚于 2026-07-28 | `smoke_*`、`preflight_*`、`tmp_*`、`aborted_*`、`unified_smoke_200`、`reward3x_smoke`、`step_balanced_smoke_300`。 | `DELETE_AFTER_CONFIRM`；删除前保留对应 config/summary，并确认不在论文/报告引用清单。|
| `milestones/` | 24,471,990,272 B / **22.791 GiB**；顶层最晚 2026-07-29 | 18 个 checkpoint（289.0 MiB）及 **2,153 个 GIF（15.344 GiB）**，是精选视觉证据，而不是普通缓存。 | `KEEP` / `MANUAL_REVIEW`。|
| milestone 的精选 rollout/GIF 线 | 11 个目录，23,441,498,112 B / **21.832 GiB** | 最大为 `2026-07-17_longwin_curriculum` 8.744 GiB（其中 `best_20rollout10gif` 8.274 GiB）与 `2026-07-23_a3_apfnew_sqrtn_curriculum` 4.448 GiB（其中 best 4.406 GiB）；此外有 A2scratch 2.468 GiB、KSL 1.464 GiB、six-lines 1.429 GiB、V0/安全线等。 | `KEEP` / `MANUAL_REVIEW`；这些正是最可能与主资产审计重叠的最终 rollout/GIF，宁可保留。|
| milestone 的原 IQN 证据 | `2026-07-02_cocap_solid_experts`、`2026-07-02_cocap_m_phase`、`2026-07-06_cocap_expert_iteration`，合计 1,030,447,104 B / **0.960 GiB** | specialist checkpoint、configs、rollouts/reports。 | `KEEP`。|
| `results/` | 509,661,184 B / **0.475 GiB**；2026-07-09 | M0/M1 same-episode / encirclement 评估 GIF、episode JSON；单 GIF 最大约 7.5 MiB。 | `KEEP`。|
| `docs/` | 68,644,864 B / **0.064 GiB**；含 2026-08-02 文档 | CE centroid/energy TODO、MS refactor、VorAdj ledger、68 MiB 设计文档；活动项目有直接引用。 | `KEEP`。|
| `reference/` | 203,018,240 B / **0.189 GiB**；2026-08-04 | 本地论文、抽取文本与综述；活动 architecture 文档有直接相对引用。 | `KEEP`。|
| `config/`、`scripts/`、`cocap/` | 分别 1.731 MiB、1.227 MiB、0.562 MiB | 全部训练配置、监督/rollout 脚本、旧训练语义；远端基线不能覆盖未跟踪部分。 | `KEEP`。|
| `TrainedModels/`、`logs/` | 48.7 MiB、166.3 MiB | 旧基线模型及运行日志；日志可随完整 archive 一起迁出。 | 模型 `KEEP`；日志 `ARCHIVE_TAR`。|

`CoCapRuns/` 的三个超过 100 MiB 的单文件为旧 episode 流：

- `voradj_stage0_pure_capture_2m_overnight_20260710_run1/episodes.jsonl` — 151,356,048 B，2026-07-10 18:28；
- `capture_expert_55m_dualparity_bp10_stationary_scratch_2m_gpu0_restart_20260705/episodes.jsonl` — 132,694,983 B，2026-07-06 18:52；
- `capture_curriculum_stage2_8p2e75_from_capture2m_20260706/episodes.jsonl` — 110,892,077 B，2026-07-08 08:01。

这些是可随已确认的旧运行 archive 迁出的高体量日志，不能因单文件大而绕过所属 checkpoint/run 的审查。

## Stage1 历史 GitHub 副本

`cocap_github_stage1_current_20260727` 的 `UPLOAD_SCOPE_STAGE1.md` 明确说明它不是随意复制：它有意保留核心代码、配置、scripts，以及 `milestones/2026-07-23_a3_apfnew_sqrtn_curriculum/` 的精选 checkpoint/GIF/summary，并排除 `CoCapRuns/`、大部分结果和 GIF。

- 与 TERL 相比，`cocap/` 相同；Stage1 的 config 少 TERL 后加的 A3 stage8/stage9 两个 YAML；`scripts/supervise_a3_apfnew_sqrtn_curriculum_20260723.py` 有实际差异（其余主要差异为 `__pycache__`）。
- Stage1 的 selected milestone 为 **0.247 GiB**；与主 worktree 的同名 artifact 有 74 个共同相对路径，其中 48 个大小相同，且均无 inode 共享。因此不能证明为完全相同副本，也不可当作可无损重建物。
- 因无 commit/remote，推荐先对整个目录做带 SHA-256 manifest 的外置 `tar`，抽样或完整恢复到临时位置，再由负责人确认是否删除原目录。该归档应连同所有 config、scripts、`UPLOAD_SCOPE_STAGE1.md` 一起保留。

## 净回收空间估算（不执行）

| 情形 | 净回收估计 | 前提 / 不包含项 |
|---|---:|---|
| 现在 | **0 B** | 活动根、PID 输出、全部配置、原 IQN/最终 rollout/GIF 均保护；本审计不建议即时清理。|
| 仅删除明确测试目录 | **约 1.920 GiB** | 仅 `DELETE_AFTER_CONFIRM` 的 28 个 TERL smoke/preflight/tmp/aborted 目录；需负责人确认其 config/summary/引用均已保留。|
| 先外置并验证 `CoCapRuns/`，再移除原目录 | **最多 47.408 GiB** | 需先将原 IQN 和 VorAdj checkpoint/run 完整外置、生成 manifest、恢复演练；这不是当前批准的删除建议。|
| 完成文档引用迁移、外置验证整个 TERL 与 Stage1 后 | **最多约 71.437 GiB** | `TERL` 71.185 GiB + Stage1 0.252 GiB；需要保留外置可恢复副本，且处理 TERL 脏改/未跟踪代码。|
| 未来停用 speedopt 后处理两个 metadata-identical artifact 组 | **最多约 0.289 GiB** | 需先逐文件 hash 验证且 speedopt 不再活动；当前完全不计入。|

把 tar 放在同一 CoCap1 文件系统只会临时增加占用，**不产生净回收**；压缩率也不可假设。Git worktree 共享已避免约 288.98 MiB object-store 的第二份复制，这不是待释放空间。

## 建议的人工确认顺序

1. 不触碰两项活动 worktree、其 `.git` common dir、以及 PID 527834 / 528005 的 stage4a/stage4c 输出。
2. 为 TERL 的 `docs/`、`reference/`、`config/`、`scripts/`、`cocap/`、两处脏改和原 IQN milestones 建立明确保留清单；先迁移活动文档的 `../../TERL/...` 引用，才讨论整个 TERL 的离线归档。
3. 审核 28 个 smoke/preflight/tmp/aborted 运行目录的 summary/config 是否已被保留；若确认，才作为首批 `DELETE_AFTER_CONFIRM` 的约 1.920 GiB。
4. 对 Stage1 做外置归档 + SHA-256 manifest + restore test；确认后才将原始 0.252 GiB 列入删除操作。
5. speedopt 结束后，hash 校验两组同名同大小 artifact，再决定是否合并为单一来源；在此之前它们均保持 `ACTIVE_PROTECT` / `MANUAL_REVIEW`。
