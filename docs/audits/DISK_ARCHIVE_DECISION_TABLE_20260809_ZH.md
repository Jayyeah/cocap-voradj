# 磁盘归档与删除确认总表（2026-08-09）

> 范围：`/home/yjq`，重点为 `/home/yjq/rl`。本轮仅做只读盘点并新增审查文档；没有删除、移动、压缩任何既有训练资产，也没有向训练进程发送信号。

## 1. 结论

- 根盘共约 915 GiB，已用约 831 GiB，用户可见空闲约 39 GiB，使用率 96%。`/home/yjq` 占约 286 GiB，其中 `/home/yjq/rl` 占约 270 GiB。
- 原 4A/4C 当前均在约 66k，进程正常。按每 25k 保存完整 replay 的现实现估算，两线到 200k 还可能新增约 70 GiB，现有空闲不足以让它们安全跑完。
- `/data/disk1`、`/data/disk2` 虽有大空间，但当前用户 `yjq` 不属于对应写权限组；本方案不依赖 sudo，也不尝试借 Docker 绕过权限。
- 最保守的首批方案是 `R1 + R2 + R3`：压缩历史长日志/W&B、对逐对 SHA 相同的 replay 做硬链接去重、清理已确认不用的旧 VS Code Server/扩展缓存。预计净释放约 57–60 GiB，且不删除任何 checkpoint/GIF；完成后空闲预计接近 96–99 GiB。
- 若目标是一次清出更大余量，可改选主仓库 `A` 批次，直接删除已完成且不再需要精确 resume 的 replay 和有权威副本的过程 checkpoint，约释放 51 GiB。`A` 与 `R2` 有重叠，实际执行时会去重重算，不能把估算简单相加。

## 2. 绝对保护范围

### 2.1 活动训练与 speedopt

| 编号 | 路径 | 状态 | 处置 |
|---|---|---|---|
| P0-A | `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-08-08_200k_reference/stage4a/` | PID 527834，原 4A | 整棵目录 `ACTIVE_PROTECT` |
| P0-C | `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-08-08_200k_reference/stage4c/` | PID 528005，原 4C | 整棵目录 `ACTIVE_PROTECT` |
| P0-S | `/home/yjq/rl/CoCap1/cocap-voradj-speedopt/artifacts/2026-08-09_focal_replay_speed_validation/` | 25k 已完成，尚待最终迁移决策 | 整棵目录暂时 `ACTIVE_PROTECT` |
| P0-G | `/home/yjq/rl/CoCap1/cocap-voradj/.git/` | 两个 worktree 共用 Git object store | 禁止清理或移动 |

### 2.2 原 IQN 最终组合版：CR-MS + VCT-LS + CE

权威发布根目录全部保留：

`/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-08-04_crms_vctls_ce_final/`

| 阶段 | best model | SHA-256 |
|---|---|---|
| 4v1 | `checkpoints/stage1_4v1_step_2000000.pt` | `2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89` |
| 8v2 | `checkpoints/stage2_8v2_step_300000.pt` | `ef58ae9bdd018633afee0d16f2242d2f157a6cc18611e41474f115aed254b87e` |
| 12v3 | `checkpoints/stage3_12v3_step_700000.pt` | `5e4173eac94c921685091d60033bffb8798180e974b72ec02437e3de265848ee` |

三个模型、manifest、最终配置和发布文档均被 Git 跟踪；其 `runs/` 来源模型已逐一验证 hash 相同。正式 rollout/GIF 没有 Git 备份证据，以下四个目录全部保留：

| 场景/lineage | 目录 | 大小 |
|---|---|---:|
| 4v1 2M | `artifacts/2026-08-02_three_line_stage_best_20rollout10gif/crms_supportapproach_4v1_s1_step_2000000/` | 380 MiB |
| 8v2 当前 CE-first best | `artifacts/2026-08-02_three_line_stage_best_20rollout10gif/crms_supportapproach_8v2_s2_step_300000_cefirst_comparison/` | 719 MiB |
| 8v2 真实 12v3 warm-start lineage | `artifacts/2026-08-02_three_line_stage_best_20rollout10gif/crms_supportapproach_8v2_s2_step_500000/` | 1.8 GiB |
| 12v3 700k | `artifacts/2026-08-02_three_line_stage_best_20rollout10gif/crms_supportapproach_12v3_s3_step_700000/` | 1.2 GiB |

### 2.3 独立 VCT-LS / CE 历史最优版

为覆盖用户所说“VCT-LS、CE、MS 版”的两种解释，以下本地-only best 也先保留：

| 类别 | 模型与正式 rollout/GIF | 大小/说明 |
|---|---|---|
| VCT-LS 4v1 support | `artifacts/2026-07-31_vct_ls_latest_20rollout10gif/4v1_supportblend_step1700k/` | 268 MiB |
| VCT-LS 8v2 warm | `artifacts/2026-07-31_vct_ls_latest_20rollout10gif/8v2_warm500k/` | 287 MiB |
| VCT-LS 4v1 plain 对照 | `artifacts/2026-07-31_vct_ls_latest_20rollout10gif/4v1_scratch_plain_step1850k/` | 352 MiB；若只留 best 可降级 |
| VCT-LS+CE support-scratch | `runs/vctls_ce4v1_support_scratch1m_20260731_run1/checkpoints/final_step_1000000.pt` + `artifacts/2026-08-02_final_20rollout10gif/vctls_support_scratch1m/` | 约 609 MiB |
| pure CE 8-agent | `runs/ce13_pure8_scratch_ce8schedule_500k_20260730/checkpoints/final_step_500000.pt` + `artifacts/2026-07-31_ce_final_20rollout10gif/ce13_pure8_final/` | 约 42 MiB |
| CE-OldMix 12v3 | `artifacts/2026-07-31_ce_oldmix_cv015_curriculum/best_20rollout10gif/stage3_12p3e3obs_step_600000/` | 257 MiB |

所有 YAML/YML、`effective_config*`、`scene_configs/`、manifest、selection、report、analysis、eval、summary、timing、run_args 和 Markdown 均默认保留，不进入任何粗删规则。

## 3. 建议先确认的可逆/低损清理

| 批次 | 对象 | 当前体积 | 预计净释放 | 可逆性/风险 | 建议 |
|---|---|---:|---:|---|---|
| R1 | 排除 P0 后的历史 `.wandb` 157 个 + `training_metrics.jsonl`/`episodes.jsonl` 564 个 | 44.972 GiB | 约 38–40 GiB | lossless gzip；使用前需解压；逐文件 `gzip -t` 校验 | 推荐 |
| R2 | CoCap1 中 36 对 standalone/bundle 同大小 replay | 15.522 GiB 可疑重复 | 最多 15.522 GiB | 执行前逐对 SHA-256；仅相同时用同文件系统硬链接保留两个原路径 | 推荐 |
| R3 | 4 个非当前 VS Code Server + 旧 OpenAI/Claude 扩展版本 + VSIX cache | 约 4.1 GiB | 约 4.1 GiB | 当前运行版本 `Stable-df53...`、OpenAI `26.803...`、Claude `2.1.226` 明确保留；旧版本可重装 | 推荐 |
| R4 | `/home/yjq/codexbackup` 的 2026-07-03 Codex 快照 | 4.5 GiB | 4.5 GiB | 可能含旧 session/log 历史；需用户确认不再需要 | 可选人工确认 |
| R5 | `CoCap1/TERL/CoCapRuns` 中 28 个 smoke/preflight/tmp/aborted 目录 | 1.920 GiB | 1.920 GiB | 先保留 config/summary 并检查文档引用 | 可选 |
| R6 | 三个干净、可按精确 Git commit 恢复的参考仓库 | 148.047 MiB | 148.047 MiB | remote 未联网复核，但本地树完全干净 | 收益小，可选 |

R1 的抽样压缩比：16 MiB `training_metrics.jsonl` 压至约 1.57 MiB（约 9.8%）；最大 W&B 文件样本压至约 1.82 MiB（约 11.4%）。估算不是保证值。执行时使用单线程、最低 CPU/I/O 优先级，逐文件完成与校验，且在活动 checkpoint 写盘窗口暂停。

R2 目前只做了大小匹配；已有一对 ladder replay 经 SHA-256 验证为字节级相同。实际执行必须逐对 hash，任何不相同或处在保护根下的文件自动跳过。

## 4. 主仓库 checkpoint/replay/GIF 删除必要性表

### A：低风险 binary 精简，约 51 GiB

| 对象 | 约占用 | 删除后保留什么 | 主要代价 |
|---|---:|---|---|
| inactive ladder replay | 20 GiB | 每线模型、eval、report、config、metrics | 旧线不能精确 replay-resume |
| CTDE contract replay | 6.3 GiB | contract 报告/config/manifest | preflight 不能精确 resume |
| P6 refactor replay | 9.2 GiB | diagnostics/report/config | 失败/过程线不能精确 resume |
| reward A1/A2/A3 replay | 4.7 GiB | 模型/eval/report/config | 消融线不能精确 resume |
| `runs/**/checkpoints/*.pt`，豁免 2.3 两个 local-only best | 约 11 GiB | metrics、配置、selection、日志；最终组合模型留 release 副本 | 不能从任意历史中间点继续训练 |

分类：`DELETE_AFTER_CONFIRM`。不触碰活动 4A/4C、speedopt、正式 GIF 和明确 best model。

### B：失败/过渡模型，约 10 GiB

| 对象 | 约占用 | 必要性 |
|---|---:|---|
| CTDE `.pt` | 2.1 GiB | contract/失败过程；报告足以保留工程结论 |
| P6 `.pt` | 2.4 GiB | 被后续合同修复取代；报告足以保留诊断 |
| reward A1/A2/smoke/重复封装 | 约 1.5 GiB | A3 root final 可留作消融代表 |
| ladder step1/smoke/第二封装 | 约 4 GiB | 正式 25k 每线最多保留一个末端模型 |

分类：`DELETE_AFTER_CONFIRM`。建议在 A 完成并复核磁盘后再决定。

### C：非最终 rollout/GIF，约 10.5 GiB

| 对象 | 约占用 | 必要性 |
|---|---:|---|
| three-line 的两个非 CR-MS 8v2 warm/scratch 对照 | 约 3.3 GiB | 消融价值，非最终组合主线 |
| `2026-08-02_final_20rollout10gif` 除 VCT-LS+CE support-scratch best 外五组 | 约 6.6 GiB | 4v1 正式消融，摘要/配置可保留 |
| CE14B 失败视觉 | 约 272 MiB | 失败展示，可留 JSON summary |
| VCT-LS plain 对照 | 352 MiB | 若只留 VCT-LS best，可删除；若保留对照则继续 KEEP |

分类：`ARCHIVE_OPTIONAL / DELETE_AFTER_CONFIRM`。这是用户明确要求最后确认的 GIF 组，不作为首批默认删除。

## 5. `/home/yjq/rl` 逐项目归档结论

非 CoCap1 共 17 个顶层目录，合计 106.494 GiB。

| 分组 | 项目 | 大小 | 判断 |
|---|---|---:|---|
| 小型 KEEP | `CoCap0`、`runtime_patch` | 32.395 MiB | 有 provenance/独有补丁，空间收益不值得风险 |
| 干净 Git，可确认删除 | `coverage_control_lpac_reference`、`rl_static_formation_maddpg_reference`、`rl_formation_fcca_reference` | 148.047 MiB | 当前 tree 干净，精确 HEAD/remote 已记录 |
| 先归档源码/软链接组 | `MTRL`、`mtrl_sg_for_terl`、顶层 `TERL`、`container_exports`、`swarm_guard_mappo`+`swarm_guard_iqn`、`swarm_guard_iqn_v2` | 5.389 GiB | 含定制源码、无 Git 资产或相对软链接；同盘 tar 不释放空间 |
| 人工 lineage 复核 | `multi_uav_encirclement` | 5.548 GiB | 29 个 tracked 改动、3,464 untracked；模型 4.127 GB、GIF 1.154 GB |
| 人工 lineage 复核 | `save` | 7.300 GiB | 历史保存集、嵌套脏 Git、部分非当前用户所有 |
| 人工 lineage 复核 | `CoCap` | 5.638 GiB | 嵌套 TERL 有 1,052 untracked；checkpoint 5.063 GiB |
| 重点人工复核 | `DualHead_Capture2Cover` | 82.444 GiB | 无 Git；checkpoint 39.771 GiB、W&B 21.217 GiB、JSONL 18.116 GiB、GIF 1.576 GiB |

`DualHead_Capture2Cover` 虽不能整体删除，但 R1 可先 lossless 压缩其 W&B/JSONL，保留全部历史数据并释放约 34–36 GiB。其 checkpoint/GIF 仍需建立模型-config-metrics lineage 后再删。

CoCap1 内旧 `TERL` 为 71.185 GiB，不是普通 clone：有本地脏改、未跟踪实现、47.408 GiB CoCapRuns 和 22.791 GiB milestones。当前只建议 R5 的 1.920 GiB 明确测试目录；整树只有在迁移独有代码/引用、保存原 IQN 证据并做恢复验证后，才可能释放最多约 71.185 GiB。

## 6. speedopt 与后续训练影响

25k speedopt 4A 已完成：25,000 transitions、5,001 updates、`all_finite=true`；eval20 capture 5/20、collision 0/20，而同 seed 原 4A 25k 为 capture 0/20、collision 0/20。训练段约 15.4k steps/h，相对原线首个 25k 约 2.3 倍，相对原 4A 的 25k→50k 段约 5.9 倍。

优化保持采样合同/分布等价，但大池 RNG 映射不同，不宣称与旧实现逐 transition 位级相同。它明显成功解决了 replay 随规模线性变慢的问题，但没有达到“25k 对原首段至少 8 倍、可立即停原线”的强替换条件。

因此：

1. 清理确认前不启动新 4A/4C，也不停止原线；
2. 首批清理后要求空闲至少约 90–100 GiB，再开 speedopt 同配置 4A/4C；
3. 新线以低优先级并行追赶，checkpoint/eval/finite/行为 gate 正常且 step 超过原线后，才逐线优雅停止旧进程；
4. 不采用现在直接关闭原 4A/4C 的方案。

## 7. 建议回复格式

推荐首批：

```text
确认执行 R1 + R2 + R3；R4/R5/R6 暂不；A/B/C 暂不。
```

如果倾向直接清理旧 binary：

```text
确认执行 A + R1 + R3；B/C 暂不。（R2 与 A 重叠，由执行脚本自动去重）
```

所有实际操作前都会再次检查 PID/CWD/打开文件，先写路径与 checksum manifest，再低优先级逐批执行；每批后复核 `df`、训练 PID、最新 step 和错误日志。任何未列入确认批次的文件都不会处理。

## 8. 子审查报告

- `COCAP_ASSET_AUDIT_20260809_ZH.md`：主 `cocap-voradj` checkpoint/replay/GIF 与最终 IQN 资产。
- `COCAP1_COPIES_AUDIT_20260809_ZH.md`：CoCap1/TERL、Stage1 快照和 worktree 关系。
- `RL_NON_COCAP1_AUDIT_20260809_ZH.md`：非 CoCap1 的 17 个顶层目录、Git/dirty/untracked/独有脚本逐项判断。
