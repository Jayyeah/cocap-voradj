# CoCap `cocap-voradj` 磁盘资产只读审查（2026-08-09）

## 0. 结论与边界

本次只读取目录项、文件大小、文本配置/报告和少量 SHA-256；没有加载大 checkpoint，没有删除/移动训练资产，也没有向训练进程发信号。

| 范围 | 大小 |
| --- | ---: |
| `/home/yjq/rl/CoCap1/cocap-voradj/artifacts` | 77 GiB |
| `/home/yjq/rl/CoCap1/cocap-voradj/runs` | 12 GiB |
| `configs` / `docs` | 1.3 MiB / 636 KiB |

配置和文档极小且决定复现语义，建议一律保留。真正的回收对象是 inactive replay、过程 checkpoint、失败 smoke checkpoint 和非最终 rollout/GIF。

分类：`ACTIVE_PROTECT`=当前训练/验收边界；`KEEP`=权威最终资产；`ARCHIVE_OPTIONAL`=有历史/消融价值；`DELETE_CANDIDATE`=用户确认后可删。全局保留所有 YAML/YML、`effective_config*`、`scene_configs/`、manifest、selection、report、analysis、eval、summary、JSONL、timing、run_args、日志和 Markdown。

保守批次只清理 inactive replay 与已有权威副本的 `runs/**/checkpoints`，约可回收 **51 GiB**；再删失败/过渡模型约 **10 GiB**；再舍弃非最终展示约 **10.5 GiB**。

## 1. `ACTIVE_PROTECT`：绝对路径

审查时仍在运行：

| PID | 线 | 保护根目录（整个子树不碰） |
| ---: | --- | --- |
| 527834 | 原 4A | `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-08-08_200k_reference/stage4a/` |
| 528005 | 原 4C | `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-08-08_200k_reference/stage4c/` |

保护包含 tag 目录、`checkpoints/`、`.tmp/`、stdout、metrics、runtime state、replay 和将来出现的新 step。两线当前合计约 5.4 GiB，并会增长；不要边跑边删旧 step。

原 speedopt PID 769401 复核时已结束，但 25k 仍在提速正确性验收，继续保护：

```text
/home/yjq/rl/CoCap1/cocap-voradj-speedopt/artifacts/2026-08-09_focal_replay_speed_validation/
```

尤其是：

```text
/home/yjq/rl/CoCap1/cocap-voradj-speedopt/artifacts/2026-08-09_focal_replay_speed_validation/stage4a_25k/stage4a_speedopt_25k_20260809/
```

在提速验收明确结束前，建议连同 smoke/final-save smoke 一并保护，避免审查与验收竞态。

## 2. 用户要求保留的原 IQN 最终资产

“VCT-LS、CE、MS 版”可能指三者组合的最终课程，也可能包含各自的历史权威版本。为防误删，组合最终版为硬 `KEEP`；独立 VCT-LS/CE 版本暂也列 `KEEP`，待用户确认只需组合版后再降级。

### 2.1 组合最终版：CR-MS + VCT-LS + CE（硬 `KEEP`）

权威发布目录：

```text
/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-08-04_crms_vctls_ce_final/
```

| 阶段 | 权威 best model 绝对路径 | 大小 | SHA-256 |
| --- | --- | ---: | --- |
| 4v1 | `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-08-04_crms_vctls_ce_final/checkpoints/stage1_4v1_step_2000000.pt` | 17,906,562 B | `2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89` |
| 8v2 | `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-08-04_crms_vctls_ce_final/checkpoints/stage2_8v2_step_300000.pt` | 17,906,437 B | `ef58ae9bdd018633afee0d16f2242d2f157a6cc18611e41474f115aed254b87e` |
| 12v3 | `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-08-04_crms_vctls_ce_final/checkpoints/stage3_12v3_step_700000.pt` | 17,906,437 B | `5e4173eac94c921685091d60033bffb8798180e974b72ec02437e3de265848ee` |

证据/配置：

```text
/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-08-04_crms_vctls_ce_final/manifest.json
/home/yjq/rl/CoCap1/cocap-voradj/docs/CRMS_VCTLS_CE_FINAL_RELEASE_20260804_ZH.md
/home/yjq/rl/CoCap1/cocap-voradj/docs/CRMS_4V1_METHOD_AND_PROGRESS_20260803_ZH.md
/home/yjq/rl/CoCap1/cocap-voradj/docs/CRMS_8V2_300K_VS_500K_FORMAL_20260804_ZH.md
/home/yjq/rl/CoCap1/cocap-voradj/configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/
```

相关正式 rollout/GIF 全部 `KEEP`：

| 资产 | 大小 | 必要性 |
| --- | ---: | --- |
| `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-08-02_three_line_stage_best_20rollout10gif/crms_supportapproach_4v1_s1_step_2000000/` | 380 MiB | 4v1 最终正式证据 |
| `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-08-02_three_line_stage_best_20rollout10gif/crms_supportapproach_8v2_s2_step_300000_cefirst_comparison/` | 719 MiB | 当前 8v2 CE-first best |
| `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-08-02_three_line_stage_best_20rollout10gif/crms_supportapproach_8v2_s2_step_500000/` | 1.8 GiB | 12v3 的真实 warm-start lineage；300k/500k 对比基线 |
| `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-08-02_three_line_stage_best_20rollout10gif/crms_supportapproach_12v3_s3_step_700000/` | 1.2 GiB | 最终 12v3 正式证据 |

不要直接删这些最终目录的 `_workers/`：抽样显示顶层只含一部分 seed，完整“20 rollout”证据分布在 worker 合并前后目录中。

Git/备份状态：发布目录、三个模型、manifest、配置和发布文档均被 Git 跟踪；当前分支与 `origin/ladder/implementation-20260807` 同步。仓库没有 Git LFS 规则，三个 `.pt` 是 `.gitignore` 显式豁免的普通 Git 资产。上述 rollout/GIF 均 untracked，本机之外是否有备份无法由 Git 证明。

三个发布模型与 `runs/` 对应 source 已逐一验证 SHA-256 完全一致，故 source 可删：

```text
runs/crms_supportapproach_ce_curr_stage1_4p1e1obs_scratch2m_20260802_run1/checkpoints/step_2000000.pt
runs/crms_supportapproach_ce_curr_stage2_8p2e2obs_700k_20260802_run1/checkpoints/step_300000.pt
runs/crms_supportapproach_ce_curr_stage3_12p3e3obs_700k_20260802_run1/checkpoints/step_700000.pt
```

### 2.2 独立 VCT-LS 历史权威版（暂列 `KEEP`）

该归档保存三种角色，不是一个跨规模唯一 best：

| 角色 | checkpoint | rollout/GIF | 整目录大小 |
| --- | --- | --- | ---: |
| 4v1 support-blend 候选 | `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-07-31_vct_ls_latest_20rollout10gif/4v1_supportblend_step1700k/checkpoint_step_1700000.pt` | 同目录 `rollouts_capture_mix_20r10g/` | 268 MiB |
| 8v2 warm 规模候选 | `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-07-31_vct_ls_latest_20rollout10gif/8v2_warm500k/checkpoint_step_500000.pt` | 同目录 `rollouts_capture_mix_20r10g/` | 287 MiB |
| 4v1 plain 对照 | `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-07-31_vct_ls_latest_20rollout10gif/4v1_scratch_plain_step1850k/checkpoint_step_1850000.pt` | 同目录 `rollouts_capture_mix_20r10g/` | 352 MiB |

证据为同根 `README.md` 与 `docs/VORONOI_COMM_TOPOLOGY_LOCAL_SENSING.zh-CN.md`。这些内容全是 untracked/local-only。若只要 best 而不要 plain 对照，可把第三项降为 `ARCHIVE_OPTIONAL`。

后续 VCT-LS + CE 4v1 该组最佳候选也应豁免：

```text
model:   /home/yjq/rl/CoCap1/cocap-voradj/runs/vctls_ce4v1_support_scratch1m_20260731_run1/checkpoints/final_step_1000000.pt
rollout: /home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-08-02_final_20rollout10gif/vctls_support_scratch1m/
config:  /home/yjq/rl/CoCap1/cocap-voradj/configs/experiments/vct_ls_ce_20260731/vctls_ce_4v1_support_scratch1m.yaml
```

其正式评估 capture/mix capture 1.00、零碰撞；checkpoint 只有 ignored `runs/` 本地副本。

### 2.3 独立 CE 历史权威版（暂列 `KEEP`）

CE 有两个任务口径：

| 口径 | best model | rollout/GIF | 证据 |
| --- | --- | --- | --- |
| pure coverage 8-agent | `/home/yjq/rl/CoCap1/cocap-voradj/runs/ce13_pure8_scratch_ce8schedule_500k_20260730/checkpoints/final_step_500000.pt` | `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-07-31_ce_final_20rollout10gif/ce13_pure8_final/`（25 MiB） | `docs/CE_COVERAGE_EXPERIMENT_LOG.zh-CN.md`：strict 1.00、collision 0 |
| CE-OldMix 12v3 | `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-07-31_ce_oldmix_cv015_curriculum/best_20rollout10gif/stage3_12p3e3obs_step_600000/step_600000.pt` | 同目录（257 MiB） | 同文档：最终 stage3-600k；mix capture/CE strict 0.90、pure strict 1.00 |

CE-OldMix stage1-600k（980 MiB）与 stage2-700k（272 MiB）是 lineage，列 `ARCHIVE_OPTIONAL`；若只要最终模型，可仅留 checkpoint/config/selection/summary。上述 CE 资产均 untracked/ignored。

### 2.4 MS / Ring-MS

2026-07-31 初版 CR-MS warm/scratch 正式 deterministic capture 均为 0，不是“最优 MS”。当前权威 MS 是 2.1 的组合最终包；依据 `docs/CAPTURE_REWARD_RING_MS_REFACTOR_20260731_ZH.md`、`docs/CRMS_4V1_METHOD_AND_PROGRESS_20260803_ZH.md`。因此旧 `crms_{warm500k,scratch1m}` 只是失败/过渡对照。

## 3. checkpoint / replay / GIF 决策表

| 范围 | 大小 | 分类 | 建议 |
| --- | ---: | --- | --- |
| 组合最终模型发布包 | 52 MiB | `KEEP` | Git-tracked 最终三阶段模型 |
| 组合最终/lineage rollout（2.1 四目录） | 约 4.1 GiB | `KEEP` | 用户明确要求；untracked/local-only |
| `2026-07-31_vct_ls_latest_20rollout10gif` | 905 MiB | `KEEP`（待解释确认） | standalone VCT-LS 权威归档 |
| VCT-LS+CE support-scratch final + 展示 | 约 609 MiB | `KEEP`（待解释确认） | 该组 best；model 本地-only |
| CE13 final + 展示 | 约 42 MiB | `KEEP`（待解释确认） | pure CE 权威版本 |
| CE-OldMix stage3-600k | 257 MiB | `KEEP`（待解释确认） | 完整 CE 课程最终 12v3 |
| CE-OldMix stage1/stage2 | 约 1.25 GiB | `ARCHIVE_OPTIONAL` | lineage；可只留 selected model/config/summary |
| three-line 的 VCT-LS+CE 8v2 warm/scratch | 约 3.3 GiB | `ARCHIVE_OPTIONAL` | 非组合最终线，但有消融价值 |
| `2026-08-02_final_20rollout10gif` 除 support-scratch | 约 6.6 GiB | `ARCHIVE_OPTIONAL` | 4v1 正式消融；保留摘要/配置即可复盘 |
| inactive ladder replay | 20 GiB | `DELETE_CANDIDATE` | 完成的 smoke/25k 不再精确 resume；保留模型/eval/config |
| inactive ladder `.pt` | 6.0 GiB | 混合 | 正式 25k 每线最多留一个末端模型；step1/smoke/第二封装可删 |
| CTDE contract replay / `.pt` | 6.3 / 2.1 GiB | `DELETE_CANDIDATE` | contract preflight，无最终策略价值；留报告/配置 |
| P6 refactor replay / `.pt` | 9.2 / 2.4 GiB | `DELETE_CANDIDATE` | 过程/失败实验；留 `*_report.json` |
| reward first batch replay | 4.7 GiB | `DELETE_CANDIDATE` | A1/A2/A3 已完成，不承担证据角色 |
| reward first batch `.pt` | 1.6 GiB | 混合 | A3 root final 约138 MiB可选留；A1/A2/smoke/step1/第二封装可删 |
| `runs/**/checkpoints/*.pt` | 11 GiB | 混合 | 豁免 2.2/2.3 本地-only best；其余已有 artifact 副本或属过程/smoke |
| 原 4A/4C 200k | 当前约 5.4 GiB | `ACTIVE_PROTECT` | 正在写入 |
| speedopt validation | 动态 | `ACTIVE_PROTECT` | 仍在验收 |

已抽样验证 ladder Stage4A seed1 的 root/checkpoint 两份 25k replay SHA-256 均为 `967fce3311837e22a25fb9fc1a1f200c28691a8f31a68113bdd12b1da870d200`，是一份约 800 MiB 的字节级重复。相应两份 `.pt` hash 不同，不能按字节重复处理。按相同目录结构估算，inactive ladder、CTDE、reward 三组 root/checkpoint replay 镜像约 15 GiB；执行前应逐对 hash，或统一删 root final replay、保留 checkpoint bundle。

## 4. `artifacts/` 逐顶层判断

| 顶层目录 | 大小 | 分类/建议 |
| --- | ---: | --- |
| `2026-08-08_200k_reference` | 5.4 GiB | `ACTIVE_PROTECT` |
| `2026-08-08_reward_first_batch` | 6.3 GiB | 混合：A3 final/eval 可选留，其余 binary 候选删 |
| `2026-08-07_positive_feedback_ladder` | 26 GiB | 混合：配置/报告/基线视觉留；replay 候选删，关键 25k 模型可留一个 |
| `2026-08-06_ctde_contract` | 8.4 GiB | binary `DELETE_CANDIDATE`；留 contract 报告/config/manifest |
| `2026-08-06_continuous_marl_formal_20rollout10gif` | 67 MiB | `ARCHIVE_OPTIONAL`，成本低 |
| `2026-08-04_continuous_marl_refactor` | 12 GiB | P6 binary `DELETE_CANDIDATE`；留 diagnostics/report |
| `2026-08-04_crms_vctls_ce_final` | 52 MiB | `KEEP` |
| `2026-08-02_three_line_stage_best_20rollout10gif` | 7.2 GiB | 4.1 GiB 组合最终/lineage留；3.3 GiB VCT-LS对照可选 |
| `2026-08-02_final_20rollout10gif` | 7.2 GiB | support-scratch 592 MiB暂留；其余消融可选 |
| `2026-08-02_cr_ms_support_approach_ce_curriculum` | 8.5 MiB | `KEEP`，最终课程元数据 |
| `2026-08-02_vct_ls_ce_support_8v2_parallel` | 5.9 MiB | `ARCHIVE_OPTIONAL`，建议留摘要 |
| `2026-07-31_ce_oldmix_cv015_curriculum` | 1.5 GiB | stage3 final留；stage1/2 lineage可选 |
| `2026-07-31_ce_final_20rollout10gif` | 296 MiB | CE13留；CE14B失败视觉约272 MiB可选删 |
| `2026-07-31_vct_ls_latest_20rollout10gif` | 905 MiB | `KEEP`（待解释确认） |
| `2026-07-31_cr_ms_ring_vctls_ce_4v1` | 2.0 MiB | `ARCHIVE_OPTIONAL`，建议留文本 |
| `2026-07-31_vct_ls_ce_coverage_4v1` | 2.7 MiB | `ARCHIVE_OPTIONAL`，建议留文本 |
| `2026-07-30_ce_energy_curriculum` | 54 MiB | `ARCHIVE_OPTIONAL` |
| `2026-07-30_ce_p3_visuals` | 19 MiB | `ARCHIVE_OPTIONAL` |
| `2026-07-30_ce_p3_deterministic_reval` | 380 KiB | `KEEP`，文本证据 |
| `2026-07-30_vct_ls_k10_warm_8v2` | 393 MiB | `ARCHIVE_OPTIONAL`，latest 已覆盖 |
| `2026-07-30_vct_ls_scratch4v1_capturefocus` | 102 MiB | `ARCHIVE_OPTIONAL` |
| `2026-07-30_vct_ls_supportblend4v1_capturefocus` | 120 MiB | `ARCHIVE_OPTIONAL` |
| `2026-07-29_ce_coverage_p2_quickscreen` | 18 MiB | `ARCHIVE_OPTIONAL`，只留 best/config也可 |
| `2026-07-29_ce_coverage_p3_followup` | 18 MiB | `ARCHIVE_OPTIONAL` |
| `2026-07-28_zonedemo_b1_adaptation` | 1.9 GiB | `ARCHIVE_OPTIONAL`，Zone 非主线 |
| `zonedemo_v0` | 363 MiB | `ARCHIVE_OPTIONAL`，Zone 非主线 |
| `2026-07-23_a3_apfnew_sqrtn_curriculum` | 245 MiB | `ARCHIVE_OPTIONAL`；有 Git 特例，勿粗删 |
| `vct_ls_smoke_visual_20260730` | 3.0 MiB | `DELETE_CANDIDATE`，正式展示已覆盖 |
| `smoke_auto_pipeline_20260802` | 1.8 MiB | `DELETE_CANDIDATE`，保留报告即可 |
| `codex_background_20260729/30` | 约 1.4 MiB | `DELETE_CANDIDATE`，收益很小 |

## 5. `runs/` 精简规则

`runs/` 共 12 GiB，其中 `checkpoints/*.pt` 约 11 GiB。不要删整个 run 目录；保留约 1 GiB 的 metrics、配置快照、selection、日志，只清 checkpoint。

硬豁免本地-only best：

```text
/home/yjq/rl/CoCap1/cocap-voradj/runs/ce13_pure8_scratch_ce8schedule_500k_20260730/checkpoints/final_step_500000.pt
/home/yjq/rl/CoCap1/cocap-voradj/runs/vctls_ce4v1_support_scratch1m_20260731_run1/checkpoints/final_step_1000000.pt
```

VCT-LS standalone 三个 selected 已复制到 `2026-07-31_vct_ls_latest_20rollout10gif`；CE-OldMix selected 已复制到其 `best_20rollout10gif`；组合 CR-MS 三个 selected 已复制且 hash 验证到 Git-tracked release。因此对应 `runs/` source 可删。`smoke_*`、`aborted_*` checkpoint 均列候选删除。

## 6. 建议确认的分批方案

### 批次 A：低风险，约 51 GiB

- inactive ladder replay：20 GiB；
- CTDE replay：6.3 GiB；
- P6 replay：9.2 GiB；
- reward replay：4.7 GiB；
- `runs/**/checkpoints/*.pt` 除上述两项本地-only best：约 11 GiB。

不删配置、报告、metrics、正式 GIF，也不触碰 active roots。代价是这些旧线不能从 replay 精确 resume，但仍能评估模型和复盘结果。

### 批次 B：失败/过渡模型，约 10 GiB

- CTDE `.pt` 2.1 GiB；P6 `.pt` 2.4 GiB；
- reward A1/A2/smoke/重复封装约 1.5 GiB；
- ladder step1/smoke/第二末端封装约 4 GiB，正式 25k 每线最多留一个 root final。

### 批次 C：非最终 rollout/GIF，约 10.5 GiB

- three-line 两个非 CR-MS 8v2 对照约 3.3 GiB；
- `2026-08-02_final_20rollout10gif` 除 support-scratch best 的五组约 6.6 GiB；
- CE14B 失败视觉约 272 MiB；VCT-LS plain 对照 352 MiB（若不需要）。

## 7. 禁止的粗删方式

- 不要全局按 `*.pt` 删除：会误删 CE13/VCT-LS 本地-only best。
- 不要删除整个 `runs/`：小型 metrics/selection/log 是结论审计证据。
- 不要删最终 rollout `_workers/`：未证明顶层含完整 seed。
- 不要在 4A/4C 运行时清理它们的旧 step/replay。
- 不要把“同大小”当作“相同文件”；只对文中明确列出的样本做了 hash。
- 不要清理任何 YAML/config snapshot。

默认推荐先批准批次 A：约 51 GiB 已足以缓解当前磁盘瓶颈，又保留全部配置、评价数字、正式 GIF 和明确 best models。
