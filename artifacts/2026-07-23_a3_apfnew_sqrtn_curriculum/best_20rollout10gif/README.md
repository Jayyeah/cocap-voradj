# A3 APFnew + CenterSqrtN Curriculum Best Rollouts

本目录保存 A3 长课程各阶段晋级后选中模型的正式 `20 rollout / 10 GIF` 结果。每个 `stage*_...` 子目录应同时包含：

- `all_summaries.json`、各场景 `batch_summary.json`、episode json 和 GIF；
- 本次 rollout 使用的 checkpoint 原文件，保持原文件名，例如 `step_900000.pt`；
- 本次训练使用的配置文件原件，保持原文件名，例如 `stage5_12p3e3obs_1400k.yaml`。

后续凡是输出到 `best_20rollout10gif` 的正式 rollout，脚本会自动备份输入 checkpoint 和 config，避免只剩 GIF 和 summary 时难以复现。

## 课程总策略

- 训练设备：`cuda:1`。
- 训练模式：`voradj_mixed_coverage`。
- 核心迭代：APF v2 修正版 `apf.version: v2_fixed`，Voronoi center 向量/奖励相关的 `center_sqrt_n_normalization_enabled: true`，coverage motion gate 为 `hard`，early speed shaping 开启。
- 筛选：stage1 从 `1M` 开始检测；stage2 及后续原计划从 `300k` 开始每 `100k` 检测；`800k` 后才允许提前晋级。若未在允许窗口内通过，则训练到上限并选择历史 quick screening 最优模型。
- 正式 rollout：每个晋级 checkpoint 跑 `capture / coverage / mix` 三个场景，各 `20 rollout / 10 GIF`。
- 通过门槛：`mix_capture >= 0.98`，`capture_success >= 0.98`，`mix_settled >= 0.85`，`coverage_settled >= 0.90`，`mix_geometric >= 0.95`，`coverage_geometric >= 0.95`，`max_collision <= 0.10`。

## 阶段配置总表

| Stage | 规模 | 训练步数 | settle window | coverage max steps | 是否 scratch | 配置文件 | run_name | warm start / 来源 |
|---|---:|---:|---:|---:|---|---|---|---|
| stage1 | `4p1e1obs` | `2.0M` | `50` | `900` | 是，真正从零训练 | `configs/experiments/voradj_a3_apfnew_sqrtn_20260723/a3_apfnew_sqrtn_4v1_scratch_mix_2m.yaml` | `voradj_a3_apfnew_sqrtn_4v1_scratch_mix_2m_20260723_run1` | 无 pretrained |
| stage2 | `6p2e2obs` | `1.2M` | `60` | `1100` | 否，课程 warm start | `configs/experiments/voradj_a3_apfnew_sqrtn_curriculum_20260723/stage2_6p2e2obs_1200k.yaml` | `voradj_a3_apfnew_sqrtn_curr_stage2_6p2e2obs_1200k_20260723_run1` | stage1 selected `step_2000000.pt` |
| stage3 | `8p2e2obs` | `1.2M` | `70` | `1200` | 否，课程 warm start | `configs/experiments/voradj_a3_apfnew_sqrtn_curriculum_20260723/stage3_8p2e2obs_1200k.yaml` | `voradj_a3_apfnew_sqrtn_curr_stage3_8p2e2obs_1200k_20260723_run1` | stage2 selected `step_1100000.pt` |
| stage4 | `10p3e3obs` | `1.4M` | `80` | `1400` | 否，课程 warm start | `configs/experiments/voradj_a3_apfnew_sqrtn_curriculum_20260723/stage4_10p3e3obs_1400k.yaml` | `voradj_a3_apfnew_sqrtn_curr_stage4_10p3e3obs_1400k_20260723_run1` | stage3 selected `step_600000.pt` |
| stage5 | `12p3e3obs` | `1.4M` | `90` | `1600` | 否，课程 warm start | `configs/experiments/voradj_a3_apfnew_sqrtn_curriculum_20260723/stage5_12p3e3obs_1400k.yaml` | `voradj_a3_apfnew_sqrtn_curr_stage5_12p3e3obs_1400k_20260723_run1` | stage4 selected `step_1000000.pt` |
| stage6 | `14p4e4obs` | `1.6M` | `100` | `1800` | 否，课程 warm start | `configs/experiments/voradj_a3_apfnew_sqrtn_curriculum_20260723/stage6_14p4e4obs_1600k.yaml` | `voradj_a3_apfnew_sqrtn_curr_stage6_14p4e4obs_1600k_20260723_run1` | stage5 selected `step_900000.pt` |
| stage7 原始线 | `16p4e4obs` | `1.6M` | `110` | `2000` | 否，课程 warm start | `configs/experiments/voradj_a3_apfnew_sqrtn_curriculum_20260723/stage7_16p4e4obs_1600k.yaml` | `voradj_a3_apfnew_sqrtn_curr_stage7_16p4e4obs_1600k_20260723_run1` | stage6 selected `step_400000.pt`，训练到约 `466k` 时服务器重启中断 |
| stage7 续训线 | `16p4e4obs` | `1.15M` remaining | `110` | `2000` | 否，断点权重续训；supervisor 中的 `scratch: true` 只是防止覆盖 resume config 的技术标记 | `configs/experiments/voradj_a3_apfnew_sqrtn_curriculum_20260723/stage7_16p4e4obs_resume_from450k_remaining1150k_20260727.yaml` | `voradj_a3_apfnew_sqrtn_curr_stage7_16p4e4obs_resume_from450k_remaining1150k_20260727_run2` | 原 stage7 `step_450000.pt`，即 stage6 -> stage7 后的中断 checkpoint |
| stage8 | `18p5e5obs` | `1.8M` | `120` | `2200` | 否，课程 warm start | `configs/experiments/voradj_a3_apfnew_sqrtn_curriculum_20260723/stage8_18p5e5obs_1800k.yaml` | `voradj_a3_apfnew_sqrtn_curr_stage8_18p5e5obs_1800k_20260723_run1` | 由 supervisor 在 stage7 晋级后写入 selected checkpoint |
| stage9 | `20p5e5obs` | `2.0M` | `130` | `2400` | 否，课程 warm start | `configs/experiments/voradj_a3_apfnew_sqrtn_curriculum_20260723/stage9_20p5e5obs_2000k.yaml` | `voradj_a3_apfnew_sqrtn_curr_stage9_20p5e5obs_2000k_20260723_run1` | 由 supervisor 在 stage8 晋级后写入 selected checkpoint |

## 已晋级并完成正式 rollout 的阶段

| Stage | 选中 checkpoint | 选择原因 | 正式 rollout 目录 | 已备份文件 |
|---|---|---|---|---|
| stage1 `4p1e1obs` | `runs/voradj_a3_apfnew_sqrtn_4v1_scratch_mix_2m_20260723_run1/checkpoints/step_2000000.pt` | 训练到上限后选择历史 quick screening 最优 | `stage1_4p1e1obs_step_2000000/` | `step_2000000.pt`, `a3_apfnew_sqrtn_4v1_scratch_mix_2m.yaml` |
| stage2 `6p2e2obs` | `runs/voradj_a3_apfnew_sqrtn_curr_stage2_6p2e2obs_1200k_20260723_run1/checkpoints/step_1100000.pt` | 训练到上限后选择历史 quick screening 最优 | `stage2_6p2e2obs_step_1100000/` | `step_1100000.pt`, `stage2_6p2e2obs_1200k.yaml` |
| stage3 `8p2e2obs` | `runs/voradj_a3_apfnew_sqrtn_curr_stage3_8p2e2obs_1200k_20260723_run1/checkpoints/step_600000.pt` | 训练到上限后选择历史 quick screening 最优 | `stage3_8p2e2obs_step_600000/` | `step_600000.pt`, `stage3_8p2e2obs_1200k.yaml` |
| stage4 `10p3e3obs` | `runs/voradj_a3_apfnew_sqrtn_curr_stage4_10p3e3obs_1400k_20260723_run1/checkpoints/step_1000000.pt` | 训练到上限后选择历史 quick screening 最优 | `stage4_10p3e3obs_step_1000000/` | `step_1000000.pt`, `stage4_10p3e3obs_1400k.yaml` |
| stage5 `12p3e3obs` | `runs/voradj_a3_apfnew_sqrtn_curr_stage5_12p3e3obs_1400k_20260723_run1/checkpoints/step_900000.pt` | 训练到上限后选择历史 quick screening 最优 | `stage5_12p3e3obs_step_900000/` | `step_900000.pt`, `stage5_12p3e3obs_1400k.yaml` |
| stage6 `14p4e4obs` | `runs/voradj_a3_apfnew_sqrtn_curr_stage6_14p4e4obs_1600k_20260723_run1/checkpoints/step_400000.pt` | 训练到上限后选择历史 quick screening 最优 | `stage6_14p4e4obs_step_400000/` | `step_400000.pt`, `stage6_14p4e4obs_1600k.yaml` |

## 当前未完成阶段

- stage7 原始线在 `2026-07-27 13:58` 左右因服务器重启中断，最新可用 checkpoint 为 `step_450000.pt`。
- stage7 续训线从 `step_450000.pt` warm-start，配置文件为 `stage7_16p4e4obs_resume_from450k_remaining1150k_20260727.yaml`，当前由 A3 supervisor 接管。
- stage8/stage9 尚未产生 handoff；实际 warm start 将以各自前一阶段最终 selected checkpoint 为准，supervisor 会在进入阶段前写入对应 config。

## 复现注意

- 每个已完成 best 目录内备份的是正式 rollout 当时使用的 config 和 checkpoint；复现该目录结果时优先使用目录内备份文件。
- stage2 及后续不是从零训练，而是课程学习：前一阶段 selected checkpoint 作为当前阶段 pretrained path。
- stage7 续训不是严格 optimizer/replay/global_step 原地 resume；当前 trainer 只支持从 checkpoint 权重 warm-start，因此该段应理解为“权重断点续训”。
- `stage8_18p5e5obs_1800k.yaml` 和 `stage9_20p5e5obs_2000k.yaml` 中的 `pretrained.path: __SET_BY_A3_CURRICULUM_SUPERVISOR__` 是占位符，真实训练开始前会被 supervisor 改为前一阶段 selected checkpoint。
