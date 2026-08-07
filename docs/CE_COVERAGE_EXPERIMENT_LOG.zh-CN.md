# CE-Coverage 实验日志

## 边界

- CE-Coverage 使用旧 `[0,120] x [0,120]` mix 地图语义，首轮仅训练 4-agent pure coverage。
- `zone_demo.enabled=false`。本线不修改 ZoneDemo B1 的任务、APF、失守定义或训练产物。
- 两个开关均默认关闭：`coverage_objective_version=legacy`、`voronoi_obstacle_mode=legacy_assign`。
- 完整理论方案保存在本地核心文档 `TERL/docs/CE_COVERAGE_CENTROID_ENERGY_TODO_20260729_ZH.md`，暂不纳入公开仓库。

## 2026-07-29：CE-P0/P1

实现：

- 障碍膨胀区从 Voronoi 自由空间中剔除，blocked owner 固定为 `-1`。
- cell 中心使用机器人所在 owner 连通分量的算术质心；若质心落入障碍或不在该分量，则投影到最近可达栅格。
- CE 中心代价为 `d_i^2`，其中 `d_i=sqrt(N)*||p_i-c_i||/map_diagonal`。
- 控制代价包含归一化速度、加速度和角速度平方项。
- PBRS 为 Potential-Based Reward Shaping（基于势函数的奖励塑形），`Phi=-kappa*d_i^2`，终止状态 `Phi=0`。
- 成功只要求 `E_rms<=0.05`、`E_max<=0.10` 连续 30 步。

验证：

```bash
PYTHONPATH=src pytest -q tests/test_ce_coverage.py
```

结果：`5 passed`。另已完成 CE2、旧 A3、ZoneDemo B1 的 reset/step 回归，后两者仍走 legacy 逻辑。

已知工具问题：

- 本机 Codex 的文件沙箱偶发 `bwrap: loopback: Failed RTM_NEWADDR`，导致内置 `apply_patch` 无法读取文件。
- 发生时改用标准 `patch` 或精确原子替换；已删除工具产生的 `.orig` 临时文件，并用 `git diff --check` 检查。

## CE-P2 配置

目录：`configs/experiments/ce_coverage_20260729/`

| 线 | 速度权重 | PBRS | 目的 |
| --- | ---: | --- | --- |
| CE0 | 0.00 | 开 | 识别无能耗约束时的高速穿越与振荡 |
| CE1 | 0.05 | 关 | 与 CE2 构成严格 PBRS 对照 |
| CE2 | 0.05 | 开 | 首选平衡候选 |
| CE3 | 0.10 | 开 | 检查较强速度代价是否诱发保守静止 |

四线均为 scratch、同 seed、300k、GPU1、4-agent pure coverage。100k/200k/300k 后续使用固定评估种子，不按训练累计 reward 直接选点。

## 2026-07-29：训练闭环与 CE-P2 启动

- CE2 1k/5k scratch smoke 均完成；5k 已跨过 replay warm-up，`loss_ema` 约从 `3.78` 降到 `2.00`，并生成 init/latest/final checkpoints、完整 effective config、metrics 和 episodes 日志。
- 新增 CE reward 的 episode 与 replay batch 分量统计：center、control、PBRS、terminal correction。
- 新增 `tools/evaluate_ce_coverage.py`：固定 seed 的 pure-random rollout，记录成功率、最后有效/全程最小中心误差、累计中心/控制代价、碰撞、收敛步数和成功后 30 步自然动作。
- 新增 `tools/supervise_ce_coverage.py`：自动等待 CE0-CE3 的 100k/200k/300k checkpoints，每点顺序跑 20 rollout，避免并发写路径冲突。
- CE0-CE3 已在 GPU1 并行启动；自动评估 watcher 已启动。评估产物写入 `artifacts/2026-07-29_ce_coverage_p2_quickscreen/`。

## CE-P2：100k 固定评估

统一为 20 rollout、seed `2026072900..2026072919`、max 900 step、成功后额外观察 30 step。

| 线 | 成功 | 碰撞 | 最小 RMS | 最小 max-error | 末态均速 | 解释 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CE0 | 0.00 | 0.70 | 0.0656 | 0.0805 | 1.5595 | 能接近中心，但高速穿越/碰撞，未保持 30 步 |
| CE1 | 0.00 | 0.00 | 0.1803 | 0.2425 | 0.0026 | 无 PBRS 时退化为原地静止 |
| CE2 | 0.00 | 0.55 | 0.1527 | 0.1898 | 0.3799 | 尚未形成可靠接近或低速平衡 |
| CE3 | 0.00 | 0.00 | 0.1749 | 0.2403 | 0.0000 | 强速度代价诱发保守静止 |

100k 后训练窗口出现跃迁：约 150k 时 CE0 最近训练成功率约 0.39、碰撞约 0.09；CE1/CE2/CE3 仍约 0.01/0.02/0.01。继续保留到 200k 固定评估，不依据训练窗口提前选线。

## CE-P2：200k 固定评估

| 线 | 成功 | 碰撞 | 最小 RMS | 末态均速 | 成功步数 | 成功后均速 | 成功后 idle action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CE0 | 0.55 | 0.05 | 0.0509 | 0.0220 | 196.5 | 0.0567 | 0.9692 |
| CE1 | 0.00 | 0.00 | 0.1756 | 0.0024 | - | - | - |
| CE2 | 0.00 | 0.00 | 0.1364 | 0.2003 | - | - | - |
| CE3 | 0.00 | 0.05 | 0.1587 | 0.0081 | - | - | - |

CE0 是唯一越过固定评估成功门槛的候选，并且成功后 30 步主要输出 `(a=0,w=0)`，说明 hold + PBRS 已产生自然静止，不依赖旧速度 shaping 或 terminal 大奖。CE1/CE3 的主要失败是过早静止，CE2 尚未形成稳定中心接近。继续训练到 300k，主要确认 CE0 稳定性与 CE3 是否晚发改善。

## CE-P2：最终选点

CE0 补评：150k 成功 0%、250k 成功 25%、300k 成功 0%。200k 的 55% 成功和成功后 96.9% idle action 均为历史最佳。CE1 300k 虽达到 35% center-hold，但成功后均速 0.798、idle action 0%，不满足低能耗自然静止目标。CE2/CE3 300k 均为 0% 成功并趋向过早静止。

P2 best 已精简归档到 `artifacts/2026-07-29_ce_coverage_p2_quickscreen/best_ce0_step200000/`，仅复制 `step_200000.pt`、完整配置和说明。

## CE-P3：follow-up

已启动四条 300k scratch：

| 线 | 改动 | 目的 |
| --- | --- | --- |
| CE4 | CE0，seed 2026071544 | 多种子复现 |
| CE5 | CE0，seed 2026071545 | 多种子复现 |
| CE6 | CE0 原 seed，200k 后 lr 1e-4 -> 3e-5 | 抑制后期策略漂移 |
| CE7 | PBRS，lambda_v=0.002 | 测试极轻全局速度能耗 |

watcher 在 150k/200k/250k/300k 各跑固定 10 rollout；每条历史最佳再做 20 rollout 确认。产物目录：`artifacts/2026-07-29_ce_coverage_p3_followup/`。

## 2026-07-30：CE-P3 完成与确定性复评

CE4--CE7 均由原训练进程正常完成 300k，未发生异常恢复，ZoneDemo 代码和
产物未被修改。训练窗口成功率不参与选点。

### IQN 评估确定性修正

昨夜首轮评估虽然设置了 `epsilon=0` 和固定环境 seed，但 `CoCapIQN.act()`
仍为每次动作随机采样 32 个 IQN 分位点。因此同一 checkpoint、同一环境 seed
仍可能选择不同动作；昨夜 `artifacts/2026-07-29_ce_coverage_p3_followup/`
中的表格只能作为 random-tau 历史记录，不能作为最终可复现实验数字。

现已为评估增加 `deterministic_quantiles=true`：使用
`tau_j=(j+0.5)/32, j=0..31` 的固定分位中点近似期望 Q。训练仍保留随机分位
采样，算法训练语义不变。重复运行同一点、同一 seed 已得到逐字段一致结果；
以下结果统一标记为 `policy_quantile_mode=fixed_midpoint_32`。

### 确定性 10-rollout 全里程碑

单元格为 `成功率 / 碰撞率`：

| 线 | 150k | 200k | 250k | 300k |
| --- | ---: | ---: | ---: | ---: |
| CE4 | 0.00 / 0.00 | 0.00 / 0.00 | 0.70 / 0.00 | 0.00 / 0.00 |
| CE5 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| CE6 | 0.00 / 0.00 | 0.10 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| CE7 | 0.00 / 0.00 | 0.00 / 0.00 | 0.10 / 0.00 | 0.30 / 0.10 |

### 确定性 20-rollout 重点确认

| 点 | 成功 | 碰撞 | RMS / max | 成功步数 | 成功后均速 | idle action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CE4-250k | 0.75 | 0.00 | 0.0508 / 0.0718 | 359.9 | 0.8362 | 0.0000 |
| CE5-300k | 0.00 | 0.00 | 0.0728 / 0.0930 | - | - | - |
| CE6-200k | 0.05 | 0.00 | 0.0727 / 0.0972 | 194.0 | 0.0022 | 1.0000 |
| CE6-250k | 1.00 | 0.00 | 0.0362 / 0.0498 | 83.5 | 0.1580 | 0.0300 |
| CE6-300k | 1.00 | 0.00 | 0.0282 / 0.0388 | 78.8 | 0.3878 | 0.6638 |
| CE7-300k | 0.45 | 0.05 | 0.0562 / 0.0803 | 104.2 | 0.1333 | 0.9333 |

CE6-250k/300k 的 `100%` 中心成功、零碰撞在确定性复评中保留，说明
“中心平方误差 + PBRS + 30-step hold”确实能在没有旧 speed shaping、settle
terminal 大奖和速度门禁时学出可靠中心收敛。CE4 只能复现中等成功且高速；
CE5 退化为未到中心就静止；CE7 从训练开始加入 `lambda_v=0.002` 后只有
45% 成功并出现 5% 碰撞，不支持继续使用全程固定能耗惩罚。

这还不是长期静止解。成功后再额外观察 30 步时，CE6-250k 的逐步几何保持率
为 `89.5%`、最后一步仍保持阈值的 rollout 比例为 `85%`；CE6-300k 分别为
`91.2%/90%`。CE6-250k 某些 seed 在完成原 30-step hold 后继续漂出阈值，
因此当前 success 应解释为“曾经连续保持”，不能解释为永久稳态。

评估器现已增加与奖励权重无关的客观累计量：
`sum(v/v_max)^2`、`sum(a/a_max)^2`、`sum(omega/omega_max)^2`，以及 active
agent-step 数。CE6-250k 的四项均值为 `35.49 / 301.30 / 288.65 / 333.8`；
CE6-300k 为 `43.97 / 127.90 / 171.35 / 315.0`。250k 的累计速度暴露较低，
但加减速和转向动作明显更多；idle action 也不能单独代表静止，因为 300k
常在已有惯性速度时输出 idle。

### 可视化与归档

确定性 GIF 位于
`artifacts/2026-07-30_ce_p3_visuals/deterministic/`：包含 CE6-200k 的稀有
自然静止、CE6-250k 的低运动/典型/高运动三个样本，以及 CE6-300k 的精度端点。
图中显示可达 cell 中心 `C0..C3` 与实时 `E_rms/E_max`，不绘制轨迹线和邻接线。

CE6-250k 继续作为 P3 平衡历史最优归档，但不是可直接接回旧 mix 的最终解。
复评 JSON 位于 `artifacts/2026-07-30_ce_p3_deterministic_reval/`。

### 下一阶段 TODO

1. 实现按全局训练步切换的 CE 能耗系数；不能把只加载模型权重的 warm start
   描述成等价续训。CE8/CE9 均应从 scratch 完整训练并保留 replay/optimizer
   的真实连续性。
2. CE8 在 0--200k 使用 `lambda_v=0`，随后使用 `0.0005`；CE9 同期切换到
   `0.001`。沿用 CE6 在 200k 将学习率 `1e-4 -> 3e-5` 的设置，其余完全一致。
3. 在 200k/225k/250k/275k/300k 使用固定分位中点做 20 rollout。晋级先要求
   成功率 `>=0.90`、碰撞率 `<=0.05`、成功后最终几何保持率 `>=0.95`；通过者
   再按客观累计能耗和收敛步数做 Pareto 比较，不把任意速度阈值写进成功判定。
4. 若两条线仍过早静止，下一次只降低/推迟能耗系数，不恢复旧速度 shaping、
   CV、center-ok、settle 窗口或 terminal 大奖。pure coverage 通过后再接回非
   zone 的旧 mix；ZoneDemo 继续作为互不干扰的并行实验。

## 2026-07-30：CE8/CE9 启动与 8v2 接口准备

### CE8/CE9 真实 schedule

训练器现支持按 `global_step` 切换 `coverage_ce_speed_weight`，并将当前权重写入
metrics、checkpoint extra 和 replay metadata。边界语义固定为：transition
`0..199999` 使用 `lambda_v=0`，因此 `step_200000.pt` 仍是纯中心目标端点；
下一次 action transition 才启用 CE8 的 `0.0005` 或 CE9 的 `0.001`。两线同时
在 200k 将学习率从 `1e-4` 降为 `3e-5`，加速度和角速度权重始终为零。

CE8/CE9 显式保持 `coverage_ce_pbrs_reset_mode=success_only_legacy`，以便与
CE6/CE7 严格比较，并保证服务器重启后不因后续 mix 修正改变训练语义。两线均为
scratch、同 seed、300k，每 25k 保存 checkpoint，不提前结束。

自动评估由 `tools/supervise_ce_energy_curriculum.py` 执行：200k/225k/250k/
275k/300k 各跑 20 rollout，固定 seed `2026072900..2026072919`、固定 32 个 IQN
分位中点、max 900 step、成功后额外观察 30 step。硬门槛为成功率 `>=0.90`、
碰撞率 `<=0.05`、成功后最终几何保持率 `>=0.95`；通过者才比较客观累计速度
平方、成功步数和成功后速度。

启动前通过 13 项测试和 CE8/CE9 各自 1k/5k smoke；5k 的 loss 从约 `6.75`
降到 `1.68`。正式 tmux 为 `ce8_energy_gpu1_0730`、`ce9_energy_gpu1_0730`、
`ce89_eval_gpu1_0730`，评估产物目录为
`artifacts/2026-07-30_ce_energy_curriculum/`。2026-07-30 15:45 两线均为
`43k/300k`，200k 前逐字段一致；训练窗口成功约 `3.8%`、碰撞约 `38.5%`，只作
早期诊断，不用于选点。

### 接回旧 mix 前的硬修正与统一地图决定

环境仍支持在同一状态缓存两张地图，便于历史兼容和独立消融。但 2026-07-30
最新实验决定是：CE10--CE12 不再保留旧围捕图，围捕阶段与 coverage 阶段统一
使用 `free_mask_projected`：

- 障碍物及机器人半径膨胀区域不归属于任何 cell。
- capture 邻接、角色、敌人 token、围捕奖励、CE 质心 observation、center cost、
  PBRS 和成功判定均读取障碍剔除后的可达自由空间图。
- `capture_adjacency_obstacle_mode` 与 `voronoi_obstacle_mode` 均固定为
  `free_mask_projected`。

因此 CE11 的 A3 checkpoint 只提供网络权重初始化。A3 原本在
`legacy_assign` 围捕图上训练，加载后立即面对新的 observation 和邻接语义，
不能把历史 capture `1.00` 当作新配置的训练前性能。正式训练前必须重新跑
zone 之外的 mix、capture/围捕诊断和 pure coverage 基线，记录地图切换本身造成
的性能变化。

最后一个 evader 被捕的 action transition 仍标记为 `pre_capture`；真正下一次
action 才是 `post_capture step=1`。全体 evader 被捕后立即清空所有 pursuer 的
K10 release counter。新版 mix 使用
`coverage_ce_pbrs_reset_mode=phase_and_all_terminal`：在 capture->coverage 边界
结算旧 phase potential，并在 timeout、碰撞、成功等所有真正终止处将 terminal
potential 置零。旧实验默认仍为 `success_only_legacy`。

新增配置目录 `configs/experiments/ce_coverage_20260730_scale_mix/`：CE10 为
8-agent pure scratch；CE11 从 A3 stage3 8v2 `step_600000.pt` warm start；CE12
为同配置 8v2 scratch pilot。A3 历史基线为 capture `1.00`、pure coverage
settled `0.90`、mix settled `0.70`、collision `0.00`，仅描述旧地图语义。
统一地图后，12 项配置与地图合同测试通过。CE11 新版 500-step smoke 的 effective
config 确认两项模式均为 `free_mask_projected`；A3 checkpoint 正常加载，首个
mix episode 在 383 步捕获全部 2 个 evader、零碰撞，并继续进入 post-capture。
这只证明新图训练闭环正常，不能替代正式训练前的独立 20-rollout 统计基线。
CE10--CE12 尚未启动正式长训，待 CE8/CE9 选出能耗系数后再决定顺序。
`git diff --check` 通过；ZoneDemo 的配置、行为与后台任务未被这些 CE 开关修改。

## 2026-07-30：CE13/CE14 并行线启动

CE8/CE9 的 pure coverage 信号已基本满足当前要求，因此先开两条互补短线：

- CE13：8 个 pursuer 的 pure coverage scratch scale test，配置为 `configs/experiments/ce_coverage_20260730_scale_mix/ce13_pure8_scratch_ce8schedule_500k.yaml`。沿用 CE8 的 delayed control-energy schedule：0--200k 只优化 cell-center 收敛，200k 后 `lambda_v=0.0005`。无旧 CV/center-ok terminal bonus、无 speed-settle shaping、无 coverage 大额终奖。目标是判断新 coverage 在 8-agent 规模是否仍可成功或接近成功。
- CE14：4v1 old mix scratch，配置为 `configs/experiments/ce_coverage_20260730_scale_mix/ce14_mix4v1_a3capture_cecoverage_scratch_500k.yaml`。capture 部分保持 A3 旧 mix 语义：旧 capture Voronoi 图、APF-v2_fixed、K10 release delay、hard boundary/death 和 safety penalties；coverage/post-capture/pure coverage 改为 CE coverage。显式 `vct_ls.enabled=false`，不使用 VCT-LS 新 capture。

CE14 的 coverage 奖励只保留 CE center cost + PBRS + delayed control-energy，不保留 A3 的 early speed shaping、settle 窗口、speed-only 成功奖励、geometric latch reward 或 terminal repeat reward。mix post 结束逻辑为：capture 后若 CE coverage 成功则停止，否则直到 `post_capture_coverage_window_steps=300`。

可视化规则同步修正：只有 new pure coverage rollout 绘制 cell center；mix 无论 pre-capture 还是 post-capture 都不绘制 cell center。`tools/rollout_voradj_visual.py` 和 `tools/batch_rollouts.py` 已传入 `draw_ce_targets=(scenario == "coverage")`。

两条线均放在 GPU0：`ce13_pure8_gpu0_0730` 与 `ce14_mix4_a3cap_cecov_gpu0_0730`。启动前通过 `tests/test_ce_scale_mix_configs.py` 与 `tests/test_ce_coverage.py` 共 14 项测试，CE13/CE14 各 500-step smoke 均能完成训练闭环。

若 CE13 和 CE14 均出现正信号，下一步开 8v2 旧 mix 课程验证：优先 4v1->8v2；若迁移不稳，再改为 4v1->6v2->8v2。课程 capture 仍保持 A3 旧 mix，coverage 仍保持 CE coverage。

## 2026-07-30：CE14B old-mix 接回重开

CE14 首次训练中途停在 302k，没有 final checkpoint；由于当前 trainer 不支持严格恢复 optimizer/replay/global_step，后续不把 `step_300000.pt` 伪装成连续续训。CE14 首次结果的关键诊断是：pure CE coverage 已经能学会，但 old-mix 接回弱。最后 100 个 `voradj_coverage` episode 中 pure success 约 0.86、collision 约 0.02；最后 100 个 `voradj` mix episode 中 capture 约 0.68、post CE coverage 约 0.14、post window timeout 约 0.54、collision 约 0.29。

因此开启 CE14B scratch，不改 CE 奖励主体，只调整训练分布与 post 窗口：

- `post_capture_coverage_window_steps: 300 -> 500`，给 capture 后从围捕构型扩散到 CE cell center 更多时间。
- replay batch 从 `64/16/16/32` 改为 `64/16/32/16`，即提高真实 `post_capture_real`，降低已经学得较好的独立 `recovery_pure`。
- recovery 初始化中 captured snapshot 比例从 `0.50 -> 0.75`，让 pure coverage 辅助样本也更接近真实 capture 后状态。
- 总步长设为 700k；仍为 scratch，capture 仍为 A3 old mix / `legacy_assign`，coverage 仍为 CE / `free_mask_projected`，不启用 VCT-LS，不恢复旧 speed shaping 或 coverage terminal 奖励。

配置：`configs/experiments/ce_coverage_20260730_scale_mix/ce14b_mix4v1_a3capture_cecoverage_postreal500_scratch_700k.yaml`。tmux：`ce14b_mix4_postreal_gpu0_0730`。


## 2026-07-31：CE13/CE14B final 20-rollout 检验

本次使用训练完成后的 final checkpoint 做独立固定分位评估：

- CE13 checkpoint：`runs/ce13_pure8_scratch_ce8schedule_500k_20260730/checkpoints/final_step_500000.pt`
- CE14B checkpoint：`runs/ce14b_mix4v1_a3capture_cecoverage_postreal500_scratch_700k_20260730/checkpoints/final_step_700000.pt`
- 评估产物：`artifacts/2026-07-31_ce_final_20rollout10gif/`
- policy 推理统一使用 `fixed_midpoint_32`，即 32 个固定 IQN 分位中点；不使用 rollout 随机 tau。

### CE13：8-agent pure coverage 通过

CE13 在 8 pursuer pure coverage 上完成 20 rollout / 10 GIF：

| 指标 | 数值 |
| --- | ---: |
| coverage success | `1.00` |
| geometric / settled | `1.00 / 1.00` |
| collision / soft OOB / timeout | `0.00 / 0.00 / 0.00` |
| 平均步数 | `65.85` |
| 末态 mean / max speed | `0.2866 / 0.9022` |
| 初始化分布 | map random `10`，inner random cluster `10` |

结论：CE coverage 在独立 pure coverage 规模测试上非常稳定。当前证据足以把
CE13 视为“pure coverage 新默认候选”：障碍剔除 Voronoi、可达 cell centroid、
CenterSqrtN、中心平方误差、PBRS、delayed `lambda_v=0.0005` 与 30-step center
hold 的组合，在 8-agent pure coverage 上不需要旧 CV、speed shaping 或 terminal
大额奖励即可完成收敛。

### CE14B：old mix 接回未通过

CE14B 训练末段窗口很乐观：final 前后日志中 old-mix capture 保持 `1.00`，recent
post CE coverage 约 `0.88`，pure coverage 可到 `1.00`。但 independent final
rollout 的 mix fast stats 明显下滑：

| 指标 | 数值 |
| --- | ---: |
| capture success | `1.00` |
| CE coverage success | `0.30` |
| episode success | `0.30` |
| geometric / settled | `0.30 / 0.30` |
| collision / soft OOB | `0.00 / 0.00` |
| 平均步数 | `496.4` |
| 末态 mean / max speed | `0.0 / 0.0` |
| expanded then stopped | `0.75` |

CE14B 的 coverage-only final fast stats 也未通过：20 rollout 中 coverage success 仅 `0.45`，平均 `704.6` 步，末态 mean/max speed 为 `0.007/0.028`，`expanded_then_stopped=0.75`。因此 CE14B final 不是“mix 接回差但 pure coverage 仍好”，而是 final 策略在 4-agent CE coverage 上也出现展开后静止、中心 hold 不稳定的漂移。<!-- CE14B coverage-only final stats -->

失败样本的典型形态是：capture 已成功、无碰撞，post-capture 阶段先扩散，随后
速度降到 0，但没有连续满足 CE center hold，最终 post window timeout。这说明
CE14B 的问题不是 old capture 被破坏，也不是 CE pure coverage 本体不能学，而是
old capture 构型接到 CE coverage 的状态分布仍未稳定覆盖。训练窗口中的 recent
success 对独立固定 seed rollout 有偏乐观，不能作为晋级依据。

因此当前不把 CE coverage 设为 old-mix 全局默认，也不启动 8v2 old-mix curriculum。
CE 可以先作为 pure coverage 分支的新默认候选；old mix 接回需要下一轮 CE14C
诊断后再决定。

### 相对旧 coverage 的核心改动

旧 coverage 同时使用 area CV、center-ok、inside ratio、几何 latch、speed-only
settle、速度 shaping、settled repeat 和 terminal 大奖；奖励与判定由多个目标拼接，
容易出现“几何成功但速度不收敛”或“先刹停导致几何丢失”的分叉。

CE coverage 将目标压缩为一个主问题：每个 pursuer 到达自身障碍剔除 Voronoi cell
的可达质心，并尽量少运动。成功判定只看：

```text
E_rms <= 0.05
E_max <= 0.10
连续 hold 30 步
```

奖励只保留：

```text
中心平方误差: -C_i
Potential-Based Reward Shaping: gamma * Phi(s') - Phi(s)
延迟开启的轻量速度能耗: -lambda_v * (v_i / v_max)^2
```

其中 PBRS 是 Potential-Based Reward Shaping（基于势函数的奖励塑形），只用于把
“中心误差下降”变成更及时的 dense 信号，不新增新的成功目标。CE 成功本身不包含
固定速度阈值；速度、累计能耗和成功后自然动作仅作为诊断指标。

### 下一步建议

CE14C 不应恢复旧 speed shaping、CV 或 terminal 大奖；应专门解决 post-capture
接回分布：

1. 提高真实 post-capture 样本占比，或单独增加 captured snapshot -> CE coverage
   的过渡课程。
2. 增加 post-capture window 到 `700--800` 做诊断，但不要把更长 timeout 误认为成功。
3. 针对失败 GIF 统计 final `E_rms/E_max`、每个 agent 到 CE center 的距离、是否有
   agent 停在错误 cell 中心附近。
4. 若仍出现“展开后停住但未满足中心”，优先调整 PBRS/center cost 的尺度或
   captured snapshot 分布，而不是重新加入速度终局奖励。

## 2026-07-31：VCT-LS + CE Coverage 融合线

根据 VCT-LS 结果，局部敌人感知和一阶 Voronoi 友邻通信已经能让 warm-start 围捕闭环成立；根据 CE13 结果，CE-Coverage 在 pure coverage 中是当前最干净、最稳定的 coverage 候选。为检查二者能否互补，开启 VCT-LS + CE-Coverage 四线实验，仍与 ZoneDemo 和 CE standalone 并行互不干扰。

公共设置：

```text
common: configs/experiments/vct_ls_ce_20260731/vctls_ce_4v1_common.yaml
capture 主体: VCT-LS/A3 4v1 old mix，K10 release delay 显式开启
coverage 主体: centroid_energy_v0
CE 成功: E_rms <= 0.05, E_max <= 0.10, hold 30 steps
CV loose: rollout 同步统计 area CV < 0.15
batch: pursuing 64 / pre_capture_cover 16 / post_capture_real 16 / recovery_pure 32
recovery_pure: 50% captured snapshot + 50% ordinary map_random，后者等同 mix 我方初始化；本组不再混 synthetic_cluster
步数限制: post_capture_coverage_window_steps=300, env episode_max_length=900, screening coverage_max_steps=500
```

四条线分别为：plain warm-start 500k、support warm-start 500k、plain scratch 1M、support scratch 1M。support 版本沿用 VCT-LS support reward blend：当自己未直接 pursuing、但一阶友邻 pursuing 时，dense task reward 为 `0.5 * capture + 0.5 * CE coverage`；自己直接 pursuing 或友邻不再 pursuing 后回到原任务奖励。

scratch 两条线每 100k 做 capture/mix/coverage 轻量 rollout 检测，输出：

```text
artifacts/2026-07-31_vct_ls_ce_coverage_4v1/screening/
```

该组的判断标准不是单独证明 CE pure coverage，而是看 CE 是否能缓解 VCT-LS scratch 初始无敌人可见时 coverage 扩散弱、post-capture 接回不稳的问题，同时不破坏局部感知围捕。

## 2026-07-31 补充：CV<0.15 loose 复核与 old-mix 课程重启

CE14B 的 strict CE 成功率确实不够高，但重新按 `final Voronoi area CV < 0.15` 统计后，均布质量明显强于旧 coverage 判定时代的常见结果。补跑 20 episode 无 GIF 复核，产物位于：

```text
artifacts/2026-07-31_ce_final_20rollout10gif/ce14b_final_cv015_20stat/
```

复核结果：

| 场景 | capture | strict CE success | CV<0.15 loose | best CV<0.15 | final CV | best CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mix | `1.00` | `0.30` | `1.00` | `1.00` | `0.0800` | `0.0773` |
| pure coverage | - | `0.45` | `1.00` | `1.00` | `0.0774` | `0.0740` |

因此修正判断为：CE14B 不能被视为 strict CE coverage 完全通过，但它已经证明 old-mix 接回后能形成很好的面积均布。`CV<0.15` 后续作为 loose 诊断/晋级指标同步统计，但不进入奖励函数，也不替代 CE 的中心误差 strict 标准。

已新增训练与 rollout 兼容改动：

- `tools/batch_rollouts.py`：按 episode frame 统计 `coverage_cv015_rate`、`coverage_cv015_best_rate`、`avg_final_voronoi_cv`、`avg_best_voronoi_cv`。
- `src/cocap_voradj/envs/voronoi_adjacency.py`：训练 episode record 增加 `coverage_cv015_success` / `coverage_cv_loose_success`，默认阈值 `coverage_cv_loose_area_cv_threshold=0.15`。
- `src/cocap_voradj/training/trainer.py`：recent diagnostics 增加 `recent_coverage_cv015_rate` 与 `recent_post_capture_cv015_rate`。

已启动验证性 old-mix + CE-coverage 快速课程：

```text
configs/experiments/ce_oldmix_cv015_curriculum_20260731/
tools/supervise_ce_oldmix_cv015_curriculum.py
tmux: ce_oldmix_cv015_curr_gpu0_0731
artifact: artifacts/2026-07-31_ce_oldmix_cv015_curriculum/
```

课程为 `4v1 -> 8v2 -> 12v3`，每阶段上限 `700k`，`300k` 起每 `100k` screening，`500k` 后允许提前晋级。硬晋级看 capture/mix capture、mix/pure coverage 的 `CV<0.15` 和碰撞率；strict CE success、final CV 和步数作为优选排序项。每个选中 checkpoint 会同步启动 `capture/coverage/mix` 三场景 20rollout10gif，并在 best rollout 目录中备份对应 checkpoint 与训练配置。

## 2026-07-31：CE-OldMix-v1 默认配置定义

后续若未特别说明，`CE-OldMix-v1` 指“旧 A3 mix capture + 新 CE coverage”的默认 solid 配置。它不改变旧 mix 的 capture 主体，只替换 coverage/post-capture/pure coverage 目标。

默认语义如下：

- capture：沿用 A3 old-mix capture，包含 APF-v2_fixed、K10 capture-to-coverage release delay、hard boundary/death、collision/boundary/emergency safety penalties、CenterSqrtN；capture Voronoi 仍用 `legacy_assign`。
- coverage：使用 `centroid_energy_v0`，Voronoi 图使用 `free_mask_projected`，障碍物占据区域不归属任何 cell，cell target 为可达连通分量质心或投影点。
- coverage reward：`10 * (-after_center_cost - control_cost + PBRS)`；center cost 为 `sqrt(N) * ||p_i-c_i|| / map_diagonal` 的平方；PBRS 为 Potential-Based Reward Shaping，`kappa=1.0`。
- control energy：默认只启用速度平方能耗；`lambda_v=0` 到 200k，200k 后 `lambda_v=0.0005`；加速度和角速度能耗权重均为 0。
- 旧 coverage 奖励关闭：无旧 CV/center terminal bonus、无 geometric latch、无 speed-settle、无 early speed shaping、无 terminal repeat reward。
- CE strict 成功：`E_rms <= 0.05` 且 `E_max <= 0.10`，连续 hold 30 步；不以速度作为成功条件。
- loose 诊断：`CV<0.15` 只用于 screening/selection 诊断，不进入 reward，也不替代 strict CE。
- mix replay batch：batch size 128，`pursuing=64`、`pre_capture_cover=16`、`post_capture_real=32`、`recovery_pure=16`，比例为 `50% / 12.5% / 25% / 12.5%`。
- pure/recovery 初始化：capture snapshot pool 有样本后，`75%` 来自真实 capture snapshot；其余 `25%` 中 map random 与 synthetic cluster 各半，即整体约 `75% / 12.5% / 12.5%`。
- 训练场景调度：`voradj` 与 `voradj_coverage` alternating 1:1；训练更新由四 replay buffer 按上面的 batch 比例采样。
- stage1 scratch optimizer：lr `1e-4 -> 3e-5 @200k`，epsilon `0.6 -> 0.05 @500k`。
- warm curriculum optimizer：lr `3e-5` from start，epsilon `0.28 -> 0.05 @420k`。
- 训练环境 episode 上限仍为 `episode_max_length=3000`；mix post-capture 实际窗口由 `post_capture_coverage_window_steps` 控制，当前 4v1/8v2/12v3 分别为 `500/600/700`。
- screening/formal rollout 截断：capture `1000` steps；coverage 4v1/8v2/12v3 分别 `1200/1500/1800` steps；mix 总 `max_steps` 分别为 `2200/2500/2800`。

当前配置文件：

```text
configs/experiments/ce_oldmix_cv015_curriculum_20260731/stage1_4p1e1obs_700k.yaml
configs/experiments/ce_oldmix_cv015_curriculum_20260731/stage2_8p2e2obs_700k.yaml
configs/experiments/ce_oldmix_cv015_curriculum_20260731/stage3_12p3e3obs_700k.yaml
```

当前 supervisor：

```text
tools/supervise_ce_oldmix_cv015_curriculum.py
```


## 2026-08-02：CE-OldMix 课程完成与 4v1 融合线 CE 侧归档（deepseek 执行）

### 课程自动晋级与完成

| 阶段 | 训练 | 选中 checkpoint | 时间 |
| --- | --- | --- | --- |
| stage1 4p1e1obs | 700k 训满 | step_600000 | 08-01 00:57 晋级 stage2 |
| stage2 8p2e2obs | 700k 训满 | step_700000 | 08-01 07:39 晋级 stage3 |
| stage3 12p3e3obs | 700k 训满 | step_600000 | 08-01 16:39 curriculum_complete |

stage2/3 配置已写入 selected_from_previous_stage，warm 起点分别为 stage1-600k、stage2-700k。

### 课程正式 20-rollout（supervisor 自动执行）

| 点 | capture | mix capture | mix CE strict | coverage CE strict | CV<0.15 | 碰撞 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stage2-700k | 0.95 | 0.95 | 0.95 | 1.00 | 0.75-0.80 | 0.05 |
| stage3-600k | 0.90 | 0.90 | 0.90 | 1.00 | 0.75 | 0.10 |

结论：CE-OldMix-v1 三规模课程闭环成立，stage3-600k 为最终选中模型。

### 4v1 融合线正式评估中的 CE 侧（deepseek 执行）

| 线 | mix CE strict | coverage CE strict | mix CV<0.15 | coverage CV<0.15 |
| --- | ---: | ---: | ---: | ---: |
| CR-MS warm | 0.00 | 0.00 | 0.00 | 0.05 |
| CR-MS scratch | 0.00 | 0.00 | 0.00 | 0.80 |
| VCT-LS plain warm | 0.00 | 0.00 | 0.00 | 0.10 |
| VCT-LS support warm | 0.00 | 0.00 | 0.00 | 0.00 |
| VCT-LS plain scratch | 0.00 | 1.00 | 0.00 | 0.95 |
| VCT-LS support scratch | 0.35 | 0.45 | 0.90 | 1.00 |

结论：CE strict 在 4v1 mix 接回后仍是最短板（最高 0.35-0.45）；CV<0.15 loose 在能围捕的线上可达 0.90-1.00；纯 coverage 分支（VCT-LS plain scratch）strict 达 1.00，与 CE13 的 pure coverage 稳定性一致。课程模型（8v2/12v3）的 CE strict 明显强于 4v1 融合线，说明规模课程 + warm curriculum 是 CE 接回更有效的路径。

## 未来方向

1. 课程模型 stage3-600k 作为 CE-OldMix-v1 默认候选。
2. 下一步把课程 4v1 起点替换为 VCT-LS support scratch 1M，验证 support blend + CE 在更大规模的叠加。
3. CE strict 在 mix 中的 0.35-0.45 仍是主要短板，按 CE14C 方向（提高真实 post-capture 样本占比、captured snapshot 占比、post window 诊断）继续修，不恢复旧 speed shaping/terminal 大奖。
