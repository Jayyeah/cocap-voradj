# ZoneDemo-v0：内外区入侵围捕演示

ZoneDemo-v0 是 A3/APFnew/CenterSqrtN 主线之后的场景 demo。它暂时不改训练奖励主线，只增加地图语义、外区敌人初始化、敌人弱目标行为和评估指标，用来观察既有 checkpoint 对“外区进入内区”的泛化能力。

## 地图语义

- 内区：`[0, 120] x [0, 120]`。这是原 A3 训练地图，也仍是 pursuer 的硬边界、Voronoi 分割区域和 coverage 几何判定区域。
- 外区：`[-20, 140] x [-20, 140]` 中扣除内区后的 20m 环带。ZoneDemo-v0 的 evader 从外区随机初始化。
- Voronoi：只在内区网格上计算。外区 evader 不作为内区 Voronoi 站点，避免它把 coverage 几何和 cell centroid 拉到错误位置。

## 观测与角色切换

- Pursuer 初始化、障碍物初始化和内区硬边界语义沿用 A3。
- Evader 在 `evader_spawn_mode: outer_ring_random` 下不会因触碰内区边界被判死亡。
- 常规 VorAdj 感知仍来自内区 Voronoi 邻接。
- 当前低权重 demo 关闭 pursuer 对外区 evader 的额外半径感知：外区敌人不会进入 pursuer 的 evader token；只有敌人进入内区并参与 Voronoi 分割后，pursuer 才能通过 VorAdj 邻接感知敌人并进入 capture 角色。
- 主分支 pursuer 仍保持硬边界。后续可单独做“我方软出边界”训练分支：出边界不死，但训练时给惩罚；外部友邻和外区协同定义后续再讨论。

## 敌人行为

- Evader 使用 APFnew 原逻辑，并额外叠加一个弱目标吸引力。
- 当 `zone_demo.enabled=true` 时，evader 的 APF boundary force 和 out-of-bounds 特殊动作逻辑全程关闭，避免内区边界项错误地主导外区入侵行为。
- 初始目标是在内区中心附近小圆内随机采样，圆半径为 `evader_goal_radius`。
- 目标吸引力由 `evader_goal_weight` 控制。当前低权重 sweep 使用 `10/30/50/80/100`。
- Evader 到达目标点即视为失守；到达后切换为纯 APF，便于继续观察后续运动。
- 失守指标和 episode 结束逻辑可以分开：`zone_breach_event` 仍按目标抵达统计；若 `zone_done_on_breach=false`，目标抵达不会立即结束回合，evader 会在无目标吸引、无边界排斥的 APF 下继续运动，直到传统 mix 成功、其他终止条件或 rollout 的 `max_steps`。

## 评估指标

Rollout JSON 的 `zone_metrics` 和 `voradj_metrics` 中会记录：

- `zone_any_evader_entered_inner`：敌人是否曾进入内区。
- `zone_breach_event`：敌人是否到达其期望目标点，当前版本以此定义失守。
- `zone_exit_after_entry_event`：敌人进入内区后是否又离开内区；当前只作为诊断指标保留，不作为默认失守定义。
- `zone_all_pursuers_inside_inner`：我方是否始终在内区内行动。
- `zone_pursuer_left_inner_event`：是否出现我方离开内区的事件。
- `zone_evader_target_reached`：敌人是否到达过中心目标点。

## 推荐 rollout 命令

当前 B-series 用来筛选 evader APF 躲避参数。共同设置：关闭 pursuer 外区半径感知、evader APF 边界项全程关闭、敌人到达中心目标算失守。

| 组 | `force_exponent` | `velocity_k` | dynamic 基础排斥 |
| --- | ---: | ---: | --- |
| B0 | 2.0 | 1.0 | 依赖 `v_ao>=0` |
| B1 | 1.5 | 1.0 | 不依赖 `v_ao` |
| B2 | 1.2 | 1.0 | 不依赖 `v_ao` |
| B3 | 1.5 | 5.0 | 不依赖 `v_ao` |
| B4 | 1.2 | 5.0 | 不依赖 `v_ao` |
| B5 | 1.2 | 10.0 | 不依赖 `v_ao` |

批量 rollout 默认通过 `tools/run_zonedemo_batch.py` 运行。所有输出都会放在：

```text
artifacts/zonedemo_v0/<batch-name>/<case>_seed<seed>/
```

8v2 goal50 B-series 示例：

```bash
python3 tools/run_zonedemo_batch.py \
  --configs 'configs/demos/zonedemo_v0/zone_v0_8v2_goal50_b*.yaml' \
  --checkpoint artifacts/2026-07-23_a3_apfnew_sqrtn_curriculum/best_20rollout10gif/stage3_8p2e2obs_step_600000/step_600000.pt \
  --batch-name bseries_8v2_goal50_seed2026072802 \
  --seeds 2026072802 \
  --device cuda:1
```

6v2 goal80 双 seed 示例：

```bash
python3 tools/run_zonedemo_batch.py \
  --configs 'configs/demos/zonedemo_v0/zone_v0_6v2_goal80_b*.yaml' \
  --checkpoint artifacts/2026-07-23_a3_apfnew_sqrtn_curriculum/best_20rollout10gif/stage2_6p2e2obs_step_1100000/step_1100000.pt \
  --batch-name bseries_6v2_goal80_seeds2026072802_2026072803 \
  --seeds 2026072802 2026072803 \
  --device cuda:1
```

6v2 goal65、目标抵达不终止、`max_steps=300` 示例：

```bash
python3 tools/run_zonedemo_batch.py \
  --configs 'configs/demos/zonedemo_v0/zone_v0_6v2_goal65_b*.yaml' \
  --checkpoint artifacts/2026-07-23_a3_apfnew_sqrtn_curriculum/best_20rollout10gif/stage2_6p2e2obs_step_1100000/step_1100000.pt \
  --batch-name bseries_6v2_goal65_seed2026072804_continue_after_breach \
  --seeds 2026072804 \
  --device cuda:1 \
  --max-steps 300
```
