# Open Encirclement CTDE L0 最终结果（3 seeds × 150k）

## 1. 完成性与口径

- 运行窗口：2026-08-23 11:50:28–12:26:19 +08:00。
- supervisor 最终状态：`COMPLETED`；6/6 runs 完成，`failed=[]`，两条自动队列均清空。
- MAPPO 每条 150,000 env steps / 750 updates；MADDPG 每条 150,000 env steps / 14,898 updates、最终 replay=150,000。
- 六条线均保留经加载验证的 `final-full.pt`；被取代的 `rolling-full.pt` 已按存储合同清理。
- 最终评估每条线包含 20 个 deterministic rollouts（seeds 601000–601019）和 20 个 stochastic/reduced-noise rollouts（seeds 701000–701019）。capture 必须同时满足三机非退化凸包包含 target 且三机距离均不超过 0.3。
- 这里是 `Audit-Corrected/corrected_roundup_v1` 结果，不能与保留 bug、且 target 也是 learned agent 的 `Upstream-Raw` 60% deterministic capture 混算。

## 2. 各线最终性能

### 2.1 Deterministic final evaluation

| run | capture | mean length | containment | all-radius | collision | throughput |
|---|---:|---:|---:|---:|---:|---:|
| MAPPO seed1 | 2/20 = 10% | 99.40 | 0.0011 | 0.0970 | 0.1190 | 273.42 steps/s |
| MAPPO seed2 | 0/20 = 0% | 100.00 | 0.0000 | 0.2270 | 0.1140 | 270.06 steps/s |
| MAPPO seed3 | 0/20 = 0% | 100.00 | 0.0000 | 0.0310 | 0.0605 | 275.92 steps/s |
| MADDPG seed1 | 1/20 = 5% | 99.20 | 0.0117 | 0.0031 | 0.2650 | 189.44 steps/s |
| MADDPG seed2 | 0/20 = 0% | 100.00 | 0.0000 | 0.0000 | 0.9145 | 192.60 steps/s |
| MADDPG seed3 | 0/20 = 0% | 100.00 | 0.0000 | 0.0000 | 0.7200 | 193.69 steps/s |

### 2.2 Stochastic/reduced-noise final evaluation

| run | capture | mean length | containment | all-radius | collision |
|---|---:|---:|---:|---:|---:|
| MAPPO seed1 | 1/20 = 5% | 99.25 | 0.0011 | 0.0650 | 0.1580 |
| MAPPO seed2 | 1/20 = 5% | 99.75 | 0.0010 | 0.2224 | 0.0400 |
| MAPPO seed3 | 0/20 = 0% | 100.00 | 0.0000 | 0.0210 | 0.0580 |
| MADDPG seed1 | 0/20 = 0% | 100.00 | 0.0075 | 0.0045 | 0.2205 |
| MADDPG seed2 | 0/20 = 0% | 100.00 | 0.0000 | 0.0000 | 0.9200 |
| MADDPG seed3 | 0/20 = 0% | 100.00 | 0.0000 | 0.0000 | 0.7550 |

### 2.3 三 seed 汇总

| algorithm | deterministic capture | stochastic capture | deterministic collision | stochastic collision |
|---|---:|---:|---:|---:|
| MAPPO | 2/60 = 3.33% | 2/60 = 3.33% | 9.78% | 8.53% |
| MADDPG | 1/60 = 1.67% | 0/60 = 0% | 63.32% | 63.18% |

## 3. 25k checkpoint 轨迹

每条线在 25k、50k、75k、100k、125k、150k 都做了独立 20-rollout deterministic milestone evaluation：

| run | 最高 milestone capture | checkpoint | 150k milestone capture |
|---|---:|---:|---:|
| MAPPO seed1 | 15% | 50k | 0% |
| MAPPO seed2 | 5% | 125k | 0% |
| MAPPO seed3 | 0% | 无 | 0% |
| MADDPG seed1 | 15% | 50k | 0% |
| MADDPG seed2 | 5% | 25k | 0% |
| MADDPG seed3 | 5% | 50k | 0% |

milestone seeds 与最终 601xxx/701xxx seeds 不同，因此“最高 checkpoint”只用于训练轨迹诊断，不能替代最终固定 seed 汇总。capture 没有随训练步数单调提升，最终 checkpoint 并非稳定最佳点。

## 4. 可支持的结论

1. **工程 Gate 通过，性能 Gate 未通过。** 相同环境合同下的训练、评估、全量恢复、自动调度和资源控制均可靠完成，但两者都没有学出稳定围捕。
2. **MAPPO 是更好的后续锚点。** MAPPO 最终 capture 略高、平均碰撞仅约 9%，seed2 有 22.7% deterministic all-radius fraction，说明它更常把三机带到目标附近。
3. **MAPPO 的主要缺口是角度/拓扑而不是纯接近。** all-radius 明显高于 containment，但 containment 近乎为零；三机常靠近却没有形成包含 target 的非退化三角形。
4. **MADDPG 有严重 seed instability/collision collapse。** seed2/seed3 deterministic collision 分别为 91.45%/72.00%。最终 actor LR 已按 upstream scheduler 衰减至约 `4.40e-6`，不同 seed 的 Q 尺度明显分叉；scheduler/Q-scale 是后续 ablation 假设，不在本轮宣称因果。
5. **下一轮不应直接扩大训练步数。** 应先对 25k checkpoint 做统一 final-seed selection，再分别验证 angular-spread/containment reward、MADDPG scheduler、碰撞惩罚与 critic/Q 归一化；任何奖励或环境变化必须升级新 spec/version。

## 5. 资产位置

- 动态结果：`artifacts/open_encirclement_ctde/l0/<algorithm>_seed{1,2,3}/`（本地保留、Git 忽略）。
- 每线最终摘要：`final_evaluation.json`、`state.json`、`manifest.json`。
- 每 25k 评估：`eval_step_*.json`；模型：`checkpoints/step_*-model.pt`。
- 最终可恢复状态：`checkpoints/final-full.pt`。
