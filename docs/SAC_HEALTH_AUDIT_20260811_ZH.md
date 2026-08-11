# Legacy-VorAdj Old-Mix SAC Health Audit（2026-08-11）

## 1. 固定证据合同

- checkpoint/replay：Baseline B 冻结的 pre-switch `resume_frozen_pre_fix_step_000100000`；trainer、replay、runtime 均为 env step `100000` / update `23751`。
- Gradient/Q/TD：从该 replay 的持久 RNG 状态开始，uniform joint、32 个固定 batch × 128 transitions，共 16,384 active-agent items。固定 transition-id SHA256 和完整分桶统计见 `artifacts/2026-08-11_sac_health_audit/pre_switch_100k_fixed_batches.json`。
- Q ranking：同一个 100k critic，200 个在线 capture states × 4 agents = 800 agent-state items；同 state 比 deterministic current policy、uniform random、oracle-like seek joint action。完整 rows 见 `artifacts/2026-08-11_sac_health_audit/q_ranking_100k.json`。
- 本审计只诊断，不修改 reward、entropy、MSE、LR、tau、UTD、batch 或 observation。

## 2. P0-A：Gradient / grad_clip=0.5

下表是 optimizer clipping 前的 global L2 norm；scale 是旧 `.5` clip 后实际乘数。

| 项 | min | p50 | p90 | p95 | p99 | max | `.5`触发率 | clip scale p50（min--max） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| critic | 144.11 | 381.37 | 1058.81 | 2511.06 | 3245.51 | 3570.16 | 100% | 0.001312（0.000140--0.003470） |
| actor | 0.932 | 1.531 | 2.513 | 2.868 | 4.864 | 5.716 | 100% | 0.3266（0.0875--0.5362） |
| alpha | 0.00061 | 0.1649 | 0.2952 | 0.3663 | 0.3873 | 0.3893 | 0% | 1.0 |

结论：旧 `.5` clip 不是偶发安全阀，而是对 critic/actor **每个 sampled batch 都改变 optimizer gradient**。critic 尤其严重，典型只保留约 `0.13%` 的 raw norm；actor 典型保留约 `32.7%`。假设 no-clip，optimizer 接收上表 raw gradient；alpha 在该 checkpoint 下本来就不会被 `.5` 触发。故 mid-run 去 clip 是明确 optimizer-rule switch，不能再把 B 全曲线描述为“no-clip from scratch”。

## 3. P0-B：Q / target-Q / TD

数值格式均为 `min / p50 / p90 / p95 / p99 / max`。collision 样本仅 12 个、near-collision 仅 72 个，尾部只作风险信号；near-collision 定义为无 collision event 且 pursuer 到边界、障碍物表面或其他 pursuer 的几何 clearance ≤2m。

| bucket（active items） | Q1 | Q2 | twin gap | target Q | abs TD error |
|---|---|---|---|---|---|
| ordinary（16,384） | -282.63 / -12.50 / -9.70 / -9.43 / -9.10 / -8.63 | -282.83 / -12.86 / -9.99 / -9.69 / -9.31 / -8.72 | 0.00004 / 0.526 / 3.56 / 5.76 / 12.86 / 155.90 | -280.73 / -12.75 / -9.95 / -9.67 / -9.33 / -0.04 | 0.00001 / 0.413 / 3.20 / 5.06 / 11.80 / 146.56 |
| pursuing/pre-capture（3,756） | -282.63 / -61.27 / -30.55 / -25.12 / -19.96 / -13.08 | -282.83 / -60.44 / -30.60 / -25.09 / -19.86 / -11.37 | 0.0001 / 2.27 / 7.36 / 10.22 / 21.58 / 122.56 | -280.73 / -60.46 / -30.92 / -25.42 / -20.38 / -0.04 | 0.00003 / 1.94 / 6.51 / 9.19 / 23.12 / 107.97 |
| pure CE（12,628） | -282.52 / -11.26 / -9.59 / -9.36 / -9.08 / -8.63 | -282.83 / -11.56 / -9.86 / -9.61 / -9.26 / -8.72 | 0.00004 / 0.413 / 1.67 / 2.88 / 8.28 / 155.90 | -280.67 / -11.51 / -9.83 / -9.59 / -9.28 / -0.04 | 0.00001 / 0.302 / 1.46 / 2.48 / 7.16 / 146.56 |
| near-collision（72） | -282.42 / -37.97 / -10.85 / -9.82 / -9.33 / -9.31 | -282.74 / -40.02 / -10.99 / -10.05 / -9.63 / -9.45 | 0.013 / 1.46 / 11.28 / 16.73 / 37.60 / 75.32 | -280.53 / -39.86 / -11.10 / -10.06 / -9.54 / -9.40 | 0.005 / 1.10 / 9.14 / 14.31 / 36.39 / 72.96 |
| collision（12） | -108.93 / -63.60 / -13.96 / -12.56 / -11.34 / -11.03 | -108.05 / -63.76 / -14.70 / -12.93 / -11.41 / -11.03 | 0.0004 / 6.02 / 18.10 / 18.43 / 18.71 / 18.78 | -114.92 / -48.09 / -0.04 / -0.04 / -0.04 / -0.04 | 11.02 / 66.69 / 101.78 / 105.05 / 107.39 / 107.97 |
| post-capture（0） | 无样本 | 无样本 | 无样本 | 无样本 | 无样本 |

结论：pure CE 主体 TD 尚可，但 pursuing 的 Q 尺度、twin gap 与 TD 尾部显著更大；collision items 的 TD p50 已达 66.69。整体大 tail 与稀少严重负事件（reward 最低 -160）足以解释 critic raw grad 的巨大、重尾分布。replay 中 post-capture 为 0，本线截至 100k 没有可供 critic 学习 capture→coverage 转换的正式数据。

## 4. P0-C：Reward / entropy scale

| bucket | reward `min/p50/p90/p95/p99/max` | `|reward|` p50/p95/p99/max | log_prob p50/p95/p99 | `|alpha log pi|` p50/p95/p99/max | `|alpha log pi|/|Q_policy|` p50/p95/p99/max |
|---|---|---|---|---|---|
| ordinary | -160 / -0.109 / -0.015 / -0.006 / 0.325 / 2.146 | 0.115 / 1.502 / 10.468 / 160 | 1.418 / 5.493 / 7.957 | 0.0575 / 0.2159 / 0.3124 / 0.4437 | 0.00298 / 0.00787 / 0.01026 / 0.02289 |
| pursuing | -160 / -0.683 / -0.023 / 0.270 / 1.050 / 2.146 | 0.728 / 9.984 / 11.522 / 160 | 3.408 / 7.031 / 8.679 | 0.1340 / 0.2761 / 0.3407 / 0.4366 | 0.00217 / 0.00541 / 0.00803 / 0.01389 |
| pure CE | -160 / -0.085 / -0.014 / -0.007 / -0.001 / 0.027 | 0.085 / 0.781 / 10.170 / 160 | 1.078 / 4.107 / 7.067 | 0.0445 / 0.1619 / 0.2779 / 0.4437 | 0.00339 / 0.00821 / 0.01055 / 0.02289 |

100k `alpha=0.03926`。按当前 critic Q 尺度，maximum-entropy 项绝对值通常只有 Q 的约 `0.2%--0.4%`（p99 约1%），属于**相对偏弱**，不是“entropy 压过价值”。这不等于现在应调 alpha/reward；本阶段只记录尺度诊断。

## 5. P0-D：Critic counterfactual Q ranking

这里按 agent-state item 判断该 agent 的 local enemy visibility，并按该 agent 到 enemy 距离分 near `<12m`、medium `12--20m`、far `>=20m`。本次当前策略 rollout 没产生 near 样本，不能对 near 作结论。

| bucket | n | `Qseek>Qpolicy>Qrandom` | `Qseek>Qpolicy` | `Qseek>Qrandom` | `Qpolicy>Qrandom` | mean seek-policy | mean policy-random |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 800 | 21.9% | 48.9% | 54.4% | 57.0% | -0.495 | +0.916 |
| enemy visible | 649 | 21.3% | 49.8% | 54.1% | 56.7% | -0.771 | +0.889 |
| enemy invisible | 151 | 24.5% | 45.0% | 55.6% | 58.3% | +0.693 | +1.033 |
| medium | 256 | 20.7% | 44.5% | 48.0% | 59.0% | -2.660 | +1.485 |
| far | 544 | 22.4% | 50.9% | 57.4% | 56.1% | +0.524 | +0.649 |

结论：critic 对 policy 相对 random 只有弱排序能力，对 seek 更没有稳定正确排序；尤其 medium 范围平均把 seek 估得比 policy 低 2.66。visible 并未优于 invisible，因此当前证据不支持把首要瓶颈归因于“Actor看不见 enemy”；更直接的证据是 **critic 尚未学出可靠动作优劣与接敌排序**。同时 rollout 缺 near 样本，不能排除 near-range geometry/collision value 的额外问题。

## 6. Post-switch safety 状态与当前判断

- exact switch：`step < 100000` 为 pre-fix Actor-Q + clip `.5`；`step >= 100000` 从同一 100k state 恢复为 `bounded_critic_vjp_v1` + no clip。
- 首个 post-switch 1k 窗口（101k）仍全部 finite，peak allocated `8045 MiB`，没有显存增长；raw/post grad 完全相同，证明 no-clip 生效。
- 但 101k 的 critic loss `355.84`、abs TD `4.69`、critic grad `4073.66`，相对 pre-switch 100k 窗口的 `31.23 / 1.37 / 711.25` 明显上升。它尚不是 NaN/Inf，但属于需要继续观察到 102k 的 no-clip 风险信号，不能把“finite”误写成“已经证明长期稳定”。

当前最可疑瓶颈排序：

1. critic 对 seek/policy/random 的动作排序仍弱，且 capture/collision tail 极重；
2. `.5` 长期把 critic gradient 缩到约千分之一，可能严重改变/拖慢 critic 学习；去 clip 后又有短期 overshoot 风险；
3. replay 没有 post-capture 样本，完整 mixed 后半任务尚无真实训练支撑；
4. entropy 相对 Q 很弱，但本轮不调整；
5. 当前 visible/invisible ranking 没显示 observability 是第一瓶颈。
