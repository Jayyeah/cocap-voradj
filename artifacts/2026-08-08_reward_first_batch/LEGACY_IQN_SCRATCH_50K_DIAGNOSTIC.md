# LEGACY IQN SCRATCH 50K DIAGNOSTIC（2026-08-08）

> 同一连续轨迹（25k→50k，`step_25000.pt`→`step_50000.pt`），与 25k 相同评估协议（20 集、1000 步、确定性、同 seed 序列）。

## 0k/random → 25k → 50k 对比

| 指标 | random（4B 基线） | IQN 25k | IQN 50k | 变化 |
|---|---:|---:|---:|---|
| capture | 0% | 0/20 | 0/20 | 无 |
| collision | 90-95% | 100% | 100% | 仍失控 |
| avg length | ~1000 | 330 | 185 | 更早撞停 |
| d1_min | ~13.4 | 2.60 | 2.61 | 保持单机逼近 |
| d2_min | - | 25.1 | 26.0 | 仍无第二机 |
| fraction_closing | ~0.49 | 0.70 | 0.70 | 保持 |
| abs_bearing_error | ~1.73 | 0.90 | 0.71 | 改善 |
| turn_direction_correct_rate | ~0.53 | 0.52 | 0.66 | 改善，首次超 random |
| 任意步单机进 8m | ~0 | 10.8% | 13.4% | 上升 |
| 任意步进 ring | ~0 | 4.7% | 6.4% | 上升 |
| 2+ 机同时进 ring | ~0 | 0.05% | 0.13% | 极低但上升 |
| 动作模式 | 随机 | 全油门+全转向（action 0/8 主导） | 中性+直行（action 4/7 主导） | 从自旋冲撞→直行冲撞 |

## 结论

1. IQN 50k capture 仍为零，collision 仍 100%；**没有出现 25k→50k 的质变跃迁**（Situation C 的部分证据：有改善但未跨过 capture threshold）。
2. 改善集中在控制质量：转向正确率超过 random、bearing error 下降、动作从“双向自旋”收敛为“直行+中性停顿”。
3. 多机协调几乎没有进展：d2_min 26、2+ ring 0.13%，说明 50k 内 IQN 也没有形成包围几何。
4. 行为解读：25k 的“猛冲”是探索+稠密奖励的粗解；50k 收敛为更可控的“单机直冲”，但仍以碰撞告终。
5. 对 ladder 的参照意义：
   - 即使原成功算法，50k 内 capture=0 且 collision 100% → 不能用“25k capture=0”单独否定 MASAC 候选；
   - 但 IQN 50k 至少把 closing/bearing/ring 连续指标做正，而 MASAC 4B seed1 这些指标仍接近 random → MASAC 早期学习确实更弱（Situation A/B 混合）；
   - MASAC 若 25k/50k 连 closing/bearing 都不改善，则不是“慢”，而是“没学对方向”。

## 文件

- 评估结果：`artifacts/2026-08-08_reward_first_batch/iqn_scratch_50k_behavior.json`
- 50k checkpoint：`runs/iqn_scratch_early_25k50k_20260808/checkpoints/step_50000.pt`
- 25k 对照：`LEGACY_IQN_SCRATCH_25K_DIAGNOSTIC.md`
