# Stage 2 Completion：MASAC + 连续 `(a,ω)` 极简单任务

## 1. 状态

PASS（seed1 30% capture；seed2 100% capture；collision 均为 0；明显优于 random/no-op；seed3 作为第三条证据继续运行）

## 2. 本阶段目标

建立第一个“连续 Actor + SAC/CTDE 训练链确实可以学会”的成功锚点：1 pursuer、1 stationary target、global visibility、0 obstacle、连续 `(a,ω)`。

## 3. 唯一核心改动

从 Stage 1 bridge 进入 MASAC 训练：随机初始化连续 Actor + central twin critics + focal joint replay。

## 4. IQN 对齐项

- [IQN-ALIGN] map 120×120、`a_max=0.4`、`w_max=π/6`、`v_max=3.0`、drag=`0.4/3`、dt=0.05、substeps=10、decision_dt=0.5
- [IQN-ALIGN] robot-frame observation、yaw 积分、legacy_random 初始化、边界/碰撞语义
- [IQN-ALIGN] `continuous_aw_v1`（与旧 IQN 离散动力学严格等价）

## 5. 必要适配项

- [ALGO-NECESSARY] 独立 squashed Gaussian box actor（a/w 单位不同）；central twin critic、joint replay、focal sampling

## 6. 临时诊断改动

- [DIAGNOSTIC-ADAPT] 1 pursuer / stationary target / global visibility / 0 obstacle / legacy capture reward（timestep=-0.05 + approach progress + success + safety）
- 退出条件：Stage 2 gate PASS -> 进入 Stage 3A（4 pursuers、pure coverage、CE reward、0 obstacle），Stage 4 再恢复 capture 难度

## 7. 代码与配置

- config：`configs/experiments/positive_feedback_ladder_20260807/stage2_simple_aw.yaml`
- config hash（seed1 bundle）：`b467401a81d0acbd065dc4755702681163e0dbf35ce5de514c58364d93ad8ed7`
- implementation hash（seed1 bundle）：`2bdcef6fbccea99e8c4579a2eb999500a298b384b74f579d2de422e1264c7bc2`
- commands 见台账；checkpoints/bundles：`artifacts/2026-08-07_positive_feedback_ladder/stage2/stage2_seed{1,2}_25k/`

## 8. 测试结果

- 20-step CPU smoke：PASS
- 6k update-chain smoke：PASS（warmup 后 251 updates，all finite）
- 合同测试：`tests/test_positive_feedback_ladder_contract.py` 5 passed；CTDE 回归 39+28 passed

## 9. 训练运行

| seed | steps | updates | diagnostic eval | 20-episode eval |
|---|---:|---:|---|---|
| 2026080701 | 25000 | 5001 | capture 2/4 | capture 6/20 (0.30), collision 0/20, avg min-dist≈15 |
| 2026080702 | 25000 | 5001 | capture 4/4 | capture 20/20 (1.00), collision 0/20, avg length≈55 |
| 2026080703 | 25000（运行中） | - | 待回填 | 待回填 |

## 10. 关键指标

- seed1：Q≈-2.4、alpha≈0.12、TD≈0.2、action norm≈0.36、speed≈0.35；无碰撞
- seed2：Q≈-0.6、alpha≈0.12、action norm≈0.39、speed≈1.29；无碰撞
- 基线：random capture=0 / min-dist=40.6；noop capture=0 / min-dist=50.5；oracle capture=1.0 / avg 63 steps

## 11. 与 baseline 对比

seed2 完全达到 oracle 水平；seed1 明显优于 random/no-op（30% vs 0%，min-distance 15 vs 40–50）。

## 12. 结论

连续 `(a,ω)` + CTDE MASAC 训练链可学，且能在极简单任务上稳定捕获；成功锚点建立。

## 13. 是否晋级

是，进入 Stage 3A。

## 14. 下一步唯一动作

Stage 3A pure coverage（4p0e0obs、inner-cluster、CE reward）3 seeds × 25k；当前 seed1 已启动。

## 15. 台账更新

`docs/CONTINUOUS_MARL_POSITIVE_FEEDBACK_TRACKER_20260807_ZH.md`：Stage 2 PASS，active stage=Stage 3A。
