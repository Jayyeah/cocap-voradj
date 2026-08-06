# Stage 1 Completion：旧 IQN -> 连续 `(a,ω)` bridge 等价验证

## 1. 状态

PASS（严格 parity：9/9 组轨迹 0 误差、0 reward 差、事件 9/9 一致）

## 2. 本阶段目标

证明旧 IQN 离散动作可精确映射到连续 `(a,ω)` API 执行，且不改变状态、奖励、事件或成功率。

## 3. 唯一核心改动

旧 IQN 仍输出离散 action index，通过旧 `action_list` 精确映射为 `a∈{-0.4,0,0.4}`、`w∈{-π/6,0,π/6}`，再由连续 `acceleration_angular_velocity_body` 环境执行。

## 4. IQN 对齐项

- [IQN-ALIGN] 同一 config / checkpoint / seed / episode / reward
- [IQN-ALIGN] map 120×120、4v1、1 obstacle、VCT-LS、CR-MS reward
- [IQN-ALIGN] `a_max=0.4`、`w_max=π/6`、`v_max=3.0`、drag=`0.4/3`
- [IQN-ALIGN] `dt=0.05`、`substeps=10`、`decision_dt=0.5`
- [IQN-ALIGN] yaw 初始化恢复 `legacy_random`（连续线旧默认 yaw=0 被本阶段明确否决）
- [IQN-ALIGN] 显式 Euler 位置积分 + “子步后 velocity 保留子步开始值”的历史语义 + 边界逐子步/实体碰撞整步检查，均与旧离散路径完全一致

## 5. 必要适配项

- [ALGO-NECESSARY] 无（本阶段不训练 MASAC）

## 6. 临时诊断改动

- [DIAGNOSTIC-ADAPT] 无

## 7. 代码与配置

- 新增：`tools/run_stage1_aw_bridge_parity.py`
- 修改：`src/cocap_voradj/dynamics/robot.py`（AW 路径改为 legacy-exact 积分与 velocity 时序）
- 修改：`src/cocap_voradj/envs/base.py`（AW 路径碰撞检查恢复整步语义）
- 新 profile：`dynamics.profile=continuous_aw_v1`（Stage 2/3A 配置使用）
- command：

```bash
PYTHONPATH=src:. python3 tools/run_stage1_aw_bridge_parity.py \
  --seeds 2026081201 2026081202 2026081203 --max-steps 400 --device cuda:1 \
  --out-root artifacts/2026-08-07_positive_feedback_ladder/stage1_aw_bridge_parity
```

## 8. 测试结果

`tests/test_positive_feedback_ladder_contract.py::test_stage1_aw_bridge_single_step_exact_parity` PASS；既有 CTDE 合同回归 39+28 passed。

## 9. 训练运行

无训练；仅固定 seed 轨迹对比。

## 10. 关键指标

| metric | capture | coverage | mix |
|---|---:|---:|---:|
| runs | 3 | 3 | 3 |
| max state error | 0.0 | 0.0 | 0.0 |
| max reward diff | 0.0 | 0.0 | 0.0 |
| capture parity | 3/3 | - | 3/3 |
| collision parity | 3/3 | 3/3 | 3/3 |
| episode success parity | 3/3 | 3/3 | 3/3 |

## 11. 与 baseline 对比

连续 bridge 执行结果与旧离散执行结果逐状态、逐奖励、逐事件完全一致；Stage 0 的成功率自动继承。

## 12. 结论

连续 `(a,ω)` API、动作映射、runner 接线和动力学未破坏旧策略行为，bridge 严格等价。

## 13. 是否晋级

是，进入 Stage 2。

## 14. 下一步唯一动作

运行 Stage 2（1 pursuer / 1 stationary target / global visibility / 0 obstacle / `(a,ω)` MASAC）3 seeds × 25k，并每 25k 回填台账。

## 15. 台账更新

`docs/CONTINUOUS_MARL_POSITIVE_FEEDBACK_TRACKER_20260807_ZH.md`：Stage 1 PASS，active stage 更新为 Stage 2。
