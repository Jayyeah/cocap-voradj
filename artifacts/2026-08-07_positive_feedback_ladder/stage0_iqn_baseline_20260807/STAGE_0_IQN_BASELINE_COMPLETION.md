# Stage 0 Completion：旧 IQN 成功基线复现

## 1. 状态

PASS

## 2. 本阶段目标

在 `continuous/masac-ctde-contract-20260806` 分支上证明旧 IQN checkpoint 仍可加载，capture/coverage/mix 主要成功行为可复现，指标与历史 20-rollout 同数量级。

## 3. 唯一核心改动

无。本阶段只读评估，不训练、不改代码、不改配置。

## 4. IQN 对齐项

- [IQN-ALIGN] config：`configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/stage1_4p1e1obs_scratch2m.yaml`
- [IQN-ALIGN] checkpoint：`artifacts/2026-08-02_three_line_stage_best_20rollout10gif/crms_supportapproach_4v1_s1_step_2000000/step_2000000.pt`，SHA256 `2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89`
- [IQN-ALIGN] 动作：`unicycle_discrete` 9 动作，epsilon=0，fixed-midpoint quantiles
- [IQN-ALIGN] 场景：capture / coverage / mix，seed 2026081201 起 20 个 episode
- [IQN-ALIGN] max steps：capture 1000、coverage 1200、mix 2200（与历史参数一致）

## 5. 必要适配项

无 `[ALGO-NECESSARY]`。

## 6. 临时诊断改动

无 `[DIAGNOSTIC-ADAPT]`。

## 7. 代码与配置

- commit：9827770（评估时 HEAD；当日 ladder 工具/配置改动不涉及旧 IQN 执行路径）
- command：

```bash
cd /home/yjq/rl/CoCap1/cocap-voradj
PYTHONPATH=src:. python3 tools/batch_rollouts.py \
  --config configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/stage1_4p1e1obs_scratch2m.yaml \
  --checkpoint artifacts/2026-08-02_three_line_stage_best_20rollout10gif/crms_supportapproach_4v1_s1_step_2000000/step_2000000.pt \
  --output-root artifacts/2026-08-07_positive_feedback_ladder/stage0_iqn_baseline_20260807 \
  --episodes 20 --gif-count 0 --scenarios capture coverage mix \
  --seed 2026081201 --device cuda:1 --max-steps 2200 \
  --capture-max-steps 1000 --coverage-max-steps 1200 --capture-evaders 1
```

## 8. 测试结果

20 episodes × 3 scenarios 全部完成；无 NaN/异常退出；checkpoint 加载成功。

## 9. 训练运行

- seeds：2026081201..2026081220（每 scenario 20 episodes）
- steps：capture avg 77.15、coverage avg 132.95、mix avg 227.45
- 无训练 checkpoint；只评估旧 checkpoint

## 10. 关键指标

| scenario | capture | coverage | mix |
|---|---:|---:|---:|
| capture success | 1.0 | 0.0 | 1.0 |
| coverage success | 0.0 | 1.0 | 0.9 |
| episode success | 1.0 | 1.0 | 0.9 |
| collision | 0.0 | 0.0 | 0.05 |
| final/best CV | - | 0.049 / 0.024 | 0.108 / 0.045 |
| avg steps | 77.15 | 132.95 | 227.45 |

## 11. 与 baseline 对比

与 `artifacts/2026-08-02_three_line_stage_best_20rollout10gif/crms_supportapproach_4v1_s1_step_2000000/all_summaries.json` 逐项一致：capture 1.0、coverage 1.0、mix 0.9、collision ≤0.05、CV 同数量级。

## 12. 结论

旧 IQN 成功基线在当前分支完全可复现；可以安全进入 Stage 1 bridge。

## 13. 是否晋级

是，进入 Stage 1。

## 14. 下一步唯一动作

完成 Stage 1 连续 `(a,ω)` fixed-seed trajectory parity 并出具 `STAGE_1_AW_BRIDGE_COMPLETION.md`。

## 15. 台账更新

`docs/CONTINUOUS_MARL_POSITIVE_FEEDBACK_TRACKER_20260807_ZH.md`：STAGE0 条目标 PASS，active stage 更新为 Stage 1。
