# Stage 3B Completion：Pure Coverage（1 obstacle）

## 1. 状态

PASS（r3 eval20：CE energy progress 0.0528（~51%）、collision 3/20（15%）、CV<0.15 6/20；明显优于 random/no-op）

## 2. 本阶段目标

验证 Stage3A 的 CE 几何收益在恢复 1 obstacle 后仍能保留，且 collision 可控、出现避障行为。

## 3. 唯一核心改动

`num_obstacles: 0 -> 1`；其余 config/seed 与 Stage3A 完全一致。

## 4. IQN 对齐项

- [IQN-ALIGN] map 120、CE centroid energy + PBRS、speed weight=0、episode 1500、inner-cluster spawn
- [IQN-ALIGN] `(a,ω)` 连续合同、`continuous_aw_v1`、robot-frame local observation、1 obstacle

## 5. 必要适配项

- [ALGO-NECESSARY] 4-agent joint replay + focal sampling；central twin critic

## 6. 临时诊断改动

无（obstacle 已恢复，属正式环境）。

## 7. 代码与配置

- config：`configs/experiments/positive_feedback_ladder_20260807/stage3b_pure_ce_obs_aw.yaml`
- 修复：active-only 索引 bug（`_active_sites`、`_voradj_coverage_potentials`），新增回归测试
- 产物：`artifacts/2026-08-07_positive_feedback_ladder/stage3b/stage3b_seed1_25k_r3/`

## 8. 训练运行

| run | steps | updates | eval20 CE progress | collision | CV<0.15 | strict |
|---|---:|---:|---:|---:|---:|---:|
| r3 seed1 | 25000 | 5001 | 0.0528（~51%） | 3/20（15%） | 6/20 | 0/20 |

## 9. 与 baseline 对比

Stage3B random：progress 0.0306 / collision 100%；noop：progress 0 / collision 0%。r3 明显优于二者，且碰撞不主导。

## 10. 结论

恢复 1 obstacle 后，连续 `(a,ω)` CTDE MASAC 仍保留大部分 CE 几何收益，碰撞可控；Stage3B 成功锚点成立。

## 11. 是否晋级

是，进入 Stage 4A（stationary/global/no-obstacle capture）。

## 12. 下一步唯一动作

启动 Stage4A seed1/seed2（并行）；按用户确认，Stage4A 启动后 4B/4C 可并行开启。

## 13. 台账更新

`docs/CONTINUOUS_MARL_POSITIVE_FEEDBACK_TRACKER_20260807_ZH.md`：Stage3B PASS，active stage=Stage4A。
