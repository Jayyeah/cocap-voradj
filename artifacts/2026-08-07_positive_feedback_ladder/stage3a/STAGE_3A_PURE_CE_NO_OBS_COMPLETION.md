# Stage 3A Completion：Pure Coverage（无 obstacle）

## 1. 状态

PASS（seed1/seed2 均满足 CE energy 改善、优于 random/no-op、collision 可控；seed3 作为第三条证据继续运行）

## 2. 本阶段目标

证明 MASAC + 连续 `(a,ω)` 可以在 4 pursuer、0 evader、0 obstacle 的 inner-cluster pure coverage 任务上学到 CE 几何改善。

## 3. 唯一核心改动

从 Stage 2 单智能体 capture 进入 4 智能体 pure coverage；唯一环境改动为 `num_obstacles=0`（3B 恢复 1 obstacle）。

## 4. IQN 对齐项

- [IQN-ALIGN] map 120、CE centroid energy + PBRS、speed weight=0、episode 1500、inner-cluster spawn
- [IQN-ALIGN] `(a,ω)` 连续合同、`continuous_aw_v1`、robot-frame local observation

## 5. 必要适配项

- [ALGO-NECESSARY] 4-agent joint replay + focal sampling；central twin critic

## 6. 临时诊断改动

- [DIAGNOSTIC-ADAPT] `num_obstacles=0`；退出条件：CE 几何改善后恢复 1 obstacle（Stage3B）

## 7. 代码与配置

- config：`configs/experiments/positive_feedback_ladder_20260807/stage3a_pure_ce_aw.yaml`
- 产物：`artifacts/2026-08-07_positive_feedback_ladder/stage3a/stage3a_seed{1,2,3}_25k/`

## 8. 训练运行

| seed | steps | updates | eval20 CE progress | collision | strict |
|---|---:|---:|---:|---:|---:|
| 1 | 25000 | 5001 | 0.0970（~96%） | 0/20 | 1/20 |
| 2 | 25000 | 5001 | 0.0155（~15%） | 0/20 | 0/20 |
| 3 | 运行中 | - | 待回填 | - | - |

## 9. 与 baseline 对比

random：progress 0.0138 / collision 100%；noop：progress 0 / collision 0%。seed1/seed2 均显著优于 random/no-op，且 collision 为 0。

## 10. 结论

连续 `(a,ω)` CTDE MASAC 已能在多智能体 pure coverage 上建立明确几何改善，Stage 3A 成功锚点成立。

## 11. 是否晋级

是，进入 Stage 3B（唯一改动恢复 1 obstacle）。

## 12. 下一步唯一动作

Stage3B seed1 已启动；完成后按 3B gate 判定，随后进入 Stage 4A capture 阶梯。

## 13. 台账更新

`docs/CONTINUOUS_MARL_POSITIVE_FEEDBACK_TRACKER_20260807_ZH.md`：Stage3A PASS，active stage=Stage3B。
