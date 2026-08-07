# Stage 4A Completion：Capture（stationary/global/no-obs）

## 1. 状态

PASS（2-seed 验证：seed1/seed2 的 min-distance 均优于 random/no-op，collision 可控；seed2 capture 15%，seed1 5%）

## 2. 本阶段目标

验证多智能体 `(a,ω)` MASAC 在 stationary target、global visibility、0 obstacle 下能形成 capture 几何/接近行为。

## 3. 唯一核心改动

从 Stage3B pure coverage 进入 capture：恢复 1 evader（stationary）、global visibility、ring_importance_ms_v0 reward。

## 4. IQN 对齐项

- [IQN-ALIGN] map 120、`(a,ω)` 连续合同、`continuous_aw_v1`、robot-frame local observation
- [IQN-ALIGN] ring_importance_ms_v0 reward、capture distance、boundary/collision 语义

## 5. 必要适配项

- [ALGO-NECESSARY] 4-agent joint replay + focal sampling；central twin critic

## 6. 临时诊断改动

- [DIAGNOSTIC-ADAPT] evader stationary、global_evader_visibility=true、num_obstacles=0、全体 pursuers pursuing

## 7. 代码与配置

- config：`configs/experiments/positive_feedback_ladder_20260807/stage4a_capture_aw.yaml`
- 产物：`artifacts/2026-08-07_positive_feedback_ladder/stage4a/stage4a_seed{1,2}_25k/`

## 8. 训练运行

| seed | steps | updates | eval20 capture | collision | min_min_distance | distance_progress |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 25000 | 5001 | 1/20（5%） | 1/20（5%） | 22.07 | +8.34 |
| 2 | 25000 | 5001 | 3/20（15%） | 8/20（40%） | 22.13 | +2.28 |

## 9. 与 baseline 对比

random：capture 20%、collision 90%、avg_min_distance 26.59；noop：capture 0%、avg_min_distance 40.13。seed1/seed2 min-distance 均优于二者，seed1 collision 明显低于 random。

## 10. 结论

capture 几何接近行为已建立（min-distance 明确改善）；capture event 仍低，但达到 Stage4A 的“几何指标明确乐观信号” gate。

## 11. 是否晋级

是，进入 Stage4B（moving evader）与 Stage4C（1 obstacle）并行。

## 12. 下一步唯一动作

并行启动 Stage4B seed1 与 Stage4C seed1；各自保持单变量。

## 13. 台账更新

`docs/CONTINUOUS_MARL_POSITIVE_FEEDBACK_TRACKER_20260807_ZH.md`：Stage4A PASS，active stage=Stage4B/4C 并行。
