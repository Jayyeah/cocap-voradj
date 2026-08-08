# 200k 早训对照进度（2026-08-08 启动）

三线对照：IQN scratch 200k（原成功算法参照）vs Stage4A 200k（stationary，原合同）vs Stage4C 200k（moving+1obs，原合同）。
每 25k checkpoint 记录相同行为指标（capture/collision/d1_min/closing/bearing/ring 访问）；MASAC 线额外记录 eval20。

## 对照表（逐 25k 回填）

| step | 指标 | IQN 200k | Stage4A 200k | Stage4C 200k |
|---|---|---|---|---|
| 25k | capture | 0/20 | - | - |
| 25k | collision | 100% | - | - |
| 25k | d1_min | 2.60 | - | - |
| 25k | fraction_closing | 0.70 | - | - |
| 25k | abs_bearing_error | 0.90 | - | - |
| 25k | any 8m steps | 10.8% | - | - |
| 25k | 2+ ring | 0.05% | - | - |
| 25k | eval20 min-dist | - | 19.4-67.8（no-op 坍缩，初始=最终） | 14.0-16.4（弱接近） |
| 50k | capture | - | 5/20 (25%) | 待评估 |
| 50k | collision | - | 70% | - |
| 50k | d1_progress（capture 集） | - | +11 ~ +21 | - |
| 50k | speed_mean | - | 0.37-0.71 | - |
| 50k | capture | 0/20 | - | - |
| 50k | collision | 100% | - | - |
| 50k | d1_min | 2.61 | - | - |
| 50k | fraction_closing | 0.70 | - | - |
| 50k | abs_bearing_error | 0.71 | - | - |
| 50k | any 8m steps | 13.4% | - | - |
| 50k | 2+ ring | 0.13% | - | - |
| 75k | capture | 0/20 | - | - |
| 75k | collision | 100% | - | - |
| 75k | d1_min / d2_min | 2.52 / 16.4 | - | - |
| 75k | fraction_closing | 0.82 | - | - |
| 75k | abs_bearing_error | 0.38 | - | - |
| 75k | any 8m steps | 19.4% | - | - |
| 75k | 2+ ring | 0.37% | - | - |
| 100k | capture | 8/20 (40%) | - | - |
| 100k | collision | 75% | - | - |
| 100k | d1_min / d2_min | 2.50 / 10.1 | - | - |
| 100k | d1_progress | +9.25 (70% 集正向) | - | - |
| 100k | fraction_closing | 0.59 | - | - |
| 100k | abs_bearing_error | 0.54 | - | - |
| 100k | any 8m steps | 19.3% | - | - |
| 125k | capture | 15/20 (75%) | - | - |
| 125k | collision | 5% | - | - |
| 125k | d1_min / d2_min | 5.51 / 7.76 | - | - |
| 125k | d1_progress | +23.9 (100% 集正向) | - | - |
| 125k | any 8m / any ring / 2+ ring | 33.7% / 25.4% / 19.9% | - | - |
| 150k | capture | 10/20 (50%) | - | - |
| 150k | collision | 0% | - | - |
| 150k | d1_min / d2_min | 6.80 / 8.07 | - | - |
| 150k | d1_progress | +25.7 (100% 集正向) | - | - |
| 150k | any 8m / any ring / 2+ ring | 46.6% / 50.6% / 10.8% | - | - |
| 175k | capture | 5/20 (25%) | - | - |
| 175k | collision | 0% | - | - |
| 175k | d1_min / d2_min | 7.11 / 8.68 | - | - |
| 175k | d1_progress | +25.3 (100% 集正向) | - | - |
| 175k | any 8m / any ring / 2+ ring | 48.8% / 65.3% / 29.6% | - | - |
| 200k | capture | 8/20 (40%) | - | - |
| 200k | collision | 0% | - | - |
| 200k | d1_min / d2_min / d3_min | 7.06 / 8.33 / 25.8 | - | - |
| 200k | d1_progress | +23.6 (100% 集正向) | - | - |
| 200k | any 8m / any ring / 2+ ring | 28.5% / 63.7% / 36.9% | - | - |

## IQN 200k 分阶段汇总（capture 主线）

| step | capture | collision | d1_min | d2_min | closing | bearing | 2+ ring |
|---|---:|---:|---:|---:|---:|---:|---:|
| 25k | 0/20 | 100% | 2.60 | 25.1 | 0.70 | 0.90 | 0.05% |
| 50k | 0/20 | 100% | 2.61 | 26.0 | 0.70 | 0.71 | 0.13% |
| 75k | 0/20 | 100% | 2.52 | 16.4 | 0.82 | 0.38 | 0.37% |
| 100k | 8/20 | 75% | 2.50 | 10.1 | 0.59 | 0.54 | 0.08% |
| 125k | 15/20 | 5% | 5.51 | 7.76 | 0.16 | 1.03 | 19.9% |
| 150k | 10/20 | 0% | 6.80 | 8.07 | 0.11 | 1.07 | 10.8% |
| 175k | 5/20 | 0% | 7.11 | 8.68 | 0.14 | 1.65 | 29.6% |
| 200k | 8/20 | 0% | 7.06 | 8.33 | 0.12 | 1.70 | 36.9% |

结论：IQN 在 ~100k 首次出现 capture，125k 峰值 75%；之后 capture 回落（25-50%）但 collision 归零、ring 多机占用持续上升（2+ ring 至 36.9%）→ 策略从“冲撞逼近”转向“受控包围但 capture 决策不稳定”。

## Stage4A/4C 200k 最新 checkpoint（2026-08-08 16:25，25k 评估）

| 指标 | Stage4A 200k @25k | Stage4C 200k @25k |
|---|---:|---:|
| eval20 capture | 0/20 | 0/20 |
| eval20 collision | 0% | 0% |
| eval20 min-dist | 19.4–67.8（初始=最终，无移动） | 10.4–21.7（多集接近，progress 正） |
| 行为 | no-op 坍缩（action≈0.03-0.07、speed=0） | 中速接近（action≈0.15、progress 正向但无 capture） |
| 对照（原 25k 正式线） | 4A seed1/2：capture 5%/15%、min-dist 22 | 4C seed1/2：capture 0%、min-dist 14.9/15.5 |

注意：4A/4C 200k 用新 seed（2026080801）跑原合同；4A 该 seed 在 25k 出现 no-op 坍缩（原 seed 无此问题），4C 有接近趋势但未达 capture。两线继续训练至 200k，后续每 25k 回填本表。

早期轨迹初检（≤27k，新诊断字段）：4A Q→-4.2、speed≈0.10、closing≈0.28-0.43、无 capture 事件；4C Q→-4.5、speed≈0.06-0.10、closing≈0.44-0.53、无 capture 事件 → 两线早期均无接近信号（与 25k eval20 一致）。

## 运行状态

- IQN 200k：`runs/iqn_scratch_200k_20260808`（cuda:1，12:12 启动；25k 已评估，结果与早前 25k 完全一致=确定性复现；50k 监控中）。
- Stage4A 200k：`artifacts/2026-08-08_200k_reference/stage4a`（cuda:0，12:31 启动，原合同 stage4a_capture_aw.yaml，seed 2026080801）。
- Stage4C 200k：`artifacts/2026-08-08_200k_reference/stage4c`（cuda:1，12:31 启动，原合同 stage4c_capture_aw.yaml，seed 2026080801）。
- 评估工具：MASAC 用 `tools/evaluate_ctde_formal.py`（eval20）+ `tools/evaluate_stage4_paired.py`（行为指标）；IQN 用 `tools/evaluate_iqn_scratch_early.py`。
- 奖励改进参考：A1/A2/A3 结果（A2 冲撞坍缩、A3 无信号、A1 待定）→ 若 A1 也无信号，则 4A/4C 200k 保持原 reward 作为严格对照；后续探索批再考虑熵/warmup 改进。
