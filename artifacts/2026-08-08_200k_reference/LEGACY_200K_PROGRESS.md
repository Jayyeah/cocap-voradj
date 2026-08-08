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
| 25k | eval20 min-dist | - | - | - |
| 50k | capture | 0/20 | - | - |
| 50k | collision | 100% | - | - |
| 50k | d1_min | 2.61 | - | - |
| 50k | fraction_closing | 0.70 | - | - |
| 50k | abs_bearing_error | 0.71 | - | - |
| 50k | any 8m steps | 13.4% | - | - |
| 50k | 2+ ring | 0.13% | - | - |
| 75k | ... | 待评估 | - | - |

## 运行状态

- IQN 200k：`runs/iqn_scratch_200k_20260808`（cuda:1，12:12 启动；25k 已评估，结果与早前 25k 完全一致=确定性复现；50k 监控中）。
- Stage4A 200k：`artifacts/2026-08-08_200k_reference/stage4a`（cuda:0，12:31 启动，原合同 stage4a_capture_aw.yaml，seed 2026080801）。
- Stage4C 200k：`artifacts/2026-08-08_200k_reference/stage4c`（cuda:1，12:31 启动，原合同 stage4c_capture_aw.yaml，seed 2026080801）。
- 评估工具：MASAC 用 `tools/evaluate_ctde_formal.py`（eval20）+ `tools/evaluate_stage4_paired.py`（行为指标）；IQN 用 `tools/evaluate_iqn_scratch_early.py`。
- 奖励改进参考：A1/A2/A3 结果（A2 冲撞坍缩、A3 无信号、A1 待定）→ 若 A1 也无信号，则 4A/4C 200k 保持原 reward 作为严格对照；后续探索批再考虑熵/warmup 改进。
