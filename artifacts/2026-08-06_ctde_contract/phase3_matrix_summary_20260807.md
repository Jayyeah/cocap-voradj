# Phase 3 单任务 25k 矩阵进展（2026-08-07）

统一合同：`a_max=0.4`、parity drag、update_every=4、grad clip 0.5、map 120、400-step diagnostic cap、每 25k bundle。

| 实验 | scene | init | snapshot | training 25k | final report/bundle | diagnostic eval | CE energy progress | collision(train) |
|---|---|---|---|---|---|---|---|---|
| pure random | pure_ce | random | no | ✅ | ✅ | ✅ 4/4 | 4/4 positive | 90 |
| pure encoder | pure_ce | legacy encoder | no | ✅ | ✅ | ✅ 4/4 | 3/4 positive | 203 |
| pure snapshot | pure_ce | random | yes | ✅ | ✅ | ✅ 4/4 | 4/4 positive | 114 |
| pure encoder+snapshot | pure_ce | legacy encoder | yes | ✅ metrics-only（用户接受） | ❌ 未落盘 | ❌ | 待补 | 3 |
| capture random | capture | random | no | ✅ metrics-only（用户接受） | ❌ 未落盘 | ❌ | 待补 | 1 |

## 已完成四条中的关键数值

### pure random（`ctde_pure_random_25k_v2`）

- updates=5001、all_finite；diagnostic eval：collision=0/4，CE energy progress 4/4 positive（initial 0.050--0.164 → final 0.007--0.021）。
- training collision=90/25000。

### pure encoder（`ctde_pure_encoder_25k`）

- updates=5001、all_finite；diagnostic eval：collision=0/4，CE energy progress 3/4 positive；action norm mean≈0.078（末段偏低），speed mean≈0.28--0.52。
- training collision=203/25000。

### pure snapshot（`ctde_pure_snapshot_25k`）

- updates=5001、all_finite；diagnostic eval：collision=0/4，CE energy progress 4/4 positive；action norm mean≈0.056--0.070，speed mean≈0.38--0.44。
- training collision=114/25000。

### pure encoder+snapshot（metrics 已到 25000，final 未落盘）

- updates=5001、all_finite；training collision=3/25000、truncated=0；action norm mean=0.265、speed mean=0.447。
- Q1 mean≈-16.44、TD mean≈1.59；critic pre-clip grad≈524.3→post 0.5。
- 缺 400-step diagnostic eval 与 step25000 bundle/report。

### capture random（metrics 已到 25000，final 未落盘）

- updates=5001、all_finite；training collision=1/25000、truncated=0；action norm mean=0.253、speed mean=0.346。
- focal pools：pursuing=6865、support=13750、pre-capture coverage=79385、post-capture=0。
- Q1 mean≈-14.23、TD mean≈1.77；critic pre-clip grad≈386.0→post 0.5。
- 缺 400-step diagnostic eval（min-distance/discovery）与 step25000 bundle/report。

## 当前结论

- pure-only 三条完整线都出现 CE energy 下降，但 strict/CV<0.15 均为 0；snapshot 线碰撞明显低于 random/encoder。
- capture random 已产生 pursuing/support 角色池，但无 post-capture 数据，需要最终 eval 看 min-distance/discovery。
- encoder+snapshot 与 capture random 的“训练完成”只反映 metrics 达到 25000，不能当作完整 25k 交付；需 resume 补 final bundle/eval，或由用户决定接受 metrics-only。

## 待办

1. 对 encoder+snapshot 与 capture random 从 step1 bundle resume 补跑至 25k 并生成最终 report（约 2h/条，并行约 2--3h）。
2. 完成后补 400-step diagnostic eval 并回填本表。
3. 整理 teacher-assisted 对比结论，决定是否进入 100k。

## 用户决策（2026-08-07）

- 用户接受两条线的 metrics-only 状态：`ctde_pure_encoder_snapshot_25k`、`ctde_capture_random_25k` 不再补 final bundle/diagnostic eval。
- 对应结果文件：`ctde_pure_encoder_snapshot_25k_metrics_only_result.json`、`ctde_capture_random_25k_metrics_only_result.json`。
