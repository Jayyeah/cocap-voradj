# CTDE 25k Offline Diagnosis

- checkpoint: `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-08-06_ctde_contract/ctde_25k/ctde_25k_step25000.pt`
- replay: `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-08-06_ctde_contract/ctde_25k/ctde_25k_replay.pkl`
- eval: `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-08-06_ctde_contract/ctde_25k/ctde_25k_eval20.json`
- report: `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-08-06_ctde_contract/ctde_25k/ctde_25k_report.json`

## 1. Collision vs reward / TD tail

- collision transitions: 140
- collision reward mean: -40.3991, p95: -39.6200
- non-collision reward mean: -0.6858
- collision action norm mean: 0.2718
- non-collision action norm mean: 0.2611

> 25k 训练日志只保留 update tail；per-transition Q/TD 的碰撞长尾需在下一轮 metrics.jsonl 中直接落盘。

## 2. Critic gradient by scene/role

- 当前 checkpoint 未按 scene/role 保存梯度，只能给出 scene 级 Q/TD 样本。
| scene | Q1 p50 | Q2 p50 | target p50 | TD mean |
|---|---:|---:|---:|---:|
| capture | -11.337 | -11.905 | -11.528 | 0.712 |
| pure_ce | -29.204 | -28.301 | -28.761 | 2.275 |
| mixed_crms | -11.762 | -12.867 | -12.159 | 0.747 |

## 3. Capture min-distance

- replay 中没有 episode id，只能按 transition 汇总；min-distance 使用 central global_state 重建。
- capture: n=8747, median=25.97, p25=19.92, p75=34.11
- mixed_crms: n=6754, median=26.02, p25=19.15, p75=33.48

> 需要 episode 级 initial/final/min/AUC；当前 eval/replay 缺少 episode id，下一轮诊断实验必须写入。

## 4. Discovery before/after

- discovery event transitions: 37
- discovery reward mean: -0.5413
- no-discovery reward mean: -0.8358
- discovery action norm mean: 0.2601
- no-discovery action norm mean: 0.2609

## 5. Pure CE energy/CV

- 25k eval：pure_ce success=0/20、collision_rate=0.55、CE strict/CV<0.15=0/20。
- replay 未保存每步 energy/CV；下一轮 runner 需在 metrics.jsonl 中输出 episode 级 energy/CV 曲线。

## 6. Actor action distribution

- action_count=25000
- near-zero (<=0.02) rate=0.0000
- radial saturation (>=0.95*a_max) rate=0.0004
- action norm mean=0.2612
- action norm p95=0.3354

## 7. Primary blocker

- 25k 无捕获、无 coverage 成功，碰撞 0.45--0.55；replay 角色池 pursuing/support 已存在，但缺少 post-capture。
- 当前最可能首要阻塞点：capture 的 credit/几何信号不足以驱动持续逼近，且碰撞惩罚主导；其次缺少 episode 级几何指标，无法量化微小改善。

## 8. Eval summary

- capture: capture=0.00, success=0.00, collision=0.45
- pure_ce: capture=0.00, success=0.00, collision=0.55
- mixed_crms: capture=0.00, success=0.00, collision=0.45
