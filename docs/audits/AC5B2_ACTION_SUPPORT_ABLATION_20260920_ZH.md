# AC-5B2 — Action-Support Ablation

日期：2026-09-20
范围：只新增 Full-Natural MC-control LQ × 2 seeds；不重跑 AC-5B1，不做 FQE/Bellman、NQ/CQ/V、Actor、PPO、GAE 或 counterfactual-supervised training。

## 1. 固定合同与 identity

- bank：`artifacts/2026-09-20_ac2b/canonical_bc_critic_bank_v2.npz`
- bank SHA256：`07e08d17ae6283d785e0d2e00307f330b13a88dcd0f8ce83f92f0d451ad7808a`
- split：沿用 AC-2B episode split，不重分、不复制 bank。
- 初始 LQ：`artifacts/2026-09-20_ac3a/checkpoints/lq_natural_seed2026091711.pt`
- 初始 checkpoint SHA256：`e8b52a32040937c2a4060e359546d1a427aee6c955dcaec501827038fac27552`
- 初始 tensor state SHA256：`825b11e6753a9eb541dd265b087b74e4ed786fbc0430d04ab6fbd9802900eada`
- canonical BC Actor SHA256：`7916a970226a0452a8f71a22235524f6ba4511aa4c9bda907a9c8368a60bf2cd`
- normalization 固定：mean `15.788457870483398`，std `52.9477653503418`
- target：exact individual-agent full-episode MC return，`gamma=0.99`
- optimizer：Adam，lr `1e-4`，weight decay `1e-6`，batch `256`，grad clip `0.5`
- budget：4 × 300 = 1200 updates/seed；自然 active-row uniform sampler with replacement。

训练 episode/row 组成：

| training distribution | episodes | mixed | coverage | active rows |
|---|---:|---:|---:|---:|
| argmax-only（AC-5B1 control） | 32 | 24 | 8 | 28,172 |
| Full-Natural（本轮新增） | 64 | 48 | 16 | 54,784 |

主测试使用完整 AC-2B test split：8 episodes、6,620 rows；同时保留 argmax-only 4-episode test split、3,564 rows用于 AC-5B1 直接比较。

## 2. Iteration-0 parity

`PASS`。初始 LQ tensor state 与 AC-5B1/AC4B 完全一致；AC4B 固定 64 anchors × 9 actions 的 6 个 ranking aggregate 指标差值均为 `0`，prediction 最大绝对差 `4.18e-05`。

AC-5B1 argmax-only 曲线直接读取既有 `artifacts/2026-09-20_ac5b1/learning_curves.json`，没有重跑。

## 3. AW9 action support

整体 train-row histogram（action index 0..8）：

| action | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| argmax count | 166 | 39 | 2,355 | 7,659 | 0 | 84 | 8,151 | 2,328 | 7,390 |
| argmax fraction | .0059 | .0014 | .0836 | .2719 | 0 | .0030 | .2893 | .0826 | .2623 |
| Full-Natural count | 339 | 131 | 4,931 | 14,118 | 0 | 199 | 15,858 | 4,569 | 14,639 |
| Full-Natural fraction | .0062 | .0024 | .0900 | .2577 | 0 | .0036 | .2895 | .0834 | .2672 |

Full-Natural 增加了 action 0、1、2、5、7、8 的绝对 support；但 action 4 在两套 training distribution 中都没有出现。因此 support 变宽是有限的，不是 9 个动作都被覆盖。

整体 entropy/effective action count：

| distribution | entropy (nats) | effective action count |
|---|---:|---:|
| argmax-only | 1.5341 | 4.6374 |
| Full-Natural | 1.5511 | 4.7168 |

phase-conditioned histogram 以 count 向量 `[a0,...,a8]` 表示：

| phase | argmax histogram；H / exp(H) | Full-Natural histogram；H / exp(H) |
|---|---|---|
| pursuing | `[24,0,317,411,0,0,355,833,407]`; 1.5795 / 4.8527 | `[64,5,627,795,0,3,856,1601,894]`; 1.6140 / 5.0229 |
| pre_capture_cover | `[111,0,361,854,0,0,1526,1242,1551]`; 1.5805 / 4.8574 | `[182,3,795,1640,0,4,3145,2404,3354]`; 1.5728 / 4.8201 |
| post_capture | `[30,39,1351,5072,0,71,4772,185,4232]`; 1.3938 / 4.0300 | `[92,121,2729,8916,0,177,8834,443,7984]`; 1.4340 / 4.1953 |
| pure_coverage | `[1,0,326,1322,0,13,1498,68,1200]`; 1.3566 / 3.8829 | `[1,2,780,2767,0,15,3023,121,2407]`; 1.3607 / 3.8987 |

AC4B 64 anchors 的描述性 global support：

- argmax-only：BC-selected action 平均 fraction `0.2391`；8 alternatives 平均 fraction `0.0951`。
- Full-Natural：BC-selected action 平均 fraction `0.2411`；8 alternatives 平均 fraction `0.0949`。

这说明 Full-Natural 的总体 action entropy 增加，但并没有使 AC4B anchor 的 BC action 与 alternatives support 关系发生大幅改变。

## 4. Fixed AC4B action-ranking

以下为两 seed 均值；两套条件均使用同一 64 anchors × 9 actions 与 empirical return matrix。

| updates | argmax-only Spearman | Full-Natural Spearman | argmax-only sign | Full-Natural sign | argmax-only regret | Full-Natural regret |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | .1776 | .1776 | .5938 | .5938 | 13.1967 | 13.1967 |
| 300 | .0212 | .1105 | .5039 | .5488 | 13.4736 | 13.1544 |
| 600 | -.0283 | .1358 | .4629 | .5508 | 14.1564 | 12.2445 |
| 900 | -.0514 | .1053 | .4551 | .5127 | 14.6701 | 12.7700 |
| 1200 | -.0346 | .0439 | .4736 | .4756 | 14.6852 | 14.0187 |

Full-Natural − argmax-only：

| updates | Δ Spearman | Δ sign | Δ regret |
|---:|---:|---:|---:|
| 0 | 0.0000 | 0.0000 | 0.0000 |
| 300 | +.0893 | +.0449 | -.3191 |
| 600 | +.1641 | +.0879 | -1.9119 |
| 900 | +.1568 | +.0576 | -1.9001 |
| 1200 | +.0785 | +.0020 | -.6664 |

Full-Natural 1200 updates 后仍是正 Spearman `0.0439`，而 argmax-only 为 `-0.0346`；但 Full-Natural 也没有保持 iteration-0 的 ranking，属于保护/减缓 collapse，不是完全消除 collapse。

## 5. MC test RMSE

| updates | Full-Natural complete test RMSE（8 episodes） | Full-Natural argmax-only test RMSE（4 episodes） |
|---:|---:|---:|
| 0 | 36.5944 | 29.8348 |
| 300 | 35.9357 | 28.8147 |
| 600 | 38.1331 | 30.9802 |
| 900 | 36.7881 | 28.2016 |
| 1200 | 36.9596 | 29.4519 |

两列是同一个 Full-Natural 模型在两个固定 test subset 上的误差；不能把 test distribution 差异误读成训练目标变化。

## 6. Counterfactual Q drift

相对 iteration 0，在 AC4B 9-action surface 上：

| updates | common-mode RMSE | relative-action RMSE | BC-action abs shift | 8 alternatives abs shift | alternative/BC |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 300 | 18.6957 | 1.3040 | 13.9421 | 13.4250 | .9628 |
| 600 | 19.4948 | 1.2797 | 13.7914 | 13.5069 | .9794 |
| 900 | 23.3072 | 1.5584 | 17.4645 | 17.1401 | .9801 |
| 1200 | 23.7552 | 1.8485 | 18.2774 | 17.7313 | .9684 |

AC-5B1 argmax-only final checkpoint 的 endpoint 对照为：common-mode RMSE `18.8718`、relative-action RMSE `2.1277`、BC-action shift `15.1092`、alternative shift `14.5254`、alternative/BC `0.9525`。

因此 Full-Natural 的 relative-action drift endpoint 小于 argmax-only endpoint；但 alternatives 的绝对 drift 没有高于 BC action，不能单独据此宣称存在强 unobserved-action drift。

## 7. Phase-conditioned ranking

1200 updates 的 Full-Natural vs AC-5B1 argmax-only：

| phase | Full Spearman / sign | argmax Spearman / sign | Δ Spearman / Δ sign |
|---|---:|---:|---:|
| pursuing | -.0307 / .4883 | -.0026 / .5547 | -.0281 / -.0664 |
| pre_capture_cover | .1109 / .4609 | -.0354 / .4082 | +.1464 / +.0527 |
| early_recovery | -.0620 / .4961 | -.2005 / .4102 | +.1385 / +.0859 |
| pure_coverage | .1573 / .4570 | .0602 / .4785 | +.0971 / -.0215 |

Full-Natural 保留了 AC-5B1 中 early-recovery ranking 的改善方向：early-recovery Spearman 从初始 `0.3021` 到最终 `-0.0620`，仍显著优于 argmax-only 的 `-0.2005`，但最终并未保持正相关。pursuing phase 没有改善，显示 action-support effect 明显 phase-dependent。

## 8. Verdict

`ACTION_SUPPORT_PROTECTS_RANKING`

依据：

- Full-Natural 相对 argmax-only 的 ranking Spearman 在 300/600/900/1200 均更高，最终高 `0.0785`。
- sign accuracy 的平均增益为 `+0.0481`；最终几乎持平（`+0.0020`）。
- Full-Natural endpoint relative-action drift `1.8485` 小于 argmax-only `2.1277`。
- 该结论只说明更宽的 behavior-policy action support 有助于保留 counterfactual ranking；不说明 sampled policy 适合 online RL，也不等同于 PPO/Actor-Critic 成功。
- 结果同时带有 `PHASE_DEPENDENT_ACTION_SUPPORT` 特征：pursuing phase 未获益，pre-capture/early-recovery/pure-coverage 获益不完全一致。

本轮没有运行 AC-5C，也没有加入 counterfactual supervised target。

## 9. Artifacts 与资源

- `artifacts/2026-09-20_ac5b2/summary.json`
- `artifacts/2026-09-20_ac5b2/learning_curves.json`
- `artifacts/2026-09-20_ac5b2/action_support.json`
- runner：`tools/run_ac5b2_action_support_ablation_20260920.py`
- 最终目录实际大小：`32,689,098 B`（约 31.2 MiB）；未复制 formal bank。
- 训练前 free disk：`93,719,363,584 B`；训练后 free disk：`93,661,028,352 B`。
- Actor optimizer steps：`0`。

本轮到此停止，等待主会话决定下一步。
