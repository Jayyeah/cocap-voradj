# AC-5A — Bellman Bootstrap Target Audit

日期：2026-09-20

范围：在 AC-4B 已完成的同一组 64 anchors × 9 AW9 branches 上，重新执行每条 branch 的 forced first step，使用真实 counterfactual next-state 计算 frozen BC next action，并比较 direct Q 与 one-step Bellman bootstrap。无任何网络训练。

## Verdict

- LQ：`BOOTSTRAP_SIGNAL_DEGRADED`；`ON_POLICY_TAIL_WEAK`；`NO_CLEAR_EXTRAPOLATION_GAP`。
- NQ：`BOOTSTRAP_SIGNAL_DEGRADED`；`ON_POLICY_TAIL_WEAK`；`NO_CLEAR_EXTRAPOLATION_GAP`。
- 两者 bootstrap 后 primary ranking 指标均下降；LQ/NQ 的 alternative tail error 没有高于 BC-action tail error，因此本轮没有标记 `COUNTERFACTUAL_NEXT_STATE_EXTRAPOLATION`。
- 本轮不进入 AC-5B，不训练 FQE/TD 或 target network。

## 1. Frozen inputs and replay identity

- canonical actor SHA256：`7916a970226a0452a8f71a22235524f6ba4511aa4c9bda907a9c8368a60bf2cd`
- environment contract：`forward-final-aw9-4v1-swept-v1`
- source anchors：AC-4B `64` anchors，组成 `16/16/16/16`
- source empirical matrix：`artifacts/2026-09-20_ac4b/counterfactual_returns.npz`
- branch partition：`64` BC-action branches，`512` alternative branches，合计 `576`
- AW9 mapping 与 AC-4B 一致；直接 Q 复算与 AC-4B summary 的所有 ranking metrics 最大差异为 `0.0`。

AC-4B 未把 simulator snapshot 落盘；本轮按同一 manifest 的 scene、seed、episode step、focal agent deterministic replay 重建内存 snapshot。结果：

- anchor observation 最大误差：`0.0`
- action mismatches：`0`
- semantic mismatches：`0`
- 16 个 replay episode 全部找到预注册 anchors

## 2. Snapshot gate and first-step outcomes

- snapshot gate：`PASS`
- checked states：`5`
- 每个 state：首步 + 5 个 continuation steps
- observation/reward 最大误差：均为 `0.0`
- event / done / truncation：一致
- 首步 terminal branches：`0/576`

每条 branch 只执行首步并保存 immediate reward、next observation、next neighbor relation、next BC action 和 terminal flags；没有保存完整 continuation trajectory。

## 3. Direct Q versus one-step bootstrap ranking

格式：mean；括号内为 bootstrap − direct。top-1 为 fraction。

| critic | method | top-1 | top-3 | Spearman | regret | raw sign | margin sign |
|---|---|---:|---:|---:|---:|---:|---:|
| LQ | Direct Q | .188 | .432 | .178 | 13.197 | .594 | .595 |
| LQ | Bootstrap | .188 | .344 | .009 | 11.191 | .506 | .502 |
| LQ | Δ | 0.000 | -.089 | -.168 | -2.006 | -.088 | -.094 |
| NQ | Direct Q | .172 | .427 | .179 | 11.702 | .592 | .593 |
| NQ | Bootstrap | .141 | .333 | -.054 | 14.587 | .438 | .435 |
| NQ | Δ | -.031 | -.094 | -.233 | +2.885 | -.154 | -.158 |

LQ 的 regret 数值下降并未抵消 top-3、Spearman 和 sign 的下降；因此 primary ranking signal 判为 degraded。NQ 的所有主要 ranking 方向均更差。

## 4. Return prediction error against empirical full return

误差 bias 定义为 prediction − empirical。

| critic | target | RMSE | MAE | bias | Pearson | Spearman |
|---|---|---:|---:|---:|---:|---:|
| LQ | Direct Q | 41.080 | 31.628 | -6.744 | .782 | .685 |
| LQ | Bootstrap | 40.563 | 31.245 | -5.395 | .785 | .719 |
| NQ | Direct Q | 42.905 | 34.230 | +1.264 | .745 | .698 |
| NQ | Bootstrap | 41.478 | 32.450 | +.964 | .765 | .702 |

Bootstrap 的 return-level RMSE/MAE 略有改善，但这不等价于 action-ranking signal 改善；ranking 结果见上表。

## 5. Tail error localization

tail target 对 nonterminal branch 定义为：

`G_tail_emp = (G_emp - r_t) / 0.99`

本轮无首步 terminal branch，因此 all、BC-action、alternative 分组分别为 `576/64/512`。

| critic | group | RMSE | MAE | bias | Pearson | Spearman |
|---|---|---:|---:|---:|---:|---:|
| LQ | all | 40.973 | 31.560 | -5.450 | .777 | .704 |
| LQ | BC-action | 41.635 | 31.849 | -6.122 | .791 | .724 |
| LQ | alternatives | 40.889 | 31.524 | -5.366 | .776 | .702 |
| NQ | all | 41.897 | 32.777 | +.974 | .757 | .693 |
| NQ | BC-action | 42.871 | 32.919 | -.536 | .763 | .681 |
| NQ | alternatives | 41.773 | 32.760 | +1.163 | .756 | .694 |

alternative/BC tail RMSE ratio：

| critic | ratio | alternative − BC RMSE |
|---|---:|---:|
| LQ | `.982` | `-0.746` |
| NQ | `.974` | `-1.098` |

因此没有证据表明 alternative next-state tail 比 BC-action tail 更差；本轮只记录 `NO_CLEAR_EXTRAPOLATION_GAP`，不宣称 online RL 已解决。

## 6. Phase-conditioned differences

单元格为 `direct Spearman → bootstrap Spearman；direct sign → bootstrap sign；tail RMSE BC-action / alternatives`。

| phase | LQ | NQ |
|---|---|---|
| pursuing | `.035 → -.085；.578 → .469；43.183 / 39.347` | `.078 → -.173；.617 → .406；45.452 / 40.761` |
| pre_capture_cover | `.068 → .004；.430 → .453；64.099 / 63.845` | `.199 → .030；.531 → .539；55.682 / 53.839` |
| early_recovery | `.302 → -.027；.773 → .555；15.323 / 16.300` | `.176 → -.023；.633 → .383；14.531 / 16.772` |
| pure coverage | `.305 → .146；.594 → .547；26.939 / 28.243` | `.264 → -.050；.586 → .422；44.431 / 46.245` |

Bootstrap degradation 主要出现在 pursuing、early recovery 与 pure coverage 的 ranking direction；pre-capture-cover 的 sign 变化较小，说明存在 phase dependence。

## 7. Immediate reward versus empirical tail contribution

下表在每个 phase 的 `144` branches 上统计 `Var(r_t)` 与 `Var(gamma * G_tail_emp)`，并给出绝对均值量级。

| phase | Var immediate | Var gamma-tail | mean |r| | mean |gamma-tail| |
|---|---:|---:|---:|---:|
| pursuing | .404 | 1117.251 | 1.727 | 122.972 |
| pre_capture_cover | .118 | 1986.988 | .296 | 61.729 |
| early_recovery | .356 | 47.577 | .756 | 11.291 |
| pure coverage | .376 | 80.003 | .808 | 13.874 |

在四个 phase 中，empirical continuation tail 的方差和绝对量级都明显高于 immediate reward；bootstrap ranking 的变化主要由 tail estimate 及其 action-dependent ordering 驱动，而非 immediate reward 单独决定。

## 8. Immutability and artifacts

- actor optimizer steps：`0`
- LQ optimizer steps：`0`
- NQ optimizer steps：`0`
- actor、LQ、NQ state hash before/after：全部 bit-exact
- critic training / iterative FQE / target-network training：均未执行
- output size：约 `311 KB`
- 磁盘 free：before `93,651,828,736 B`；after `93,646,630,912 B`

Artifacts：

- `artifacts/2026-09-20_ac5a/summary.json`
- `artifacts/2026-09-20_ac5a/branch_bootstrap_predictions.npz`
- runner：`tools/run_ac5a_bellman_bootstrap_audit_20260920.py`

本轮 AC-5A 完成后停止，是否进入 AC-5B 由主会话决定。
