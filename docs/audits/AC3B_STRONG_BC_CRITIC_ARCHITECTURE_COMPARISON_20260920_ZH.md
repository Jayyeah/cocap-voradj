# AC-3B — Canonical Strong-BC Critic Architecture Comparison

日期：2026-09-20
范围：canonical strong Full-Task BC bank；Natural sampler；只比较 MC-supervised LQ/NQ/CQ/V 的 realized-return prediction。未做 bootstrap、TD/GAE、PPO、counterfactual action ranking 或 actor update。

## Verdict

- `LOCAL_STILL_COMPETITIVE`
- `MIXED_PHASE_DEPENDENT`

这是 realized-return fitting / information sufficiency 的描述性结论，不是 actor-update usability 结论。

## 1. 固定数据与训练合同

- bank：`artifacts/2026-09-20_ac2b/canonical_bc_critic_bank_v2.npz`
- bank SHA256：`07e08d17ae6283d785e0d2e00307f330b13a88dcd0f8ce83f92f0d451ad7808a`
- 80 episodes；train/validation/test = 64/8/8 episodes；按 episode split，不按 transition 重切分。
- test 是同一组 8 个 held-out Natural episodes；所有 critic 的目标都是同一 individual-agent full-episode MC return，`gamma=0.99`，无 bootstrap。
- train-only target normalization：mean `15.788457870483398`，std `52.9477653503418`，fit 后冻结。
- Natural sampler：所有 train active-agent rows 上均匀、有放回采样；batch `256`，`1200` updates，validation 每 `100` updates；seed `2026091711/2026091712`。
- Adam，LR `1e-4`，weight decay `1e-6`，grad clip `0.5`；best checkpoint 只按 validation MC-RMSE 选择。

CQ/V 严格复用旧 A1：CQ 为 centralized entity state + joint AW9 action，按 focal agent 取 `Q_i`；V 为 action-free centralized state，按 focal agent 取 `V_i`。CQ 参数量 `3,238,145`，V 参数量 `3,236,865`。两者 forward smoke 均 finite；optimizer steps 仅发生在本轮 CQ/V fit，actor updates = 0。

LQ/NQ 数值直接读取 AC-3A Natural artifact，未重训、未覆盖：`artifacts/2026-09-20_ac3a/summary.json`，其 SHA256 由 AC-3B summary 记录。

## 2. Consolidated Natural held-out test

以下均为 `mean ± seed SD`（两 seed）；RMSE/MAE 越低越好，EV/Pearson/Spearman 越高越好，ECE 越低越好。

| critic | input | params | RMSE | MAE | EV | Pearson | Spearman | slope | ECE | early-rec RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LQ | local | 4,030,225 | 36.69 ± 0.10 | 22.66 ± 0.50 | .472 ± .005 | .688 ± .004 | .505 ± .010 | .974 ± .033 | 4.89 ± .56 | 20.05 ± .12 |
| NQ | local + neighbor | 4,228,113 | 37.00 ± 0.18 | 23.74 ± 0.92 | .467 ± .001 | .687 ± .001 | .471 ± .010 | .907 ± .005 | 5.73 ± 1.23 | 18.45 ± 2.37 |
| CQ | central + joint action | 3,238,145 | 41.68 ± 2.03 | 20.69 ± .02 | .315 ± .067 | .593 ± .051 | .543 ± .018 | .749 ± .032 | 7.58 ± .07 | 21.11 ± 1.56 |
| V | central state | 3,236,865 | 40.82 ± 1.09 | 21.27 ± .24 | .346 ± .037 | .614 ± .031 | .553 ± .090 | .774 ± .006 | 8.09 ± 1.23 | 19.99 ± 4.43 |

这里 slope 是 target-on-prediction calibration slope；intercept 另列于 summary 和下表。

## 3. Early recovery 与 calibration

Early recovery 共 `n=960` rows。表中 bias = prediction mean − target mean；括号为 seed SD。

| critic | RMSE | MAE | bias | calibration slope | intercept | ECE |
|---|---:|---:|---:|---:|---:|---:|
| LQ | 20.05 ± .12 | 13.50 ± .85 | +12.32 ± .45 | .946 ± .156 | -12.24 ± .39 | 12.32 ± .45 |
| NQ | 18.45 ± 2.37 | 12.34 ± 2.60 | +6.54 ± 3.82 | .643 ± .048 | -8.35 ± 2.73 | 8.39 ± 3.24 |
| CQ | 21.11 ± 1.56 | 10.43 ± .06 | +5.76 ± 1.32 | .771 ± .349 | -6.76 ± 3.24 | 6.07 ± 1.01 |
| V | 19.99 ± 4.43 | 10.17 ± 1.09 | +1.92 ± 3.86 | .783 ± .593 | -1.84 ± 9.07 | 6.01 ± .23 |

Interpretation is phase-specific: NQ has the lowest early-recovery RMSE; V has the smallest mean bias and V/CQ have lower early-recovery ECE, but their seed spread and calibration intercept spread are larger. This does not establish counterfactual action ranking.

## 4. Phase/event-conditioned RMSE

All rows are from the same Natural test split; `n` is shown explicitly.

| phase/event | n | LQ | NQ | CQ | V | best observed |
|---|---:|---:|---:|---:|---:|---|
| early recovery | 960 | 20.05 ± .12 | **18.45 ± 2.37** | 21.11 ± 1.56 | 19.99 ± 4.43 | NQ |
| late recovery | 2,428 | 10.98 ± 1.06 | 14.70 ± .78 | **3.40 ± .55** | 5.17 ± .14 | CQ |
| pure coverage | 1,164 | 12.73 ± 1.35 | 16.77 ± 2.27 | **3.53 ± .10** | 5.21 ± .29 | CQ |
| ring2 | 328 | 56.81 ± 1.80 | **53.62 ± .43** | 65.51 ± 2.32 | 62.43 ± 3.72 | NQ |
| ring3 / capture | 24 | 66.69 ± .29 | 68.69 ± 6.55 | 69.19 ± 4.20 | **64.45 ± 4.02** | V |

Semantic-class RMSE shows the same split: pursuing `65.93/65.16/85.17/78.98` for LQ/NQ/CQ/V; pre-capture-cover `60.76/60.16/67.18/67.81`; post-capture-real `14.17/15.86/11.62/11.55`; recovery-pure `12.73/16.77/3.53/5.21`.

## 5. Relative-to-LQ deltas

Deltas use test means; RMSE/ECE percentage/absolute signs are interpreted as lower-is-better, EV/Spearman as higher-is-better.

| critic | ΔRMSE % | ΔEV | ΔSpearman | ΔECE |
|---|---:|---:|---:|---:|
| LQ | 0.00% | 0.000 | 0.000 | 0.00 |
| NQ | +0.86% | -0.005 | -0.034 | +0.85 |
| CQ | +13.59% | -0.157 | +0.038 | +2.70 |
| V | +11.26% | -0.127 | +0.049 | +3.20 |

Overall natural-test fitting 因而没有显示 central context 对 realized-return RMSE/EV/calibration 的统一改善；NQ 的主要收益集中在 early recovery 与 ring2，CQ/V 的优势集中在 late/pure recovery 及 V 的 ring3/capture 小样本。phase 最优者不同，所以保留 `MIXED_PHASE_DEPENDENT`，同时 LQ overall 仍具竞争力，保留 `LOCAL_STILL_COMPETITIVE`。

## 6. Artifact / forbidden checks

- 新增 runner：`tools/run_ac3b_natural_matched_critic_20260920.py`。
- 新增结果：`artifacts/2026-09-20_ac3b/summary.json`。
- 本地 checkpoint 仅用于本轮结果复核，不提交；未复制 bank。
- `LQ_retrained=false`，`NQ_retrained=false`，bootstrap/GAE/PPO/counterfactual action ranking 均 false，actor updates = 0。
- AC-3B output directory actual size：`51,916,489` bytes（约 `49.5 MiB`，含四个本地 checkpoint、summary、progress）；不包含既有 bank。

## 7. 限定结论

本轮只证明四种架构在 canonical strong-BC 行为分布上对 realized full-episode MC return 的拟合差异。它没有证明任何 critic 能对未执行 AW9 action 做正确排序，也没有证明可直接用于 actor update；这些问题留给后续阶段。
