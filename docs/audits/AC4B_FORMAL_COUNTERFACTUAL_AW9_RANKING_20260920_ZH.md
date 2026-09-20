# AC-4B — Formal Counterfactual AW9 Action-Ranking Audit

日期：2026-09-20

范围：在 canonical strong Full-Task BC 的 fresh natural argmax rollout 上，收集 phase-balanced anchors，并对 focal agent 的 9 个 AW9 actions 做 full-continuation empirical return branching。全程无 actor/critic training、bootstrap、GAE、PPO 或 optimizer update。

## Verdict

- LQ：`LQ_RANKING_CONFIRMED`（总体方向为正，但 phase-dependent）。
- NQ：`NQ_RANKING_CONFIRMED`（总体方向为正，但 phase-dependent）。
- CQ：`WEAK_RANKING_SIGNAL`；未判定为 action-ranking winner。
- AC4A→AC4B：`AC4A_SIGNAL_DIRECTION_REPLICATED`。LQ/NQ 的正向 Spearman 与 BC sign direction 在更大集合上仍存在；CQ 仍弱。
- 本轮不进入 AC-5/bootstrap。

## 1. Frozen contract and checkpoints

- canonical actor requested path：`artifacts/2026-09-08_forward_final/c2_distillation/actor_epoch_030.pt`
- actual loaded path：`artifacts/2026-09-17_critic_identifiability_audit/frozen_policy/actor_epoch_030.pt`
- actor SHA256：`7916a970226a0452a8f71a22235524f6ba4511aa4c9bda907a9c8368a60bf2cd`
- environment：`forward-final-aw9-4v1-swept-v1`
- AW9：`0..8 = [(a,w) for a in (-0.4,0,0.4) for w in (-pi/6,0,pi/6)]`
- return：focal individual reward，`gamma=0.99`，full continuation，无 bootstrap/V/Q tail。
- branch：当前 step 只强制 focal action；teammates 使用 anchor BC argmax；从 `t+1` 起所有 agent 恢复 canonical BC argmax。

固定使用 AC-4A 的 validation-selected Natural critic：

| critic | seed | checkpoint | SHA256 |
|---|---:|---|---|
| LQ | 2026091711 | `artifacts/2026-09-20_ac3a/checkpoints/lq_natural_seed2026091711.pt` | `e8b52a32040937c2a4060e359546d1a427aee6c955dcaec501827038fac27552` |
| NQ | 2026091711 | `artifacts/2026-09-20_ac3a/checkpoints/nq_natural_seed2026091711.pt` | `53a194b17d9e963c5a63fce137d44725df19dc59420b624a5657c87d290a5052` |
| CQ | 2026091711 | `artifacts/2026-09-20_ac3b/checkpoints/cq_natural_seed2026091711.pt` | `361ace40555e478ec5ad9eb6e5cf5f8b7daafca1fb11daed49ec67e5b63db4e0` |

## 2. Snapshot gate

- fresh states：5
- 每个 state：同一首步 action + 5 个 continuation steps
- observation max absolute error：`0.0`
- reward max absolute error：`0.0`
- event / phase / done / truncation：全部一致
- gate：`PASS`

## 3. Anchor composition and independence

使用 AC4A seed base 之外的新固定 seed manifest；anchor 只来自 natural canonical BC argmax rollout，不使用 random action、policy noise、epsilon exploration 或 adversarial spawn。

| semantic class | anchors | unique episodes | max same-class / episode | minimum same-class step gap |
|---|---:|---:|---:|---:|
| pursuing | 16 | 8 | 2 | 10 |
| pre_capture_cover | 16 | 8 | 2 | 10 |
| early_recovery | 16 | 8 | 2 | 10 |
| recovery_pure / pure coverage | 16 | 8 | 2 | 10 |
| total | 64 | 16 | 6 total / episode | — |

总 branches：`64 × 9 = 576`。没有保存 simulator snapshot 或完整 trajectory；anchor metadata 与 9-action outcome summary 保存在小型 manifest 中。

## 4. Empirical action separation

| quantity | mean | median | SD | IQR |
|---|---:|---:|---:|---:|
| best − second-best | 2.862 | 0.491 | 4.359 | 4.600 |
| best − worst | 34.253 | 8.643 | 38.058 | 59.365 |
| return SD across 9 actions | 11.353 | 2.741 | 12.824 | 20.120 |

`best − second-best <= 1e-3` 的 near-tie anchors：`5/64 = 7.81%`。

重复相同 snapshot/action 的 12 次 numerical-noise check 最大 return 差异为 `0.0`；因此在 critic inference 前固定 margin tolerance 为 `1e-3`。margin-filtered sign 只计入 empirical alternative-vs-BC absolute return difference 大于该 tolerance 的 pairs。

## 5. Canonical BC empirical action quality

| metric | value |
|---|---:|
| empirical top-1 | 21.875% (`14/64`) |
| empirical top-3 | 46.875% (`30/64`) |
| empirical rank mean / median / IQR | 4.203 / 4.000 / 4.000 |
| regret mean / median / SD / IQR | 13.313 / 3.730 / 18.411 / 22.213 |

这只是 strong BC 附近的局部 action-quality 描述，不是对 BC 全局质量的否定。

## 6. Overall critic ranking

top-1 为 exact empirical-best agreement；其余指标为 anchor-level mean，`±` 后为 SD；Spearman、regret、sign 同时给出 median/IQR。

| critic | top-1 | top-3 overlap | Spearman | regret | raw sign vs BC | margin-filtered sign |
|---|---:|---:|---:|---:|---:|---:|
| LQ | 18.75% (`12/64`) | .432 ± .299 | .178 ± .473; med .217, IQR .696 | 13.197 ± 18.677; med .863, IQR 21.796 | .594 ± .282; med .625, IQR .500 | .595 ± .283 |
| NQ | 17.19% (`11/64`) | .427 ± .279 | .179 ± .454; med .200, IQR .658 | 11.702 ± 20.374; med 1.225, IQR 15.771 | .592 ± .249; med .625, IQR .375 | .593 ± .251 |
| CQ | 7.81% (`5/64`) | .344 ± .263 | .079 ± .375; med .033, IQR .575 | 11.008 ± 17.424; med 1.285, IQR 12.817 | .490 ± .235; med .500, IQR .281 | .493 ± .238 |

## 7. Phase-conditioned results

每个 phase `n=16`；单元格格式为 `top1 / top3 / Spearman / regret / raw sign`。

| phase | BC rank / regret mean | LQ | NQ | CQ |
|---|---:|---|---|---|
| pursuing | 4.250 / 26.877 | .063 / .313 / .035 / 31.430 / .578 | .063 / .375 / .078 / 25.344 / .617 | .063 / .292 / -.009 / 23.469 / .531 |
| pre_capture_cover | 5.938 / 23.689 | .188 / .396 / .068 / 19.844 / .430 | .250 / .458 / .199 / 19.366 / .531 | .250 / .417 / .181 / 18.372 / .461 |
| early_recovery | 2.813 / 2.293 | .313 / .500 / .302 / 1.159 / .773 | .125 / .417 / .176 / 1.598 / .633 | .000 / .375 / .215 / 1.697 / .531 |
| pure coverage | 3.813 / .392 | .188 / .521 / .305 / .354 / .594 | .250 / .458 / .264 / .498 / .586 | .000 / .292 / -.069 / .495 / .438 |

结果具有明显 phase dependence；不据此做 architecture winner 判定。

## 8. Branch outcomes

| outcome | count | denominator / scope |
|---|---:|---|
| collision | 1 | 576 |
| unsafe incomplete | 1 | 576 |
| capture failure | 1 | 432 mixed branches |
| CE failure | 0 | mixed captured or pure coverage |
| extreme delay (`>=90%` horizon) | 0 | 576 |
| catastrophic unique | 1 | 576 |

唯一 catastrophic branch：anchor `27`，`pre_capture_cover`，forced action `0`，boundary collision，未 capture、未 safe complete，length `104`，empirical return `-2.298`。该 branch 保留在 manifest/summary 中，没有删除或过滤。

## 9. AC4A → AC4B replication

| critic | AC4A Spearman / sign | AC4B Spearman / sign | status |
|---|---:|---:|---|
| LQ | .364 / .734 | .178 / .594 | positive direction replicated |
| NQ | .265 / .680 | .179 / .592 | positive direction replicated |
| CQ | .011 / .453 | .079 / .490 | signal remains weak |

formal replication status：`AC4A_SIGNAL_DIRECTION_REPLICATED`。AC4B 没有把 AC4A 的 LQ/NQ 正向 signal 完全抹掉，但 magnitude 下降，且 phase-specific 结果不均匀。

## 10. Immutability and artifacts

- actor optimizer steps：`0`
- critic optimizer steps：`0`
- actor state hash before/after identical：`d626fe2cd19b1cc40fadabf85fdbff8d8e151969c707d9a777f081d5b893cfa0`
- LQ/NQ/CQ state hashes before/after：全部 identical
- bootstrap / GAE / PPO / TD3 / MADDPG / SAC：均未执行
- contract/reward：未修改
- output directory：约 `3.33 MiB`；未复制 checkpoint、bank 或完整 trajectory

Artifacts：

- `artifacts/2026-09-20_ac4b/summary.json`
- `artifacts/2026-09-20_ac4b/anchor_manifest.json`
- `artifacts/2026-09-20_ac4b/counterfactual_returns.npz`
- `artifacts/2026-09-20_ac4b/seed_manifest.json`
- runner：`tools/run_ac4b_formal_counterfactual_aw9_20260920.py`

本轮完成 AC-4B 后停止，下一步是否进入 bootstrap 或保留多 critic 由主会话决定。
