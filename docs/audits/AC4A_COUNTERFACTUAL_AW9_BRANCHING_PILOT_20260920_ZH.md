# AC-4A — Counterfactual AW9 Branching Pilot

日期：2026-09-20
范围：canonical strong Full-Task BC；16 anchors；每个 anchor 只改变 focal agent 的当前 AW9 action，之后所有 agent 使用 frozen BC argmax continuation。无 critic training、bootstrap、GAE、PPO、BC training 或 actor update。

## Verdict

`COUNTERFACTUAL_PIPELINE_PASS`

描述性 pilot labels：

- `LQ_RANKING_SIGNAL_PRESENT`
- `NQ_RANKING_SIGNAL_PRESENT`
- `NO_CLEAR_RANKING_SIGNAL`（CQ overall）

这些不是 architecture winner，也不代表 actor-update usability。

## 1. Contract and frozen inputs

- canonical BC actor：`artifacts/2026-09-17_critic_identifiability_audit/frozen_policy/actor_epoch_030.pt`
- canonical requested path：`artifacts/2026-09-08_forward_final/c2_distillation/actor_epoch_030.pt`
- actor SHA256：`7916a970226a0452a8f71a22235524f6ba4511aa4c9bda907a9c8368a60bf2cd`
- environment contract：`forward-final-aw9-4v1-swept-v1`
- return：focal individual reward full continuation，`gamma=0.99`；无 TD/bootstrap/V tail/learned-Q tail。
- branch：anchor 当前 step 强制一个 focal action；teammate 当前 BC actions 固定；从下一 step 起所有 agent 回到 deterministic BC argmax。
- AW9 mapping：`0..8 = [(a,w) for a in (-0.4,0,0.4) for w in (-pi/6,0,pi/6)]`，runtime match。

加载的 validation-selected checkpoint 为各 Natural 两 seed 中 validation MC-RMSE 最优者：

| critic | seed | validation RMSE | checkpoint SHA256 |
|---|---:|---:|---|
| LQ | 2026091711 | 35.378 | `e8b52a32040937c2a4060e359546d1a427aee6c955dcaec501827038fac27552` |
| NQ | 2026091711 | 34.640 | `53a194b17d9e963c5a63fce137d44725df19dc59420b624a5657c87d290a5052` |
| CQ | 2026091711 | 30.927 | `361ace40555e478ec5ad9eb6e5cf5f8b7daafca1fb11daed49ec67e5b63db4e0` |

## 2. Snapshot / restore gate

实现为内存 snapshot，不写 simulator snapshot 文件。snapshot 包含：deep-copied `VorAdjEnv`（包括 entity/runtime/counters/cache/collision bookkeeping/env-owned RNG）、observations，以及 Python/NumPy/Torch RNG state。

- states checked：10
- 每个 state：同一 action 首步 + 5 个 canonical BC continuation steps
- observation max absolute error：`0.0`
- reward max absolute error：`0.0`
- done/truncation/event/phase：全部一致
- gate：`PASS`

## 3. Anchors and branches

| semantic class | anchors |
|---|---:|
| pursuing | 4 |
| pre_capture_cover | 4 |
| early recovery | 4 |
| pure coverage / recovery_pure | 4 |
| total | 16 |

总 branches：`16 × 9 = 144`。anchor 来自 fresh canonical BC argmax rollout；没有随机动作、噪声 actor 或环境难度改变。完整 branch trajectory 未落盘。

## 4. BC action empirical quality

empirical target 是同一 anchor 上 9 个 full-continuation return。

- empirical rank=1：`6/16 = 37.5%`
- empirical top-3：`9/16 = 56.25%`
- mean BC regret：`2.843`

该小样本表明 BC action 常在 empirical top-3，但并非普遍 empirical rank-1；不能从 16 anchors 推断 strong BC 的全局 action optimality。

## 5. Critic ranking smoke（mean ± anchor SD）

| critic | top-1 agreement | top-3 overlap | Spearman | regret | BC empirical rank | sign accuracy vs BC |
|---|---:|---:|---:|---:|---:|---:|
| LQ | 37.5% | 0.583 ± .300 | .364 ± .429 | 4.674 ± 12.063 | 3.625 ± 2.713 | .734 ± .242 |
| NQ | 31.25% | 0.500 ± .354 | .265 ± .566 | 10.218 ± 17.633 | 3.625 ± 2.713 | .680 ± .316 |
| CQ | 6.25% | 0.229 ± .227 | .011 ± .298 | 13.330 ± 19.958 | 3.625 ± 2.713 | .453 ± .229 |

LQ 在本 pilot 的总体 top-1、top-3、Spearman、regret 和 sign accuracy 最好；NQ 仍有正向 ranking signal；CQ overall 接近无序/较弱，暂记 `NO_CLEAR_RANKING_SIGNAL`。这些结果不是显著性检验，也不是最终 winner 判定。

## 6. Phase-conditioned trends

每个 phase 只有 4 anchors，不做显著性检验。

| phase | critic | top-1 | top-3 | Spearman | regret | sign accuracy |
|---|---|---:|---:|---:|---:|---:|
| pursuing | LQ | 1.00 | .750 | .567 | 0.000 | 1.000 |
| pursuing | NQ | .500 | .500 | .538 | 26.770 | .938 |
| pursuing | CQ | 0 | .167 | -.092 | 45.292 | .219 |
| pre-capture-cover | LQ | 0 | .417 | .250 | 17.180 | .594 |
| pre-capture-cover | NQ | 0 | .333 | -.038 | 12.292 | .500 |
| pre-capture-cover | CQ | .250 | .333 | .217 | 5.884 | .531 |
| early recovery | LQ | 0 | .417 | -.033 | 1.408 | .531 |
| early recovery | NQ | .250 | .333 | -.150 | 1.474 | .406 |
| early recovery | CQ | 0 | .333 | -.008 | 1.257 | .625 |
| pure coverage | LQ | .500 | .750 | .671 | .109 | .813 |
| pure coverage | NQ | .500 | .833 | .708 | .338 | .875 |
| pure coverage | CQ | 0 | .083 | -.071 | .888 | .438 |

Phase differences are mixed: LQ is strongest on pursuing and close to strongest on pure coverage; NQ has useful pure-coverage ranking and positive overall signal; CQ does not show a stable action-ranking signal in this pilot.

## 7. Branch anomaly and immutability checks

- catastrophic branches：`0/144`（collision 或 unsafe/incomplete branch 均为 0）。
- returns and critic predictions：仅保存 `counterfactual_returns.npz` 的小矩阵，不保存 144 条 trajectory。
- actor state hash before/after identical：`d626fe2c…893cfa0`
- actor updates：`0`
- critic optimizer steps：`0`
- output directory actual size：`671,466 bytes`（约 `0.64 MiB`）。
- 既有 critic checkpoint、formal bank、simulator snapshot 均未复制或提交。

## 8. Artifacts

- `artifacts/2026-09-20_ac4a/summary.json`
- `artifacts/2026-09-20_ac4a/anchor_manifest.json`
- `artifacts/2026-09-20_ac4a/counterfactual_returns.npz`
- runner：`tools/run_ac4a_counterfactual_aw9_pilot_20260920.py`

本轮只确认 counterfactual snapshot/branching methodology 可运行且输出可解释 pilot signal；不进入 AC-4B，不据此启动 actor update 或 PPO。
