# AC-5B1 — MC continuation vs iterative Bellman LQ

日期：2026-09-20
范围：只做 LQ critic；不做 NQ/CQ/V、Actor、PPO、bootstrap actor update，也没有生成新的 trajectory bank。

## 1. Identity 与数据合同

- formal bank：`artifacts/2026-09-20_ac2b/canonical_bc_critic_bank_v2.npz`
- bank SHA256：`07e08d17ae6283d785e0d2e00307f330b13a88dcd0f8ce83f92f0d451ad7808a`
- 只取 bank 中 `policy_mode_id=bc_argmax` 的 episode；没有把 sampled policy 混入训练。
- episode split 保持 AC-2B 的 episode-level split，不按 transition 重分：

| split | 总 episode | mixed | pure coverage | active-agent rows |
|---|---:|---:|---:|---:|
| train | 32 | 24 | 8 | 28,172 |
| validation | 4 | 3 | 1 | 2,768 |
| test | 4 | 3 | 1 | 3,564 |

训练只使用 train rows；test 是固定 argmax held-out rows。sampler 是自然 active-row uniform sampling with replacement，batch size 256。

共同初始 LQ checkpoint：

`artifacts/2026-09-20_ac3a/checkpoints/lq_natural_seed2026091711.pt`

- checkpoint SHA256：`e8b52a32040937c2a4060e359546d1a427aee6c955dcaec501827038fac27552`
- tensor state SHA256：`825b11e6753a9eb541dd265b087b74e4ed786fbc0430d04ab6fbd9802900eada`
- 四个 arm 均从该同一 state 开始。
- 固定 normalization：mean `15.788457870483398`，std `52.9477653503418`。
- architecture：historical A1 LocalQ，LegacyVorAdjFeatureBackbone，hidden 256、8 heads、4 layers、dropout 0。
- optimizer：Adam，lr `1e-4`，weight decay `1e-6`，grad clip `0.5`；每 arm 4 个 outer blocks、每 block 300 updates，共 1200 updates。
- reward/return：individual per-agent reward，`gamma=0.99`。

两个 target 合同：

1. `mc_control`：固定使用完整 episode 的 realized individual MC return。
2. `iterative_bellman`：每个 outer block 开始复制前一 block 的 online Q 为 frozen target，更新 300 次；target 为
   `r + 0.99 * (1-done) * Q_target(next local observation, frozen canonical BC argmax action)`。

## 2. Iteration-0 parity

`PASS`。初始 state 与 AC4B LQ state SHA 完全相同。固定 AC4B 64 anchors × 9 actions 的 prediction 最大绝对差为 `4.18e-05`；6 个 ranking aggregate 指标差值均为 `0`，因此后续 ranking 曲线可以直接与 AC4B 固定分支比较。

iteration 0 的 test MC metrics：RMSE `29.8348`，MAE `19.3550`，EV `0.6148`，Pearson `0.7848`，Spearman `0.5092`，ECE/decile calibration error `3.1082`。

## 3. Test MC metrics 与固定 action-ranking

下表是两个 seed 的均值；ranking 使用 AC4B 已存在的 64 anchors/576 branches，没有重新分支。

| updates | MC RMSE | MC rank Spearman | MC sign | MC regret | FQE RMSE | FQE rank Spearman | FQE sign | FQE regret |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 29.8348 | 0.1776 | 0.5938 | 13.1967 | 29.8348 | 0.1776 | 0.5938 | 13.1967 |
| 300 | 27.7948 | 0.0212 | 0.5039 | 13.4736 | 28.1862 | 0.0478 | 0.5020 | 14.2068 |
| 600 | 27.8351 | -0.0283 | 0.4629 | 14.1564 | 28.2584 | -0.0366 | 0.4678 | 13.0912 |
| 900 | 27.7488 | -0.0514 | 0.4551 | 14.6701 | 27.5607 | -0.0514 | 0.4424 | 13.5755 |
| 1200 | 28.0380 | -0.0346 | 0.4736 | 14.6852 | 27.9627 | -0.0546 | 0.4521 | 14.1354 |

FQE − MC paired delta：

| updates | ΔRMSE | Δrank Spearman | Δsign | Δregret |
|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 300 | +0.3915 | +0.0266 | -0.0020 | +0.7332 |
| 600 | +0.4233 | -0.0083 | +0.0049 | -1.0651 |
| 900 | -0.1881 | 0.0000 | -0.0127 | -1.0946 |
| 1200 | -0.0752 | -0.0199 | -0.0215 | -0.5498 |

## 4. Early-recovery phase curve

test early-recovery 固定为 480 rows；数值为两 seed 均值，括号内为 `Spearman`。

| updates | MC RMSE (Spearman) | iterative Bellman RMSE (Spearman) |
|---:|---:|---:|
| 0 | 12.7880 (0.4207) | 12.7880 (0.4207) |
| 300 | 12.8667 (0.5851) | 14.1162 (0.4257) |
| 600 | 16.9523 (0.6021) | 13.1026 (0.4339) |
| 900 | 16.5154 (0.6929) | 13.3069 (0.4503) |
| 1200 | 19.3740 (0.6371) | 13.3430 (0.4601) |

## 5. Bellman target drift

以下是 iterative Bellman target 在每个 outer block 使用的 `y_k`，在固定 train argmax rows 上相对 realized `G_MC` 的 seed 均值；第一个 block 没有前一 target，因此 `Δy` 不适用。

| online updates | `y_k-G_MC` RMSE | bias | Pearson | target std | `y_k-y_{k-1}` RMSE |
|---:|---:|---:|---:|---:|---:|
| 300 | 26.0237 | +4.0469 | 0.8464 | 37.5333 | — |
| 600 | 26.2394 | +6.0148 | 0.8496 | 37.2200 | 6.3533 |
| 900 | 26.1091 | +5.1188 | 0.8498 | 36.6316 | 4.1795 |
| 1200 | 26.1023 | +4.9683 | 0.8499 | 36.4228 | 3.4088 |

结论：Bellman target 没有发散，且 target drift 随 outer block 减小；但它相对 realized MC 仍有约 26 的 RMSE 和正 bias，不能把 iterative Bellman 解释为 MC target 的等价替代。

## 6. Q scale drift

固定 test argmax rows 上的 physical-Q 统计（两 seed 均值）：

| updates | arm | mean | std | min | max |
|---:|---|---:|---:|---:|---:|
| 0 | both | 21.7997 | 36.0329 | -55.6201 | 161.4626 |
| 300 | MC | 16.2647 | 37.8773 | -70.2424 | 168.4780 |
| 300 | FQE | 24.2192 | 35.7116 | -57.8613 | 156.8615 |
| 600 | MC | 16.7593 | 42.6628 | -92.9154 | 186.0736 |
| 600 | FQE | 23.6308 | 35.6042 | -50.1419 | 152.9812 |
| 900 | MC | 18.2940 | 42.5619 | -82.0971 | 187.3424 |
| 900 | FQE | 23.2891 | 35.4118 | -56.2477 | 153.8608 |
| 1200 | MC | 18.1590 | 42.4907 | -86.8107 | 187.4570 |
| 1200 | FQE | 23.9095 | 36.0618 | -51.7701 | 155.9865 |

## 7. Verdict 与资源边界

- `MC_CONTROL_STABLE`：所有曲线 finite；最终 test RMSE 相对 iteration 0 为 `0.9398×`，Q std 为 `1.1792×`。
- `ITERATIVE_BOOTSTRAP_STABLE`：所有曲线 finite；最终 test RMSE 相对 iteration 0 为 `0.9373×`，Q std 为 `1.0008×`。此处 stable 只表示本轮数值稳定，不表示 target 已经等价于 MC，也不表示可直接用于 Actor update。
- Actor optimizer updates：`0`。
- NQ/CQ/V/PPO/GAE/TD3/MADDPG/SAC：`0`。
- 四个 final LQ checkpoint 只保存在本地 `artifacts/2026-09-20_ac5b1/checkpoints/`，未纳入提交；metrics/summary 总计约 0.8 MB。AC-5B1 输出目录实际约 `65,009,871` bytes（约 62.0 MiB）。运行记录的 free disk：before `94,147,776,512` bytes，after `93,911,846,912` bytes。

## 8. Artifacts

- summary：`artifacts/2026-09-20_ac5b1/summary.json`
- learning curves：`artifacts/2026-09-20_ac5b1/learning_curves.json`
- runner：`tools/run_ac5b1_mc_vs_iterative_bellman_lq_20260920.py`

本轮到此结束；不进入 AC-5B2 或 AC-5C。
