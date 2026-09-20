# AC-3A — Critic Sampling Contract Audit

日期：2026-09-20  
分支：`audit/ac-bc-critic-lineage-20260920`  
范围：仅 LQ / NQ；Natural vs IQN-reference phase-balanced；无 CQ/V、bootstrap、GAE、PPO 或 action-ranking 主实验。

## Verdict

- LQ：`NATURAL_PREFERRED`
- NQ：`SAMPLER_NEUTRAL`
- 统一建议：`NO_CLEAR_WINNER`
- `BALANCED_OVERSAMPLING_OVERFIT`：未触发预注册判据，但 Balanced 的 pursuing train error 大幅下降，LQ 的 natural test overall 与 calibration 反而变差。

## Fixed contract

使用 AC-2B formal bank：

- path：`artifacts/2026-09-20_ac2b/canonical_bc_critic_bank_v2.npz`
- SHA256：`07e08d17ae6283d785e0d2e00307f330b13a88dcd0f8ce83f92f0d451ad7808a`
- 16964 transitions、67856 active-agent rows；episode split `64/8/8`，test 始终为相同 natural held-out 8 episodes
- target：individual realized full-episode MC return，`gamma=0.99`，无 bootstrap
- A1 architecture：Legacy VorAdj backbone，hidden 256、4 layers、8 heads、dropout 0；Adam `1e-4`、weight decay `1e-6`、grad clip `0.5`、batch 256、1200 updates；train-only mean/std normalization，冻结；validation MC RMSE 选 best checkpoint
- seeds：`2026091711`、`2026091712`；每个 critic × sampler 两个 seed，共 8 fits

Balanced 保持 A1 的 batch 256：每个 batch 内使用两个独立 128-row quota blocks，每个 block 为 `[64,16,32,16]`，即实际 batch 配额 `[128,32,64,32]`，比例为 `50%/12.5%/25%/12.5%`。没有 ring3/capture 单独 quota，也没有修改 raw bank。

## Natural test overall（mean ± seed SD）

| critic / sampler | RMSE | MAE | EV | Pearson | Spearman | slope | intercept | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LQ Natural | 36.69±0.10 | 22.66±0.50 | .472±.005 | .688±.004 | .505±.010 | .974±.033 | -1.86±.25 | 4.89±.56 |
| LQ Balanced | 37.96±0.38 | 21.50±0.62 | .433±.012 | .667±.006 | .443±.015 | .862±.019 | 2.99±.47 | 6.23±.48 |
| NQ Natural | 37.00±0.18 | 23.74±0.92 | .467±.001 | .687±.001 | .471±.010 | .907±.005 | -1.30±2.13 | 5.73±1.23 |
| NQ Balanced | 37.88±0.52 | 22.91±1.60 | .438±.013 | .667±.008 | .419±.004 | .889±.012 | .90±2.00 | 5.60±.94 |

## Early recovery / semantic class test

| critic / sampler | early recovery n=960 RMSE | post_capture_real n=3388 RMSE | recovery_pure n=1164 RMSE | pursuing n=617 RMSE |
|---|---:|---:|---:|---:|
| LQ Natural | 20.05±0.12 | 14.17±0.63 | 12.73±1.35 | 65.93±0.46 |
| LQ Balanced | 17.99±2.48 | 12.46±1.41 | 11.20±0.10 | 68.80±0.52 |
| NQ Natural | 18.45±2.37 | 15.86±1.30 | 16.77±2.27 | 65.16±1.10 |
| NQ Balanced | 18.48±1.43 | 14.68±2.44 | 17.57±3.91 | 65.86±0.30 |

Additional natural-test event RMSE: LQ Natural/Balanced are ring2 `56.81/61.03` (`n=328`) and ring3/capture `66.69/68.64` (`n=24`); NQ Natural/Balanced are ring2 `53.62/59.33` (`n=328`) and ring3/capture `68.69/68.65` (`n=24`). These rare subsets are reported, not quota-balanced.

## Sampling actually seen

Each run drew `307200` rows (`1200 × 256`) with replacement.

| condition | pursuing | pre_capture_cover | post_capture_real | recovery_pure |
|---|---:|---:|---:|---:|
| Natural ratio | 8.834% | 21.004% | 53.644% | 16.517% |
| Balanced ratio | 50.000% | 12.500% | 25.000% | 12.500% |

Reuse ratio is defined as `draw_count / unique_count` within the train active-agent rows. Pursuing: Natural `5.63±0.01` (`27139.5` draws, `4818` unique); Balanced `31.70±0.00` (`153600` draws, `4845` unique). Balanced therefore repeatedly visits essentially the full available pursuing pool.

## Train/validation overfit check

| critic / sampler | train overall RMSE | validation overall RMSE | train pursuing RMSE |
|---|---:|---:|---:|
| LQ Natural | 33.77±2.63 | 35.58±0.20 | 55.79 |
| LQ Balanced | 30.08±0.48 | 36.70±0.06 | 21.26 |
| NQ Natural | 29.45±0.28 | 34.79±0.15 | 43.24 |
| NQ Balanced | 24.93±1.01 | 36.25±0.24 | 13.77 |

Balanced lowers train pursuing error, but the LQ natural test overall RMSE changes `36.69 → 37.96`, ECE `4.89 → 6.23`, and pursuing test RMSE `65.93 → 68.80`. The pre-registered `>=10%` test degradation overfit flag is therefore false, while the calibration trade-off supports choosing Natural for LQ. NQ has nearly unchanged early-recovery RMSE (`18.45 → 18.48`) and no clear natural-test win for either sampler.

## Storage and artifacts

- disk free before fits：`14661423104` bytes（约 13.66 GiB，`df -h` 显示 14G）
- AC-3A output actual：`132447112` bytes（约 126 MiB；8 local checkpoints plus summary/progress）
- formal bank was reused in place；没有复制 bank；大 checkpoint 未提交 Git
- summary：`artifacts/2026-09-20_ac3a/summary.json`
- runner：`tools/run_ac3a_critic_sampler_audit_20260920.py`

本轮到此停止，不进入 AC-3B。
