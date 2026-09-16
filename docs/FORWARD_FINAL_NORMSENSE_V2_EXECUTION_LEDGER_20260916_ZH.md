# Forward-Final NormSense V2 执行 Ledger（2026-09-16）

## 启动结论

最新本地已验证的 Agent B audit ref 为
`origin/experiment/density-normalized-sensing-v2-20260915@2ebfde3ff0b753cf0a8895916c0652c03bfc266b`，五项启动 Gate 均满足：

- `SENSING_V2_CONTRACT: PASS`
- `NORM_PURE_CAPTURE_PARITY: PASS`
- `FULLMIX_ORIGINAL_VS_DOWNWEIGHT05_PARITY: PASS`
- `ALPHA1_REWARDBALANCED: SUPERSEDED_NO_EFFECT`
- `THREE_NEW_RUNS: READY`

用户随后明确授权“直接启动，不 push”。因此三条新线按已审计合同在本地启动；本次启动后未产生有效训练更新。随后用户明确要求不再修改执行层、不重跑，仅记录失败并同步仓库。

启动时本地 launcher HEAD：`cd8a5c9`。

## Legacy 保护

`LEGACY_R20_PURE_CAPTURE` 保持原样：不改代码、不改 config、不切换 sensing、不 resume 到 V2、不停止、不覆盖 checkpoint。仅读取 PID/GPU/step 做资源管理。本次未对旧线发出 kill、resume 或 checkpoint 写入操作。

不启动第五条 V2 Pure-Coverage，也不启动 RewardBalanced、alpha=.25、PCGrad、CAGrad、phase-wise advantage normalization 或 200k Full-Mix 扩训。

## 三条新线启动快照

| line | kind / alpha | budget | tmux | PID | physical GPU | launch status | step-0 checkpoint |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| NormSense-PureCapture | Pure-Capture / 1.0 | 250k | `normsense_pure_20260916` | 1108314 | 0 | `FAILED_STOP_IMPLEMENTATION at step 0` | `runs/2026-09-15_normsense_v2/NormSense-PureCapture/step_000000.pt` |
| NormSense-Original-FullMix | Full-Mix / 1.0 | 100k | `normsense_original_20260916` | 1103790 | 1 | `FAILED_STOP_IMPLEMENTATION at step 0` | `runs/2026-09-15_normsense_v2/NormSense-Original-FullMix/step_000000.pt` |
| NormSense-CaptureDownweight05-FullMix | Full-Mix / 0.5 | 100k | `normsense_downweight05_20260916` | 1103798 | 1 | `FAILED_STOP_IMPLEMENTATION at step 0` | `runs/2026-09-15_normsense_v2/NormSense-CaptureDownweight05-FullMix/step_000000.pt` |

三条线各自独立 output、PID/tmux、RNG、optimizer、ValueNorm、checkpoint 和日志。Full-Mix 两条使用相同 seed / 初始 hash / eval seeds；唯一核心实验变量是 audited contract 中的 `alpha_capture=1.0` 对 `0.5`。

## 启动层异常与失败原因

本轮没有进入任何 rollout/update；三条线均在 step 0 初始 eval 之后、第一次训练 transition 的有限值检查处停止。失败是执行层 bug，不是算法、config、sensing、seed、alpha、GPU OOM、NaN/Inf 或旧线干扰。

1. Original Full-Mix 与 CaptureDownweight05 Full-Mix 的首批进程加载了 launcher 中的 `finite_tree()` 调用，但 scratch 模块 `tools/preflight_forward_final_scratch_20260914.py` 没有该符号，触发：
   `AttributeError: module ...preflight_forward_final_20260914 has no attribute finite_tree`。
2. Pure-Capture 首次启动也触发同一缺失符号；该失败目录保留为
   `runs/2026-09-15_normsense_v2/NormSense-PureCapture.failed_20260916_0022`。
3. 在用户要求“不做任何改动”前，曾有一次仅用于诊断的 launcher 兼容性尝试；Pure-Capture 重启后进一步暴露 `np.isfinite()` 直接作用于 object ndarray 的 `TypeError`。该次正式输出目录的 `failure.json` 和日志均保留；本次不再修复、不重跑。

三条当前正式目录均有 `failure.json`，`progress.json` 最后为 `step=0 / INITIAL_CHECKPOINT_EVAL`；均没有 `learning.jsonl`、`gradient_logging.jsonl` 或 step 25k+ checkpoint。失败目录不覆盖任何 OLD checkpoint。

## 实际产生的数据：仅 step-0 eval 基线

以下数据来自各线独立的 `eval_step_000000.json`，只能作为 random-init / pre-update 基线，不能解释为训练结果。

### Pure-Capture step 0

| mode | episodes | capture | ring2 | ring3 | hold mean | collision | CE success | mean return | CE RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| argmax | 20 | 0.00 | 0 | 0 | 0 | 1.00 | 0.00 | -66.0791 | 0.21696 |
| sample | 20 | 0.00 | 0 | 0 | 0 | 1.00 | 0.00 | -233.7110 | 0.21036 |

### Full-Mix step 0

Original 与 Downweight05 使用相同初始 hash、seed 和 eval seed，因此这次未更新的行为指标完全相同；alpha 只改变了 eval reward component 的缩放记录，尚未有 PPO update 可比较。

| line | mode/task | capture | ring2/ring3 | hold mean | pure CE | safe | collision | mean episode time | CE RMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | argmax / mixed | 0/20 | 0/0 | 0 | 0/20 | 0/20 | 20/20 | 96.65 | 0.20552 |
| Original | argmax / coverage | — | 0/0 | 0 | 0/20 | 0/20 | 12/20 | 1222.35 | 0.20457 |
| Original | sample / mixed | 0/20 | 0/0 | 0 | 0/20 | 0/20 | 20/20 | 387.25 | 0.18021 |
| Original | sample / coverage | — | 0/0 | 0 | 0/20 | 0/20 | 20/20 | 403.10 | 0.19718 |
| Downweight05 | argmax / mixed | 0/20 | 0/0 | 0 | 0/20 | 0/20 | 20/20 | 96.65 | 0.20552 |
| Downweight05 | argmax / coverage | — | 0/0 | 0 | 0/20 | 0/20 | 12/20 | 1222.35 | 0.20457 |
| Downweight05 | sample / mixed | 0/20 | 0/0 | 0 | 0/20 | 0/20 | 20/20 | 387.25 | 0.18021 |
| Downweight05 | sample / coverage | — | 0/0 | 0 | 0/20 | 0/20 | 20/20 | 403.10 | 0.19718 |

### Perception exposure step 0

| line | eval records | reset enemy visible | zero-detector fraction | detector count mean | first-detection latency mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pure-Capture | 40 | 0.9500 | 0.06852 | 0.93148 | 2.667 |
| Original Full-Mix | 80 | 1.0000 | 0.02799 | 0.97201 | 1.000 |
| Downweight05 Full-Mix | 80 | 1.0000 | 0.02799 | 0.97201 | 1.000 |

没有训练期 reward components、GAE/raw-normalized advantage、critic EV/loss、entropy/KL/clip、actor/value grad 或 phase-conditioned gradient；这些字段应记为 `NA_NOT_REACHED`，绝不能填零。

## 结案判定

- Full-Mix matched table：`NOT_AVAILABLE — no training update`。
- PureCapture OLD vs NormSense：`NOT_AVAILABLE — NormSense did not train`。
- Downweight causal classification：`INCONCLUSIVE — no post-update data`。
- NormSense classification：`NO_MEANINGFUL_NORMSENSE_GAIN` 不可宣称；当前严谨状态为 `INCONCLUSIVE — execution failure before learning`。
- `Legacy Pure-Coverage success != V2 Pure-Coverage learnability proof` 仍然成立。

## 固定停止策略

- Pure-Capture：每 25k checkpoint/eval，250k `STOP_FOR_MASTER_REVIEW`；不因早期 zero capture 停止，不自动延长到 500k。
- 两条 Full-Mix：每 25k checkpoint/eval，100k STOP；不自动继续。
- 每条线必须保留 perception exposure、phase-wise/full-mix gradient diagnostics、训练统计、matched eval 和 resume metadata。

## 启动时资源检查

启动时 GPU0 上旧线和 Pure-Capture 新线，GPU1 上两条 Full-Mix 新线；RTX A6000 显存余量充足，RAM 充足，无 swap/OOM。非本项目 Isaac Sim 进程未被抢占或修改。启动后新进程显存约 366--376 MiB，旧线保持约 2104 MiB；后续仅在 OOM、严重 swap 或总吞吐明显恶化时按优先级排队新线，绝不 kill OLD。

## 解释边界

Legacy Pure-Coverage checkpoint 在 V2 argmax 下的 transfer shift 已由 Agent A 证明，但这不阻塞两条 random-init Full-Mix A/B；最终报告必须写明：

`Legacy Pure-Coverage success != V2 Pure-Coverage learnability proof`

训练未完成；不自动进入下一实验，不启动任何 alpha=.25、PCGrad、CAGrad、phase-wise advantage normalization、V2 Pure-Coverage 或 200k Full-Mix。失败日志与 step-0 eval 作为审计证据保留。用户已重新允许 push，本次文档变更将同步到 GitHub。

## GitHub 同步

已 fetch 并确认远端目标分支无落后提交；失败记录已推送至：
`origin/experiment/density-normalized-sensing-v2-20260915`，同步提交为 `3007a02`（后续仅补充本节状态记录）。
