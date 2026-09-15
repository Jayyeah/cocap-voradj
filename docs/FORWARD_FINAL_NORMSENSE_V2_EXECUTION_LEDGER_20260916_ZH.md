# Forward-Final NormSense V2 执行 Ledger（2026-09-16）

## 启动结论

最新本地已验证的 Agent B audit ref 为
`origin/experiment/density-normalized-sensing-v2-20260915@2ebfde3ff0b753cf0a8895916c0652c03bfc266b`，五项启动 Gate 均满足：

- `SENSING_V2_CONTRACT: PASS`
- `NORM_PURE_CAPTURE_PARITY: PASS`
- `FULLMIX_ORIGINAL_VS_DOWNWEIGHT05_PARITY: PASS`
- `ALPHA1_REWARDBALANCED: SUPERSEDED_NO_EFFECT`
- `THREE_NEW_RUNS: READY`

用户随后明确授权“直接启动，不 push”。因此三条新线按已审计合同在本地启动；本次不执行 push，也不把未 push 当作运行时阻塞。

启动时本地 launcher HEAD：`cd8a5c9`。

## Legacy 保护

`LEGACY_R20_PURE_CAPTURE` 保持原样：不改代码、不改 config、不切换 sensing、不 resume 到 V2、不停止、不覆盖 checkpoint。仅读取 PID/GPU/step 做资源管理。本次未对旧线发出 kill、resume 或 checkpoint 写入操作。

不启动第五条 V2 Pure-Coverage，也不启动 RewardBalanced、alpha=.25、PCGrad、CAGrad、phase-wise advantage normalization 或 200k Full-Mix 扩训。

## 三条新线启动快照

| line | kind / alpha | budget | tmux | PID | physical GPU | launch status | step-0 checkpoint |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| NormSense-PureCapture | Pure-Capture / 1.0 | 250k | `normsense_pure_20260916` | 1108314 | 0 | `RUNNING_INITIAL_EVAL` | `runs/2026-09-15_normsense_v2/NormSense-PureCapture/step_000000.pt` |
| NormSense-Original-FullMix | Full-Mix / 1.0 | 100k | `normsense_original_20260916` | 1103790 | 1 | `RUNNING_INITIAL_EVAL` | `runs/2026-09-15_normsense_v2/NormSense-Original-FullMix/step_000000.pt` |
| NormSense-CaptureDownweight05-FullMix | Full-Mix / 0.5 | 100k | `normsense_downweight05_20260916` | 1103798 | 1 | `RUNNING_INITIAL_EVAL` | `runs/2026-09-15_normsense_v2/NormSense-CaptureDownweight05-FullMix/step_000000.pt` |

三条线各自独立 output、PID/tmux、RNG、optimizer、ValueNorm、checkpoint 和日志。Full-Mix 两条使用相同 seed / 初始 hash / eval seeds；唯一核心实验变量是 audited contract 中的 `alpha_capture=1.0` 对 `0.5`。

## 启动层异常与修复

首个 Pure-Capture 进程在 step 0 eval 后暴露执行层兼容错误：旧 scratch 模块没有被调用方假定的 `finite_tree` 符号。该错误不涉及算法、config、sensing、seed、alpha 或 checkpoint；失败目录完整保留为
`runs/2026-09-15_normsense_v2/NormSense-PureCapture.failed_20260916_0022`。

随后将有限值递归检查实现留在 launcher 执行层，重新使用同一正式输出名和合同启动，当前 PID 为 `1108314`。重启后日志已建立，step-0 checkpoint / metadata 可读，当前仍在初始 eval；截至快照未发现 Traceback、OOM、NaN 或 Inf。两条 Full-Mix 从未受此异常影响。

## 固定停止策略

- Pure-Capture：每 25k checkpoint/eval，250k `STOP_FOR_MASTER_REVIEW`；不因早期 zero capture 停止，不自动延长到 500k。
- 两条 Full-Mix：每 25k checkpoint/eval，100k STOP；不自动继续。
- 每条线必须保留 perception exposure、phase-wise/full-mix gradient diagnostics、训练统计、matched eval 和 resume metadata。

## 启动时资源检查

启动时 GPU0 上旧线和 Pure-Capture 新线，GPU1 上两条 Full-Mix 新线；RTX A6000 显存余量充足，RAM 充足，无 swap/OOM。非本项目 Isaac Sim 进程未被抢占或修改。启动后新进程显存约 366--376 MiB，旧线保持约 2104 MiB；后续仅在 OOM、严重 swap 或总吞吐明显恶化时按优先级排队新线，绝不 kill OLD。

## 解释边界

Legacy Pure-Coverage checkpoint 在 V2 argmax 下的 transfer shift 已由 Agent A 证明，但这不阻塞两条 random-init Full-Mix A/B；最终报告必须写明：

`Legacy Pure-Coverage success != V2 Pure-Coverage learnability proof`

训练完成后只生成 matched table 与证据分类，不自动进入下一实验。推送被用户明确取消；本 ledger / MASTER / SESSION_HANDOFF 仅做本地记录。
