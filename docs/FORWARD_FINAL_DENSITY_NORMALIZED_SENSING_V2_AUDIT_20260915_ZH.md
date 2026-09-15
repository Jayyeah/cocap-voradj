# Density-Normalized Sensing V2 独立合同审计（2026-09-15）

本审计基于 Agent A delivery branch `experiment/density-normalized-sensing-v2-20260915` 的 HEAD `42bb9bc`，远端同步后未发现更新提交。审计只准备合同，不启动训练；旧 Pure-Capture 作业保持运行，未停止、resume、修改或读取其 checkpoint。

## 最终 Gate

机器状态见 [MASTER.json](../artifacts/2026-09-15_normsense_v2/MASTER.json) 和 [contracts.json](../artifacts/2026-09-15_normsense_v2/contracts.json)：

- `SENSING_V2_CONTRACT: PASS`
- `NORM_PURE_CAPTURE_PARITY: PASS`
- `FULLMIX_ORIGINAL_VS_DOWNWEIGHT05_PARITY: PASS`
- `ALPHA1_REWARDBALANCED: SUPERSEDED_NO_EFFECT`
- `COVERAGE_V2_FROZEN_TRANSFER: BREAKS_POLICY_NONBLOCKING_FOR_SCRATCH_AB`
- `THREE_NEW_RUNS: READY`

### Sensing V2

合同为 `k=0.8715`，通过的最小 k 为 `0.83`，R4/R8/R12 的 runtime resolved surface radius 为：

| stage | radius range (m) | reference (m) |
|---|---:|---:|
| R4 | 52.210051–52.224597 | 52.2173245 |
| R8 | 36.856311–36.882061 | 36.8769126 |
| R12 | 30.046745–30.076221 | 30.0720122 |

enemy 与 obstacle 使用统一 onboard radius，判定为 `center_distance - observer_radius - target_radius <= R`；禁止 global enemy broadcast。friendly topology 仍是历史 global-friendly VorAdj；V3 local-friendly 尚未实现。geometry 72-layout artifact 的 worst blind 为 0，已有 regression 为 30 passed。详见 [sensing_v2_audit.json](../artifacts/2026-09-15_normsense_v2/sensing_v2_audit.json)。

## alpha 的限定解释

Agent A 的 `alpha_capture=1.0` 来自 frozen BC Actor、cold/random critic、initial ValueNorm、16×256 on-policy windows、保留真实 phase occupancy、capture 缺席窗口按零贡献计入，以及 occupancy-weighted trimmed-mean gradient norm。因此它回答的是：

> 在该 frozen policy 的真实 phase visitation 下，Capture 是否主导整体更新？

它不回答：

> Capture transition 自身的 reward/gradient scale 是否相对 Coverage 偏大？

Capture 只出现在 6/16 窗口，Coverage 出现在 16/16 窗口；同时含两者窗口的 descriptive `median ||g_capture|| / ||g_coverage|| ≈ 2.02`，对应 descriptive candidate `alpha≈0.495`。该 candidate 不是最优 alpha，也不是 performance-selected 的 optimally balanced reward。

因此旧 `NormSense-RewardBalanced-FullMix`（alpha=1）保留为历史 artifact，但状态是 `SUPERSEDED_NO_EFFECT_ALPHA1`，不得训练，也不算 negative result。

## 正式 causal ablation

新增 [NormSense-CaptureDownweight05-FullMix.yaml](../configs/experiments/forward_final_normsense_v2_20260915/NormSense-CaptureDownweight05-FullMix.yaml)，固定 `alpha_capture=0.5`。它是 `predeclared causal downweight ablation`，不是 `optimally balanced reward`，禁止动态 alpha 或按训练表现调 alpha。

乘 0.5 的 runtime consumer：

- direct capture-specific dense reward；
- support reward 的 capture component；
- capture-specific terminal reward。

保持 1.0 的 consumer：support coverage component、normal coverage CE center、CE PBRS、CE control/speed、coverage/post-coverage terminal、collision、boundary、safety/proximity、entropy 和其他算法项。源码逐项核对为 `static_capture_scale` 只包住 direct capture、support-capture blend 与 capture terminal；CE 和 safety 不经过该 scale。

## Parity

完整机器 diff 在 [fullmix_original_vs_downweight05_parity.json](../artifacts/2026-09-15_normsense_v2/fullmix_original_vs_downweight05_parity.json)。Original 与 Downweight05 的唯一 effective runtime leaf 差异是：

- `mixed.reward.static_capture_scale: 1.0 → 0.5`；
- `coverage.reward.static_capture_scale: 1.0 → 0.5`。

random Actor、central V、ValueNorm initial state、optimizer、seed、scene schedule、recovery/reset、PPO/GAE、rollout length、eval seeds、horizon、collision semantics 和 teacher dependency 全部相同，`UNEXPLAINED=0`。

Pure-Capture 的唯一主要变量是 Legacy R20 enemy+obstacle sensing → V2 onboard sensing；reward、spawn、PPO、terminal、support、horizon、seed 无额外漂移，`UNEXPLAINED=0`。详见 [pure_capture_parity_audit.json](../artifacts/2026-09-15_normsense_v2/pure_capture_parity_audit.json)。

## Coverage transfer 的边界

冻结 Legacy Pure-Coverage@200k 的 argmax 为 100% success，而 V2 argmax 为 10% success、75% collision；V2 sample 仍为 100% success，同时 obstacle token occupancy 从约 `.06084` 变为 `.36883`。因此保留 `COVERAGE_TRANSFER_BREAKS_POLICY`：这是真实 observation distribution shift。

该 blocker 不阻塞 scratch Full-Mix A/B，因为两条 run 都从 random init 开始、使用完全相同 V2，问题是 V2 内部 Original vs Downweight05 的因果比较。它仍禁止旧 checkpoint 迁移结论，不能声称旧 Pure-Coverage checkpoint 已验证 V2；后续应补 V2 Pure-Coverage confirmation。

## 预算与 logging

- `NormSense-PureCapture`: 0→250k，每25k eval；250k 由 MASTER 决定是否扩到500k；禁止因50k/100k zero capture早停，禁止自动扩训。
- `NormSense-Original-FullMix`: 0→100k，每25k eval，100k STOP。
- `NormSense-CaptureDownweight05-FullMix`: 0→100k，每25k eval，100k STOP。

两个 Full-Mix 已加入 `normsense-gradient-logging-v1`：每次 update 并在 checkpoint 邻近持久化 occupancy-weighted 与 phase-conditioned 的 capture/support/coverage 梯度范数、pairwise cosine、row counts、raw reward、raw/normalized advantage、GAE/value target 和 EV。phase 样本不足记 `NA_INSUFFICIENT_ROWS`，不伪造等量样本。该 side-channel 不更新 optimizer、ValueNorm、不消费 RNG，也不改变 PPO 输入。schema 见 [gradient_logging_schema.json](../artifacts/2026-09-15_normsense_v2/gradient_logging_schema.json)。

三条 run 的正式 manifest：

- [NormSense-PureCapture.json](../artifacts/2026-09-15_normsense_v2/NormSense-PureCapture.json)
- [NormSense-Original-FullMix.json](../artifacts/2026-09-15_normsense_v2/NormSense-Original-FullMix.json)
- [NormSense-CaptureDownweight05-FullMix.json](../artifacts/2026-09-15_normsense_v2/NormSense-CaptureDownweight05-FullMix.json)

审计结论：三条合同 READY，但本审计没有启动任何训练；提交并 push 后才可由后续 Agent C 按 MASTER 合同启动。
