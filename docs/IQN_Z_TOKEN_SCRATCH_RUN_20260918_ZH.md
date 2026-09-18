# IQN Z-token matched scratch 运行记录（2026-09-18）

本记录已按 2026-09-19（Asia/Shanghai）正式 matched 启动事实更新。仓库 runtime artifact 优先于聊天摘要。

## 当前状态

- 分支：`experiment/iqn-z-token-scratch-20260918`
- 启动 HEAD：`57bb0aa2e5e80fb0b34da5880e059bbcf60d7b6c`
- 配置：`configs/experiments/iqn_token_scratch_20260918/z_token.yaml`
- tmux：`iqn_z_token_matched_20260919`
- 初始 supervisor PID：`2935727`；25k evaluator 工程修复后 exact-resume PID：`2949712`
- GPU：物理 GPU1；进程内 `cuda:0`
- 正式启动：`2026-09-19T01:09:49+0800`
- 目标：200,000 environment steps；25k 间隔 checkpoint + formal evaluation
- 状态入口：`artifacts/2026-09-18_iqn_z_token_scratch/{launch,status,heartbeat,scratch_gate}.json`

启动后在 16k 稳定窗口观察到：约 `28.82 steps/s`、`3278` 次真实 optimizer updates、replay `64000`、loss `1.3790`、loss EMA `2.6811`，均为有限值；target 已在 step 10k 同步一次。动作窗口九个动作均有采样，不存在固定动作实现错误。

25k milestone 已完成 resumable checkpoint 与 60 局 formal evaluation；报告严格 JSON 可读。评估写出曾因“未发生事件”的 `inf` 时间值被严格 JSON writer 拒绝，修复为 `null` 后从 25k exact-resume，仅重跑 evaluator，不改变训练状态。当前 25k strict CE、capture 和 safe-complete 均为 0，属于允许的早期性能，不触发早停。

## Matched 合同

Z 与 ROLE 共同继承 `configs/experiments/iqn_token_scratch_20260918/common.yaml`，共同使用 NormSense-V2：

- policy：`forward-final-density-normalized-sensing-v2`
- runtime observation hash：`6c2af0df8ebb4fbca9c52a29fb4008a84e0ff90a5804ba633054745bef6f2bcd`
- matched non-token contract hash：`12029ad757c1efe975ea3d04971d7d12f0fb8e00ca5209782dbdd95534777843`
- Z resolved config hash：`15d36f4b1454e7dd9424390a47af02ea6ed115eaf70896151e1c463e79040b38`
- non-token diff count：`0`

Z policy 输入为 `[physical_self, z_i]` 与 `[relative_friend_physical, z_j]`。不存在 `is_pursuing` policy 输入、`pursuing_embed`、late fusion、特殊 z branch、z-dependent friend ordering 或 global target state。

Z-v1 固定：`lambda=0.95`、`eta=0.85`、direct 当前 decision 立即生效、neighbor 只读 previous-decision `z_j`、同步更新、reset 为 0、无 hard floor，且 z 完整进入 checkpoint/resume state。本轮禁止修改这些科学变量。

## Scratch 与 preflight

正式 scratch gate 已确认：`global_step=0`、四类 production replay 全为 0、optimizer state 为空、epsilon 为 `0.6`，online/target model SHA 均为 `22a7deaa4b7265909a8b6c340f84d8741b69194fd3cc5c643f487ddeb3908ec8`。ROLE/Z 初始化参数 key、shape 与 tensor bit-exact。

双边 preflight 均通过：NormSense-V2 runtime assertion、observation shape、ROLE binary token、10 个 Z friend `z_j` mapping checks、10 个 physical friend rows、15 次真实 optimizer update、finite loss、参数变化、target sync、非固定动作、checkpoint exact load、optimizer/replay/global-step/epsilon resume、Z state bit-exact resume、formal evaluator smoke 与 runtime fail-closed trigger。

机器证据：

- `artifacts/2026-09-18_iqn_z_token_scratch/preflight/ROLE_vs_Z_resolved_config_diff.json`
- `artifacts/2026-09-18_iqn_z_token_scratch/preflight/startup_sanity.json`
- `artifacts/2026-09-18_iqn_z_token_scratch/STARTUP_AUDIT.json`

首次 matched startup 曾到 3k，但因 coordinator 读取 stale prelaunch status 而主动终止；它已归档到 `aborted_matched_startup_coordinator_status_bug_20260919T005609+0800/`，标记为 `ABORTED_MATCHED_STARTUP_NOT_FOR_RESULTS`，禁止恢复。本次正式 run 是状态新鲜度修复后重新从 step 0 启动。

## ETA 与恢复

16k 短窗口估计（early/unstable）：25k `2026-09-19T01:24:17+0800`、50k `01:38:44`、100k `02:07:39`、200k `03:05:28`。

首个 production full-resume 已在 25k milestone 生成并验证可读。此后仅在 HEAD、config/runtime hash 不变且 resume 可读时自动 exact-resume；合同漂移、训练 NaN/Inf、replay/checkpoint 损坏或 wrong GPU 一律 fail closed。

Z-v2 shorter-half-life 与 epsilon hard floor 仅保留为未来 TODO，本轮不实施。

```bash
jq '{status,current_step,phase,latest_metrics}' artifacts/2026-09-18_iqn_z_token_scratch/heartbeat.json
tmux attach -t iqn_z_token_matched_20260919
CUDA_VISIBLE_DEVICES=1 python3 tools/iqn_token_matched_20260919.py supervise --arm z --device cuda:0 --output artifacts/2026-09-18_iqn_z_token_scratch
```
