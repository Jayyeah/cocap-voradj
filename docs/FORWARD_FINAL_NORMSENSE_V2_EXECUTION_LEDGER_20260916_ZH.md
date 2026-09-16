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

最终检查时间 `2026-09-16T09:54:16+0800`：OLD 进程与 `cocap_single_task_20260915` tmux 已不在；本 Agent 未停止或修改该线。旧工作树最后可见 checkpoint 为 `cocap-voradj-small-step-ac/artifacts/2026-09-15_single_task/capture/step_450000.pt`，因此不能把它报告为本轮由 Agent C 完成的 500k 结果。

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

## 2026-09-16 execution recovery / relaunch

状态序列：`FAILED_AGENT_C_EXECUTION -> BUG_FIXED -> SMOKE_PASS -> FORMAL_RELAUNCH`。这只是成功恢复执行并正式重启，不是实验成功。

启动时重新 fetch；远端最新 HEAD 为 `f1a9216c8fce801603b731aa9f300158ec1905a7`。修复与 regression 已先提交并推送为 `81682667dbe8316154d5d78e6baecf3e736fe361`。

两个执行层 root cause：

1. 初版 formal launcher 的本地 `finite_tree()` 包装器委托给 scratch preflight module，但该 module 从未导出这个 symbol，因此第一次 training transition 抛出 `AttributeError`。
2. 后续兼容实现把所有 ndarray 无条件传给 `np.isfinite`；真实 transition 含 `gradient_phase` Unicode metadata ndarray，object ndarray 也可能包含 string / None / nested containers，因此抛出 `TypeError`。新公共 helper 只检查数值 tensor / ndarray / scalar，递归处理 mapping / list / tuple / object ndarray，并忽略非数值 metadata；真正 NaN/Inf 仍 fail closed。

Gate 结果：

- Gate A targeted regression：`PASS (23 passed)`，覆盖 torch tensor、float/int ndarray、object ndarray、nested dict/list/tuple、string/None metadata、真实 NormSense transition，以及真实 NaN/Inf。
- Gate B CPU：`PASS`，256 real transitions、1 PPO update；actor/value loss 与 advantage/return finite。
- Gate C CUDA0：`PASS`，256 real transitions、1 PPO update；optimizer step、loss、advantage/return、checkpoint/log 均正常。
- Gate D resume：`PASS`；model、optimizer、ValueNorm、step、RNG、metadata round-trip 一致。

旧 `runs/2026-09-15_normsense_v2` step-0 failed artifacts 保留且未覆盖。新 lineage 固定为 `RECOVERED_AFTER_EXECUTION_BUG_FIX`，输出根为 `runs/2026-09-16_normsense_v2_execution_recovery`。

`2026-09-16T10:52:26+08:00` 快照：

| line | tmux | PID | physical GPU | current step | updates | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| NormSense-PureCapture | `normsense_v2_recovered_pure_20260916` | 1293736 | 0 | 14000 | 54 | `RUNNING` |
| NormSense-Original-FullMix | `normsense_v2_recovered_original_20260916` | 1293741 | 1 | 0 | 0 | `INITIAL_CHECKPOINT_EVAL 69/80` |
| NormSense-CaptureDownweight05-FullMix | `normsense_v2_recovered_downweight05_20260916` | 1293745 | 1 | 0 | 0 | `INITIAL_CHECKPOINT_EVAL 69/80` |

PureCapture 已明确产生训练 update；其 actor/value loss、advantage/return 均 finite。三进程均存活，GPU0/1 显存约 4.75/2.87 GiB，无 Traceback/OOM/NaN/Inf。两条 Full-Mix 仍在合同规定的 step-0 eval，不能把 eval 中写成已训练。

ETA（只作资源规划，不是结果承诺）：PureCapture launcher 当前 ETA 约 5 小时 27 分；两条 Full-Mix 预计约 3--6 分钟进入训练，100k 完成粗估各 5--7 小时（包含后续 25k eval）。按用户最新要求，本次只做一次状态确认，不再长期监控；进程继续运行，25k 节点由后续人工检查。

机器可读记录见 `artifacts/2026-09-15_normsense_v2/execution_recovery_20260916.json`。禁止项、预算、seed、alpha、reward、网络、transition semantics 和 rollout budget 均未改变。

## 2026-09-16 historical results update

### NormSense-PureCapture

`NormSense-PureCapture` 使用 audited Pure-Capture 合同（seed `2026091501`、alpha_capture `1.0`、250k budget、25k eval cadence、V2 `k=0.8715`）。step 100k checkpoint 已写入，共 390 PPO updates；随后在 100k eval 的第 10/40 条记录处停止：

```
AssertionError: telemetry.observe: all(m['phase'] != 'post_capture' for m in metas)
```

这是评估/telemetry 执行失败，不是训练 NaN、Inf 或 OOM；不将 100k eval 写成完整结果，也不继续恢复该线。

| eval step | argmax capture / CE | argmax collision | argmax CE RMS | argmax total return | sample capture / CE | sample collision | sample CE RMS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 / 0 | 1.00 | 0.21696 | -66.08 | 0 / 0 | 1.00 | 0.21036 |
| 25k | 0 / 0 | 0.00 | 0.20105 | 24.07 | 0 / 0 | 0.70 | 0.20015 |
| 50k | 0 / 0 | 1.00 | 0.26343 | -53.11 | 0.05 / 0 | 0.95 | 0.26490 |
| 75k | 0 / 0 | 1.00 | 0.21482 | -123.01 | 0 / 0 | 1.00 | 0.25220 |
| 100k | incomplete (10/40) | — | — | — | incomplete | — | — |

截至完整的 75k eval，没有稳定 capture 或 CE success；50k sample 的 capture 率 0.05 未在 75k 保持。训练状态确实发生变化（actor hash 由 `965553…` 变为 `795f99…`），但不能据此宣称 PureCapture 学习成功。

### 两条 Full-Mix 的历史性能与参数变化

两条线均为 random-init、seed `2026091401`、100k budget、25k cadence；当前均完成 step 50k / 195 updates，50k checkpoint 已写入，50k eval 尚未完成。

| line / alpha | eval step | argmax mixed: capture / collision / CE RMS | argmax coverage: collision / CE RMS | sample mixed: capture / collision / CE RMS | sample coverage: collision / CE RMS |
|---|---:|---|---|---|---|
| Original / 1.0 | 0 | 0 / 20 / 0.20552 | 12 / 0.20457 | 0 / 20 / 0.18021 | 20 / 0.19718 |
| Original / 1.0 | 25k | 0 / 0 / 0.19705 | 0 / 0.29735 | 0 / 18 / 0.23471 | 18 / 0.21332 |
| Downweight05 / 0.5 | 0 | 0 / 20 / 0.20552 | 12 / 0.20457 | 0 / 20 / 0.18021 | 20 / 0.19718 |
| Downweight05 / 0.5 | 25k | 0 / 0 / 0.19705 | 0 / 0.29735 | 0 / 19 / 0.22370 | 20 / 0.21414 |

25k 时两线 argmax mixed/coverage 均无 capture/CE success；collision 下降伴随 episode 达到 horizon，coverage CE RMS 从约 0.205 增至约 0.297。sample 指标出现轻微分叉，但尚不足以作 causal 结论。50k eval 仍为 `NOT_YET_WRITTEN`，不能填零。

训练/参数状态节点：

| line | actor loss (step 256 → 49,920) | value loss | entropy | explained variance | ValueNorm mean / std |
|---|---:|---:|---:|---:|---:|
| Original | -0.022464 → -0.021123 | 0.65449 → 0.01230 | 2.19716 → 2.10151 | 0.138 → 0.221 | -28.61/53.63 → -56.25/40.49 |
| Downweight05 | -0.022464 → -0.021571 | 0.65433 → 0.01042 | 2.19716 → 2.14218 | 0.140 → 0.697 | -28.37/53.81 → -65.22/40.82 |

actor state hashes 已从相同 initial hash 分叉：

- Original：`54f502…` → 25k `7b199c…` → 50k `ecb25a…`
- Downweight05：`54f502…` → 25k `7fd066…` → 50k `3ad01c…`

除预注册的 `alpha_capture=1.0` 对 `0.5` 外，重要合同参数没有变化：sensing V2/`k=0.8715`、seed、network dimensions、PPO（actor LR `3e-5`、critic LR `1e-4`、3 epochs/2 minibatches、clip `0.2`、gamma `0.99`、GAE lambda `0.95`）、ValueNorm、rollout 256、mixed→coverage schedule、reset pool、eval seeds、horizon 和 collision semantics 均保持 audited contract。两条线的 50k eval 与 100k STOP 仍待各自自然完成；不添加第四条实验、不扩预算、不修改参数。

机器可读明细：`artifacts/2026-09-15_normsense_v2/execution_recovery_results_20260916.json`。

## Pure-Capture V2 窄范围恢复（2026-09-16）

状态标记：`BASELINE_100K_EVAL_FIXED`、`BASELINE_RESUMED_TO_250K`、`CONSERVATIVE300K_STARTED`。

100k 历史断言已用原 seed `2026191511` 精确重放。触发点是 tick 213 的 **非 capture** terminal loss（`evader collision`）：所有 agent 均 `done=true`、`terminated=true`、`truncated=false`，metadata 因 terminal successor state 被标为 `post_capture`，且 episode 没有产生下一条 transition。因此“100k evaluator crash 是否由 capture terminal 触发”的答案为 **否**。修复仅调整 evaluator/telemetry 的 terminal interpretation：只允许全体已 terminated、无 truncation 的单条 terminal `post_capture` metadata；仍拒绝任何非 terminal `post_capture` 或 terminal 后再次 `observe()`。环境 transition/reward/sensing 语义未改。相关 16 个 Pure-Capture 回归测试通过，修复提交为 `29d8498`。

75k/100k 均以 matched seed base `2026191501` 完成 50 argmax + 50 sample 重评估：

| checkpoint / mode | capture（normal/stationary） | ring2 / ring3 | longest ring3 hold mean | collision episodes / events | AA / obstacle / boundary events | capture time mean (s) | total / discounted return mean | visible fraction / first detection |
|---|---|---|---:|---|---|---:|---|---|
| 75k argmax | 0%（0%/0%） | 68% / 14% | 1.80 | 100% / 54 | 23 / 0 / 0 | — | -97.438 / -39.798 | 0.9769 / 2.40 |
| 75k sample | 0%（0%/0%） | 18% / 0% | 0.00 | 100% / 50 | 40 / 2 / 8 | — | -256.143 / -13.787 | 0.9957 / 2.38 |
| 100k argmax | 4%（4%/0%） | 78% / 26% | 1.54 | 96% / 51 | 27 / 0 / 4 | 61.00 | -185.093 / -37.312 | 0.9900 / 2.40 |
| 100k sample | 8%（8%/0%） | 40% / 10% | 0.18 | 92% / 46 | 37 / 6 / 0 | 141.25 | -180.249 / -0.119 | 0.9963 / 2.38 |

100k 全量重评估中有 4 个 terminal-loss episode 出现 terminal `post_capture` metadata，capture episode 中为 0；实际 post-capture rollout transition 为 0。每类 capture、ring3 near-success、collision episode 已保存最后 50–100 action-step diagnostics。

Baseline `NormSense-Final-PureCapture-Baseline` 从真实 step 100000 full-resume checkpoint 恢复，Actor/Critic/optimizer/ValueNorm/RNG/training step 与 160-step partial rollout 均保留；PID `1423372`，physical GPU 0，记录快照 step `122100` / update `476`，目标 250k，下一 checkpoint 125k。首个恢复 update 的 finite/ValueNorm/entropy/KL/checkpoint gate 为 PASS。

Scratch 线 `NormSense-Final-PureCapture-Conservative300k` 已启动；PID `1423380`，physical GPU 0，记录快照 step `18200` / update `71`，目标 300k，下一 checkpoint 25k。step 256 健康 gate（finite losses、ValueNorm、entropy/KL、checkpoint write/resume read）为 PASS。

机器可读 parity 为 PASS、`UNEXPLAINED=0`。唯一科学改动组为 `episode_max_length 3000→1000`（并显式启用 `pre_capture_max_length=1000`）与 `actor.dropout 0.1→0.0`；其余 Final Pure-Capture 合同一致。两条 scratch 初始 Actor/Critic/ValueNorm hashes 完全一致。审查同时确认 trainer 始终将 Actor 保持在 `eval()`，故 baseline 中配置的 dropout 实际也未在 rollout/PPO update 生效；该实现事实不扩大 ablation。Full-Mix Original/Downweight 进程未被停止或修改。

本轮机器可读总结果：`artifacts/2026-09-15_normsense_v2/pure_capture_v2_recovery_results_20260916.json`；parity：`artifacts/2026-09-15_normsense_v2/pure_capture_baseline_vs_conservative300k_parity.json`。

## 已完成线交付与未完成线状态（2026-09-16 22:30）

本节覆盖当前快照，优先于本 ledger 中较早的运行中描述。三条线已完成并停止在 master review：Pure baseline `250k/250k`，Original Full-Mix `100k/100k`，CaptureDownweight05 Full-Mix `100k/100k`。Conservative300k 尚未完成，不能填入最终性能结论。

### 已完成线的合同与最终 eval

| line | budget / eval | seed | 唯一实验参数 | actor hash（final） | 关键结果 |
|---|---:|---:|---|---|---|
| NormSense-Final-PureCapture-Baseline | 250k，20 argmax + 20 sample | 2026091501 | CR-MS `ring_importance_ms_v0`；horizon 3000；dropout .1 | `77bf7001…` | argmax/sample capture 均 5%；ring2/ring3 = 65/15%、50/25%；collision episode 均 95% |
| NormSense-Original-FullMix | 100k，20×scene×mode | 2026091401 | `alpha_capture=1.0`；horizon 3000 | `500c2f2a…` | argmax mixed/coverage capture、CE 均 0；sample mixed collision 16/20，coverage collision 15/20 |
| NormSense-CaptureDownweight05-FullMix | 100k，20×scene×mode | 2026091401 | `alpha_capture=0.5`；horizon 3000 | `b2061d17…` | argmax mixed/coverage capture、CE 均 0；sample mixed collision 17/20，coverage CE success 1/20、collision 18/20 |

Pure baseline 250k 的细项为：argmax capture `5%`（normal `5%`、stationary `0%`），capture time mean `22.5s`，enemy-visible `0.9654`，ring3 hold mean `0.2`，collision events `23`（AA/obstacle/boundary `4/0/1`），total/discounted return `-51.272/-20.770`；sample capture `5%`（normal `5%`、stationary `0%`），capture time mean `143.5s`，enemy-visible `0.9922`，ring3 hold mean `1.3`，collision events `19`（AA/obstacle/boundary `18/0/1`），total/discounted return `-250.093/-0.986`。

Full-Mix 最终 eval 的 CE RMS / area CV：Original argmax mixed/coverage `0.19705/0.39028`、`0.29735/0.40882`；sample mixed/coverage `0.20813/0.30873`、`0.20856/0.30402`。Downweight05 对应为 `0.19705/0.39028`、`0.29735/0.40882`；`0.20081/0.32542`、`0.18949/0.33707`。两个 Full-Mix eval 均为 80/80 完整，Actor/Critic/ValueNorm/RNG resume metadata 保留，未改变 audited sensing、transition、collision、PPO 或 ValueNorm 合同。

### 未完成线的如实状态

`NormSense-Final-PureCapture-Conservative300k`：PID `1423380`，GPU 0，快照 `step=287100/300000`、`training_updates=1121`、约 `11.54 step/s`，状态 `RUNNING`；训练剩余约 `1118s`（约 19 分钟），300k final eval 尚未开始。该线仅预注册 horizon `3000→1000`（含 `pre_capture_max_length=1000`）和 Actor dropout `.1→0`，不得把当前 287.1k 状态当作最终结果。

机器可读完成线汇总：`artifacts/2026-09-15_normsense_v2/normsense_v2_completed_lines_results_20260916.json`。原始最终 eval 文件仍保留在各 run 目录；本节只记录已完成线可复核的关键字段，不替代原始 episode records。
