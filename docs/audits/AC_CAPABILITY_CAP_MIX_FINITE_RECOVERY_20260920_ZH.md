# AC-CAPABILITY-RECOVERY：CAP/MIX `finite=0` 修复审计

日期：2026-09-21（Asia/Shanghai）  
模型：Luna

## 结论

`ROOT_CAUSE_IDENTIFIED`、`FIX_IMPLEMENTED`、`CAP_SHORT_GATE_PASS`、
`MIX_SHORT_GATE_PASS`、`CAP_500K_RELAUNCHED`、`MIX_500K_RELAUNCHED`、
`COV_UNTOUCHED`。

修复没有改变 actor-critic 方程、reward、gamma、tau、actor/critic LR、
gradient clip、replay ratio 或 exploration。CAP/MIX 使用同一 learner fix
commit；MIX 的 post→pure cold-start runner 保留。

## 1. 原始失败复现

### CAP

在 CAP 原始提交 `1ae25b6` 上启用一次性逐阶段 forensic gate，raw replay
batch 的 obs、next_obs、action、reward、terminated/truncated、active mask、
replay provenance、behavior/actor probability 与 epsilon 均 finite。

本地 deterministic replay 在 env step 832、update 21 首次失败（原始长训已知
失败区间约 3000 steps）。唯一 offending row 为 61，`terminated=true`，其
`next_obs.masks` 全为 false；首个非有限 operation 是：

```text
target_actor_encoder -> features
```

### MIX

在 MIX 原始提交 `ead4e57` 上复现到用户报告的精确位置：env step 636、update
110。batch composition 为 `64 pursuing / 16 pre_capture_cover / 0
post_capture_real / 48 recovery_pure`。offending row 为 68，边界 collision
后 `terminated=true`、`next_obs.masks` 全为 false；首个非有限 operation 同为
`target_actor_encoder -> features`。

小型 forensic 文件（不属于正式 run）保存在：

```text
/tmp/ac-capability-finite-repro/cap_finite_repro_debug/artifacts/
/tmp/ac-capability-finite-repro/mix_finite_repro_debug/artifacts/
```

## 2. Root cause

终止 transition 的 inactive `next_obs` 由 zero-filled observation 表示，所有
token mask 都为 false。PyTorch Transformer 对全 mask row 执行 all `-inf`
attention，softmax 产生 NaN。learner 当时先计算 target actor/critic，再使用

```text
y = r + gamma * (1 - terminated) * E_pi[Q_target]
```

因此 `0 * NaN` 仍为 NaN；`finite=0` 不是 reward、Q scale 或 optimizer
发散，也不是 MIX sampler/cold-start 造成的。

## 3. 精确修复

在共享 `LocalObservationEncoder` 中，检测 all-masked row，并仅为 encoder
计算临时启用零值 self token，避免 Transformer 的全 `-inf` attention。有效
observation 的 mask、token、输出均不变；terminated row 仍由 learner 的
`(1 - terminated)` 终止 bootstrap。

同时保留可控的 `AC_FIRST_NONFINITE_DEBUG=1` forensic gate：按
replay raw batch、target actor/critic、expected next Q、TD target、critic/
actor forward/loss/backward、optimizer、Polyak 顺序检查；首次失败只写一个
小型 `first_nonfinite_batch.npz` 与 `first_nonfinite_report.json`。

新增 all-masked terminal observation finite regression test。

## 4. COV healthy 对照

COV live control 未停止、未 checkout、未改 run/checkpoint/replay。只读的 COV
25k telemetry 显示 all finite；以下为同一 learner 的健康量级（mean）：

| quantity | COV healthy 25k | CAP/MIX fail |
| --- | ---: | --- |
| critic loss | 0.9115 | 未到 loss，先在 target actor encoder NaN |
| actor loss | 9.7239 | 未到 actor objective |
| actor entropy | 0.1819 | 未到 actor objective |
| critic grad norm | 12.8685 | 未到 backward |
| actor grad norm | 0.0374 | 未到 backward |
| q mean | -9.7396 | 未生成 finite q batch |
| TD target mean | -10.3996 | 未生成 finite target |
| TD target std | 8.0121 | 未生成 finite target |
| finite | 1.0 | 0.0（原始实现） |

这解释了为什么 Coverage 正常而涉及 capture terminal/inactive semantics 的
两条线失败：Coverage replay 没有把 all-masked terminal next row 送入同一
target attention path；CAP/MIX 会产生该 transition。

## 5. Regression 与 short gate

共享 learner unit/integration test：CAP、MIX 各 `6 passed`；COV 独立 CPU
smoke 为 1200 env steps、297 updates，finite 全为 1，actor/critic/target
均改变，action4 被采样。

CAP short gate（正式 learner/env 配置，独立 run name）：

```text
steps=6000
updates=1313
finite=1.0 for all updates
actor/critic/target changed=true
action4_sampled=true
```

MIX short gate（正式 learner/env/replay 参数；仅 gate 预评估缩为 1 episode，
不改变训练参数）：

```text
steps=3000
updates=701
finite=1.0 for all updates
scheduler=voradj <-> voradj_coverage, 1:1
replay rows=12000
first learner update=200
substitution batches=64/16/0/48
```

MIX 首个和末个 composition telemetry 都确认：
`64 pursuing / 16 pre_capture_cover / 0 post_capture_real / 48 recovery_pure`。

## 6. Branch 与 push

```text
repair branch: fix/ac-capability-cap-mix-finite-20260920
shared fix commit: 766283d
repair forensic commit: f69b0ef

CAP branch: experiment/ac-capability-cap-stage1-20260920
CAP launch code HEAD / origin: 6ba8404

MIX branch: experiment/ac-capability-mix-stage1-20260920
MIX launch code HEAD / origin: 93ab659
```

CAP/MIX 均通过 fast-forward push，local HEAD 与对应 origin HEAD 一致；MIX
原有未提交的 AC audit 配置/产物未被覆盖。

## 7. Formal relaunch

两条正式命令均为 clean-scratch 500k，未从 short gate checkpoint resume：

```text
tmux ac_cap_500k
PID 52278 / GPU0 / ac_capability_cap_stage1 / total=500000

tmux ac_mix_500k
PID 52272 / GPU1 / ac_capability_mix_stage1 / total=500000
```

CAP 的 runner 只在 checkpoint/report 时写 env step；本次观察到 0-step eval
文件已于 00:44:15 完成，PID 52278 仍 alive、无 finite traceback。MIX 当前可
由 telemetry 直接观测到 env step 3592、849 learner updates，已经超过原
636 failure point，PID 52272 alive，且仍为 `64/16/0/48` substitution。

## 8. COV 与资源状态

```text
COV: PID 13302, tmux ac_cov_500k, GPU0, alive
CAP: PID 52278, tmux ac_cap_500k, GPU0, alive
MIX: PID 52272, tmux ac_mix_500k, GPU1, alive
```

COV 没有被 kill、pause、restart、checkout、文件修改或 run/checkpoint/replay
删除。GPU0 上 COV/CAP 并发，GPU1 上 MIX 并保留既有 Z 任务；两块 GPU 均有
余量。磁盘余量以最终回复中的 `df -h /home/yjq` 为准。

## 9. Final classifications

```text
CAP_RELAUNCH_PASS
MIX_RELAUNCH_PASS
```

CAP classification 基于 0-step eval 完成、正式 PID 仍 alive、未出现 immediate
recurrence；其 runner 当前不暴露 live env-step counter。MIX classification
有 telemetry 直接支撑已超过 636 failure point。
