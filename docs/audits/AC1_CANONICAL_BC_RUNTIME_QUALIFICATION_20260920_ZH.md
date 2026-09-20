# AC-1 — Canonical Strong BC Runtime Qualification

日期：2026-09-20
本轮仅做 checkpoint identity、合同审计、offline inference smoke 与有限 CPU runtime qualification；未训练、未生成 trajectory bank、未运行 critic/PPO。

## Identity

Canonical historical strong Full-Task BC：

```text
artifacts/2026-09-08_forward_final/c2_distillation/actor_epoch_030.pt
```

当前 audit worktree 实际加载副本：

```text
artifacts/2026-09-17_critic_identifiability_audit/frozen_policy/actor_epoch_030.pt
```

两者 SHA-256 均为：

```text
7916a970226a0452a8f71a22235524f6ba4511aa4c9bda907a9c8368a60bf2cd
```

结论：`checkpoint identity VERIFIED`。严格 state_dict 加载通过，critical key 无 missing/unexpected；actor 是 Final IQN decision representation 的 86-key frozen backbone + 9-action categorical policy head。

## Contract

结论：`CONTRACT_MATCH`。

使用的 canonical contract：`forward-final-aw9-4v1-swept-v1`，来源为：

- environment：`configs/experiments/forward_final_mappo_20260908/stage1_4v1.yaml`
- historical base：`configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/stage1_4p1e1obs_scratch2m.yaml`
- evaluator：`tools/run_forward_final_bridge_20260908.py`
- resolved contract：`artifacts/2026-09-20_ac1/AC1_BC_RESOLVED_CONTRACT.json`

live/runtime facts 与历史 BC C3 合同一致：120×120；mixed 4v1、pure coverage 4v0；1 obstacle；horizon 3000；capture radius 8、k=3；capture 后不 terminal、post window 500；CE enabled、minimum active pursuers 4；synchronized swept collision semantics。

Sensing 为 enemy/obstacle 20m surface clearance、friend-only Voronoi/free-mask projected、无 global enemy token；不是 Z-token、unified-decay、NormSense、Pure-Capture 或 critic-audit 特制合同。

Action contract 为 AW9 index `0..8`：

```text
[(a,w) for a in (-0.4, 0, 0.4) for w in (-pi/6, 0, pi/6)]
```

runtime action grid 与 environment action list bitwise-equivalent within `1e-6`。

## Offline smoke

历史 C1 NPZ dataset shard 当前 checkout 不可用，因此：`offline historical dataset unavailable`；没有伪造 teacher-action agreement/top-3 overlap。

仍完成了 live environment 上的 bounded inference smoke：

- actor `eval()`，forward finite，无 NaN/Inf；
- categorical probability sum 最大误差 `1.19e-7`；
- deterministic argmax 与 sampled action 均可执行，index 均在 `0..8`；
- backbone trainable parameter 数为 `0`；
- tensor checksum 前后均为 `d626fe2cd19b1cc40fadabf85fdbff8d8e151969c707d9a777f081d5b893cfa0`；
- optimizer updates：`0`。

## Rollout

使用 CPU、固定 seed base `2026092101`，每个 mode 每个 scene 20 episodes；coverage 使用 base+100000。复用 canonical Forward Final evaluator，未改环境、reward、sensing 或 action mapping。

### Pure Coverage

| mode | CE success | safe completion | collision | mission time mean / P50 / P90 (s) |
|---|---:|---:|---:|---:|
| argmax | 20/20 | 20/20 | 0/20 | 75.225 / 65.5 / 109.8 |
| sampled | 20/20 | 20/20 | 0/20 | 79.55 / 65.25 / 111.5 |

### Pure Capture

`not part of canonical BC qualification`。历史 strong BC qualification 使用 mixed full-task，其中 capture 是 mixed episode 的真实 prefix；本轮未人为创建独立 pure-capture 合同。

### Mixed Full Task

| mode | capture | CE success | safe completion | collision | capture time mean / recovery time mean / mission time mean (s) |
|---|---:|---:|---:|---:|---:|
| argmax | 20/20 | 19/20 | 19/20 | 0/20 | 40.8 / 65.16 / 105.89 |
| sampled | 20/20 | 20/20 | 20/20 | 0/20 | 42.48 / 85.0 / 127.48 |

两种模式均达到本轮 positive-control gate 的 `>=18/20`；没有异常 collision mode、runtime assertion failure 或 recovery system-wide failure。任务时间仍处于历史 C3 数量级（历史 100-episode BC mixed mission mean：argmax 117.02s、sampled 112.98s）。

## Argmax vs sampled

历史验证过两种执行模式；本轮两者均通过：

- argmax：mixed 19/20 safe，coverage 20/20 safe；
- sampled：mixed 20/20 safe，coverage 20/20 safe；
- 两者 collision 均为 0/40，总体无采样导致的安全回归。

## Action mapping

`MATCH`。actor 输出 index `0..8` 与 canonical Final IQN AW9 的 `(a,w)` ordering、longitudinal acceleration、angular velocity、unicycle coordinate convention 完全一致；runtime assertion 通过。

## Verdict

```text
BC_QUALIFIED
```

这只表示 canonical historical strong BC 在当前 resolved contract 下通过小样本 runtime qualification；不表示已完成 critic training，也不进入 AC-2。
