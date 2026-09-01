# AC / CTDE Gap 系统审计（2026-08-30）

## 1. 状态与结论

`STATUS: AUDIT_COMPLETE / MAPPO-9-v2_THREE_SEED_COMPLETE_GATE_PASS / TD3_THREE_SEED_COMPLETE`

`IMPLEMENTATION_COMMIT: dda2a23f20c4b857bc958c856d76bf869b50df02`

本轮结论不是“PPO 一定不适合”，而是旧 MAPPO-9 不能作为严格 IQN→Actor-Critic 对照：它把 IQN 的完整 decision feature 简化成了 `self token + mean pooling`，同时使用 10 epochs、actor LR `1e-4`，没有 target-KL 和 ValueNorm。旧 MAPPO-AW 进一步出现 KL/clip 失控和饱和坏吸引子。TD3-AW 则是另一类故障：target 公式与 twin-Q 基本正确，但 Q 绝对值、twin gap 与裁剪前 critic gradient 随训练扩大，且 actor 使用 centralized joint-action gradient，不是历史 all-agent MASAC 的 focal-gradient 语义。

因此本轮建立的新锚点是：

```text
corrected Pure-Capture AW9
+ exact legacy IQN decision-feature backbone
+ categorical shared actor
+ action-free centralized V
+ explicit terminal/truncation/active masks
+ ValueNorm
+ 3 PPO epochs / 2 minibatches / actor LR 3e-5
+ target-KL 0.02
```

三 seed 均为 400k，每 25k formal eval。只有该线 PASS，才进入 MAPPO-AW-v2；若优化健康但任务仍失败，则转向 discrete centralized-Q / counterfactual CTDE，不盲调 PPO。

参考实现只用于公式与工程习惯核对：MAPPO 论文为 <https://arxiv.org/abs/2103.01955>，官方实现为 <https://github.com/marlbenchmark/on-policy>。本仓库没有复制其环境或网络。

## 2. 变量分类

| 项目 | 旧 MAPPO-9 | MAPPO-9-v2 | 分类 |
| --- | --- | --- | --- |
| environment/reward/K10/APF/spawn/horizon | corrected Pure-Capture | 完全相同 | `SAME` |
| action | categorical AW9 | categorical AW9 | `SAME` |
| decentralized information | local VorAdj，无 global enemy | 相同 | `SAME` |
| actor representation | local Transformer 后 `self+mean` | exact legacy IQN decision feature | `TASK_REQUIRED_FIX` |
| learning algorithm | PPO/GAE | PPO/GAE | `ALGORITHM_REQUIRED` |
| centralized critic | action-free V | action-free V | `ALGORITHM_REQUIRED` |
| PPO intensity | 10 epochs、4 minibatches、LR `1e-4` | 3 epochs、2 minibatches、actor LR `3e-5` | `OPTIMIZER_REQUIRED_FIX` |
| target-KL | 无 | `0.02`，minibatch 后 early stop | `OPTIMIZER_REQUIRED_FIX` |
| value scale | raw | ValueNorm (`beta=.99999`) | `OPTIMIZER_REQUIRED_FIX` |
| checkpoint | 25k milestone + rolling resume | 相同并含 ValueNorm/runtime | `SAME/INFRA` |

## 3. Representation parity

### 3.1 IQN 的真实 decision feature

`CoCapIQN` 的 `voradj_single_head` 不是简单 pooling。它依次包含：

1. self/pursuer/evader/obstacle typed entity encoders；
2. local entity Transformer；
3. self、masked mean、masked max；
4. evader target query/key/value attention；
5. mean/max/target 三角色 summary attention；
6. `is_pursuing` scalar embedding；
7. `summary_fusion + single_action_feature`；
8. 最后才进入 quantile embedding 与 Q head。

旧 MAPPO-9 actor 只保留第 1–3 项中的 `self+mean`，遗漏 target、max、summary role 与 pursuing role，故不是严格桥接。

### 3.2 新 Legacy backbone

实现：`src/cocap_voradj/models/continuous/local_entity_token_encoder.py::LegacyVorAdjFeatureBackbone`。

- 复制且仅复制 IQN decision feature 所需模块；
- state-dict key 与 IQN 原 key 对齐；
- 不移动、重命名或修改 `CoCapIQN`，历史 IQN checkpoint loading 不变；
- categorical head 为 `legacy feature -> MLP -> logits[9]`；
- policy MLP 使用 orthogonal `sqrt(2)`，logits head gain `0.01`；backbone scratch 初始化仍与 IQN 模块默认一致。

`test/test_mappo_v2_contract.py` 把同一 IQN state dict 严格装入 backbone，在 eval mode 对随机 observation 做 bit-exact feature parity（`rtol=0, atol=0`），同时确认 IQN 原 state-dict keys 未改变。

## 4. GAE 与 mask

当前实现计算：

```text
delta_t = r_t + gamma * (1 - terminated_t) * V(s_{t+1}) - V(s_t)
A_t = delta_t + gamma * lambda * (1 - episode_end_t) * A_{t+1}
return_t = A_t + V(s_t)
```

语义：

- true terminal：不 bootstrap；
- time-limit truncation：从 `next_value` bootstrap，但停止跨 reset 的 GAE recursion；
- inactive/dead row：从 advantage、loss、normalization 中 mask，且递归状态归零；
- rollout boundary 若不是 episode end，可继续用显式 `next_value`；
- `episode_end` 强制并入 terminal/truncation，不能因 caller 漏字段而泄漏。

旧公式的核心方向基本正确，但 rollout 没有单独持久化 `truncated`。v2 已显式保存，旧 in-memory rollout 缺字段时只做向后兼容的全 false fallback。单测分别证明 terminal advantage 为 `r-V`，truncation advantage 为 `r+gamma*V_next-V`，且不跨 episode 递归。

## 5. PPO ratio、advantage 与 update

- old log-prob 在 rollout 时保存，update 时 detach；
- categorical actor 以当时采样的 action index 重新计算 log-prob；
- `ratio = exp(new_logp-old_logp)`；
- surrogate 为标准 clipped minimum；
- advantage 只在 active rows 上做 mean/std normalization；
- approx-KL 使用非负二阶近似 `mean((ratio-1)-log_ratio)`；
- 同时记录 mean KL、max minibatch KL、clip fraction、entropy、grad norm、explained variance、PPO epochs、minibatch updates；
- actor 参数更新量记录绝对/相对 L2；
- minibatch KL 超 `0.02` 即结束本 rollout 后续 update；连续 formal log 若 KL `>0.05` 或 clip `>0.3` 标 WARN，KL `>0.1` 连续三点则 supervisor 暂停 seed。

配置从 10 epochs/4 minibatches 降到 3/2；actor LR 从 `1e-4` 降到 `3e-5`，critic LR 保留 `1e-4`。这不是 sweep，而是针对旧线长期高 KL/clip 的单一保守 recipe。

## 6. ValueNorm 审计与 smoke 中发现的问题

ValueNorm 保存 running mean、mean-square、debias term；variance floor 为 `1e-2`，避免早期极小 variance 把 target 放大数百倍。状态被纳入 `MAPPOTrainer.state_dict()` 和 rolling resume。

第一次 GPU smoke 发现：若 rollout 保存 raw value，更新 running stats 后再重新 normalize old value，value clipping 可能落在 clamp 的零梯度支路；首个 update 出现非零 value loss 但 `value_grad_norm=0`。已修为官方语义一致的两尺度合同：

1. rollout 保存 critic 原始的 normalized prediction；
2. GAE 在更新 stats 前用旧 ValueNorm 反归一化；
3. returns 计算后更新 stats并 normalize target；
4. PPO value clipping 对比 rollout 保存的原 normalized prediction。

修复后全新 2-step GPU smoke：value loss `0.1106`、裁剪前 value grad norm `6.1316`、actor update L2 `0.05059`、max KL `1.38e-4`，resume/milestone/formal eval 均成功。

## 7. Centralized V

`CentralValueNetwork` 是 action-free training-only critic。每个 focal pursuer 的 global state 包含：

- focal self token（独立 type 0，提供 focal identity）；
- 所有 pursuer world tokens 与 focal-conditioned pursuer mask；
- all evader/obstacle tokens及 mask；
- active mask。

输出为每个 pursuer 一个 `V_i(s)`；inactive rows 为零。Actor 仍只接受 local observation，未泄漏 global state。Actor/critic backbone 不同是 CTDE 算法要求，不是 representation parity 违规。

## 8. TD3 / centralized-Q 审计

### 8.1 正确项

- target：`r + gamma*(1-terminated)*min(Q1',Q2')`；truncation bootstrap；
- inactive rows从 critic loss/Q metrics排除；
- twin critics独立；
- target policy smoothing 先在 normalized action scale 加噪，再按 `(a,w)` scale，最后物理 bounds clip；
- actor/target soft update只发生在 delayed policy step；
- checkpoint含 actor、critics、两 target、两 optimizer 与 update count。

`tests/test_td3_centralized_audit.py` 覆盖 terminal/truncation、dead-agent reward mask、target noise、delayed update和梯度语义。

### 8.2 Gradient semantics

当前 actor loss 为：

```text
-mean_i Q_i(s, pi(o_1), ..., pi(o_N))
```

所以每个 `Q_i` 对所有 agent action branch 反传，是 centralized joint gradient。单测构造显式 cross-agent critic，证明 teammate branch 没有 detach。这不是数学上非法，但与历史 all-agent MASAC 的 focal contract不同：后者对 `Q_i` 只让 `a_i` 可微，`a_-i` stop-gradient。共享 actor下 joint gradient 会把同一参数经 `N×N` action dependency重复累计，更容易放大 critic credit/scale 问题。

本轮不直接把 TD3 改成 focal：这会成为新算法变量。若后续做 TD3-v2，必须把 `joint` 与 `focal` 做明示的单变量消融。

### 8.3 数值证据

| run | step | capture | collision | critic loss | Q1/Q2 | twin gap | raw critic grad |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| seed1 final | 300k | 0% | 55% | 79.08 | -70.27 / -70.45 | 4.54 | 1156.9 |
| seed2 final | 300k | 0% | 85% | 64.99 | -48.05 / -49.18 | 4.60 | 1165.9 |
| seed3 final | 300k | 0% | 65% | 49.10 | -51.77 / -52.44 | 4.20 | 903.4 |

梯度实际 clip 为 `0.5`，表中是裁剪前值。三 seed 均已自然完成且 final capture 都为 0；seed3 deterministic final 另有 collision `65%`、visited2+ `40%`、visited3+ `0%`、mean return `-569.55`。没有 NaN，但三 seed 的 Q/gradient 尺度与任务失败共同排除了“只差更多步数”；TD3-v2 不自动启动。

旧日志每 1000 env steps恰落在 critic-only update，导致 actor loss/grad 表面恒为零。现已在不改 update 的前提下新增 `actor_updated_this_step`、`last_actor_loss/grad/update_count`，并纳入 resume；旧 `actor_loss` 仍严格表示当前 update。

## 9. Reward fidelity：success 与 high-3+/hold failure

### 9.1 历史同批 deterministic20

三 seed deterministic20 合并：

| checkpoint/group | n | capture | mean return | mean 3+ hold | collision | mean length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 225k captured | 42 | 42/42 | +99.08 | 3.79 | 0% | 195.7 |
| 225k failed but visited 3+ | 8 | 0/8 | -159.14 | 11.50 | 100% | 256.1 |
| 300k captured | 5 | 5/5 | +170.79 | 11.80 | 0% | 310.6 |
| 300k failed but visited 3+ | 46 | 0/46 | +51.76 | 36.85 | 45.7% | 780.4 |

225k→300k 的跨种子 deterministic 曲线为：capture `70%→8.3%`，visited3+ `45%→81.7%`，mean max 3+ hold `27.3→75.7`，return `+33→+51`。这批同评估数据先暴露了“失败回合变长但 return 上升”的异常，但 role-level 全回合均值不足以定位尾段奖励来源。

### 9.2 冻结新 seeds 的 paired reward-tail audit

为避免用原20回合曲线反复解释，本轮对统一 `225k` 与 `300k` checkpoint 使用相同、独立的新 evaluation seeds，各做 `3×20` 回合；IQN 使用 fixed midpoint τ，并逐回合保存最后100/200 environment steps的 reward component、role 与碰撞事件。聚合源为 `aggregate_uniform225_tailaudit.json` 与 `aggregate_uniform300_tailaudit.json`。

| checkpoint/group | n | capture | mean return | mean 3+ hold | collision | mean length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 225k captured | 38 | 38/38 | +57.87 | 4.87 | 0% | 238.92 |
| 225k failed but visited 3+ | 9 | 0/9 | -231.28 | 11.22 | 77.78% | 429.67 |
| 300k captured | 8 | 8/8 | +241.19 | 12.25 | 0% | 170.75 |
| 300k failed but visited 3+ | 36 | 0/36 | +20.19 | 34.64 | 47.22% | 773.08 |

最后100步的 agent-step 均值进一步给出机制：

| checkpoint/group | approach | mean-shift | front | capture shaping | terminal | total | direct role |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 225k captured | .15484 | .33838 | .03937 | .53260 | 1.70770 | 1.20739 | 78.75% |
| 225k failed 3+ | .05395 | .31868 | .03896 | .41159 | 0 | -1.24952 | 82.53% |
| 300k captured | .28274 | .44905 | .03165 | .76344 | 2.02071 | 2.59710 | 77.63% |
| 300k failed 3+ | .00196 | .23541 | .04241 | .27977 | 0 | -.52648 | 88.72% |

200步尾窗得到同方向结果：225k→300k 的 failed-3+ 中，approach `.08392→.01500`、mean-shift `.28186→.26064`、capture shaping `.40755→.31445`、terminal 始终为0。300k 失败组不是获得了更高的瞬时 shaping；它的回合约为225k失败组的 `1.80×`、3+ hold约为 `3.09×`，在 approach 几乎消失后仍长时间维持 mean-shift/ring shaping。其最后100步 direct-capture role 占比反而升到88.72%，说明角色标签不等于 terminal progress。

因此可归因的结论是 **proxy/time-horizon mismatch**：更长的 nonterminal dwell 让全回合 return 仍可为正，而 true capture terminal 没发生。碰撞率同时从77.78%降到47.22%，也排除了“300k只因更频繁即时碰撞而失败”的简单解释。本任务只增加观测，不改 reward；后续 reward 改动必须另做单变量实验。

当前动作是：

- 不改 reward；
- fixed midpoint τ、冻结新 eval seeds，对精确 best 做独立 100 episodes；
- Full-VXY 第一轮保持 Final-AW reward；
- 后续若专门改 reward，必须把 terminal、ring progress、support、collision/boundary逐项单变量消融。

## 10. MAPPO-9-v2 正式合同

配置：`configs/experiments/mappo9_v2_20260830/{common,seed1,seed2,seed3}.yaml`。

```text
SEED: 2026083001 / 002 / 003
STEPS: 400k each
CHECKPOINT: 25k
EVAL: deterministic20 + stochastic20
ROLLOUT: 256
PPO: epochs=3, minibatches=2, clip=.2
LR: actor=3e-5, critic=1e-4
TARGET_KL: .02
VALUE_NORM: beta=.99999, variance floor=.01
```

supervisor：`tools/supervise_mappo9_v2_20260830.py`。三 seed 均已自然完成400k，累计0次重启；supervisor于2026-08-31 13:28 CST正常写入 `gate_decision.json` 后结束。三seed best分别为：seed1 250k capture10%/collision90%，seed2 250k capture50%/collision50%，seed3 400k capture5%/collision95%；mean best capture `21.67%`、mean collision `78.33%`、all-seed nonzero=true、optimization healthy=true。

预声明gate结果为 `PASS_TO_MAPPO_AW_V2`。这是“离散Actor-Critic桥接已达到最低可迁移门槛”，不是强策略结论：seed1/3碰撞仍高，且final分别为0%/100%、5%/95%、5%/95%。下一步已路由到MAPPO-AW-v2，但本轮状态检查时尚未启动，因此没有可报告的训练ETA。

- 三 seed重复非零且平均 best capture ≥10%、collision <80%：`PASS_TO_MAPPO_AW_V2`；
- 优化健康但任务失败：`HEALTHY_FAIL_TO_DISCRETE_COUNTERFACTUAL_Q`；
- 优化不健康：暂停实现审计。

## 11. 验证状态

- legacy decision feature bit-exact parity：PASS；
- GAE terminal/truncation/mask：PASS；
- ValueNorm roundtrip/variance/resume：PASS；
- target-KL early stop：PASS；
- actor update magnitude：PASS；
- first-update critic nonzero gradient：PASS；
- TD3 target/noise/twin/delay/joint-gradient：PASS；
- MAPPO-v2 2-step GPU smoke + checkpoint + full resume + dual formal eval：PASS。
- MAPPO-v2 long-run：3×400k COMPLETE，0 restart，optimization healthy；预声明gate为 `PASS_TO_MAPPO_AW_V2`。
- TD3-AW seed3 300k + final dual formal eval：COMPLETE；三 seed 原配方任务性能最终失败。

三种子正式结果已完成；结论是桥接门槛PASS但策略仍弱，后续连续动作因果结论必须等待MAPPO-AW-v2。


### 11.1 MAPPO-9-v2 运行产物归档复核（2026-09-01）

本地归档根目录为 `artifacts/2026-08-30_mappo9_v2/`。三 seed 的 `status.json` 均为 `complete / 400000`，每 seed 均保留16个25k checkpoint、16份对应deterministic20+stochastic20评估、rolling resume、manifest、effective config、episodes与learning metrics；根目录 `gate_decision.json` 与supervisor终态文件完整存在。大体积运行产物按仓库策略仅保存在本机，Git台账归档的是可审计路径、汇总结论与gate，不上传checkpoint。
