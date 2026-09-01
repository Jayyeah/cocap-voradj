# PPO-CF / Counterfactual Central-Q 审计（2026-09-01）

## 1. 问题与实验边界

本线只回答一个问题：MAPPO-9-v2 的 action-free centralized `V_i(s)` 是否因缺少联合动作条件而给不出“第三架是否真正补位”的细粒度 credit。

父实验固定为 MAPPO-9-v2。Pure-Capture corrected 4P1E1obs、local Legacy-VorAdj、IQN decision feature、AW9、shared categorical decentralized Actor、K3/K10、fixed swept collision、reward、spawn/APF/horizon、PPO clipping/target-KL/entropy/LR/epochs/minibatches、ValueNorm、三 seed 与评估协议均不变。唯一主要变量为：

```text
centralized V_i(s) + GAE
                 ↓
centralized Q_i(s, a_-i, a_i=0..8) + exact counterfactual advantage
```

因此命名为 `ppo_cf` / PPO-CF，而不写成 canonical MAPPO。

## 2. Central-Q 实现

`CentralCounterfactualQNetwork` 复用现有 centralized entity Transformer 的成熟结构：focal self token、focal-relative pursuer tokens、global evader/obstacle tokens、type embedding、padding/active mask。每个 teammate pursuer token附加其实际 AW9 action embedding；focal 对角 action embedding被显式 mask，故 `Q_i` 输入不泄漏 `a_i`。

Transformer focal feature经一个 head一次输出9个数：

```text
Q_i(s, a_-i, a_i=0...8) ∈ R^9
```

AW9 索引顺序继续使用 MAPPO-9-v2/IQN 的 `[(a,w) for a in (-.4,0,.4) for w in (-π/6,0,π/6)]`，没有改 Actor head、动作物理语义或 observation。

## 3. Counterfactual advantage 与 critic target

rollout保存实际 `action_indices`、行为策略9维概率、下一状态 teammate action/probability、global/local observations以及 terminal/truncation/active mask。对实际动作：

```text
Q_chosen = Q_i[a_i]
b_i = Σ_k π_i(k|o_i) Q_i(k)
A_i_CF = Q_chosen - b_i
```

9动作精确枚举，无 Monte-Carlo、Gumbel 或混合 `A_GAE`。Actor仍用父实验 PPO ratio、clip、entropy、target-KL、3 epochs×2 minibatches和 actor LR `3e-5`；仅主要 advantage 来源改为标准化后的 `A_CF`。

Critic第一版保留 per-agent reward，用实际 chosen action的 `Q_i` 拟合 TD(λ) return。terminal不bootstrap；truncation和rollout边界bootstrap；dead/padded agent置零。ValueNorm沿用父实验尺度。未选择的8个Q没有伪标签，只通过共享参数泛化。

第一版有意不引入 twin/target Q 或 team reward。这是一条 on-policy PPO 隔离线，固定 rollout 开始时的 online Q产生bootstrap/advantage，再在PPO epoch内更新；若长训出现 Q/grad失控，先归因 implementation/scale audit，不能当作 counterfactual credit失败。

## 4. 恢复与设备隔离

PPO-CF rolling full-resume包含 Actor/Q、双 optimizer、ValueNorm、环境、未满rollout、Python/NumPy/Torch/runner RNG，并增加 `ppo-cf-rollout-v1` schema，拒绝字段语义不匹配的旧恢复包。

GPU RNG只保存/恢复 assigned device。GPU1进程不调用 `get_rng_state_all/set_rng_state_all`，避免初始化或占用GPU0 CUDA context。对下一行为动作的preview也只快照GPU1 RNG并恢复；非terminal transition的preview与下一步真实采样一致，terminal时该值被no-bootstrap mask消除。

## 5. 验证

- exact baseline、focal action indexing：PASS；
- focal action排除、teammate action conditioning、inactive/padding与非法索引：PASS；
- 人工 trajectory：terminal、truncation、rollout boundary、active/dead、TD(λ)：PASS；
- Actor/Q均更新、finite grad、checkpoint state roundtrip：PASS；
- 父 MAPPO-v2/AW-v2与central critic定向回归：PASS；
- CPU 2→4 full-resume：PASS；
- GPU1 CUDA 2→4 full-resume（含assigned-device RNG修正后fresh重跑）：PASS。

最终CUDA smoke的 step4 指标为：actor loss `-0.02126`、critic loss `.90653`、KL约 `2e-6`、actor/critic raw grad `3.987/7.092`、Q chosen/baseline `.0486/.0649`、raw `A_CF` mean/std `-.0163/.0779`、9-action Q span `.2837`，均finite。

## 6. 正式训练与 gate

配置：`configs/experiments/ppo_counterfactual_q_20260901/{common,seed1,seed2,seed3}.yaml`。seed与 MAPPO-9-v2 paired：`2026083001/2/3`；每seed 400k，每25k deterministic20 + stochastic20，normal/stationary、collision types、2+/3+、hold/angular、return与PPO/Q健康指标全部保留。

GPU1 supervisor逐seed串行，要求独占GPU1、至少27 GiB根盘空闲，支持rolling resume和最多3次异常重启。连续3点 KL>`.1`或 Q critic loss/grad/Q绝对尺度越过紧急阈值会暂停，避免把数值崩溃解释成算法结论。

预声明主要比较对象为 MAPPO-9-v2（best deterministic三seed `10%/50%/5%`，mean capture `21.67%`、mean collision `78.33%`）：

- 三seed各至少两个nonzero deterministic节点，mean best capture至少30%且高于21.67%，mean collision低于78.33%：`PASS_COUNTERFACTUAL_CREDIT_IMPROVEMENT`；
- 优化健康但没有清晰超过parent：`HEALTHY_NO_CLEAR_IMPROVEMENT_OVER_MAPPO9_V2`，停止继续堆Q；
- Q/grad/return尺度失控：`UNHEALTHY_Q_IMPLEMENTATION_OR_SCALE_AUDIT`，先停机审计。

长训runtime状态在本文件后续提交中追加；在正式结果未知前不自动延长到500k。

