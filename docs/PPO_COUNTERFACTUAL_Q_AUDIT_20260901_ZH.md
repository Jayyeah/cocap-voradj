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

## 7. 正式启动状态（2026-09-01 21:58 CST）

```text
STATUS: seed1 ACTIVE / optimization healthy
COMMIT_AT_LAUNCH: 4f7216fb8ec786a58be2e922a7ccae6dcd720267
SUPERVISOR_TMUX/PID: cocap_ppo_cf_supervisor_gpu1 / 265176
TRAIN_TMUX/PID: cocap_ppo_cf_seed1_gpu1 / 263896
SEED: 2026083001
STEP: 4k / 400k
RESTARTS: 0
CHECKPOINT: 首个25k尚未到；2→4 full-resume已由独立CUDA smoke验证
```

step4k状态：throughput `32.56 step/s`、runner ETA约3.38小时（不含25k formal评估）；entropy `2.1891`、max KL `2.87e-5`、clip `0`、Actor/Q raw grad `3.626/3.064`、critic loss `1.924`、Q chosen/baseline `-15.09/-15.11`、raw `A_CF` std `1.616`、Q span `4.974`，全部finite，health streak均为0。

GPU1仅有训练PID 263896，约5.99 GiB VRAM；assigned-device RNG验证后GPU0没有PPO-CF compute context。启动时根盘46.35 GiB空闲、RAM约88 GiB可用。GPU温度瞬时85°C，低于93°C max operating与95°C slowdown阈值，继续由每60秒资源状态监控。

supervisor重启合同已现场验证：只重启supervisor时能识别自己的active seed tmux并以 `already_running` 接管，不会重启child；每个新seed或异常restart前都会重新检查GPU1无外部compute app和free VRAM阈值。

## 8. 已完成种子与剩余ETA（2026-09-02）

seed1/seed2均自然完成 `400000/400000`，各16个25k评估节点、轻量checkpoint完整落盘，terminal full-resume按配置删除；两者均0 restart、Q/critic/PPO指标finite且未触发health暂停。

| seed | wall time | best deterministic capture/collision | final Q chosen/baseline | final Q span | final critic loss | explained variance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.11 h | 0 / 0 | 12.73 / 12.73 | 1.34 | .0080 | .976 |
| 2 | 8.25 h | 0 / 0 | 9.27 / 9.26 | 1.15 | .0203 | .826 |

两seed的 centralized-Q 拟合和梯度健康，但 deterministic policy没有出现capture；因此当前证据不支持“仅把V换成counterfactual Q就缩小MAPPO gap”。3-seed gate仍不能提前封版，也不能把这两个seed的健康优化误写成算法PASS。

seed3当前未启动，supervisor状态为 `waiting_for_exclusive_gpu1`。GPU1唯一compute app是范围外的 OmniVLA 服务：PID `487833`，命令 `limo_adapter.server_omnivla_full`，占用约17,006 MiB；未对其执行kill或重启。按seed1/2观测，GPU1一旦释放，seed3的400k训练预计约 `8.2–8.5 h`，再加最后评估与gate约数分钟；释放时间本身不可从仓库推定，故当前ETA是“外部GPU1释放后约8.5小时”，不是确定日历时间。supervisor会自动保留等待并在GPU独占后启动。

## 9. 2026-09-03 实际状态复核

截至 `2026-09-03 08:52 CST`，本节只记录现场状态，不改写第8节的结果：

```text
GPU1: 17029 MiB used / 31511 MiB free / 0% util
PPO-CF supervisor: tmux=cocap_ppo_cf_supervisor_gpu1, PID=265176
supervisor state: waiting_for_exclusive_gpu1, seed=3
blocking compute app: PID=487833, python -m limo_adapter.server_omnivla_full, ~17006 MiB
seed1: 400k COMPLETE; seed2: 400k COMPLETE; seed3: not started
```

seed1/2的400k终点评估均为 deterministic capture `0/20`、collision `0/20`；Q chosen/baseline分别约 `12.73/12.73` 与 `9.27/9.26`，Q span `1.34/1.15`，critic loss `.0080/.0203`，explained variance `.976/.826`，`A_CF` std约`.266`，没有出现NaN/Inf、KL/clip或Q/grad失控。结论仍是 `HEALTHY_NO_CLEAR_IMPROVEMENT_OVER_MAPPO9_V2` 的候选，而不是 counterfactual credit PASS；正式三seed gate必须等待seed3。

GPU0已空闲，未发现本线训练子进程。seed3只有在外部PID释放且GPU1独占检查通过后才会自动启动；按seed1/2含评估墙钟约 `8.2–8.5 h`，另加收尾 formal/gate 数分钟。外部服务释放时刻不可从仓库推定，因此不报日历 ETA。当前磁盘约 `40.3 GiB` 可用、RAM available 约 `100.6 GiB`。

## 10. 2026-09-03 阶段结论复核

最新 `status.json` 与 400k evaluation 确认 seed1/seed2 均自然完成，没有 restart 或 health pause，且 deterministic capture 都是 `0/20`。两 seed 末端的数值仍健康：

| seed | critic loss | explained variance | Q chosen / baseline | Q span | A_CF std | approx KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `.0080` | `.976` | `12.73 / 12.73` | `1.34` | `.266` | `.000146` |
| 2 | `.0203` | `.826` | `9.27 / 9.26` | `1.15` | `.266` | `.000016` |

这支持“centralized-Q 拟合/优化健康，但简单 V→counterfactual Q 尚未带来 capture 改善”的两 seed provisional 判断；不能提前改写成三 seed final negative。seed3 尚未启动，supervisor 为 `waiting_for_exclusive_gpu1`，外部 GPU1 PID `487833` 未被触碰。其条件 ETA 是外部进程释放后约 `8.2–8.5 h`，再加最后 formal/gate 数分钟。

若 seed3 也在健康优化下保持 deterministic capture `0/20`，再将 gate 封为 `HEALTHY_NO_CLEAR_IMPROVEMENT_OVER_MAPPO9_V2`，停止无依据地继续增加 critic 复杂度；若出现 Q/grad/return 尺度失控，则先按实现/尺度问题审计，不能直接解释为算法失败。
