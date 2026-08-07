# CoCap-VorAdj 连续 MARL 正反馈阶梯推进方案

**版本：** 2026-08-07 v1  
**适用仓库：** `Jayyeah/cocap-voradj`  
**当前工作分支：** `continuous/masac-ctde-contract-20260806`  
**基准分支：** `main`  
**目标：** 在不破坏旧 IQN 成功主线的前提下，用每次只改变一个核心变量的方式，逐步建立连续控制成功锚点，最终推进到完整的 CTDE MASAC、mixed capture→coverage 和二维连续加速度版本；若最终版本尚未成功，至少获得明确、可重复的乐观学习信号与清晰阻塞定位。

---

## 0. 本文的地位

本文是后续“正反馈阶梯”路线的最高优先级执行合同。

现有：

`docs/CONTINUOUS_ACTION_MARL_EXPERIMENT_TRACKER_20260804_ZH.md`

仍保留历史实验，但其中 local SAC、旧 ax/ay、body/world frame、70/20/10 sampler 等早期结论可能已经过时。后续 Codex Agent 必须新建并持续维护：

`docs/CONTINUOUS_MARL_POSITIVE_FEEDBACK_TRACKER_20260807_ZH.md`

新台账只记录本文定义的阶梯路线。旧台账不得删除，但不得作为当前路线的唯一执行依据。

---

# 1. 核心判断

当前项目不是“已经证明 MASAC 不适合”，而是一次性同时改变了过多因素：

- IQN → MASAC；
- 离散动作 → 连续动作；
- `(a,\omega)` → world-frame `[a_x,a_y]`；
- 标量前向运动 + yaw → 全平面运动；
- 原旋转归一化结构 → yaw 恒 0 的固定世界轴结构；
- agent-level replay → joint/focal replay；
- local value structure → centralized twin critic；
- 单阶段成功配置 → capture、coverage、mixed 长时联合任务。

因此，当前 25k 无成功不能直接归因于算法。

新的主线是：

> **先找到第一个可靠的连续控制成功锚点，然后每次只增加一个难点。**

推荐主路径：

```text
旧 IQN 成功复现
→ 旧 IQN 通过连续 (a,ω) bridge 复现
→ MASAC + 连续 (a,ω) + 极简单任务
→ MASAC + 连续 (a,ω) + pure coverage
→ MASAC + 连续 (a,ω) + capture
→ MASAC + 连续 (a,ω) + 双任务交替
→ MASAC + 连续 (a,ω) + mixed capture→coverage
→ body-frame [ax,ay]
→ world-frame [ax,ay]
```

算法替换不是第一步。只有在同一简化任务、同一动作、同一动力学、同一观测和同一奖励下，MASAC 明确失败而 MATD3/MADDPG 明确成功，才允许把算法作为主因。

---

# 2. 三类参数：必须明确标注

任何实验配置、提交和台账条目都必须给每个参数打上以下标签之一。

## 2.1 `[IQN-ALIGN]`：必须与原成功 IQN 对齐

除非某阶段明确写了临时诊断简化，否则以下项目必须与 `main` 的成功配置保持一致：

### 环境

- map：120×120；
- pursuer 初始速度：0；
- evader 行为；
- capture distance；
- related distance；
- boundary 处理；
- collision 处理；
- obstacle 几何；
- spawn edge margin；
- pursuer/evader 最小间距；
- capture 与 coverage 的 spawn 类型；
- reward 各分项及其符号、尺度；
- termination 语义；
- post-capture coverage 语义；
- VCT-LS 实体字段及顺序。

### 感知

- perception range：20；
- FOV：2π；
- enemy sensing radius：20；
- obstacle sensing radius：20；
- surface-distance sensing；
- `global_evader_visibility=false`，除非处于明确的 capture 诊断阶段；
- Actor 不得读取 centralized global state；
- 不得向 Actor 输入 oracle phase、event、origin 或 replay role 标签。

### 原 `(a,\omega)` 动力学

- forward acceleration bound：`a_max=0.4`；
- angular velocity bound：`\omega_max=\pi/6`；
- `v_max=3.0`；
- physics dt：0.05；
- 10 substeps；
- decision interval：0.5 s；
- drag coefficient：`0.4/3.0`；
- yaw 积分；
- robot-frame observation；
- body-frame forward acceleration 语义。

### 优化中应继续对齐的值

- batch size：128；
- gamma：0.99；
- update frequency：每 4 env transitions 更新一次；
- grad clip norm：0.5；
-主要网络 hidden dim：256；
-训练总步数按阶段设置，但超过 25k 时必须每 25k 保存。

## 2.2 `[ALGO-NECESSARY]`：MASAC 必须适配

这些变化是算法差异，不应被误判为“未对齐”：

- stochastic continuous Actor；
- twin centralized critics；
- target critics；
- Polyak `tau=0.005`；
- automatic entropy temperature；
- `alpha_init=0.2`；
- target entropy：
  - 连续 `(a,\omega)`：默认 `-2.0`；
  - 二维 `[a_x,a_y]`：默认 `-2.0`；
- actor/critic/alpha lr：`1e-4`；
- joint replay；
- focal-agent training item；
- Actor 更新时仅 focal action 保留梯度，其他 agent action detach；
- Critic 使用完整 global state、joint action、active/entity masks；
- replay capacity：4-agent joint replay 默认 250000；
- warmup：默认 5000 joint transitions；
- checkpoint 保存 Actor、双 critic、双 target、alpha 和全部 optimizer。

任何适配项都必须在台账中注明：

```text
类型：[ALGO-NECESSARY]
必要性：连续随机策略与 centralized soft-Q 所需
未改变：环境、奖励、感知和物理任务
```

## 2.3 `[DIAGNOSTIC-ADAPT]`：仅限某阶段的必要简化

允许临时改变，但必须说明必要性、适用阶段和恢复条件：

- 无 obstacle；
- stationary evader；
- global evader visibility；
- 单智能体；
- 简化角色；
- 缩短单任务训练 horizon；
- 只训练 pure coverage；
- 只训练 capture；
- 使用旧 IQN snapshot curriculum；
- Encoder transfer；
- uniform-disk warmup；
- diagnostic evaluation cap=400。

每个临时改动必须写：

```text
类型：[DIAGNOSTIC-ADAPT]
必要性：该阶段只验证……
退出条件：达到……后恢复……
禁止外溢：不得自动进入最终 formal 配置
```

---

# 3. 全路线强制执行规则

## 3.1 一次只改变一个核心变量

同一个实验对照中禁止同时改变：

- 算法；
- 动作空间；
- 动力学；
- 观测坐标；
- reward；
- map；
- obstacle；
- visibility；
- replay quota；
- update ratio；
- grad clip。

若必须连带改变，例如从 `(a,\omega)` 改为 `[a_x,a_y]` 必须同时更换动作头和动作缩放，应在台账中声明它们属于同一个不可分割动作合同，而不是两个独立实验变量。

## 3.2 每 25k 必须保存

所有超过 25k transition 的训练，必须在以下里程碑保存：

```text
25k, 50k, 75k, 100k, 125k, ...
```

不得只在训练结束后保存。

每个里程碑必须包含：

- Actor；
- critic1/critic2；
- target critic1/target critic2；
- log alpha；
- actor/critic/alpha optimizer；
- replay；
- focal indexes；
- recovery/snapshot pool；
- env transition step；
- update count；
- RNG state；
- effective config；
- config hash；
- implementation commit；
- action/dynamics/observation/replay contract hash；
- metrics history；
- 400-step diagnostic evaluation；
- manifest。

保存必须是原子的。checkpoint、replay 和 runtime step 不一致时禁止 resume。

## 3.3 训练检查频率

训练运行时采用低频监控：

- 约每 10 分钟检查一次；
- 不得 30 秒或 1 分钟高频轮询；
- 若训练提前完成或报错，立即处理；
- 每次检查仅记录：
  - 进程是否存活；
  - 当前 step；
  - 最近 checkpoint；
  - 是否出现 NaN/Inf/OOM；
  - GPU/磁盘是否异常；
  - stdout 最后若干行；
- 不因正常训练速度慢而反复重启。

建议使用 `tmux`、`screen` 或 `nohup`，并将 PID、命令和日志路径写入台账。

## 3.4 每阶段必须单独输出完成说明

每个阶段结束，无论通过、部分通过还是失败，Codex 必须在对话中单独输出：

```text
【阶段 X 完成说明】

状态：PASS / OPTIMISTIC_PARTIAL / FAIL / CONTRACT_ERROR
本阶段唯一核心改动：
与 IQN 对齐项：
必要适配项及理由：
运行 seed / steps：
关键 checkpoint：
关键指标：
对照结果：
发现的问题：
是否允许进入下一阶段：
下一步唯一动作：
台账更新位置：
```

不得只说“已完成”或只贴测试结果。

## 3.5 实验台账实时维护

专用台账必须在以下时刻更新：

1. 阶段开始前；
2. 代码或配置修改完成后；
3. 训练命令启动后；
4. 每个 25k milestone 后；
5. 训练异常后；
6. 阶段结论后。

每条 run 至少记录：

- stage；
- run id；
- date；
- branch/commit；
- dirty status；
- config path/hash；
- seed；
- initialization；
- teacher/snapshot dataset hash；
- action contract；
- observation contract；
- dynamics contract；
- reward contract；
- exact command；
- PID/log；
- milestone checkpoint；
-主要连续指标；
- PASS/PARTIAL/FAIL；
-下一步。

## 3.6 评估规则

早期诊断保留用户指定的：

```yaml
diagnostic_rollout_cap: 400
```

用途是避免无行为策略产生过长无效 rollout。

要求：

- 400-step 结果明确标记为 diagnostic；
- 不将 400-step 未成功等同于 full-horizon 失败；
- 当出现积极连续指标后，再增加 full-horizon evaluation；
- 训练 horizon 与 diagnostic evaluation cap 分开；
- 每个关键 gate 至少 3 seeds；
- 工程 gate 可 1 seed；
- 性能晋级原则上要求 3 seeds 中至少 2 个出现一致方向。

## 3.7 当前速度终局奖励保持关闭

在 capture 与 coverage 尚未形成明确几何行为前：

```yaml
coverage_ce_speed_weight: 0.0
```

速度仍作为诊断指标记录，但不进入 reward。

恢复条件：

- capture geometry 或 coverage geometry 已持续接近成功；
- 失败主要表现为末态速度过大、无法停稳；
- 由用户确认后单变量开启。

---

# 4. 固定的 MASAC 基线超参

除非阶段明确说明，否则所有 MASAC 阶段使用：

```yaml
masac:
  gamma: 0.99
  tau: 0.005
  actor_lr: 0.0001
  critic_lr: 0.0001
  alpha_lr: 0.0001
  alpha_init: 0.2
  auto_entropy: true
  target_entropy: -2.0

training:
  batch_size: 128
  warmup_joint_transitions: 5000
  update_every_env_steps: 4
  gradient_steps: 1
  grad_clip_norm: 0.5
  replay_capacity_joint: 250000
  checkpoint_interval_env_steps: 25000
  metrics_flush_interval_env_steps: 1000
  diagnostic_eval_interval_env_steps: 25000
```

网络：

```yaml
actor:
  hidden_dim: 256
  num_heads: 8
  num_layers: 4
  dropout: 0.0
  shared_parameters: true

central_critic:
  twin: true
  hidden_dim: 256
  num_heads: 8
  num_layers: 4
  dropout: 0.0
  max_agents: 12
```

这些超参不得在同一任务诊断中与动作、场景或算法同时改变。

---

# 5. 正反馈阶梯

---

## Stage 0：旧 IQN 成功基线复现

### 目标

证明仓库、环境和旧 checkpoint 仍能复现已知成功结果。

### 唯一任务

在 `main` 或明确基于 `main` 的只读 worktree 中运行旧 IQN checkpoint。

### `[IQN-ALIGN]`

全部使用原成功：

- config；
- checkpoint；
-离散 9 动作；
- `(a,\omega)` 动力学；
- reward；
- spawn；
- perception；
- episode；
- evaluation seeds。

### 禁止

- 不修改旧代码；
- 不改 episode；
- 不用新 Actor；
- 不用新 replay；
- 不用 current CTDE runner。

### 通过条件

- checkpoint 可加载；
- 指标与历史报告同数量级；
- capture、pure coverage、mixed 至少能复现主要成功行为；
- collision 不出现明显回归；
- 生成 20-episode evaluation 和至少若干可视化轨迹。

### 失败处理

停止后续学习实验，先修环境或 checkpoint 回归。

### 完成产物

- `STAGE_0_IQN_BASELINE_COMPLETION.md`
- 评估 JSON；
- config/checkpoint hash；
- 台账条目。

---

## Stage 1：旧 IQN → 连续 `(a,\omega)` bridge 等价验证

### 目标

证明新的连续 action API、runner 和动力学接线不会破坏旧策略。

### 唯一核心改动

旧 IQN 仍输出离散 action index，但将其精确映射为连续数值：

```text
a ∈ {-0.4, 0, 0.4}
ω ∈ {-π/6, 0, π/6}
```

随后通过新的连续 `(a,\omega)` API 执行。

### `[IQN-ALIGN]`

必须保持：

- 同一 observation；
- 同一 robot frame；
- 同一 yaw；
- 同一 drag；
- 同一 substeps；
- 同一 reward；
- 同一 spawn；
- 同一 seed；
- 同一 episode；
- 同一 target policy。

### `[ALGO-NECESSARY]`

无。此阶段不训练 MASAC。

### 通过条件

对固定 seed 进行 trajectory parity：

- action 映射完全一致；
- position、velocity、theta 在数值容差内一致；
- reward term 一致；
- event/capture/collision 一致；
- 成功率与 Stage 0 同数量级。

建议：

- 单步单元测试容差尽量接近浮点误差；
- 长轨迹若因积分顺序存在微小偏差，必须证明不改变行为结论。

### 失败处理

不能进入 Stage 2。定位 bridge、积分、yaw 或 action mapping。

### 完成产物

- `STAGE_1_AW_BRIDGE_COMPLETION.md`
- parity report；
- fixed-seed trajectory diff；
- 台账条目。

---

## Stage 2：MASAC + 连续 `(a,\omega)` 极简单任务

### 目标

建立第一个“连续 Actor + SAC/CTDE 训练链确实可以学会”的成功锚点。

### 场景

优先使用与现有环境结构最接近的简单任务：

```text
1 pursuer
1 stationary evader/target
global visibility
0 obstacle
无 teammate
无 support/coverage 角色
连续 (a,ω)
```

Actor 仍使用 robot-frame local observation。Central critic 在 N=1 时退化为单智能体 centralized critic，但仍走同一 MASAC 代码路径。

### `[IQN-ALIGN]`

- map 120×120；
- physics dt 0.05；
- 10 substeps；
- drag `0.4/3`；
- `a_max=0.4`；
- `ω_max=π/6`；
- `v_max=3.0`；
- robot-frame observation；
- yaw update；
- boundary；
- acceleration/turning semantics。

### `[DIAGNOSTIC-ADAPT]`

- target stationary；
- global visible；
- no obstacle；
- N=1；
- reward只保留距离 progress、成功和安全基本项；
- episode horizon 可缩为 400～600。

必要性：只验证连续 `(a,\omega)`、Actor、critic target、replay、终止和 evaluation 是否可学。  
退出条件：成功后恢复多智能体、原任务 reward 和障碍。

### 动作 Actor

两个独立 squashed Gaussian 输出，不使用 radial disk：

```yaml
action:
  mode: acceleration_angular_velocity_body
  bound_type: independent_box
  low: [-0.4, -0.5235987756]
  high: [0.4, 0.5235987756]
```

必要性：`a` 与 `ω` 单位不同，不能共享 L2 圆盘。

### 通过条件

工程 gate：

- finite；
- replay/target/done 正确；
- action 合法；
- checkpoint 可恢复。

学习 gate，3 seeds 中至少 2 个满足：

- 25k～50k 内成功率明显高于 random/no-op；
- 推荐目标：20 episodes success ≥80%；
- final distance、distance AUC 和 episode length 明显改善；
- collision 可控；
- deterministic Actor 有明确转向和推进。

### 失败处理

若 3 seeds 到 50k 均无方向性信号：

1. 不换复杂场景；
2. 检查 reward、done、target Q、action scaling、observation、replay；
3. 运行 MATD3 同合同对照；
4. MASAC 与 MATD3 都失败则判定为实现问题；
5. MATD3 成功而 MASAC 失败才进入算法原因分析。

### 完成产物

- `STAGE_2_SINGLE_AGENT_AW_COMPLETION.md`
- 3-seed 曲线；
- 25k/50k checkpoints；
- trajectories；
- 台账结论。

---

## Stage 3：MASAC + 连续 `(a,\omega)` Pure Coverage

### Stage 3A：无障碍 pure coverage

#### 场景

- 4 pursuers；
- 0 evader；
- 0 obstacle；
- inner-cluster spawn；
- robot-frame VCT-LS；
- continuous `(a,\omega)`；
- CTDE MASAC；
- CE reward；
- speed reward=0。

#### `[IQN-ALIGN]`

除 obstacle 临时关闭外，全部对齐旧 pure coverage：

- map；
- spawn；
- local observation；
- CE geometry；
- drag/yaw；
- action bounds；
- episode 1500；
- reward尺度；
- batch、grad clip 和 update frequency。

#### `[DIAGNOSTIC-ADAPT]`

`num_obstacles=0`。

必要性：先排除 obstacle avoidance 对早期学习的干扰。  
退出条件：出现稳定 CE 几何改善后恢复 1 obstacle。

#### 通过条件

不强制 strict success。3 seeds 至少 2 个满足：

- median CE/Lloyd energy 从初始到末态下降；
- 建议工程 gate：末态 energy 相对初始改善 ≥15%；
- 相对 random/no-op baseline 有明显优势；
- area CV 或 center-distance 至少一项持续改善；
- collision 不主导 episode；
- action 具有朝 centroid 移动和转向的方向性。

状态可标：

- `PASS`：已有稳定成功或明确几何达标；
- `OPTIMISTIC_PARTIAL`：无 strict success，但连续指标稳定改善；
- `FAIL`：连续指标与 random/no-op 无差异。

### Stage 3B：恢复 1 obstacle

唯一改动：

```text
num_obstacles: 0 → 1
```

其他配置、初始化和 seed suite 不变。

通过条件：

- 保留 Stage 3A 大部分几何收益；
- collision 没有不可接受增加；
- 至少出现 obstacle-aware 轨迹。

### 失败处理

- 3A 失败：检查 CE reward、坐标、Actor转向和 shared credit；
- 3A 成功、3B 失败：问题主要在避障/安全，不换算法；
- 可使用旧 IQN coverage snapshots 作为 curriculum，但必须由新环境重新生成 transition。

### 完成产物

- `STAGE_3A_PURE_CE_NO_OBS_COMPLETION.md`
- `STAGE_3B_PURE_CE_OBS_COMPLETION.md`

---

## Stage 4：MASAC + 连续 `(a,\omega)` Capture 难度阶梯

每个子阶段只增加一个难点。

> 并行规则（用户确认 2026-08-07）：Stage 4A 完成 gate 并启动后，Stage 4B 与 Stage 4C 可作为两个独立的单变量分支并行开启；4B 只改 stationary→moving evader，4C 只改 num_obstacles 0→1，二者互不叠加。

> Seed 数量调整（用户确认 2026-08-07）：Stage 4A 起，性能验证 seed 数由 3 改为 2（至少 1 个明显优于 random/no-op 即可晋级），以加速验证；如需更强证据可追加第 3 seed，但不阻塞晋级。

### Stage 4A：stationary evader + global visibility + no obstacle

目标：证明多智能体可以接近并形成 capture geometry。

临时简化：

- evader stationary；
- global visible；
- 0 obstacle；
- 全体 active pursuers 先统一 pursuing；
- 不引入 support/coverage 角色混合。

通过条件：

- min-distance、ring geometry 或 capture event 显著优于 random；
- 推荐 50k 内 capture rate ≥30%，或连续几何指标达到明确乐观信号。

### Stage 4B：moving evader + global visibility + no obstacle

唯一改动：

```text
stationary evader → 原 IQN evader behavior
```

通过条件：

- capture 或 distance/ring progress 保持正向；
- 若成功率下降但仍显著优于 random，可标 `OPTIMISTIC_PARTIAL`。

### Stage 4C：恢复 1 obstacle

唯一改动：

```text
num_obstacles: 0 → 1
```

通过条件：

- capture progress 保留；
- collision 可控。

### Stage 4D：恢复 local visibility

唯一改动：

```text
global_evader_visibility: true → false
```

恢复原 VCT-LS local sensing。

关注：

- discovery rate；
- discovery step；
- discovery 前后行为；
- discovery 后是否继续接近。

若 global 成功、local 失败，瓶颈是搜索/感知，不应换 MASAC。

### Stage 4E：恢复 support/coverage 角色分工

唯一改动：

- 恢复原 task assignment、support、pre-capture coverage；
- focal replay 使用 64/8/8 配额；
- post-capture 缺失时 32 fallback 到 pure/recovery 是正确逻辑。

通过条件：

- capture progress 没有完全消失；
- support/coverage agents 有合理行为；
- role reward 和 focal sampling 正常。

### 完成产物

每个子阶段单独报告，不得合并成一句“capture 阶梯完成”。

---

## Stage 5：双任务与 mixed

### Stage 5A：Capture/Pure-CE episode-level alternating

### 目标

先证明共享 Actor 能在不同 episode 中同时保留两项技能，不立即要求一个 episode 内 phase transition。

### 场景

训练 episode 按固定比例交替：

```yaml
task_mix:
  capture: 0.5
  pure_ce: 0.5
```

保持连续 `(a,\omega)`、CTDE MASAC、局部观测和完整 obstacle。

### 初始化

优先从表现最稳定的 Stage 4 checkpoint 初始化 Actor。

Critic 是否复用必须做一次小规模对照：

- `actor_only_init`：复用 Actor，重置 critics/targets/optimizers；
- `full_trainer_init`：完整恢复。

默认优先 `actor_only_init`。

必要性：mixed reward/return 分布改变，旧 critic 可能引入偏置；Actor 已学到运动技能，值得复用。

### 通过条件

- capture 指标保留单任务最佳结果的主要部分；
- CE energy/CV 也出现改善；
- 不允许一种任务完全覆盖另一种；
- 若出现 catastrophic forgetting，先调整 episode 比例或 replay balance，不立即修改网络。

### Stage 5B：真实 mixed capture→coverage

### 目标

恢复一个 episode 中 capture 后进入 coverage。

### `[IQN-ALIGN]`

- pre-capture horizon 1000；
- post-capture window 500；
- total 1500；
- local sensing；
- obstacle=1；
-原 reward；
-真实 capture event；
- post-capture 数据仅来自真实捕获或合法 snapshot reset。

### Replay

当前 fallback 逻辑保留：

- post-capture pool 为空时，32 quota 转 pure/recovery；
- 一旦真实 post-capture 数据出现，逐步恢复 64/8/8/32/16；
- 不伪造标签。

### 通过条件

乐观信号可以是：

- capture rate 非零；
- post-capture pool 开始增长；
- capture 后 CE energy 有下降；
- 或 capture geometry 与 pure CE geometry 同时保持正向。

完整通过：

- mixed capture 与 post-capture coverage 都出现稳定成功。

---

## Stage 6：从 `(a,\omega)` 迁移到 body-frame `[a_x,a_y]`

### 前置条件

Stage 5 至少达到 `OPTIMISTIC_PARTIAL`，且 `(a,\omega)` 已建立可靠成功锚点。

### 核心动作合同

Actor 输出 robot/body frame 二维加速度：

\[
\mathbf a_b=[a_x,a_y],\qquad \|\mathbf a_b\|_2\le0.4
\]

环境执行：

\[
\mathbf a_w=R(\theta)\mathbf a_b
\]

### yaw 合同

不能继续全程 yaw=0。

首选：

```text
speed > epsilon：
    yaw = velocity heading
speed <= epsilon：
    hold last valid yaw
```

或保留可独立定义的 heading state，但必须保证：

- observation 在 robot frame；
- action 也在 robot frame；
- 场景整体旋转时，策略输入输出保持一致；
- 低速时 yaw 不抖动。

### `[ALGO-NECESSARY]`

- radial/squashed Gaussian L2 disk；
- action head 从 box `(a,\omega)` 改为 disk `[a_x,a_y]`；
- dynamics 从 unicycle-like 改为 planar acceleration。

### `[IQN-ALIGN]`

仍保留：

- map；
- reward；
- perception；
- spawn；
- drag；
- dt/substeps；
- v_max；
- episode；
- tasks；
- obstacle。

### 子阶段

- 6A：pure coverage；
- 6B：capture；
- 6C：mixed。

每个子阶段从对应 `(a,\omega)` 成功任务开始，不允许直接跳到 mixed。

### 通过条件

至少能复现 `(a,\omega)` 的方向性趋势；若只在 body `[a_x,a_y]` 失败，定位为动作表征/动力学问题。

---

## Stage 7：world-frame `[a_x,a_y]`

### 前置条件

body-frame `[a_x,a_y]` 至少在单任务获得积极信号。

### 目标

测试最终期望的 world-frame 平面加速度。

### 结构风险

world-frame action 与 robot-frame observation不天然旋转等变。

允许两条严格对照：

#### 7A：world-frame observation + world-frame action

- 相对位置和速度均保持 world axes；
- yaw 不再作为局部坐标旋转基准；
- 可使用随机全局旋转数据增强；
- 这是坐标一致的 world-frame baseline。

#### 7B：旋转等变/增强方案

- 保留局部相对几何；
- 网络显式接收 `sin(yaw), cos(yaw)` 或采用等变表示；
- 必须论证 Actor 能完成 local→world 变换；
- 不允许只随机 yaw 而不提供方向信息。

当前“yaw 恒 0 + robot-frame函数退化为world axes”可作为 7A 的一种退化 baseline，但必须明确它失去旋转归一化和泛化效率。

### 子阶段

- 7A1：pure coverage；
- 7A2：capture；
- 7A3：mixed；
- 只有前一级有信号才推进。

### 完整目标

最终实现：

- local/decentralized Actor；
- centralized twin critic；
- continuous world `[a_x,a_y]`；
- capture、pure coverage、mixed；
-真实 post-capture replay；
- 4v1，并可继续 8v2/12v3 curriculum。

若 world-frame 最终未成功，但 body-frame `[a_x,a_y]` 或连续 `(a,\omega)` 成功，也应保留为有效成果和后续研究基线。

---

# 6. 旧 IQN 教师辅助路线

该路线用于缩短探索，不得污染新动力学 Bellman transition。

## 6.1 可直接做

- exact-shape Encoder transfer；
- 旧 IQN 生成 geometry snapshots；
- 新环境从 snapshot reset；
- 新 Actor 在新环境执行；
- 新 replay 保存新 action/reward/next state。

## 6.2 不可直接做

- 不把旧离散 action 写入新 replay；
- 不把旧 `(a,\omega)` transition 当成 `[a_x,a_y]` transition；
- 不直接用旧 replay 训练 centralized critic；
- 不伪造 post-capture label。

## 6.3 触发时机

若某阶段：

- bridge 证明旧策略在该任务成功；
- 新 MASAC random-init 连续两个 25k 无方向性进展；
- 工程链路和 reward 已验证；

则依次尝试：

1. Encoder transfer；
2. geometry snapshot curriculum；
3. Encoder + snapshot；
4. 只有动作翻译 gate 通过后才做 BC/demo prefill。

教师辅助属于初始化/数据分布改动，必须单独对照 random-init。

---

# 7. 如果一直 25k 不过门槛

## 7.1 不允许无限重复 25k

同一合同、同一初始化方式最多：

- 3 seeds；
- 每个 seed 最多先观察至 50k；
- 若三条均无连续指标改善，不再只换 seed 堆算力。

## 7.2 决策树

### A. Stage 2 极简单任务也失败

优先认为实现或训练链错误。

检查：

- reward sign/scale；
- action scaling；
- done/truncated；
- next state；
- target Q；
- replay；
- Actor gradient；
- deterministic eval；
- observation normalization；
- action实际执行值。

此时可以跑 MATD3 同合同对照，但不能先改任务。

### B. Stage 2 成功，Stage 3 失败

问题在多智能体 coverage、CE reward、credit、碰撞或 shared Actor，不是连续控制基础。

逐项诊断，不换动作。

### C. Stage 3 成功，Stage 4 失败

按 stationary/global/moving/local/support 阶梯定位。

### D. 单任务成功，Stage 5 失败

问题在多任务共享、replay balance、phase transition 或 catastrophic forgetting。

先做 episode-level alternating，再做 mixed。

### E. `(a,\omega)` 成功，body/world `[a_x,a_y]` 失败

问题主要是动作表征、yaw、旋转等变性或新动力学，不优先换算法。

### F. MASAC 失败但 MATD3 成功

只有满足以下条件才允许判断算法因素：

- 同一 stage；
- 同一 config；
- 同一动作；
- 同一 observation；
- 同一 reward；
- 同一 replay数据量；
- 3 seeds；
- MATD3 稳定成功，MASAC 稳定失败。

此时再比较：

- entropy；
- alpha；
- target entropy；
- stochastic action collision；
- deterministic exploration。

---

# 8. 算法替换规则

## 8.1 首选 MATD3，而不是直接 MADDPG

原因：

- 当前已有 twin critics；
- MATD3保留双 critic；
- 只需 deterministic Actor、target smoothing、delayed update；
- 能更干净地测试“最大熵/随机策略是否是问题”。

建议必要适配：

```yaml
matd3:
  actor_lr: 0.0001
  critic_lr: 0.0001
  gamma: 0.99
  tau: 0.005
  policy_delay: 2
  target_policy_noise: 0.1
  target_noise_clip: 0.2
  exploration_noise_std: 0.1
```

这些属于 `[ALGO-NECESSARY]`，必须与 MASAC 同场景比较。

MADDPG可作为第二对照，但单 critic 更容易高估，不应优先替换现主线。

## 8.2 允许切算法的 gate

只有以下情况之一：

1. Stage 2 MASAC 经过实现审计仍失败；
2. 同任务 MATD3 明显成功；
3. SAC entropy/alpha 明确导致持续随机碰撞；
4. 用户确认。

不得因为完整 mixed 25k 为零就直接换算法。

---

# 9. 乐观信号定义

最终成功之前，以下均可构成有价值的正反馈。

## Coverage

- CE energy持续下降；
- area CV下降；
- center-distance下降；
- trajectory开始朝 centroid；
- collision下降；
- obstacle恢复后仍保留几何收益。

## Capture

- discovery rate提升；
- discovery后 min-distance下降；
- ring geometry改善；
- stationary/global阶段出现 capture；
- moving/local阶段仍保持接近；
- support agent行为不再随机。

## Mixed

- capture rate非零；
- post-capture pool开始增长；
- capture后 CE energy下降；
- 两项单任务技能没有完全遗忘。

## 优化层

只作为辅助，不单独算成功：

- finite；
- Q/target Q同量级；
- TD error下降；
- gradients不持续爆炸；
- action不长期近零或饱和；
- alpha不异常坍缩。

---

# 10. 阶段完成报告模板

每阶段必须创建：

`artifacts/<date>_positive_feedback_ladder/<stage>/STAGE_<ID>_COMPLETION.md`

模板：

```markdown
# Stage X Completion

## 1. 状态
PASS / OPTIMISTIC_PARTIAL / FAIL / CONTRACT_ERROR

## 2. 本阶段目标

## 3. 唯一核心改动

## 4. IQN 对齐项
- [IQN-ALIGN] ...

## 5. 必要适配项
- [ALGO-NECESSARY] ...
- 必要性：

## 6. 临时诊断改动
- [DIAGNOSTIC-ADAPT] ...
- 退出条件：

## 7. 代码与配置
- commit:
- config:
- hash:
- commands:

## 8. 测试结果

## 9. 训练运行
- seeds:
- steps:
- checkpoints:

## 10. 关键指标
- optimization:
- capture:
- coverage:
- collision:
- action:

## 11. 与 baseline 对比

## 12. 结论

## 13. 是否晋级

## 14. 下一步唯一动作

## 15. 台账更新
```

---

# 11. Codex Agent 的持续推进边界

Codex 可在本文范围内自动推进：

- 当前阶段测试；
- 当前阶段 3-seed runs；
- 25k checkpoint；
- 失败诊断；
- 下一阶梯子阶段；
- 台账维护；
- 阶段报告。

以下情况必须停止并向用户确认：

- 修改 reward结构或主要权重；
- 修改 map尺寸；
- 修改 `a_max/ω_max/v_max`；
- 修改 drag；
- 修改 observation字段；
- 向 Actor加入全局信息或 oracle标签；
- 改 focal quota；
- 改 grad clip/update ratio；
- 从 MASAC 正式切换到 MATD3/MADDPG；
- 跳过失败 gate；
- 删除失败产物；
- 同时改变两个以上核心变量。

---

# 12. 最终成果分级

## Level A：完整版

- MASAC CTDE；
- local Actor；
- centralized twin critics；
- world-frame `[a_x,a_y]`；
- capture、pure coverage、mixed；
-真实 post-capture coverage；
- 4v1成功，并具备扩到8v2/12v3的合同。

## Level B：强可用版本

- body-frame `[a_x,a_y]`；
- 保留旋转一致性；
- 单任务和mixed出现稳定成功。

## Level C：连续控制可靠版本

- continuous `(a,\omega)` MASAC；
- pure coverage和capture成功；
- mixed至少有乐观信号。

## Level D：明确乐观信号

- 极简单任务成功；
- pure CE或capture至少一项连续指标稳定改善；
- 已明确阻塞位于任务、动作表征或多任务，而不是整个连续训练链不可用。

即使暂时只达到 Level C 或 D，也远优于在最终版本上反复无信息地跑25k，因为它为后续动作空间和算法升级提供可靠成功锚点。

---

# 13. 立即执行顺序

1. 创建新专用台账；
2. Stage 0 复现旧 IQN；
3. Stage 1 验证连续 `(a,\omega)` bridge；
4. Stage 2 建立极简单任务成功锚点；
5. Stage 3 pure coverage；
6. Stage 4 capture；
7. Stage 5 alternating + mixed；
8. Stage 6 body `[a_x,a_y]`；
9. Stage 7 world `[a_x,a_y]`；
10. 仅在相同简化合同下需要时比较 MATD3/MADDPG。

每完成一个阶段，必须先提交阶段完成说明、更新台账，再进入下一阶段。
