# CoCap-VorAdj：IQN → MASAC（CTDE）重构任务合同对齐审计

**审计日期：2026-08-06**  
**修订版本：v2（2026-08-06）——将 replay 配额从“joint transition 数量”修正为“focal-agent training item 数量”，补充同一时刻多角色并存时的 CTDE 采样与 loss 合同。**  
**原成功项目：** `main`  
**待整改分支：** `continuous/current-20260806`  
**目标：** 保留原环境、任务与奖励语义，将离散 IQN 改为连续二维加速度 MASAC；训练采用 CTDE，执行时仅使用局部观测 Actor，训练时使用 centralized twin critic。

---

## 0. 最终结论

当前版本不是“原成功项目仅替换 IQN 为 MASAC”，而是同时改变了：

1. critic 信息结构；
2. episode 时间尺度；
3. 场景构造与 spawn；
4. 障碍物数量；
5. replay 语义与采样；
6. 动力学模型；
7. 动作幅值；
8. batch、梯度裁剪、更新频率和 replay 容量；
9. 配置加载方式；
10. 评估长度与评估样本数。

因此目前训练失败不能主要归因于 MASAC 本身。**现阶段首先要恢复“任务合同”，再判断算法是否能学会。**

### 最可能导致当前训不出来的原因排序

| 优先级 | 原因 | 判断 |
|---|---|---|
| 1 | episode 被训练脚本和 screening 双重硬编码为 128 步 | 直接破坏捕获、捕获后 coverage 和长时探索，属于最严重问题 |
| 2 | `p6_formal_psa_masac` 实际 `critic_mode: local`，CLI 默认也是 local + smoke | 当前正式入口不是目标 CTDE MASAC |
| 3 | 场景合同未合并旧版 `tasks.*`；pure coverage spawn 错误，capture 与 mixed 基本同构 | 训练数据分布与旧成功配置严重不一致 |
| 4 | 障碍物被强制改为 0 | 感知、避障、安全奖励和 Voronoi obstacle 语义均未训练 |
| 5 | replay 的“20% regime”采样基本等价于 uniform，旧版四语义缓冲与 capture-snapshot recovery 未迁移 | 多任务样本平衡丢失 |
| 6 | 动力学从“带阻力的单轴推进”改为“无阻力二维积分”，同时 `a_max` 从 0.4 改为 0.8 | 奖励和终局静止条件对应的物理系统已经改变 |
| 7 | central schema、critic、replay 和 runner 多处硬编码 `max_agents=4` | 无法延续 4v1→8v2→12v3 curriculum |
| 8 | batch 64、grad clip 10、replay 30k、update_every 1 等与旧基线同时发生变化 | 增加优化不稳定性和 replay 过拟合风险 |
| 9 | CE PBRS 的 `gamma` 仍从 `iqn.gamma` 读取 | 算法与奖励配置存在隐藏耦合 |
| 10 | 评估每场景默认仅 2 个 episode，且最多走 128 步 | 评估结果几乎不能用于判断学习是否发生 |

---

## 1. “MASAC + CTDE”目标合同

### 1.1 算法身份

目标算法必须满足以下条件，才允许命名为 `MASAC_CTDE`：

- 所有 pursuer 共享一个 stochastic Actor；
- Actor **只输入单机局部观测**；
- 训练使用两套完全独立的 centralized soft-Q critics；
- critic 输入：
  - 全体 pursuer 的全局状态；
  - 全体 evader 的全局状态；
  - 障碍物全局状态；
  - 全体 pursuer 的 joint action；
  - active / entity masks；
  - focal-agent query；
- critic 输出每个 focal pursuer 的 `Q_i`；
- Actor 更新时，优化 `Q_i` 对自身动作的梯度，其他 agent 动作分支 stop-gradient；
- target critic 使用 Polyak soft update；
- 执行、screening、部署时丢弃 critic，只运行局部 Actor；
- formal 训练入口禁止默认或静默退回 local critic。

当前代码中的 `CentralSACTrainer` 和 `CentralAttentionCritic` 已具有 CTDE 主体结构，可保留并修正接线；问题主要在正式配置和 runner 没有使用它。

---

## 2. 冻结后的 Stage-1 任务合同

### 2.1 环境与实体

```yaml
env:
  width: 120.0
  height: 120.0
  num_pursuers: 4
  num_evaders: scene_dependent
  num_obstacles: 1
  capture_distance: 8.0
  related_distance: 18.0
  spawn_edge_margin: 12.0
  spawn_cluster_radius: 10.0
  pursuer_spawn_min_sep_capture: 15.0
  pursuer_spawn_min_sep_coverage: 7.0
  evader_spawn_min_sep: 15.0
  min_pursuer_evader_init_dis: 15.0
  init_speed: 0.0
  enforce_hard_boundary: true
  boundary_collision_death: true
```

**地图最终敲定为 120×120。**  
`base.py` 中的 55 只能作为历史 fallback，不应在 formal 路径生效；formal 配置缺少 `width/height` 时应直接报错，而不是静默采用 55。

### 2.2 感知合同

```yaml
perception:
  observation_mode: voronoi_adjacency
  topology: friendly_voronoi_comm_v0
  range: 20.0
  angle: 6.283185307179586
  enemy_sensing_radius: 20.0
  obstacle_sensing_radius: 20.0
  local_sensing_uses_surface_distance: true
  self_feature_dim: 9
  max_pursuer_num: 8
  max_evader_num: 8
  max_obstacle_num: 5
  global_evader_visibility: false
```

要求：

- Actor 沿用旧成功 VCT-LS 局部信息合同；
- 不向 Actor 泄露全局敌人位置；
- centralized critic 可以使用训练期全局真值；
- 必须保留 1 个障碍物进行训练，否则 obstacle token、避障和安全项都没有数据覆盖；
- 每次运行把 resolved perception 参数写入 manifest。

### 2.3 三个物理上不同的场景

#### A. `capture`

- 1 个 evader；
- pursuer：`map_random`；
- evader：`map_random`；
- 捕获成功后结束；
- pre-capture 最大 1000 步。

#### B. `pure_ce`

- 0 个 evader；
- formal 训练使用 recovery 分布；
- 最大 1500 步；
- 独立评估至少包含：
  - `inner_random_cluster`；
  - `map_random`；
  - capture-snapshot recovery。

#### C. `mixed_crms`

- 1 个 evader；
- 初始 spawn 与 capture 场景一致；
- 捕获后继续执行 coverage；
- pre-capture 最多 1000 步；
- 捕获后保留完整 500 步 coverage window；
- 总长度上限 1500 步。

这样既满足用户要求的 1000–1500 步，也保留旧成功配置中的 500 步 post-capture window。不得再用统一的 `num_evaders` 差异来冒充三个场景。

### 2.4 Recovery spawn

对齐旧最终成功配置：

```yaml
recovery:
  capture_state_pool_capacity: 1000
  captured_state_ratio: 0.75
  map_random_ratio_within_non_capture: 0.5
```

对应概率：

- 75%：capture snapshot；
- 12.5%：inner random cluster；
- 12.5%：map random。

capture snapshot pool 必须真实维护和使用，不能只把这些字段留在继承配置中。

---

## 3. 连续动作与动力学合同

### 3.1 动作语义

正式动作定义为：

\[
a_t=[a_x,a_y]
\]

建议采用物理加速度圆盘：

\[
\|a_t\|_2 \le a_{\max}
\]

首轮 parity 配置：

```yaml
action:
  mode: acceleration_2d_world
  dimension: 2
  bound_type: l2_disk
  a_max: 0.4
```

说明：

- 当前 radial actor 的圆盘动作是可行的；
- 若坚持每轴独立 `Box[-a_axis_max,a_axis_max]^2`，必须另设 `vector_a_max`，不能把“每轴上限”和“向量模长上限”混为一谈；
- replay 中保存的必须是环境实际执行的 `[ax, ay]`，不是 latent、候选速度或裁剪前动作；
- 禁止 formal 配置继续接受 `velocity_2d_body` 并静默解释为 acceleration。

### 3.2 坐标系

当前实现名为 body-frame，但 yaw 固定为 0，所以数值上等价于 world-frame。为了避免概念混乱，正式合同采用：

- 公共 API、配置、replay：`acceleration_2d_world`；
- 如果内部暂时复用 body-to-world 函数：
  - 必须断言 pursuer yaw 恒为 0；
  - 必须有 world/body 等价单元测试；
  - 不得在 manifest 中继续宣称可变 body-frame。

### 3.3 时间离散

```yaml
dynamics:
  physics_dt: 0.05
  substeps_per_action: 10
  decision_dt: 0.5
  v_max: 3.0
  integration: trapezoidal
  speed_clip_each_substep: true
  collision_check_each_substep: true
  boundary_check_each_substep: true
```

### 3.4 阻力：必须显式选择，禁止无声改变

旧离散动力学包含：

\[
\dot v = a-kv,\quad k=\frac{0.4}{3.0}
\]

当前二维 acceleration 路径是：

\[
\dot{\mathbf v}=\mathbf a
\]

即没有阻力。该差异会改变：

- 零动作是否自动减速；
- coverage 静止终局的难度；
- 速度奖励和加速度惩罚的尺度；
- 从旧 IQN 行为到连续策略的可比性。

建议实现两个明确 profile：

```yaml
dynamics_profile: continuous_parity_v1
linear_drag_coefficient: 0.1333333333
a_max: 0.4
```

和：

```yaml
dynamics_profile: continuous_no_drag_ablation
linear_drag_coefficient: 0.0
a_max: 0.8
```

**首轮恢复训练必须使用 `continuous_parity_v1`。**  
无阻力、`a_max=0.8` 只能作为 parity 跑通后的消融，不应与 CTDE、episode、spawn 等修复同时混入。

---

## 4. 网络合同

### 4.1 Actor

```yaml
actor:
  shared_parameters: true
  observation: local_vct_ls_only
  hidden_dim: 256
  num_heads: 8
  num_layers: 4
  self_feature_dim: 9
  max_pursuers: 8
  max_evaders: 8
  max_obstacles: 5
  distribution: radial_squashed_gaussian
  log_std_min: -5.0
  log_std_max: 1.0
  dropout: 0.0
```

`dropout=0` 是 SAC 特有的稳定性选择，不必机械对齐 IQN 的 0.1；它应被记录为“算法必要差异”。

### 4.2 Centralized twin critic

```yaml
critic:
  mode: central
  twin: true
  hidden_dim: 256
  num_heads: 8
  num_layers: 4
  max_agents: 12
  max_evaders: 8
  max_obstacles: 5
  use_joint_actions: true
  output: per_focal_agent_q
  dropout: 0.0
```

要求：

- `max_agents=12` 必须贯穿：
  - critic config；
  - central schema；
  - replay；
  - runner；
  - checkpoint manifest；
- Stage-1 只激活前 4 个槽位；
- 当前 P5 YAML 的 `max_agents: 12` 不得再被 runner 的硬编码 4 覆盖；
- formal central trainer 必须读取 `masac` 和 `central_critic` 配置块，不能继续读取 `local_sac`。

---

## 5. 训练与优化合同

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
  total_env_steps: 2000000
  warmup_joint_transitions: 5000
  batch_size: 128  # 128 个 focal-agent training items，不是 128 条互斥 joint transitions
  max_focal_items_per_joint_transition: 4
  update_every_env_steps: 4
  gradient_steps: 1
  grad_clip_norm: 0.5
  replay_capacity_joint: 250000
```

说明：

- batch 128、grad clip 0.5、update cadence 4 首轮按旧成功配置恢复；
- `tau`、自动温度和 twin critic 是 SAC 必需差异；
- 旧 replay 1,000,000 是 agent-level transition；4-agent joint replay 的近似等效容量是 250,000，因此不应机械申请 1,000,000 个高维 joint transition；
- 当前 30,000 joint replay 仅相当于约 120,000 个 4-agent transition，明显低于旧基线；
- 首轮 parity 通过后，再单独消融：
  - update_every 1；
  - grad clip 1 或 10；
  - batch 64/256；
  - `a_max=0.8`；
  - no-drag。

---

## 6. Joint replay 与 focal-agent 采样的正确迁移

### 6.1 当前问题

当前 sampler 的 20% regime pool 为：

```python
active_target_pool + coverage_only_pool
```

两者基本覆盖整个 replay，因此这 20% 与 uniform 抽样几乎没有区别。与此同时：

- `phase` 虽被写入 metadata，但 sampler 不按 phase 建索引；
- 只保存单个 `task_label` 字符串，无法描述同一时刻不同 agent 的 pursuing、support 与 coverage 角色；
- 没有真正实现旧版 capture snapshot recovery；
- 不能把旧版 agent-level 的 64/16/32/16 配额直接解释为 128 条互斥的 joint transitions。

### 6.2 两个基本单位必须分开

#### 存储单位：joint transition

环境每前进一步，只向 replay 写入一次完整联合转移：

\[
\mathcal{T}_t=
(S_t,O_t,A_t,R_t,S_{t+1},O_{t+1},M_t,\mathrm{metadata}_t)
\]

其中包括：

- 全局实体状态；
- 全体 agent 的局部观测；
- 全体 agent 的 joint action；
- 每个 agent 的 reward；
- active/entity masks；
- terminated/truncated；
- phase、scene、spawn origin 和每个 agent 的角色。

#### 训练与配额单位：focal-agent training item

每个训练项定义为：

```text
(joint_transition_id, focal_agent_id, bucket_id)
```

同一条 joint transition 可以根据不同 focal agent 同时生成多个训练项。例如某一时刻：

```text
agent 0 = pursuing
agent 1 = support
agent 2 = coverage
agent 3 = coverage
```

该 transition 只存储一次，但可以产生：

```text
(t, agent 0, pre_capture_pursuing)
(t, agent 1, pre_capture_support)
(t, agent 2, pre_capture_coverage)
(t, agent 3, pre_capture_coverage)
```

四个训练项都把该时刻完整的 global state、joint action 和 masks 送入 centralized critic。区别只在于当前查询和优化的 focal agent 不同。

因此：

> **central critic 的全局输入要求，与按 focal agent 做角色平衡并不冲突。joint transition 是存储单位，focal-agent item 是采样和 loss 单位。**

### 6.3 新 replay schema

每个 joint transition 至少保存：

```yaml
phase_id: pre_capture | post_capture | pure_recovery
scene_id: capture | mixed_crms | pure_ce
origin_id: online | capture_snapshot | inner_cluster | map_random

agent_role_id: uint8[max_agents]
# 0 = inactive
# 1 = pursuing
# 2 = support
# 3 = coverage

active_mask: bool[max_agents]
terminated_mask: bool[max_agents]
truncated_mask: bool[max_agents]

event_ids:
  - discovery
  - capture
  - collision
  - ce_success
```

角色合同：

- 每个 active agent 在一个 transition 中必须且只能有一个 `primary role`；
- inactive agent 的 role 必须为 `inactive`；
- `pursuing/support/coverage` masks 从 `agent_role_id` 派生，避免多个布尔 mask 不一致；
- 同一 transition 中可以同时存在多种角色；
- role、phase、origin 默认只用于 replay 索引、loss routing 和诊断，不作为 Actor 的 oracle 输入；
- 首轮也不要求把 role label 额外输入 critic，critic 继续使用既定的完整全局实体状态与 joint action。

### 6.4 replay 索引

Replay 仍是一个 joint ring buffer，同时维护以下引用索引：

```python
pre_capture_pursuing_index: list[(slot_id, generation_id, agent_id)]
pre_capture_support_index: list[(slot_id, generation_id, agent_id)]
pre_capture_coverage_index: list[(slot_id, generation_id, agent_id)]
post_capture_coverage_index: list[(slot_id, generation_id, agent_id)]
pure_recovery_coverage_index: list[(slot_id, generation_id, agent_id)]
```

要求：

- 同一个 `slot_id` 可以因不同 `agent_id` 出现在多个索引；
- 同一个 `(slot_id, agent_id)` 只能进入一个 primary-role bucket；
- ring buffer 覆盖旧 slot 时，必须通过 `generation_id` 或显式删除机制使旧索引失效；
- sampler 必须过滤 stale references；
- 不要复制完整 transition 到五个物理 buffer。

### 6.5 首轮 optimizer batch 目标

`batch_size=128` 的含义修正为：

> 每次更新采样 128 个 focal-agent training items，而不是 128 条互斥 joint transitions。

初始配额：

| focal-agent bucket | 数量 |
|---|---:|
| pre-capture pursuing agent | 64 |
| pre-capture support agent | 8 |
| pre-capture coverage agent | 8 |
| real post-capture coverage agent | 32 |
| pure/recovery coverage agent | 16 |
| **总计** | **128** |

说明：

- 原来的 `16 pre-capture coverage/support` 明确拆分为 support 8、coverage 8；
- 同一 joint transition 可以贡献多个不同 focal agents；
- 同一个 `(transition, agent)` 在单个 batch 中最多出现一次；
- Stage-1 每条 joint transition 在单个 batch 中最多贡献 4 个 focal items；
- 扩展到 8/12 agents 时，默认仍限制 `max_focal_items_per_joint_transition=4`，避免少数 transition 支配梯度；
- sampler 记录：
  - `num_focal_items`；
  - `num_unique_joint_transitions`；
  - 各 bucket item 数；
  - 每条 transition 最大贡献数；
  - pursuing/support/coverage 的实际 agent 数。

当某个 bucket 数据不足时，按以下顺序处理：

1. 先在该 bucket 内有放回补足，并记录 `replacement_count`；
2. 若该 bucket 为空，按配置的显式 fallback matrix 转移到最接近的 bucket；
3. 不允许静默退化为全局 uniform；
4. screening 报告必须展示目标配额、实际配额和 fallback 次数。

固定配额是有意进行的多任务目标重加权，因此首轮不再额外乘一次 64/8/8/32/16 权重，避免重复加权。

### 6.6 sampler 输出合同

Sampler 建议返回：

```python
batch = {
    # 完整 joint context
    "global_state":       ...,       # [B, ...]
    "local_obs":          ...,       # [B, N, ...]
    "joint_action":       ...,       # [B, N, 2]
    "reward":             ...,       # [B, N]
    "next_global_state":  ...,
    "next_local_obs":     ...,
    "active_mask":        ...,       # [B, N]
    "next_active_mask":   ...,

    # focal routing
    "focal_agent_id":     ...,       # [B]
    "focal_mask":         ...,       # [B, N], one-hot
    "focal_reward":       ...,       # [B]
    "focal_terminated":   ...,       # [B]
    "focal_truncated":    ...,       # [B]

    # sampling/diagnostics only
    "bucket_id":          ...,       # [B]
    "phase_id":           ...,
    "origin_id":          ...,
    "transition_id":      ...,
}
```

其中：

```python
focal_reward[b] = reward[b, focal_agent_id[b]]
focal_mask[b, focal_agent_id[b]] = True
```

### 6.7 centralized critic 与 loss

对于每个 focal item \(i\)，critic 仍读取完整联合上下文：

\[
Q_i(S_t,A_t,M_t)
\]

可使用两种等价接口：

1. critic 输入 `focal_agent_id`，直接输出 `[B]`；
2. critic 一次输出 `[B,N]`，再按 `focal_agent_id` gather。

Critic TD target 使用 focal agent 自身的 reward 和终止标记：

\[
y_i =
r_i+
\gamma(1-d_i)
\left[
\min(Q'_{1,i},Q'_{2,i})
-\alpha\log\pi_i(a'_i|o'_i)
\right]
\]

Actor 更新时：

- focal agent 的 sampled action 保留梯度；
- 其他 agent 的 sampled action 仍作为 centralized critic 的 joint-action context，但必须 detach；
- 共享 Actor 最终由不同角色的 focal items 共同更新；
- global context 不等于 team reward，team reward 可以写入所有 agent，角色奖励仍保留在各自 `reward[i]` 中。

不得使用一个 transition-level `loss_mask[B,N]` 同时平均所有角色后再声称满足 64/8/8/32/16，因为这样无法精确控制 focal-agent 配额。正式实现应显式返回 `focal_agent_id` 或等价 one-hot focal mask。

---

## 7. 已确认的问题清单

### P0-01：正式 P6 不是 CTDE

`p6_formal_local_sac_4v1.yaml`：

```yaml
algorithm: p6_formal_psa_masac
critic_mode: local
```

名字是 MASAC，实际是 local SAC。必须创建独立 formal central 配置，并让 formal 入口只允许 central。

### P0-02：CLI 默认选择错误路径

当前默认值：

```text
--critic-mode local
--trainer-profile smoke
--batch-size 64
--replay-capacity 30000
```

即使用户以为启动了正式训练，也会进入 local smoke 路径。formal 命令应：

- 默认读取唯一 formal YAML；
- 不提供 local fallback；
- 若 resolved `critic.mode != central`，启动即失败。

### P0-03：配置“双源真相”

runner 同时使用：

- P5：构建环境；
- P6：读取 formal trainer；
- CLI：再次覆盖 critic、a_max、batch 等。

这使 manifest 中的 config、trainer config 和有效场景配置不一致。必须改为：

1. 一个 formal YAML；
2. 一次递归 resolve；
3. scene 仅 deep-merge `tasks.<scene>`；
4. 输出一个 `effective_config.yaml`；
5. 所有模块只从 resolved config 读取。

### P0-04：episode 128 被双重硬编码

- `_scene_config()` 写死 128；
- `_screen()` 又写死 `for _ in range(128)`。

即使修改 YAML 也不会生效。两处都必须删除。

### P0-05：场景没有合并旧任务 override

旧版：

- capture：map random；
- coverage：inner random cluster。

当前 `_scene_config()` 只改 `num_evaders`，没有 deep-merge `tasks.voradj` / `tasks.voradj_coverage`。因此 pure CE 很可能仍继承 root `map_random`。

同时 `capture` 与 `mixed_crms` 都仅设置 `num_evaders=1`，没有独立 termination/post-capture 合同，从代码上看环境配置基本相同。

### P0-06：障碍物被强制删除

`config["env"]["num_obstacles"] = 0` 必须删除并恢复 1。否则：

- obstacle observation 永远是 padding；
- obstacle sensing radius 无训练意义；
- collision/safety reward 分布变化；
- masked Voronoi obstacle 逻辑未训练。

### P0-07：central 路径硬编码 4 agents

虽然 central schema 和 P5 config 支持 12，但 runner 在以下位置写死 4：

- critic `max_agents=4`；
- replay `max_agents=4`；
- `build_central_global_obs(... max_agents=4)`；
- manifest `max_agents=4`。

必须统一读取 config 的 12。

### P0-08：replay sampler 没有真正阶段平衡

20% regime 抽样近似等于全池抽样；phase 和逐 agent 角色没有形成 focal 索引；旧 agent-level 四语义配额被错误套用到 joint transition。必须按第 6 节重构。

### P0-09：动力学与旧环境不等价

当前 acceleration 路径：

- 无水阻；
- `a_max=0.8`；
- yaw 固定；
- 二维向量加速度；
- trapezoidal integration。

旧路径：

- `a=±0.4/0`；
- 标量前向速度；
- 带 `k=0.4/3` 阻力；
- 角速度控制；
- 不同的积分顺序。

必须先跑 parity profile，再做物理模型消融。

### P0-10：PBRS gamma 仍绑定 IQN

CE PBRS 两处仍读取：

```python
self.config["iqn"]["gamma"]
```

应改为统一的：

```yaml
discount:
  gamma: 0.99
```

IQN 和 MASAC 均引用该字段；禁止奖励函数依赖算法命名空间。

### P1-01：P5 central critic 配置实际被忽略

P5 写了 `central_critic.max_agents: 12`，但 `_make_trainer()` 根据 generic encoder 字段重新创建 critic，并硬编码 4。formal central 必须直接解析 `central_critic`。

### P1-02：评估不足

默认 2 episode × 128 步不具统计意义。建议：

- unit/smoke：2 episodes；
- 25k/100k screening：每场景 20 episodes；
- checkpoint selection：每场景至少 100 episodes，或 5 seeds × 20；
- 固定 seed suite；
- 报告均值、标准差、Wilson interval 或 bootstrap CI。

### P1-03：动作别名存在语义陷阱

`velocity_2d_body` 被静默映射为 acceleration。formal 路径应拒绝历史 velocity 名称，避免旧配置在新代码中产生完全不同的物理含义。

### P1-04：base fallback 55 可能隐藏配置断链

当前 formal 继承链应得到 120，但 `base.py` 缺少 width 时默认 55。formal schema 应要求 width/height 明确存在。

### P1-05：resume 不是精确环境恢复

当前 checkpoint 保存 episode step，但恢复后从新的 seeded episode boundary 开始。可以保留，但必须：

- 明确写入报告；
- 不把它称为 exact resume；
- checkpoint selection 不跨越不一致的 replay/episode 语义；
- 若需要无损训练，后续保存完整 env state。

### P2-01：Actor 更新时可冻结 critic 参数

当前 actor backward 会为 critic 参数计算无用梯度，虽不会错误更新，但浪费显存与计算。Actor loss 期间临时 `requires_grad_(False)` critic，再恢复。

### P2-02：focal critic 可后续向量化

当前每个 focal agent 单独 forward critic，4v1 可接受；12v3 会明显变慢。先保证正确，再向量化，不应在本轮同时重写算法。

---

## 8. 必须新增的验收测试

### 8.1 配置测试

1. formal resolved config 中：
   - critic mode 必须 central；
   - map 必须 120×120；
   - episode 不是 128；
   - obstacles=1；
   - max_agents=12；
   - batch=128；
   - grad clip=0.5。
2. 三个 scene 的 config hash 必须不同。
3. capture 与 mixed 的 termination/post-capture 字段必须不同。
4. formal 启动时检测到 local critic、smoke profile 或 missing task override 必须失败。

### 8.2 CTDE 测试

1. Actor 输出对仅 critic 可见的全局状态扰动不变化；
2. 改变 teammate action 时 `Q_i` 必须变化；
3. Actor 参数中不存在 global-state encoder；
4. critic 输入缺少 joint action 或 mask 时必须失败；
5. inactive/padded agent 不参与 loss；
6. time-limit truncated 仍 bootstrap，真实 terminated 不 bootstrap；
7. 4/8/12 agents 的 shape 和 mask 测试全部通过。

### 8.3 动力学测试

对无碰撞、无边界、无洋流场景：

1. 从静止开始恒定加速度，验证解析/数值速度和位移；
2. `||a|| > a_max` 被拒绝或按唯一合同裁剪；
3. 达到 `v_max` 后每个 substep 均不超速；
4. parity drag 下零动作速度指数衰减；
5. no-drag profile 下零动作保持速度；
6. world-frame 与内部 body-frame（yaw=0）严格等价；
7. 每个 substep 都执行 boundary/collision 检测；
8. replay action 等于环境实际执行动作。

### 8.4 Spawn 与数据分布测试

至少采样 10,000 次 reset，验证：

- capture/mixed 为 map_random；
- pure recovery 比例接近 75% / 12.5% / 12.5%；
- min separation、edge margin 和 obstacle clearance 满足配置；
- capture snapshot 不是空池时才抽样；
- optimizer batch 恰好包含 128 个 focal-agent items，实际配额为 64/8/8/32/16；
- 每个 focal item 都能映射到有效的 `(transition_id, focal_agent_id)`；
- 同一 transition 可贡献不同角色 agent，但同一 `(transition, agent)` 不重复；
- centralized critic 对每个 focal item 仍接收完整 global state 与 joint action；
- 每个 active agent 的 primary role 唯一，inactive role 与 active mask 一致；
- ring buffer 覆盖后不存在可被采样的 stale role index；
- 每批记录 unique joint transitions、最大 items/transition、各角色数量与 fallback 次数。

### 8.5 学习门槛

不得只检查“loss finite”。建议门槛：

#### 5k 数据链 gate

- 无 NaN/Inf；
- action rejection≈0；
- 当前场景应出现的五个 focal-agent 索引池均开始有数据；不可出现的池必须有明确解释；
- Q、target Q、alpha、entropy、grad norm 数量级正常；
- scene/spawn/episode 分布日志正确。

#### 25k 学习信号 gate

- deterministic policy 不再长期接近零动作；
- action norm、速度和状态访问范围明显区别于随机初始化；
- capture distance progress 或 CE energy 至少一项有持续改善；
- 不要求最终成功率，但不得三场景所有核心指标完全不动。

#### 100k gate

- capture 或 pure CE 至少一项出现稳定非零成功；
- collision rate 不持续恶化；
- central critic teammate-action sensitivity 保持非零；
- 若仍完全失败，先做 reward-scale/Q-target 诊断，不立即同时改多个超参。

---

## 9. 推荐实施顺序

### Phase A：只修合同，不训练

- 新建唯一 formal central YAML；
- 删除 runner 的 local/smoke 默认；
- 删除所有 128、0 obstacle、4 agents 等硬编码；
- 恢复 scene deep-merge；
- dump effective config；
- 完成配置、spawn、动力学和 CTDE 单元测试。

### Phase B：恢复物理和 replay

- 增加 parity drag profile；
- `a_max=0.4`；
- 新 joint replay schema；
- capture snapshot recovery；
- phase/role stratified sampler；
- PBRS gamma 解耦 IQN。

### Phase C：小规模验证

- 5k data-chain；
- 25k learning-signal；
- 每场景 20 episodes screening；
- 只允许修 bug，不做大范围奖励重写。

### Phase D：正式 Stage-1

- 2M env steps；
- 100k checkpoint；
- 固定评估 seed suite；
- 选历史最优而非最后 checkpoint。

### Phase E：单因素消融

依次只改一个：

1. `a_max: 0.4 → 0.8`；
2. drag → no-drag；
3. update_every 4 → 1；
4. grad clip 0.5 → 1/10；
5. replay sampler；
6. map 120 → 55 curriculum。

---

## 10. 主要证据路径

### 原成功版本

- `configs/experiments/voradj_a3_apfnew_sqrtn_20260723/a3_apfnew_sqrtn_4v1_scratch_mix_2m.yaml`
  - 120×120、1 obstacle、episode 3000；
  - perception 20 m / 2π；
  - batch 128、replay 1M、train freq 4、grad clip 0.5；
  - capture/coverage 独立 spawn。
- `configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/common.yaml`
  - episode 3000；
  - post-capture window 500；
  - 历史 agent-level replay counts 64/16/32/16（新 joint replay 中改为 focal-item 配额）；
  - recovery ratio 0.75 和 capture-state pool 1000。

### 当前分支

- `configs/experiments/continuous_marl_20260804/p6_formal_local_sac_4v1.yaml`
  - algorithm 名称为 MASAC，但 critic local；
  - grad clip 10、a_max 0.8。
- `tools/run_continuous_p6_screening.py`
  - P5 环境 + P6 trainer 双配置；
  - episode 128；
  - obstacles 0；
  - central/replay/schema 多处 max_agents 4；
  - CLI 默认 local/smoke/batch64/replay30k；
  - screening 仍固定 128 步。
- `src/cocap_voradj/training/continuous/joint_replay.py`
  - 70/20/10 sampler；
  - regime pool 实际重新合并全部 regime；
  - phase 不参与索引。
- `src/cocap_voradj/dynamics/robot.py`
  - acceleration_2d 路径没有 drag；
  - old/continuous `(a,w)` 路径保留阻力。
- `src/cocap_voradj/envs/voronoi_adjacency.py`
  - CE PBRS 仍读取 `iqn.gamma`。

---

## 11. 供 Coding Agent 使用的执行提示词

见同名报告附录或聊天中的完整提示词。Agent 必须先完成静态合同与测试，再进行训练；禁止在合同未通过时通过改奖励、缩图或扩大动作幅值“碰运气”。


---

## 12. v2 修订摘要：多角色并存时如何采样

本次修订不改变 centralized critic 的全局信息合同，而是修正 batch 的统计单位：

- joint transition 仍是 replay 的唯一存储单位；
- 同一时刻可同时包含 pursuing、support、coverage agents；
- `(transition_id, focal_agent_id)` 是采样和优化单位；
- batch 128 指 128 个 focal-agent items；
- 初始配额为 64 pursuing、8 support、8 pre-capture coverage、32 post-capture coverage、16 pure/recovery coverage；
- 每个 focal item 的 critic 都读取完整 global state 和 joint action；
- Actor loss 只对 focal action 保留梯度，其他 agent action detach；
- role label 不作为 Actor 的 oracle 输入；
- 使用 generation-aware 引用索引，避免 ring buffer 覆盖后 stale index 被采样。

---

# 附录：Coding Agent 完整提示词

你现在要整改仓库 `Jayyeah/cocap-voradj` 的分支 `continuous/current-20260806`。

目标不是继续试超参，而是完成一次可审计的“任务合同对齐”：在不破坏 legacy IQN 主线的前提下，把 pursuer 策略改为连续二维加速度 MASAC，并采用真正的 CTDE。

请先阅读项目根目录中的审计文档：
`cocap_voradj_masac_ctde_task_contract_audit_20260806_zh.md`

## 一、不可变目标

1. Actor：共享参数，只输入每个 agent 的局部 VCT-LS observation。
2. Critic：两套独立 centralized soft-Q critic，输入全局实体状态、joint actions、active/entity masks，输出每个 focal agent 的 Q。
3. 执行时只运行 Actor，critic 仅训练期使用。
4. 动作：连续二维 `[ax, ay]`，正式公共语义为 world-frame acceleration；首轮使用 L2 disk，`||a||<=0.4`。
5. Stage-1：4 pursuers、1 evader、1 obstacle、120×120 map。
6. perception：range=20、FOV=2π、VCT-LS、global_evader_visibility=false，实体 padding 与旧成功配置一致。
7. 不修改 legacy IQN 的行为、checkpoint、训练入口和已有成功配置。
8. 在任务合同通过前，不重写 reward，不缩小地图，不把 `a_max` 提高到 0.8/1.6，不改为全局 Actor。

## 二、必须先修复的 P0 问题

### 1. 建立唯一配置源

创建正式配置，例如：

`configs/experiments/continuous_marl_20260804/p6_formal_central_masac_4v1.yaml`

要求：

- `algorithm: masac_ctde`
- `critic_mode: central`
- 一个 YAML 完整包含 env、tasks、actor、central_critic、masac、replay、training、evaluation；
- runner 只加载和 resolve 这一个 YAML；
- scene 仅通过 deep-merge `tasks.<scene>` 生成；
- 输出完整 `effective_config.yaml`；
- 删除 P5 作为 env、P6 作为 trainer、CLI 再覆盖的“双源真相”；
- formal 入口发现 local critic 或 smoke profile 时直接报错。

### 2. 修复 episode 和 scene

删除所有硬编码的 128，包括 `_scene_config()` 和 `_screen()`。

使用以下场景：

- `capture`：
  - 1 evader；
  - pursuer/evader `map_random`；
  - 捕获后终止；
  - 最大 1000 步。
- `pure_ce`：
  - 0 evader；
  - formal recovery spawn；
  - 最大 1500 步。
- `mixed_crms`：
  - 1 evader；
  - map_random；
  - 捕获后继续 500 步 coverage；
  - pre-capture timeout 1000；
  - 总上限 1500。

必须保证三个 scene 的 resolved config hash 不同。

### 3. 恢复旧环境分布

- map：120×120；
- obstacles：1，删除 runner 中 `num_obstacles=0`；
- capture spawn：map_random，edge margin 12，min sep 15；
- coverage inner cluster：cluster radius 10，pursuer min sep 7；
- recovery：
  - capture state pool capacity 1000；
  - 75% capture snapshot；
  - 剩余 25% 中 50% map_random、50% inner cluster；
- 如果 formal config 缺少 width/height，直接失败，禁止 fallback 55。

### 4. 真正启用 CTDE

保留并修正 `CentralSACTrainer` / `CentralAttentionCritic`：

- Actor local only；
- twin central critics；
- critic 读取完整 global entity schema 与 joint action；
- actor loss 对 focal action 保留梯度，其他 agent action detach；
- target critic soft update；
- terminated 不 bootstrap，time-limit truncated 继续 bootstrap；
- inactive agent 不计入 loss；
- formal 路径不允许 LocalSACTrainer。

把以下所有硬编码 4 改为 config 驱动的 `max_agents=12`：

- CentralCriticConfig；
- build_central_global_obs；
- JointReplayBuffer；
- runner；
- checkpoint/manifest contract。

Stage-1 active mask 只激活前 4 个槽位。

### 5. 修复动力学合同

当前 acceleration_2d 路径无 drag，而旧成功动力学有水阻。实现显式 profile：

`continuous_parity_v1`：
- world-frame `[ax, ay]`；
- L2 disk `a_max=0.4`；
- `v_max=3.0`；
- physics dt=0.05；
- 10 substeps；
- decision dt=0.5；
- linear drag coefficient `0.4/3.0`，推广到二维向量；
- trapezoidal position integration；
- 每 substep speed cap、boundary check、collision check。

`continuous_no_drag_ablation`：
- drag=0；
- 仅供后续消融，不能作为首轮 formal 默认。

公共配置和 replay 使用 `acceleration_2d_world`。若内部暂时复用 body-frame 函数，必须断言 yaw=0 并加入 world/body 等价测试。formal 路径拒绝 `velocity_2d_body` 等历史别名。

### 6. 修复 joint replay 与 focal-agent 采样

当前 70/20/10 sampler 的 regime pool 基本重新合并了全 replay，且无法表达同一时刻不同 agent 分别为 pursuing、support、coverage。不要把 64/16/32/16 解释为 128 条互斥 joint transitions。

#### 存储单位

环境每步只写入一条完整 joint transition，包含：

- global state；
- all-agent local observations；
- joint action；
- per-agent rewards；
- next state/observations；
- active/entity masks；
- terminated/truncated；
- phase、scene、origin；
- 每个 agent 的 primary role。

使用单一角色字段：

```yaml
agent_role_id: uint8[max_agents]
# 0 inactive, 1 pursuing, 2 support, 3 coverage
```

约束：

- 每个 active agent 在同一 transition 中必须且只能有一个 primary role；
- inactive agent 的 role 必须为 inactive；
- 同一 transition 可以同时包含 pursuing、support 与 coverage agents；
- role/phase/origin 用于 replay 索引、loss routing 和日志，不输入 Actor；
- 首轮不额外把 oracle role label 输入 critic。

#### 训练单位

优化器 batch 由 128 个：

```text
(joint_transition_id, focal_agent_id, bucket_id)
```

组成，而不是 128 条互斥 joint transitions。

维护引用索引，不复制 transition：

- pre-capture pursuing；
- pre-capture support；
- pre-capture coverage；
- real post-capture coverage；
- pure/recovery coverage。

索引项至少为：

```text
(slot_id, generation_id, agent_id)
```

ring buffer 覆盖 slot 后，旧 generation 的索引必须失效并在采样时被过滤。

#### 首轮 focal-item 配额

- 64 个 pre-capture pursuing-agent items；
- 8 个 pre-capture support-agent items；
- 8 个 pre-capture coverage-agent items；
- 32 个 real post-capture coverage-agent items；
- 16 个 pure/recovery coverage-agent items。

总计 128 个 focal items。

采样约束：

- 同一个 joint transition 可按不同 focal agent 进入多个 bucket；
- 同一 `(transition, agent)` 在一个 batch 中最多出现一次；
- `max_focal_items_per_joint_transition=4`；
- bucket 不足时，先在 bucket 内有放回补足；bucket 为空时只能按照显式 fallback matrix 转移；
- 禁止静默退化为全局 uniform；
- 记录目标/实际配额、replacement、fallback、unique transitions 和最大 items/transition；
- 配额已经定义任务重加权，首轮不要再按 64/8/8/32/16 额外乘 loss 权重。

Sampler 返回完整 joint context，以及：

- `focal_agent_id[B]`；
- `focal_mask[B,N]`；
- `focal_reward[B]`；
- `focal_terminated[B]`；
- `focal_truncated[B]`；
- bucket/phase/origin/transition ids。

Centralized critic 对每个 focal item 始终读取完整 global state、joint action 与 masks。可直接按 focal query 输出 `[B]`，或输出 `[B,N]` 后 gather。

Critic target 使用 focal agent 自身 reward 和 done 标记。Actor loss 只对 focal agent 的 sampled action 保留梯度；其他 agent sampled actions 继续作为 joint-action context，但必须 detach。共享 Actor 由所有 focal roles 的训练项共同更新。

Joint replay capacity 设为 250000。实现真实 capture snapshot pool，不能只继承 YAML 字段。

### 7. 修复优化参数和隐藏耦合

首轮 formal：

- batch_size=128；
- actor/critic/alpha lr=1e-4；
- gamma=0.99；
- tau=0.005；
- alpha_init=0.2；
- target_entropy=-2；
- warmup=5000 joint transitions；
- update_every=4；
- gradient_steps=1；
- grad_clip_norm=0.5；
- total env steps=2M。

把 reward 中所有 `config["iqn"]["gamma"]` 改为统一 `discount.gamma`，IQN 与 MASAC 都从该字段读取。不要把 CE PBRS 绑定到具体算法命名空间。

formal central trainer 必须读取 `central_critic` 和 `masac` block，不得读取 `local_sac` block。

## 三、必须新增的测试

在开始任何正式训练前，完成并运行：

1. formal resolved config 断言：
   - central critic；
   - map 120；
   - obstacle 1；
   - max_agents 12；
   - batch 128；
   - grad clip 0.5；
   - episode 非 128。
2. 三 scene hash 不同，spawn/termination/post-capture 字段正确。
3. Actor global-leak test：仅修改 critic-only global state，Actor 输出不变。
4. central sensitivity test：改变 teammate action，`Q_i` 变化。
5. 4/8/12 agent shape、padding、active mask 测试。
6. terminated/truncated bootstrap 测试。
7. 从静止恒加速度的速度/位移解析测试。
8. parity drag 下零动作减速；no-drag 下零动作保持速度。
9. speed cap、collision/boundary per-substep 测试。
10. replay action 与环境实际执行动作一致。
11. 10,000 reset 的 spawn 比例与 min-separation 统计测试。
12. focal batch 恰好为 128 items，配额为 64/8/8/32/16。
13. 同一 joint transition 可对应多个不同 focal agents，且 critic 始终收到完整联合上下文。
14. 同一 `(transition, agent)` 不重复、每条 transition 最多贡献 4 items。
15. primary role 唯一性、inactive-role 一致性测试。
16. ring buffer overwrite 后 stale role index 不可采样。
17. bucket replacement/fallback 统计和禁止静默 uniform 测试。
18. capture snapshot pool 非空、保存、恢复和采样测试。
19. checkpoint 与 replay manifest 不匹配时拒绝 resume。

## 四、训练门槛

合同和测试全部通过后才运行：

### 5k data-chain run

输出：
- effective config；
- scene config hashes；
- joint replay size 与五个 focal index pool sizes；
- 目标/实际 focal 配额、sampled role counts、unique transition 数、replacement/fallback 次数；
- action norm、speed、speed-limit rate；
- Q1/Q2/target Q；
- TD error；
- alpha、entropy、log_std；
- actor/critic grad norm；
- terminated/truncated/collision counts。

### 25k learning-signal run

每场景至少 20 个固定 seed deterministic evaluation episodes，使用真实 1000/1500 horizon，而不是 128。

通过条件不是必须成功捕获，而是：
- policy 不再长期输出近零动作；
- capture distance progress 或 CE energy 至少一项持续改善；
- 当前场景应出现的 focal role pools 均存在数据，缺失池必须有明确原因；
- Q/gradient/entropy 无异常；
- collision 不持续恶化。

### 100k gate

capture 或 pure CE 至少一项出现稳定非零成功；若仍完全为零，先定位 reward scale、target Q、状态覆盖和动作分布，不要同时改多个参数。

## 五、交付格式

完成后给出：

1. 修改文件列表；
2. 每个修改对应哪个合同问题；
3. 新 formal 启动命令；
4. 完整 effective config 摘要；
5. pytest 结果；
6. 5k/25k 报告路径和核心指标；
7. 尚未完成或仍有风险的项目；
8. 明确说明 legacy IQN 测试是否全部通过。

不要只写设计文档；需要实际修改代码、配置和测试。不要以“loss finite”作为训练成功结论。

