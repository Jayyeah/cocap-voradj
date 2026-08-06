# CoCap-VorAdj MASAC-CTDE 重构 Agent 提示词（v2）

**配套审计文档：** `cocap_voradj_masac_ctde_task_contract_audit_20260806_zh.md`

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
