# CoCap 连续动作 MARL 重构：核心逻辑浓缩版

> 目标线：CR-MS + CE + VCT-LS 有限半径感知
>
> 本文只保留核心改动、数据流和实现顺序。旧离散 IQN 线仍作为独立 baseline，不在本线中删除。

> **2026-08-05 当前实现修订：** 本文早期的“连续速度命令 + feasible velocity layer”仅作历史设计。实际代码已采用显式 body-frame `a_x,a_y`：Actor 只限制 `||a||<=a_max`，环境负责 `v_max` 速度上限。原因、优劣和迁移规则见 `docs/CONTINUOUS_ACTION_AXAY_REFACTOR_20260805_ZH.md`。

> **动态状态优先级：** 本文是架构参考，不记录当前训练结果。六条 500k、AW bridge、深度复盘规则和结果依赖型 TODO 以 `docs/CONTINUOUS_ACTION_MARL_EXPERIMENT_TRACKER_20260804_ZH.md` 及 `docs/CONTINUOUS_ACTION_MARL_EXECUTION_LOG_20260805_ZH.md` 为准。

## 1. 为什么要重构

当前追捕者使用 9 个离散动作：纵向加速度 3 档 × 转向角速度 3 档。它能工作，但有三个瓶颈：

1. 速度方向和幅值只能从有限动作集合中选择，难以精细控制 CE 覆盖和 ring-MS 围捕。
2. 当前网络本质是“参数共享的独立 IQN”：每架机只看自己的局部观测，Critic/Q 不看全局状态和其他队友动作。
3. 逐 agent replay 无法完整保存同步的联合状态和联合动作，因此不是标准 CTDE。

重构目标是：

```text
离散 9 动作 + IQN
        ↓
二维连续加速度命令 `(a_x,a_y)` + 参数共享 Actor
        ↓
训练期全局 twin critics + joint replay
        ↓
执行期仍只使用局部 Actor
```

算法采用项目内实现的 PSA-MASAC：Parameter-Shared Attentive MASAC。它不是某篇论文的固定算法名，而是“共享局部 Transformer Actor + 集中式实体注意力 twin critics + off-policy joint replay”的组合。

## 2. 总体网络结构

### 执行阶段

每个 pursuer 使用同一个 Actor：

```text
局部 self / pursuer / evader / obstacle token
        ↓
共享 Local Token Transformer
        ↓
二维随机加速度策略头
        ↓
加速度圆盘约束 `||a||<=a_max`
        ↓
环境积分与 `v_max` 速度上限
```

Actor 只能读取本机 VCT-LS 允许的有限观测，不能读取全局敌人位置、全体联合动作或训练期 Critic 特征。

### 训练阶段

```text
全局 pursuer / evader / obstacle / boundary state
全体 body-frame acceleration actions + active mask
        ↓
独立 Global Encoder 1 → Critic 1 → Q_1^i
独立 Global Encoder 2 → Critic 2 → Q_2^i
```

Actor 和 Critic 可以使用相同的 token-attention 编程范式，但不是同一个网络，也不共享 learned encoder。两个 twin critics 从 encoder 到输出 head 都独立。

“focal pursuer i”表示当前正在计算哪一架机的回报。它不是 leader，也不是固定的 direct pursuer；每个 active pursuer 都可以作为 focal。一次前向可以同时得到所有 Q_i。

## 3. 连续动作和动力学

> **当前实现以本段为准：** Actor 直接输出加速度动作，不再根据当前速度生成 `v_cmd`。旧版本节中的速度投影公式只解释为何被废弃。

当前数据流：

```text
z ~ Gaussian(mu, sigma)
a_cmd^B = a_max * tanh(||z||) * z / max(||z||, eps)
a_cmd^B -> body/world rotation -> env integration
```

```text
v_pre = v_t + R(yaw) a_cmd^B * DeltaT
v_{t+1} = clip_norm(v_pre, v_max)   # env only
```

SAC 的 log-prob、critic action 和 replay `actions` 全部针对 `a_cmd^B`；不存在 `raw velocity` 与 `final feasible velocity` 两套动作。

### 3.1 坐标系（当前 ax/ay 仍适用）

首版动作在生成观测时的机体系中表达：

```text
u_body = [u_x, u_y]
```

原因是当前局部几何、速度、CE centroid 和 boundary 已经旋转到机体系，而 self token 没有绝对 yaw 信息。若直接输出世界系 vx、vy，同一个局部观测可能对应多个世界方向，策略会产生坐标歧义。

决策开始时只做一次 body → world 旋转，整个 0.5 秒决策窗口保持同一个世界系 command。

### 3.2 径向随机动作（当前输出加速度）

不要分别对 `a_x、a_y` 做 tanh，因为这样会得到方形加速度域，对角线加速度可能超过 `a_max`。

推荐先产生无界二维高斯变量，再压缩到单位圆盘：

```text
z = mu + sigma * noise
r = sqrt(z_x^2 + z_y^2)
u = tanh(r) / max(r, eps) * z
```

它的含义很直观：

- 方向保持 z 的方向；
- 半径从 0 到无穷压缩到 0 到 1；
- 因此 norm(u) < 1；
- SAC 的 log-prob 在 u 空间计算。

确定性评估使用 `u = radial_squash(mu)`，不进行采样。

第一版不引入 normalizing flow，也不恢复 IQN 的 quantile critic，以免同时引入过多变量。

### 3.3 历史 velocity feasible-layer 方案（已废弃）

以下公式只记录为何切换到显式 ax/ay，不得用于新训练：

决策周期为：

```text
substep dt = 0.05 s
substeps   = 10
DeltaT     = 0.5 s
```

设：

```text
A = a_max * DeltaT
```

当前速度先转到 body frame，然后由策略模块生成最终可行 command：

```text
v_cur_body = rotate(-yaw, v_cur_world)
v_pre_body = v_cur_body + A * u_body
v_cmd_body = project_to_speed_circle(v_pre_body, v_max)
v_cmd_world = rotate(yaw, v_cmd_body)
```

速度圆盘投影为：

```text
if norm(v_pre_body) <= v_max:
    v_cmd_body = v_pre_body
else:
    v_cmd_body = v_pre_body * v_max / norm(v_pre_body)
```

由于欧氏投影不会放大两个点之间的距离，而当前速度本身已经在速度圆盘内，所以：

```text
norm(v_cmd - v_cur)
    <= norm(A * u)
    <= a_max * DeltaT
```

这意味着最终给环境的 v_cmd 已经满足 max v 和 max a。环境不再负责隐式裁剪，只做数值断言。

环境在 10 个子步内执行线性速度参考：

```text
v[k] = v_cur + k / 10 * (v_cmd_world - v_cur)
p[k+1] = p[k] + 0.5 * (v[k] + v[k+1]) * dt
```

这样每个子步的实际加速度天然不超过 a_max。若发生碰撞、扰动或底层执行误差，实际速度可以偏离 command，但这属于环境转移，不能把实际速度偷换为策略 action。

`a_max` 是可调配置，而不是固定沿用旧的 0.4。第一轮建议测试：

```text
a_max = 0.4, 0.8, 1.6 m/s²
```

重点观察制动距离、90/180 度转向、CE 停稳、碰撞和 command 投影频率。

## 4. yaw、有限感知和 boundary

### 4.1 当前 yaw 事实

当前 CR-MS effective config 是：

```text
感知半径约 20 m
perception.angle = 2π
```

因此当前 VCT-LS 是“有限半径的全向局部感知”，不是扇形 FOV。首版建议：

```text
yaw.mode = hold
yaw_init = 0  # P0 smoke：所有 agent 与世界轴对齐
```

即保持 episode 内的 body/sensor yaw。P0 smoke 先让所有 agent 的 yaw_init=0，使 body frame 与世界坐标重合，降低首轮变量数量；这不是最终的旋转泛化方案。二维全向速度控制允许 UAV 保持机头方向、同时向侧方运动。后续再加入共同旋转初始化或随机 yaw，检查旋转鲁棒性。

注意：当前源码在未显式传入 theta 时仍默认随机初始化 yaw；`yaw_init=0` 是连续动作线需要新增的配置/初始化分支，不代表现有旧线已经改变。

需要区分：

```text
yaw              = 机体/传感器朝向
velocity heading  = 实际地速方向
command heading   = v_cmd 方向
```

零速时没有速度方向，不能把 atan2(0, 0) 强行定义成世界 x 轴；应保持上一有效 yaw。

不要让 body frame 每一时刻都跟随速度方向，否则会出现：

- 低速噪声导致坐标系大幅翻转；
- self velocity 的横向分量被人为变成 0；
- 动作改变速度，速度又改变动作坐标系，形成循环依赖；
- 未来扇形 FOV 下可能绕过真实 yaw-rate 约束。

未来如果真的启用小于 2π 的扇形 FOV，再单独加入受限 command-heading tracker；不要把它和第一版连续动作重构同时引入。

### 4.2 Boundary 表示

当前 CR-MS strongest config 已使用：

```text
boundary_feature_mode = nearest_vector_robot_oob
```

也就是最近边界向量已经旋转到机体系，并保留 out-of-bound flag。旧的 `axis_signed` 分支才是世界 x/y 轴语义。

当前最近边界向量的不足是：只看最近一面墙，并且在两面墙等距处可能发生 hard switch。**本线初版冻结当前 self token，不修改 boundary 维度和语义。** 4 个 boundary tokens 只作为后续独立消融方案，每个 token 可包含：

```text
signed clearance
局部墙法向 nx, ny
该墙是否已经越界
```

四面墙同时输入 attention，可以自然表达角点，也可扩展到多边形和未来的外出—再进入机制；但不进入初版主线，避免动作、Critic、观测三处同时变化。

## 5. Joint replay 和 phase 语义

### 5.1 最小 transition

每个 environment decision 只保存一条 joint transition：

```text
actor_obs_t / actor_obs_tp1
global_state_t / global_state_tp1
agent_mask_t / agent_mask_tp1
      actions = a_cmd_body / a_max
reward vector
terminated / truncated / bootstrap mask
active target mask
role mask
event bits
team size / target count
```

正式 replay 不需要逐条保存：

```text
raw Gaussian z
历史 u
body/world 两份 command
executed velocity
executed acceleration
Python reward dict
```

这些内容要么能由状态和配置恢复，要么只应作为低频 debug metrics 保存。

### 5.2 不再固定 64/16/32/16

旧比例是逐-agent IQN 的项目经验，不是 MASAC 的标准要求。新版：

```text
一个 joint ring
每条 transition 对所有 active pursuer 做 masked loss
semantic/event index 只保存 transition_id
```

推荐初始采样混合：

```text
70% uniform joint sampling
20% active-target / coverage-only regime sampling
10% discovery / capture / collision / CE-success event sampling
```

具体实现是：每条 joint transition 入 ring 时，同时写入 `regime` 和 `event_bits`，并把 transition_id 放入轻量索引。

- uniform：从整个 ring 的有效 transition_id 等概率抽取；
- regime：从 `active_target` 和 `coverage_only` 两个索引中抽取，建议各占一半，并设置 coverage-only 最低 batch 占比；
- event：先随机选择一个非空事件池，再从该事件池抽 transition_id。事件池只保存 ID，不复制 transition。

事件定义建议为：

```text
discovery       敌人首次进入该队伍可见集合，或 agent 首次从 coverage 切到 capture
capture         当前步发生 loose/stationary capture；同时加入前后 K 步窗口
collision       pursuer/evader/obstacle collision 或 boundary breach
CE-success      CE strict/settle success latch 首次翻转；可另建 near-success 池
```

例如 batch=64 时，按 70/20/10 可先生成约 45/13/6 个 source slots；coverage-only 最低比例是额外的硬 floor，若当前 batch 中不足 16 条，就从 uniform slots 中替换为 coverage-only transition。因此最终 source 数可能偏离 70/20/10，但 coverage 不会被追捕数据淹没。每个被抽到的 joint transition 仍对所有 active pursuer 做 masked loss；event 标签不作为网络输入。某个事件池为空时回退到 uniform/regime，并记录 fallback 次数。首版先不用 PER；后续若启用，必须做 importance correction。

### 5.3 phase 是否被网络感知

phase 写入 metadata 不等于网络看到了 phase。新版默认：

```text
post_capture + pure_coverage → control_regime = coverage_only
```

两者的 `origin` 只用于采样、诊断和消融，不输入 Actor。Critic 也不应使用“这个 episode 曾经捕获过敌人”这种历史捷径。

但当前代码中两者仍可能存在 reset 来源、PBRS reset、coverage window 和 termination 差异。新线应优先统一这些合同；如果某个 timer 真正影响最优动作，则该 timer 必须是部署可得状态，并同时输入 Actor/Critic。

## 6. replay 持久化

### 优点

- 崩溃或重启后可以直接恢复，不必重新 warm-up；
- 稀有 discovery、final capture、CE success 不容易丢失；
- 课程切换时可少量复用旧规模经验；
- 便于审计 replay 分布和比较不同网络。

### 风险

- joint transition 占用显著大于逐 agent transition；
- 旧策略数据过多会拖慢新策略适应；
- action、动力学、reward 或 observation schema 变化会污染 Bellman target；
- Python 对象存储会带来额外内存和 I/O 开销。

按当前 12-agent observation 粗略估算：

```text
100k joint steps: 约 1.6–2.0 GB 原始 float32 数据
250k joint steps: 约 4–5 GB
1M joint steps:   约 16–20 GB
```

首版建议 hot ring 从 200k–300k joint steps 起步，replay shard 与模型 checkpoint 分开保存。跨训练复用必须检查 action contract、vmax、amax、决策周期、yaw/FOV、boundary、reward、termination 和 normalization；旧数据初始占比控制在约 20%–30%。

## 7. 可执行实现顺序

### P0：冻结旧线

- 保存 CR-MS 当前 4v1/8v2/12v3 baseline、配置和 GIF；
- 保证 `unicycle_discrete` 旧模式完全不变；
- 增加旧线固定 seed regression。

### P1：连续动作环境合同

- 新增 canonical `acceleration_2d_body` action mode；
- Actor/adapter 只验证 `||a||<=a_max`，不做速度 projection；
- 环境执行加速度积分并负责 `v_max` 速度上限；
- 维护 acceleration、velocity、speed、yaw、trajectory 等字段；
- 做零速、制动、反向、90 度转向和碰撞测试。

### P2：radial Actor

- 新增与旧 Transformer 前端功能/权重兼容的 Local Token Encoder，首版不修改 `CoCapIQN`；连续线跑通后才考虑共享抽象；
- 新增 radial Gaussian Actor 和稳定 log-prob；
- 做 scratch 与 IQN encoder transfer 两种初始化；
- 检查 4v1/8v2/12v3、mask、permutation 和旋转一致性。

### P3：joint replay + local SAC smoke

- 建立 joint ring 和最小 transition schema；
- 先训练 local shared SAC，隔离动作和 replay 数据链问题；
- 先跑 pure CE，再跑 capture，最后跑 mixed CR-MS；
- batch 先用 32/64，实测显存后再决定是否增大。

### P4：central attention twin critics

- 新建 global encoder；
- 输入全局实体集合、joint acceleration actions 和 masks；
- 输出每个 focal pursuer 的 Q_i；
- 使用 scalar twin soft-Q、Polyak target 和自动温度；
- 全部 loss 按 active mask 求均值。

### P5：4v1 screening 与正式训练

- 对前两名 a_max 做 25k smoke；
- 50k/100k 开启 capture、coverage、mix screening；
- 晋级组合训练到正式预算；
- 阶段最优模型自动做 20-rollout/10-GIF，尾迹默认关闭。

### P6：8v2、12v3 课程

- 从 4v1 最优模型 warm-start；
- 继续使用 mask，不改成固定拼接 head；
- 混合规模 replay，验证 loss、entropy 和显存随规模稳定；
- 测试随机失活 pursuer 的 mask 鲁棒性。

### P7：归因消融

至少比较：

```text
离散动作集合 vs 连续 acceleration
local SAC vs centralized MASAC
MASAC vs Set-Attention MATD3
scratch vs IQN encoder transfer
nearest boundary vector vs four-wall tokens
```

首版暂不同时加入 GRU、normalizing flow、intrinsic reward、复杂 PER、TQC 或 FACMAC。

## 8. 最低验收标准

动作层：

- 任意 Actor action 都满足 `norm(a_cmd_body) <= a_max`；
- 环境转移速度满足 `norm(v_next) <= v_max`，并记录 `speed_limited`；
- 环境不把速度裁剪后的结果写回 replay action；
- zero command 能稳定停车；
- body/world 旋转一致，旧 unicycle 模式不回归。

网络层：

- Actor 不能读取 global state；
- Critic 对 agent permutation 等变；
- padded/失活 agent 不贡献 loss；
- twin critics 参数独立；
- deterministic evaluation 可复现且无 NaN。

训练层：

- joint replay 能混合 4/8/12 规模；
- active-agent masked loss 与 entropy 不随规模简单放大；
- capture → coverage transition 的 bootstrap 语义正确；
- replay manifest 能阻止不兼容数据混入。

最终目标不是一次性把所有新模块堆上去，而是按“动作合同 → local SAC → centralized critic → 课程扩规模”的顺序逐层验证。
