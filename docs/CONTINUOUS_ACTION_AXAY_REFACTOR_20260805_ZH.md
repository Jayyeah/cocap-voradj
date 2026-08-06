# 连续动作 ax/ay 合同修订（2026-08-05）

> 本文只冻结 ax/ay 合同；不包含 AW bridge 的训练结果。当前六条长训状态、AW 实现入口、IQN replay 分析和接手 TODO 以 [`CONTINUOUS_ACTION_MARL_EXPERIMENT_TRACKER_20260804_ZH.md`](./CONTINUOUS_ACTION_MARL_EXPERIMENT_TRACKER_20260804_ZH.md) 为准。

## 1. 当前唯一动作定义

连续线不再让策略直接输出期望速度 `v_x,v_y`。每个 pursuer 的 Actor 输出机体系二维加速度：

```text
a_cmd^B = [a_x, a_y],       ||a_cmd^B||_2 <= a_max
```

Actor 使用径向 squashed Gaussian，把无界二维高斯变量压到半径为 `a_max` 的圆盘；因此策略输出从源头满足 `a_max`。`a_max` 是可调实验参数，首轮主值为 `0.8 m/s²`，`0.4/1.6` 仅作对照。

环境执行动力学时把一次加速度命令旋转到世界系并积分：

```text
a_cmd^W = R(yaw) a_cmd^B
v_pre   = v_t + a_cmd^W * DeltaT
v_{t+1} = clip_norm(v_pre, v_max)       # 仅环境速度上限
p_{t+1} = p_t + 0.5 * (v_t + v_{t+1}) * DeltaT
```

其中 `v_max=3.0 m/s` 是环境物理边界，不进入 Actor 的动作参数化，也不由策略侧做速度圆盘投影。环境速度达到上限时记录 `speed_limited=true` 和实际加速度，但不把该速度裁剪伪装成策略动作。

代码中的 canonical 名称为：

| 旧/历史名称 | 当前名称 | 语义 |
|---|---|---|
| `velocity_2d_body` | `acceleration_2d_body` | 当前配置动作模式；旧字符串仅作兼容别名 |
| `RadialActorConfig` | `AccelerationActorConfig` | Actor 配置，不再含 `v_max` |
| `RadialSquashedGaussianActor` | `RadialSquashedGaussianAccelerationActor` | 输出 `a_x,a_y` |
| `FeasibleVelocityAdapter` | `AccelerationActionAdapter` | 只验证/记录 `a_max`，不投影速度 |
| `final_commands` | `actions` | joint replay 中实际发送的 body-frame 加速度 |

旧 velocity 名称保留别名是为了读取已有代码路径；它们不恢复旧的“策略输出速度、adapter 投影速度”语义。runtime `JointReplayBuffer.schema_version=3`；旧 velocity replay 不允许直接续训。历史记录中的“schema v2”只指动作字段语义，不是当前可加载版本。

## 2. 为什么选 ax/ay

旧的 `v_x,v_y + policy-side a_max` 方案把一个动作同时解释为“目标速度”和“加速度增量”。当当前速度接近零、速度又被投影到边界时，不同 latent 可能映射成相同执行速度，导致执行动作与 SAC 的 log-prob/entropy 不一致，训练梯度失真。

改为显式 `a_x,a_y` 后，Actor 的随机变量、replay 中的 action、env 接收的 command 是同一个物理量；SAC 的 Q target、policy loss、entropy 也都针对同一个动作定义。这样去除了 velocity feasible layer 的非可逆映射，修正了原 P6 25k 中 raw/final action mismatch 和后续径向 Jacobian 为零的问题。

## 3. 约束边界和实现责任

- 策略侧：只负责 `||a|| <= a_max`，使用圆盘 squash；越界的外部/测试动作直接报错。
- 环境侧：负责 `v_max`、碰撞、边界和子步积分；速度超过上限时做物理速度裁剪并写入诊断。
- replay：保存 `actions`（body-frame acceleration），不保存 `final feasible velocity`。
- critic：joint action 输入维度仍为每个 agent 的 2 维，但含义改为 acceleration；inactive agent 继续用 mask。
- yaw：首版保持 hold，连续线 `yaw_init=0`；body→world 在决策窗口只旋转一次。

## 4. 优势、代价与已知风险

### 优势

1. 动作、动力学和 SAC log-prob 一致，避免不可逆速度投影造成的梯度/熵错配。
2. `a_max` 直接控制机动能力，转向、制动和 jerk 更容易解释和做单变量实验。
3. `v_max` 作为环境物理上限统一约束所有策略、脚本和未来算法，不需要每个 Actor 重复实现速度投影。
4. replay schema 简单：一个 `actions` 字段即可跨 local/central critic 使用。

### 代价和风险

1. 策略不再直接指定速度，必须通过多步积分才能形成期望速度；探索初期可能更慢。
2. 环境的 `v_max` 裁剪会使实际速度增量小于 `a_cmd*DeltaT`，因此必须同时记录 `speed_limited` 和实际加速度，不能把二者混为策略 action。
3. `a_max` 过小会导致制动距离过长，过大则易碰撞；先用 scripted/oracle 筛选，再进入 SAC。
4. 暂不加入 jerk 约束；若后续需要低层平滑，应增加可观测的 controller state 或显式 jerk action，不能偷偷在 env 滤波。

## 5. 实验影响和迁移规则

1. 之前基于 velocity action 的连续线 P4/P6 报告、replay 和 checkpoint 只作为历史/故障诊断，不得与 ax/ay 结果合并，也不得直接 resume。
2. 所有连续配置、runner、smoke、oracle 和 central critic 必须使用 `action_mode: acceleration_2d_body` 与 `actions` 字段；canonical P1 配置为 `configs/experiments/continuous_marl_20260804/p1_acceleration_contract_4v1.yaml`。
3. 重新从 4v1 scratch 做 action contract、local SAC、central SAC 和 screening；通过后再开 8v2/12v3 课程。
4. 每个大阶段最优模型继续自动生成 20-rollout/10-GIF，尾迹默认关闭；该展示流程与动作合同无关。

## 6. 最小验收公式

对每个策略动作，必须满足：

```text
finite(a_cmd^B)
||a_cmd^B||_2 <= a_max + atol
```

环境转移必须满足：

```text
finite(v_{t+1}, p_{t+1})
||v_{t+1}||_2 <= v_max + atol
```

其中最后一条是环境速度上限，不是 Actor 的动作投影合同。两条约束分别由 Actor/adapter 测试和 Robot/env 动力学测试覆盖。
