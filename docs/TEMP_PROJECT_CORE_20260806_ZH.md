# CoCap-VorAdj 连续动作 MARL 重构 — 临时核心文档

> 建立日期：2026-08-06 15:xx CST  
> 用途：给完全不了解本项目的新 agent / 维护者快速建立全局理解。本文是**临时汇总**，不是唯一权威台账。  
> 权威状态、合同、run 台账、TODO 与判定规则以 [`CONTINUOUS_ACTION_MARL_EXPERIMENT_TRACKER_20260804_ZH.md`](./CONTINUOUS_ACTION_MARL_EXPERIMENT_TRACKER_20260804_ZH.md) 为准；逐轮命令与故障证据见 [`CONTINUOUS_ACTION_MARL_EXECUTION_LOG_20260805_ZH.md`](./CONTINUOUS_ACTION_MARL_EXECUTION_LOG_20260805_ZH.md)。

---

## 1. 一句话项目简介

这是一个多智能体“围捕 + 区域覆盖”强化学习项目（CoCap-VorAdj）。旧主线用离散动作 + IQN/DQN 已经达到很高的正式成绩（4v1/8v2/12v3 捕获率、覆盖成功率均接近 100%）。当前实验把追捕者的离散控制改为**连续二维加速度动作**，并用 **PSA-MASAC**（参数共享局部 Transformer Actor + 集中式实体注意力 twin critics + joint replay）重新训练，目标是在不损失任务成功率的前提下验证连续控制可行性，同时为后续 CTDE 协同做准备。

项目纪律非常严格：不许改动旧 IQN 基线；每个实验阶段有独立 gate；失败结果原样保留；一次只做一个变量；所有结论必须有命令、产物、seed、hash 证据。

---

## 2. 领域与任务需求

### 2.1 环境设定

- 平面矩形区域（约 55×55），圆形机器人（pursuer/evader），无障碍物（P6 实验 `num_obstacles=0`）。
- 每架 pursuer 有有限半径全向感知（约 20 m，2π），能看到 VCT-LS 允许的友方/敌人/障碍 token。
- 任务分三类：
  - `pure_ce`：只有 4 架 pursuer 的纯覆盖任务，按 Voronoi 分区让每架飞机靠近自己 cell 的质心（centroid）并停稳；
  - `capture`：4 追 1，要求形成围捕环并捕获 evader；
  - `mixed_crms`：先捕获再覆盖（post-capture coverage）的混合任务。
- P6 单集上限 128 步；决策周期 `DeltaT=0.5 s`，物理子步 10 个。

### 2.2 旧基线（对照标准）

- 旧算法：9 个离散动作（纵向加速度 3 档 × 转向角速度 3 档）+ 分布式 IQN（每个 pursuer 一套共享参数 Q 网络）。
- 正式 20-rollout/10-GIF 结果（CR-MS 阶段最优）：

| 规模 | capture | pure CE strict | mix capture | mix CE strict | collision |
|---|---:|---:|---:|---:|---:|
| 4v1 | 100% | 100% | 100% | 90% | ≤5% |
| 8v2 | 100% | 100% | 100% | 100% | 0% |
| 12v3 | 100% | 100% | 100% | 95% | 0% |

- CE strict 是“中心覆盖 + 面积均衡 + 停稳”的严格成功；`CV<0.15` 只是较松的面积均匀指标，**不能替代 strict**。

### 2.3 新实验目标与判定

- 连续动作必须达到对旧基线的**非劣门槛**：capture、mix capture、pure/mix CE strict 各不低 5 个百分点；collision 不高于 2 个百分点；并报告 steps、settle、CV、path、acceleration、jerk。
- 结果排序固定：**真实 capture event → CE strict → collision → steps → CV<0.15**。
- 最终成功需要 3 seeds PSA-MASAC 达标；若只有 local SAC 达标，判“部分成功”（连续动作可行、CTDE 未成功）。

### 2.4 用户硬约束

1. 不删除/重写旧 IQN/DQN，旧 `unicycle_discrete` 配置行为不变；
2. 不连续化 evader、不改 reward/termination/boundary 表示；
3. 不加 phase/event/global state 到 Actor；
4. 前置 gate 不过不升级（100k→200k→central→a_max=1.6→P7 全部冻结）；
5. 失败 run 原样保留，不换 seed、不覆盖报告；
6. 每轮只做一个最小 TODO，改动前检查 `git status`，修改后回填文档。

---

## 3. 技术方案与实现方法

### 3.1 动作与动力学合同（当前 canonical）

**ax/ay 线（主实验）：**

```text
Actor 输出机体系二维加速度 a_cmd^B = [a_x, a_y]，||a|| <= a_max = 0.8 m/s²
环境把命令旋转一次到世界系并积分：
  v_pre  = v_t + R(yaw) a_cmd^B * DeltaT
  v_next = clip_norm(v_pre, v_max=3.0)      # 环境物理速度上限
  p_next = p_t + 0.5*(v_t+v_next)*DeltaT    # 10 子步梯形积分
```

- 策略侧只保证 `a_max`（径向 squash 输出天然满足）；环境负责 `v_max`、碰撞、边界；速度被裁剪**不写回 action**，只记录 `speed_limited` 诊断。
- yaw 首版 hold，连续线 `yaw_init=0`（body 与世界轴对齐），旧线 reset 不变。
- 代码：`src/cocap_voradj/dynamics/continuous_action.py`（`AccelerationActionAdapter` 只验证）、`robot.py`（`update_state_acceleration_body`）、`envs/base.py`（连续分支）。

**AW bridge 线（独立对照，结果不得与 ax/ay 合并）：**

- Actor 输出连续 `(a, w)`：前向加速度 `a ∈ [-0.4, 0.4]`，yaw 角速度 `w ∈ [-pi/6, pi/6]`；保留旧水阻尼和 yaw 积分。
- 这是为将来 IQN teacher transfer 保留的桥接动作，不作为最终部署合同。

### 3.2 网络结构

**共享局部 Actor（部署可用）：**

```text
局部 self/pursuer/evader/obstacle token
  → LocalEntityTokenEncoder（Transformer，hidden=256，4 层 8 头）
  → self_token + mean_context 拼接 → MLP → mean/log_std
  → radial squashed Gaussian：a = a_max * tanh(||z||) * z/||z||
```

- 径向 squash 保证圆盘约束；`log_prob` 包含径向 Jacobian，SAC 梯度正确。
- AW 用 box actor：两个分量各自 `tanh`，同样含 Jacobian。

**Critic（仅训练期）：**

- local 模式：两个完全独立的 `LocalQCritic`（局部 encoder + action + MLP head）。
- central 模式：两个完全独立的 `CentralAttentionCritic`，输入训练期全局实体集合 + 全体 joint action + mask，输出每个 focal pursuer 的 `Q_i`；Actor 梯度只作用于自己的 action 分支（队友分支 detach）。
- Actor 与 Critic 参数完全独立，不复用特征。

### 3.3 SAC 训练配置

| 项 | 值 |
|---|---|
| 网络规模 | hidden=256，encoder 4 层 8 头（formal profile） |
| 优化器 | Adam，lr=1e-4（actor/critic/alpha） |
| warmup | 5000 步（期间只收集数据不更新） |
| grad clip | 10 |
| batch | 64，每步 1 次更新 |
| gamma / tau | 0.99 / 0.005 |
| alpha_init / target_entropy | 0.2 / -2.0 |
| log_std bounds | [-5, 1]（默认） |

### 3.4 Joint replay 与采样

- 一个 joint ring（`JointReplayBuffer.schema_version=3`），容量 100000；每条 transition 保存：
  `local_obs/next_local_obs、global_state/next_global_state、actions(4,2)、rewards、active_mask、terminated/truncated、metadata`。
- 采样 70% uniform + 20% regime（active_target/coverage_only）+ 10% event（discovery/capture/collision/CE-success），另有 coverage-only 最低占比 25%；事件池为空时回退 uniform 并记录 fallback。
- metadata 只允许 `regime/event_ids/phase/scene/task_label/coverage_only/active_target`；禁止把 phase/event 当网络输入。
- replay 与 checkpoint 分文件保存；manifest 严格匹配，含 config/implementation hash、RNG 状态。

### 3.5 Runner、screening、持久化与 resume

- 入口：`tools/run_continuous_p6_screening.py`。
- 场景集：`full`（三类循环）、`pure_coverage`、`broadcast_capture`（可加 `--global-evader-visibility` 解除敌人可见限制）、`broadcast_mixed`。
- 长训参数：`--critic-mode local --trainer-profile formal_p6 --total-steps 500000 --replay-capacity 100000 --screen-interval 100000 --screen-episodes 4`。
- **持久化现状**：六条运行线启动时 checkpoint/replay 与 screening 绑定，即 step1 + 每 100k 保存；2026-08-06 起 runner 新增 `--save-interval`（默认 25000）把保存与 screening 解耦，**新 run 每 25k 保存**。运行中的六条线不受影响，仍 100k 一存。
- **停机续训**：只能从最近一次保存的 checkpoint+replay 用 `--resume-checkpoint/--resume-replay/--resume-step` 恢复，`continuation_mode=seeded_episode_boundary`，`exact_env_state=false`（不恢复逐子步环境状态）。manifest 不一致会被严格拒绝。
- 运行中六条线若需续训，必须使用启动时版本 runner 备份 `tools/run_continuous_p6_screening.py.pre_25k_save_20260806.bak`（SHA256 `4303cc…`），否则新版 runner 的 implementation hash 不匹配。

### 3.6 环境与奖励（P6 有效配置）

- 奖励基于旧 CR-MS/CE/VCT-LS 配置，P6 只把 episode 上限改为 128、去掉障碍：
  - **CE 覆盖奖励**：`-中心距离代价 * 10` + PBRS（kappa=1），当前 speed/accel 正则权重为 0；
  - **capture 奖励**：ring_importance_ms_v0（`omega_ring_ms=2.0`），捕获成功给 `120 * factor` 终止奖励；
  - **碰撞惩罚**：`-80` 并永久失活；边界也有惩罚；
  - 无 timestep penalty（`capture_timestep_penalty=0`，CE 分支本身不带时间惩罚）。
- 终止语义：128 步超时 = `truncated`（允许 bootstrap）；碰撞/失活/capture 完成 = `terminated`（关闭 bootstrap）；capture 后进入 post-capture 不算终止。

### 3.7 已修复的数据链 bug（2026-08-05）

| bug | 问题 | 修复 |
|---|---|---|
| BUG-04 | 失活 pursuer 的 next obs 为 None 导致 transition 不入池，碰撞奖励丢失 | 固定 4 agent slot + 零 padding + 动作前 active mask |
| BUG-05 | 所有 done 都写成 terminated，128 步超时也被当吸收态 | 区分 terminated/truncated，SAC bootstrap 只由 terminated 关闭 |
| BUG-06 | regime/event 按 scene 硬编码、首步伪标 discovery | 从动作前状态 + env outcome 真实派生，并记录 source/fallback |
| BUG-07 | manifest 缺网络/优化/hash，resume 语义不明确 | schema2 checkpoint + schema3 replay、RNG 与 runtime state、seeded episode-boundary 合同 |

---

## 4. 实验进展与当前现状

### 4.1 阶段状态（P0–P9）

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | 旧基线冻结 + 三规模 regression | [x] |
| P1 | 连续动作/动力学合同 | [x] |
| P2 | a_max=0.4/0.8/1.6 oracle 筛选（0.8 主候选） | [x] |
| P3 | Local Token Encoder + radial Actor | [x] |
| P4 | joint replay + local SAC smoke | [x]（底层通过；P6 在线集成曾重新打开） |
| P5 | central twin critics / PSA-MASAC smoke | [x] |
| P6 | 4v1 筛选与正式训练 | [~]（六条 500k 运行中；两条 100k 已复盘失败） |
| P7 | 8v2/12v3 课程 | [ ]（冻结） |
| P8 | 消融 | [S] |
| P9 | 正式评测/展示/交接 | [ ] |

### 4.2 六条 500k scratch 长训台账

| 线 | 动作 | scene | seed | GPU | 启动 | 最新产物 |
|---|---|---|---|---|---|---|
| p6_axay_500k_mix | ax/ay a_max=0.8 | full 非 global | 2026081501 | GPU0 | 08-05 23:26 | step1；未到 100k |
| p6_axay_500k_pure | ax/ay a_max=0.8 | pure_coverage | 2026081502 | GPU1 | 08-05 23:26 | **step100000（15:01）** |
| p6_axay_500k_global | ax/ay a_max=0.8 | broadcast_capture + global | 2026081503 | GPU0 | 08-05 23:26 | step1；未到 100k |
| p6_aw_500k_mix | AW a_max=0.4,w=pi/6 | full 非 global | 2026081601 | GPU0 | 08-05 23:35 | step1；未到 100k |
| p6_aw_500k_pure | AW a_max=0.4,w=pi/6 | pure_coverage | 2026081602 | GPU1 | 08-05 23:35 | **step100000（12:32）** |
| p6_aw_500k_global | AW a_max=0.4,w=pi/6 | broadcast_capture + global | 2026081603 | GPU0 | 08-05 23:35 | step1；未到 100k |

所有进程截至 2026-08-06 15:10 均存活、无 traceback；无任何最终 `report.json`（即无训完的线）。

### 4.3 资源与 ETA

- GPU0：4 条线 + 外部任务，显存约 20 GB / 49 GB，利用率约 63%；GPU1：2 条线，约 11 GB / 49 GB，利用率约 22%。
- 磁盘剩余约 91 GB。
- ETA（按已实测 100k 吞吐推算，当前负载下）：

| 线 | 预计 100k | 预计 500k 完成 |
|---|---|---|
| AW pure | 已完成（12:32） | 约 08-08 19:00 |
| ax/ay pure | 已完成（15:01） | 约 08-09 05:00 |
| AW mix / AW global | 约 08-06 17:00–20:00 | 约 08-09 12:00–16:00 |
| ax/ay mix / ax/ay global | 约 08-06 17:00–20:00 | 约 08-09 16:00–20:00 |

估算受外部负载影响，100k 落盘后会按真实吞吐修正。

### 4.4 关键历史结论（为什么现在才到长训）

- 最初 velocity 动作方案因“策略输出速度、replay 存 feasible command”的 raw/final 不一致作废（BUG-20260805-02/03），2026-08-05 用户确认改为显式 `a_x,a_y`。
- 修复后的 ax/ay 25k（r2）工程 gate 通过（finite、resume 通过、六点 screening 齐全），但 capture/CE strict 全 0，学习 gate 失败。
- formal matched 5k 双线（pure coverage 与 broadcast capture）：工程/数值通过，但 `CV<0.15=0/2` 且 collision=100%、capture=0/2，任务 gate 失败，因此不重开 25k，改为用户要求的六条 500k scratch 长程数据积累。
- Oracle 诊断证明环境可达：同一连续环境用脚本目标在 55/60 步内无碰撞达到 CE strict、center RMS≈0.003。所以**失败不是环境不可解，而是 SAC 学习信号问题**。

---

## 5. 失败原因分析（当前核心结论）

### 5.1 现象汇总

| 阶段/线 | 现象 |
|---|---|
| formal matched 5k pure | 全程 collision=100%，CV<0.15=0/2，末速 2.2–3.0 |
| formal matched 5k global capture | capture=0/2，collision=100%，末速 2.7–3.0 |
| ax/ay r2 25k | 工程通过；后段 collision 0 但 128 步空跑、capture/CE strict 全 0；alpha 从 0.2 塌缩到约 0.054 |
| **ax/ay pure 100k（08-06）** | deterministic 探局第 14/16 步碰撞、末速 3.0、CV 0/2；replay 中 action 近饱和（p95=0.80）、speed 后 10% 比前 10% 更高（2.30 vs 1.60）、collision event=2403 |
| **AW pure 100k（08-06）** | 探局全程速度 0、无碰撞、CV 0/2；逐步动作显示 mean a≈-0.32（持续制动）、w 转一下后归零；replay 中 collision=1、speed≈0.0016 |

### 5.2 ax/ay 线解读：高速乱撞 / 无停稳信号

- 策略学到的不是“接近质心后制动”，而是“持续满油门加速到 v_max 直到碰撞”。
- 证据链：action norm 长期接近 0.8 上限；速度单调冲到 3.0；碰撞集中在 14–16 步（从零速加速到 3 m/s 大约就是这个时间尺度）；训练后期速度反而更高。
- 这说明 **CE 的“接近质心”正信号没能压过“高速移动”的副作用**，或 **PBRS/奖励尺度使近端梯度过弱**；同时 SAC 探索温度塌缩后没有机会尝试“低速停稳”区域。

### 5.3 AW 线解读：原地制动 no-op（不动最优）

- 策略发现“不动”比“移动”更优：移动会增大中心距离代价并冒 -80 碰撞风险，而不动只承受初始位置的小代价（replay reward mean -0.39）。
- 证据链：确定性策略持续输出负加速度（在速度为 0 时无效果），w 只在前 10 步转动一下；速度恒 0；collision=1/100k；后 10% 训练数据速度≈0。
- 这是典型的 **reward 景观鼓励静止 + 无 timestep penalty + 碰撞惩罚过大 + 探索提前塌缩** 的组合退化，不是环境 bug。

### 5.4 交叉验证与排除

- **排除环境不可解**：oracle 脚本 55–60 步 CE strict=1、无碰撞。
- **排除数据链**：BUG-04–07 已修复并有回归测试；100k checkpoint/replay 可加载，rollout 无 NaN，replay 事件/终止计数合理。
- **排除 action 合同**：所有 rollout 动作满足 `||a||<=a_max`，无 action validation 报错。
- **指向 SAC 学习动力学**：alpha/log-std 塌缩（r2 已观察到 alpha 0.2→0.054）、奖励尺度不平衡（CE 每步约 -0.4 vs 碰撞 -80 vs 捕获 +120）、稀疏终止奖励、无时间惩罚导致的不动最优。

### 5.5 下一步建议（严格单变量）

1. 等四条未到 100k 的线落盘后，先做与 pure 线相同的“逐 episode + replay 行为”复盘；
2. 候选单变量诊断（一次只做一个，短 run 对照）：
   - 探索层：`actor_log_std_max` / alpha target / alpha 下限（已有 `--actor-log-std-max` 开关；之前 -1.0 同 seed 5k 未改善，需换方向）；
   - 奖励尺度：CE 每步代价缩放、碰撞惩罚分级、或增加轻微时间/移动成本破坏“不动最优”；
   - bootstrap/终止：确认 128 步 truncation 与 capture 后 post-capture 的 credit 是否正确传导；
   - critic 稳定性：TD/grad 是否被安全惩罚主导（需从最终 report 的 update tail 读）。
3. 禁止同时改 reward、网络、环境；禁止在无 100k 复盘前扩 central、a_max=1.6 或 P7。

---

## 6. 代码与文档索引

### 6.1 文档

- 核心台账（唯一状态）：`docs/CONTINUOUS_ACTION_MARL_EXPERIMENT_TRACKER_20260804_ZH.md`
- 执行日志：`docs/CONTINUOUS_ACTION_MARL_EXECUTION_LOG_20260805_ZH.md`
- ax/ay 合同修订：`docs/CONTINUOUS_ACTION_AXAY_REFACTOR_20260805_ZH.md`
- 架构浓缩版：`docs/CONTINUOUS_ACTION_MARL_ARCHITECTURE_REFACTOR_20260804_ZH_COMPACT.md`
- 完整架构：`docs/CONTINUOUS_ACTION_MARL_ARCHITECTURE_REFACTOR_20260804_ZH.md`
- 旧项目背景：`docs/PROJECT_HANDOFF_20260802_ZH.md`、CR-MS/CE/VCT-LS 相关文档

### 6.2 代码

- Runner：`tools/run_continuous_p6_screening.py`
- 训练：`src/cocap_voradj/training/continuous/{local_sac,central_sac,joint_replay,central_schema}.py`
- 网络：`src/cocap_voradj/models/continuous/{local_entity_token_encoder,radial_actor,box_actor,central_attention_critic}.py`
- 动力学/环境：`src/cocap_voradj/dynamics/{continuous_action,robot}.py`、`src/cocap_voradj/envs/{base,voronoi_adjacency,coverage_ce}.py`
- 配置：`configs/experiments/continuous_marl_20260804/`（P1/P3/P4/P5/p6_formal）
- 测试：`tests/test_continuous_*`、`tests/test_joint_replay_contract.py`、`tests/test_central_*`
- 产物根：`artifacts/2026-08-04_continuous_marl_refactor/p6_screening/`

### 6.3 常用复验命令

```bash
cd /home/yjq/rl/CoCap1/cocap-voradj
PYTHONPATH=src:. python3 -m pytest -q -p no:cacheprovider \
  tests/test_continuous_action_contract.py tests/test_continuous_actor_contract.py \
  tests/test_joint_replay_contract.py tests/test_local_sac_smoke.py \
  tests/test_central_sac_smoke.py tests/test_continuous_p6_simplified_task_contract.py \
  tests/test_p6_save_interval_contract.py
```

---

## 7. 接手检查清单（只读）

```bash
cd /home/yjq/rl/CoCap1/cocap-voradj
git status --short
pgrep -af 'run_continuous_p6_screening.py'
nvidia-smi
df -h /home/yjq
find artifacts/2026-08-04_continuous_marl_refactor/p6_screening -maxdepth 1 \
  -name '*500k*report.json' -o -name '*500k*step*.pt' -o -name '*500k*replay_step*.pkl'
```

规则：

- 进程存活 → 只读记录 PID/GPU/日志/最新 checkpoint，不杀、不重启、不换 seed/tag；
- 某线退出 → 先读 stdout 尾部和 checkpoint manifest，判断正常完成/合同错误/OOM/NaN/环境异常，再决定是否恢复；
- 恢复只允许 `seeded_episode_boundary`，且运行中的旧线必须用 `.pre_25k_save_20260806.bak` 版本的 runner；
- 任何 100k/final 结果都必须按“工程合同 → 训练动力学 → 行为轨迹 → 任务结果”四层复盘，不能只看 success_rate；
- 每轮结束回填核心台账与执行日志。

---

## 8. 已知风险与注意事项

1. **磁盘**：新 run 若全部用 25k 保存，100k 容量 replay 快照约 0.7 GB/个，六条 500k 约需 85 GB+，当前剩余 91 GB，较紧张；运行中六条线按 100k 保存不受影响。
2. **manifest 严格性**：任何代码改动都会改变 `implementation_hash`；运行中线的续训必须先还原启动时版本（备份文件），否则 resume 被拒。
3. **无 exact env-state resume**：`seeded_episode_boundary` 只能从 episode 边界恢复，不能声称逐步确定性复现。
4. **外部负载**：机器 128 核 load 常年在 100+，GPU 也有其他用户任务；ETA 会随负载波动。
5. **六条 500k 只是数据积累**：进程存活、finite、甚至无碰撞都不等于学习通过；判定只认真实任务事件 + 行为轨迹证据。

