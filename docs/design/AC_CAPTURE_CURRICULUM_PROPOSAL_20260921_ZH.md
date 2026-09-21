# A3：Final Capture 初始化课程设计提案（2026-09-21）

状态：**仅设计，未实现，未启动训练，等待用户审查**。
TASK_ID：`A3-CAPTURE-CURRICULUM-DESIGN-20260921`

## 1. 目的与边界

本提案只回答一个问题：在不改变 Final Capture 的奖励、观测、动作、物理和终止合同的前提下，是否可以先用更容易的**初始状态分布**让 AC 学会接近/成环，再逐步恢复 Final 的随机初始化。

本轮明确不做以下事情：

- 不改 reward 数值、reward 开关、capture detector、support 归一化、CE/PBRS 或 terminal bonus；
- 不改 pursuer/target 控制器、动作集合、感知半径、网络形状、训练器、replay 或 spawn 实现；
- 不把当前 AC-CAP capability-only 配置直接命名为 Final：当前 AC-CAP 使用 `legacy_end_step`、`episode_max_length=1000`、zero coverage/PBRS 和 capture-terminal，这些是能力基线差异，不是 canonical Final；
- 不把环形初始化当作 Final 结果。最终必须回到 Final `map_random`、无 ring quota 的 held-out 评估；
- 不启动 formal run。以下数字只是候选配置，只有用户 review gate 通过后才可实现。

“初始化课程优先”的含义是：四个 stage 的任务形状保持一致，只改变 reset 时的 pursuer-target 初始几何/可见性。这样能够把收益归因于 initialization curriculum，而不是人数、地图、奖励或动力学变化。

## 2. 最新事实与 canonical Final 依据

本提案基于 bounded worktree 在 HEAD `668d5f7` 的代码/文档，以及同步快照 `docs/ops/TRAINING_PERFORMANCE_SYNC_20260921_ZH.md`（证据截止 `2026-09-21 15:59:58+08:00`）。

当前 AC 事实：

| 线 | 最新事实 | 对本提案的含义 |
|---|---|---|
| AC-CAP Stage1 | formal 300k；normal capture `0`、2+ ring `0`、collision `0`；状态为 `IN_PROGRESS_OR_POST_FORMAL` | 先解决初始化/稀疏 capture visitation；不能把 ring-only 信号写成 capture 学会 |
| AC-COV Stage1 | formal 500k；strict CE `0`，分类 `NO_CLEAR_SUSTAINED_SIGNAL` | 本提案不混入 coverage 课程，也不借用 coverage 结果证明 capture |
| AC-MIX Stage1 | 150k formal capture `0`；后续 telemetry 才出现 first capture | 说明 mixed phase visitation 仍可能是瓶颈；本提案仍要求最终回到 Final 分布 |
| Final IQN reference | B0 reference PASS；Final matched evaluator mixed capture `20/20`，normal capture `1.0`，collision `0`；C3 formal100 mixed capture `1.0`、CE `0.99`、collision `0.01` | capture objective 在 canonical Final contract 中可达，但不能把旧 AC capability-only 合同的零结果直接归因于 backbone |

canonical Final 的冻结依据为：

- `docs/B0_FINAL_IQN_REFERENCE_20260917_ZH.md`：Final VorAdj/VCT-LS sensing、AW9、CR-MS/support/coverage/CE-PBRS/post-capture contract；
- `configs/experiments/forward_final_mappo_20260908/stage1_4v1.yaml`：Forward-Final 只将 canonical historical collision 切换为 `synchronized_swept_v1`；
- `configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/common.yaml`：Final CR-MS、VCT-LS、support、CE 和 recovery 数值；
- `docs/FORWARD_FINAL_SINGLE_TASK_LEDGER_20260915_ZH.md`：capture strong screen 为连续三点 `argmax/sample capture >= 0.50` 且 collision `<= 0.20`，并要求报告完整曲线而非只挑 best checkpoint。

## 3. 不可改变的 Final Capture 合同

以下合同在每一个候选 stage 都相同；任何实现若改变其中一项，都不再是本提案的 initialization-only curriculum。

### 3.1 环境、感知和动作

- `num_pursuers=4`、`num_evaders=1`、`num_obstacles=1`；地图 `120.0 m × 120.0 m`。
- pursuer/target 初始半径沿用 Final：pursuer `obs_r_range=[1.0,1.1]`；不新增尺寸课程。
- local surface sensing 半径 `20.0 m`；friendly VorAdj/VCT-LS 使用 `free_mask_projected`；不输入 unconditional global enemy coordinate。
- 观测保持 `self=9`、friendly token `7`、enemy token `7`、obstacle token `5`；padding 上限沿用 `max_pursuers=8`、`max_evaders=8`、`max_obstacles=5`。
- AW9：`a ∈ {-0.4, 0.0, 0.4}`，`w ∈ {-π/6, 0.0, π/6}`，其中 `π/6=0.5235987755982988`；不因 stage 改动作。
- target 的动态 `max_speed=3.5 m/s`；每个 stage 的**reset 初始 target speed range 均为 `[0.0,0.0] m/s`**，heading 独立为 `U[0,2π)`。这与当前 Final reset 的 `init_speed=0.0` 一致；不把 target speed 课程伪装成初始化课程。

### 3.2 Final reward contract（逐 stage 不变）

capture 与完整 Final 生命周期使用同一份 reward contract：

- capture：`capture_reward_mode=ring_importance_ms_v0`、`omega_ring_ms=2.0`、`ring_ms_progress_clip=3.0`、`ring_ms_inner_extra_margin=0.5`、`ring_ms_preferred_center_radius=8.0`、`ring_ms_outer_center_radius=10.5`、`ring_ms_radial_sigma=2.0`、`ring_ms_cell_size=1.5`；
- moving-target angle weighting：`ring_ms_angle_alpha=0.25`、weight clamp `[0.75,1.25]`，static threshold `0.30 m/s`、full threshold `1.00 m/s`；occupancy mode `cartesian_disk`、radius `4.0 m`；
- capture event：`capture_distance=8.0 m`、`k_required=3`、`max_angle_gap=π`、`max_angle_ratio=3.0`；`goal_reward=120.0`；正常 capture 与 stationary capture 必须分开记录；
- support：`support_reward_blend_enabled=true`、capture/coverage weights `0.5/0.5`、`support_reward_capture_target_mode=neighbor_visible`、`support_reward_capture_component_mode=approach_only`、approach weight `1.0`、clip `3.0`；surviving capture weights 仍按 Final normalization 归一化；不把 uninformed agent 改成 oracle；
- coverage/CE：`coverage_ce_reward_scale=10.0`、`coverage_ce_pbrs_enabled=true`、`coverage_ce_pbrs_kappa=1.0`、PBRS reset mode `phase_and_all_terminal`；speed weight schedule 为 step `0→0.0`、step `200000→0.0005`；strict CE 为 RMS `0.05`、max `0.10`、hold `30`；
- safety：`timestep_penalty=-1.0`、`collision_penalty=-160.0`、`emergency_penalty=-10.0`；hard boundary/death 开启，boundary proximity distance `4.0 m`、penalty `-10.0`，`d_safe=4.0 m`；
- lifecycle：`episode_max_length=3000`；4v1 post-capture coverage window `500` steps；capture event 不因课程而提前 terminal，`capture_episode_ends_on_capture=false`，capture 后继续 canonical post-capture/recovery。若未来另做 capture-only diagnostic，必须另命名、另合同、另 gate，不能混入本提案结果。

### 3.3 Collision constraints（逐 stage 不变）

- `collision_semantics=synchronized_swept_v1`；必须按同步 swept trace 检查 action interval 内的 agent-agent、agent-target、agent-obstacle 和 boundary contact。
- reset 时 pursuer-pursuer center distance `>=15.0 m`，pursuer-target center distance 至少达到该 stage 的 `d_min`；不允许初始 contact。
- obstacle placement 必须满足现有 valid-position 的非重叠与 `4.0 m` safety clearance 约束，且所有实体中心在地图内；受控 ring sampler 若生成初始 contact，必须拒绝该样本并重采样，而不是改变 collision semantics。
- `enforce_hard_boundary=true`、`boundary_collision_death=true`；spawn edge margin `12.0 m`，不会用越界/软边界作为课程捷径。
- 评估同时报告 collision episode rate、boundary collision、agent-agent、agent-target、obstacle collision；stationary capture 不得掩盖 normal capture 失败。

## 4. 四个 exact numeric candidate stages

下表是**候选**，不是已注册配置。`d` 指每个 active pursuer 到 target 的初始 center distance；角度 `φ` 指以 target 为原点的 pursuer bearing，`θ` 指机器人 heading。

| stage | 预期 budget（environment decision steps） | pursuer / target / obstacle | map size | 初始 pursuer-target distance | angular distribution | target speed range | target visibility assumption | initial ring condition | horizon / reward |
|---|---:|---|---|---|---|---|---|---|---|
| I0 `RING_ANCHOR` | `100000`；25k 间隔；每点 20 argmax + 20 sample | `4 / 1 / 1` | `120.0×120.0 m` | 每个 `d_i∈[12.0,13.0] m` | `φ_i={0°,90°,180°,270°}+U[-5°,5°]`；随机整体 phase `U[0°,360°)`；`θ_p,θ_t iid U[0°,360°)` | reset `[0.0,0.0] m/s`；dynamic max `3.5 m/s` | 4/4 direct local target visible：center d 最大13，surface clearance 保证在20m sensing 内；无 global enemy | 4 active；`d_i>8.0`，所以 reset 不 capture；bearing gaps nominal `80°–100°`；角度几何已满足 `max_gap≤π`、`max_gap/min_gap≤3`，但必须先接近到8m内才可 capture | `3000`；完整 Final CR-MS/support/CE/PBRS/recovery，不改任何 reward |
| I1 `RING_WIDE` | `150000` continuation；25k 间隔；20+20 | `4 / 1 / 1` | `120.0×120.0 m` | 每个 `d_i∈[13.0,16.0] m` | `φ_i={0°,90°,180°,270°}+U[-7.5°,7.5°)`；随机整体 phase；`θ_p,θ_t iid U[0°,360°)` | reset `[0.0,0.0] m/s`；dynamic max `3.5 m/s` | 4/4 direct local target visible：center d≤16；无 global enemy | 4 active；`d_i>8.0`；bearing gaps `75°–105°`；无 reset capture、无 reset collision；不强制 obstacle-free line of sight | `3000`；同 I0 Final reward contract |
| I2 `RING_PARTIAL_VIS` | `200000` continuation；25k 间隔；20+20 | `4 / 1 / 1` | `120.0×120.0 m` | 2 个 direct lane：`d_i∈[17.5,19.0] m`；2 个 hidden lane：`d_i∈[23.0,26.0] m`；lane assignment 每 episode 随机 | nominal `φ_i={0°,90°,180°,270°}+U[-12.5°,12.5°)`；随机整体 phase；`θ_p,θ_t iid U[0°,360°)` | reset `[0.0,0.0] m/s`；dynamic max `3.5 m/s` | 恰好 2/4 direct、2/4 out of 20m local sensing at reset；support 只能通过 Final neighbor-visible semantics 获得；无 global enemy | 4 active；`d_i>8.0`；bearing gaps `65°–115°`；环骨架仍存在但 target visibility 不再全知；不得以 visibility quota 作为 Final 评估假设 | `3000`；同 I0 Final reward contract |
| I3 `FINAL_INIT` | `300000` continuation；25k 间隔；20+20 | `4 / 1 / 1` | `120.0×120.0 m` | canonical `map_random`：hard lower bound `d_i≥15.0 m`，不设 sampler upper clamp；在 `12m` spawn margin 的普通有效区域内报告几何 envelope `[15.0,135.7645] m`，该 upper 仅为报告上界，不是 Final rejection 条件 | pursuer/target positions independent `map_random`，bearing 不做 ring conditioning；`φ_i` 为 map/boundary 诱导分布，`U[0°,360°)` 只是无偏 null；`θ_p,θ_t iid U[0°,360°)` | reset `[0.0,0.0] m/s`；dynamic max `3.5 m/s` | 不设 direct/hidden quota；每个 pursuer 仅在 local surface clearance≤20m 时看到 target；无 global enemy；记录 reset visible count 分布 | 4 active；仅要求 reset 无 collision 且 `d_i≥15.0>8.0`，不要求 ring gap、ring occupancy 或初始 target visibility；这是 Final initialization，不是 ring task | `3000`；同 I0 Final reward contract；capture 后保留 Final post-capture/recovery |

I0–I2 的 ring 是 reset curriculum scaffolding，不是 capture success label；任何 reset 发生的 capture、contact 或 boundary event 都计为 contract failure。I3 才解除 ring geometry 与 visibility quota，作为 Final transfer 前的必要阶段。

### 4.1 每 stage 的 collision 细化

为避免“表中只写 count/map，执行时偷偷改变安全约束”，四个 stage 的 collision candidate 完整值均为：

```yaml
collision_semantics: synchronized_swept_v1
enforce_hard_boundary: true
boundary_collision_death: true
spawn_edge_margin: 12.0
pursuer_spawn_min_sep: 15.0
min_pursuer_evader_init_dis: 15.0  # I0/I1/I2 受上表 d 区间替代；I3 原样使用
boundary_proximity_distance: 4.0
boundary_proximity_penalty: -10.0
d_safe: 4.0
emergency_penalty: -10.0
collision_penalty: -160.0
```

I0–I2 的 ring placement 是有意放宽 Final lower bound 的初始化差异，用于提供可控的 `d` 分布；它不降低 pursuer-pursuer 或 obstacle clearance。I3 恢复 canonical `map_random` sampler。

## 5. Difficulty rationale 与预期学习作用

| transition | 只增加的难度 | 预期解决的问题 |
|---|---|---|
| I0→I1 | ring 半径从 `12–13` 扩到 `13–16`，角度 jitter 从 `±5°` 扩到 `±7.5°` | 先让 capture credit 有稳定但非终点的接近轨迹，再扩大径向/角向误差 |
| I1→I2 | 保留 4-agent ring，但引入恰好 2 direct / 2 hidden 的 local visibility | 让直接追击与 one-hop support 在同一 Final 信息合同下出现；不注入 global target |
| I2→I3 | 取消 ring bearing、取消 visibility quota，恢复 independent `map_random` | 检查策略是否学到 capture 机制，而非固定 cardinal slots、固定 4-visible pattern 或 target-centered shortcut |

人数、target 数、obstacle 数、地图尺寸、target max speed、AW9、collision、horizon 与 reward 全阶段固定。这样 I0→I3 是 initialization/visibility distribution curriculum，而不是同时改变 MDP 的多变量课程。

## 6. Promotion metrics 与 rollback gate

### 6.1 评估协议

- 每 `25,000` environment steps 保存 checkpoint 并评估；每个 checkpoint 使用固定、独立于训练 RNG 的 `20 argmax + 20 sample` episodes。
- 记录 `normal_capture_rate`、`stationary_capture_rate`、`capture_rate`、collision episode rate、boundary/agent-agent/agent-target/obstacle collision、2+ ring、3+ ring、capture steps；I2 额外按 direct/hidden lane 分层。
- I3 及最终 Final transfer 额外记录 post-capture CE、safe complete、capture→CE time 和完整 episode time；不能只报告 capture prefix。
- 所有 promotion 都要求“最后连续三个 evaluation checkpoints + stage terminal checkpoint”满足 gate；不允许用单个历史 best 替代稳定窗口。

### 6.2 各 stage promotion gate

| stage | promotion gate（每个条件都必须满足） | 失败处置 |
|---|---|---|
| I0 | 最后三点及 terminal：argmax normal capture `≥0.50`；sample normal capture `≥0.60`；collision `≤0.10`；2+ ring `≥0.80`；3+ ring `≥0.60`；mean normal capture steps `≤250` | 留在 I0；若连续三点不满足，回滚到 I0 最近一个通过点并不扩预算 |
| I1 | argmax normal capture `≥0.50`；sample normal capture `≥0.50`；collision `≤0.15`；2+ ring `≥0.75`；3+ ring `≥0.50`；mean steps `≤300` | 回滚 I0 promoted checkpoint；不跳过 I1 直接进入 I2 |
| I2 | argmax normal capture `≥0.40`；sample normal capture `≥0.50`；collision `≤0.20`；总 2+ ring `≥0.60`；hidden-lane normal capture `≥0.30`；mean steps `≤350` | 回滚 I1 promoted checkpoint；检查 visibility-stratified telemetry，不改 reward |
| I3 | argmax **normal** capture `≥0.50`；sample normal capture `≥0.50`；collision `≤0.20`；stationary capture 另报且不得替代 normal；最后三点与 terminal 均过 | 未通过则回滚 I2；禁止把 ring stage 的结果宣布为 Final capability |

### 6.3 通用 rollback gate

相对于最近一个 promoted checkpoint，任意 stage 触发以下任一条件即回滚：

1. normal capture 绝对下降 `>0.15`，或 collision episode rate 上升 `>0.10`，并连续两个 evaluation 仍未恢复；
2. 连续三个 evaluation 未达到本 stage promotion gate；
3. 出现 reset contract violation（初始 collision、初始 capture、错误 visibility、错误 horizon/reward hash、non-finite loss）一次即停止该 stage 并回滚；
4. 仅有 2+/3+ ring 增长、stationary capture 增长或 return 增长，但 normal capture 没有增长时，不得晋级，且不能通过调 reward 修复。

回滚对象必须是带有 optimizer/RNG/config/source manifest 的最近 promoted checkpoint。不得仅加载 actor、不得挑选 single best、不得在失败后隐式改变 seed 或 reward。

### 6.4 Final transfer gate（I3 之后的用户审查材料）

I3 通过只代表初始化课程在 controlled Final-Init 上稳定；它还不是 Final task 结论。下一步若获授权，必须在 canonical Final `map_random` full lifecycle 上用独立 fixed seeds 做 paired audit，至少满足：

- normal capture：argmax 与 sample 各 `≥0.50`；
- collision episode rate `≤0.20`；
- post-capture CE 与 safe complete 各 `≥0.30`，并报告完整 capture→CE/mission time；
- 全部 reward component、observation contract、collision semantics、horizon hash 与 canonical Final 一致；
- 不能用 I0/I1/I2 的 ring-only success 替代该 gate。

## 7. Shortcut / bias risks 与保护措施

1. **固定 cardinal ring shortcut**：I0–I2 使用四个 nominal bearings，策略可能记住 slot。每 episode 随机整体 phase，逐 pursuer jitter，且 I3 完全取消 ring conditioning；最终另做 arbitrary-angle held-out report。
2. **target always visible shortcut**：I0/I1 是 4/4 direct。I2 固定 direct/hidden lane 会引入 quota bias；lane assignment、pursuer identity、整体 phase 必须随机，I3 不设 visibility quota，并报告 visible-count histogram。
3. **stationary-target shortcut**：所有 reset speed 都是 canonical `0.0`，可能被误解为静止目标任务。target dynamic max 始终 `3.5`，normal/stationary capture 分开；不得以 stationary event 充当 Final normal-capture 证据。
4. **obstacle-free/ring-safe shortcut**：ring sampler 只拒绝初始 contact，不删除 Final obstacle sensing，也不把 line-of-sight 清空；I3 必须回到 map-random obstacle placement，并单独报告 obstacle collision。
5. **人数/形状 shortcut**：四个 stage 都是 `4/1/1`，observation padding 和网络形状不变；避免 agent count 变更掩盖 initialization effect。
6. **地图尺度 shortcut**：地图全程 `120×120`，避免 distance normalization 随 map size 改变；I3 的唯一分布变化是 reset geometry，不引入 map-size OOD。
7. **reward/replay shortcut**：不将 AC-CAP 当前 zero-coverage/PBRS capability reward 解释为 Final reward；本提案每 stage 使用同一 Final reward hash。任何 reward curriculum 必须另开独立 candidate，不能与本线联动。
8. **metric selection bias**：只看 capture 或只看 ring 会复现当前 AC-CAP 的 `RING_SIGNAL_ONLY` 风险；promotion 绑定 normal capture、collision、连续窗口和 I3 Final transfer。

## 8. 与 Final task 的关系 / diff

### 相同项

- `4 pursuers / 1 target / 1 obstacle / 120×120`；Final local VorAdj/VCT-LS、20m sensing、no global enemy；AW9；target max speed3.5、reset init speed0；hard boundary/swept collision；episode horizon3000；Final CR-MS/support/CE-PBRS/post-capture reward contract。
- capture detector 的 `8.0m / k=3 / max_gap=π / max_gap_ratio=3.0` 与 normal/stationary 分离完全相同。

### 有意 diff（仅 reset initialization）

- I0–I2 使用受控 target-centered ring geometry；I2 进一步使用 2 direct + 2 hidden 的 visibility stratification；这些分布不属于 Final map-random。
- I3 恢复 Final `map_random`、`d_i≥15.0m`、无 bearing/visibility quota；其 upper envelope `135.7645m` 仅是 120m 地图和 12m 普通 spawn margin 下的报告上界，不是新增 Final sampler clamp。

### 不允许的 diff

- 不得继承当前 AC-CAP 的 `legacy_end_step`、horizon1000、coverage/PBRS=0、`capture_episode_ends_on_capture=true` 作为“Final Capture”实现；
- 不得改成 historical legacy pure-capture reward、修改 `goal_reward`、提高 ring/capture bonus、删除 collision penalty、增加 target visibility 或 global enemy input；
- 不得把 I0–I2 的成功率直接与 B0 Final mixed formal 数字作同一任务 Gate；只能作为初始化课程的阶段性诊断。

## 9. Expected budget 与运行纪律

候选总训练预算为每条 line `100k + 150k + 200k + 300k = 750k environment decision steps`；stage continuation 从上一个 promoted checkpoint 继承，不能把四段重复算成四条独立 scratch 结论。每 stage 额外有 25k checkpoint/eval cadence、每 checkpoint `20 argmax + 20 sample`。

若 I3 通过且用户另行批准三 seed confirmation，预留 `3×300k=900k` Final-Init/Final-transfer confirmation steps；这不是本轮预算承诺，也不自动启动。总计因此分为：

- proposal line：`750k` env steps；
- optional post-review confirmation：`900k` env steps；
- 本轮实际执行：`0` env steps、`0` GPU formal run。

GPU/PID/tmux 只在用户批准并产生正式 launch 后登记；当前设计不得占用或干预 AC-CAP/AC-MIX、IQN-Z05/Z07 等现有进程。

## 10. Future independent reward-curriculum candidate（不实施）

奖励课程只能作为未来独立实验候选：冻结 I3 Final initialization、Final observation/action/physics/collision/horizon，另立 reward-ablation config，单独比较 ring shaping/terminal reward 的固定系数或 schedule，并使用同一 paired seeds。它不属于本提案、没有实现数字、没有启动条件，也不能与 initialization curriculum 同时改变。当前结论不支持现在启动 reward curriculum。

## 11. User review gate（必须由用户明确批准）

在任何代码、配置或 formal run 之前，用户需要明确审查并选择：

1. 是否批准 I0–I3 的 exact numeric table、`750k` continuation budget 和三点+terminal promotion/rollback gate；
2. 是否确认 canonical Final，而不是当前 AC-CAP capability-only 变体：`synchronized_swept_v1`、horizon `3000`、capture 不提前 terminal、coverage/PBRS/post-capture reward 保持开启；
3. 是否批准 I3 以 canonical `map_random`、无 ring/visibility quota 作为最终 transfer gate；
4. 是否批准之后才创建实现配置/initializer、跑 CPU contract smoke，并在另一个用户授权下安排 formal run。

建议用户回复固定短语：`APPROVE A3 CAPTURE CURRICULUM INITIALIZATION-ONLY`，或指出要修改的 stage/metric/budget。未收到明确批准前，本任务保持 design-only；本文件不代表批准、不代表 launch authorization，也不代表任何 formal result。

classification: CAPTURE_CURRICULUM_AWAITING_USER_APPROVAL
