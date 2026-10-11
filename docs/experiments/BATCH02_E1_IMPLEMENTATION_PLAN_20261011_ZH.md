# E1代码实施计划

目标：以成功IQN-CoCap为oracle，将 `TERL MarineEnv` 接口桥接到 `CoCap VorAdj Forward-Final V2`。当前状态 **PLAN_READY_WAITING_AUTHORIZATION**：已完成代码定位和8个参考transition fixture，尚未编写生产bridge或开启Batch02 ARM训练。

## 实施边界

E1先交付可独立实例化的环境adapter：reset、step、AW9、active masks、terminated/truncated、未重置next state、snapshot/restore和完整任务metadata。它调用现有成功CoCap环境，先不连接PPO optimizer。

CoCap环境本来就同时产出task-native observation/reward/event；E1原样保留这些输出用于oracle比较。R2/N2分别负责learner的reward/observation消费与验证，不在E1里发明“临时TERL-like CoCap reward”或伪造19-slot CoCap输入。E1本身包含physics/current/APF/collision/boundary/spawn/terminal合同切换；不能从一次E1结果归因其中某一轴。

锁定的Batch01 worktree、102-file Core源码和六条运行不修改。后续新实现放新Batch02 worktree；源起点为当前V3r2 candidate，成功IQN相同的env代码复用。`default.yaml`是V2入口声明，不是可直接训练的完整任务YAML；必须经已验证IQN `resolved→scene_config→VorAdjEnv` 链得到实际profile，同时保持 `forward_final_v2` 的NormSense入口语义。不得从失败MAPPO PureCapture config补齐缺失字段。

## 代码diff map（已有文件已由搜索/diff确认）

| 旧模块/函数（实际存在） | 新模块/函数（实际存在） | 修改原因/合同类别 |
|---|---|---|
| `vendor/terl/environment/env.py:MarineEnv`；`src/terl_mappo/native.py:NativeStage1` | `src/cocap_voradj/envs/voronoi_adjacency.py:VorAdjEnv`；`training/forward_final_v2.py`；成功IQN `tools/iqn_token_matched_20260919.py:resolved/make_env` | E1环境provider切换，恢复成功task definition，避免失败runner默认 |
| `vendor/terl/environment/env.py:update_rob_state/get_current_velocity`；`robots/robot.py:update_state` | `envs/base.py:_move_robot`；`dynamics/robot.py:update_state` | 恢复零current、.05×10的AW9物理；不重新编写积分器 |
| `vendor/terl/thirdparty/APF.py:ApfAgent`；`MarineEnv.get_evaders_observation` | `control/apf.py:ApfAgent`；`envs/base.py:get_evader_observations_for_apf` | APF-v2与theta/speed self解释配对，多个E逐个维护agent |
| `vendor/terl/robots/pursuer.py:perception_output` 调用 `robots/robot.py:check_collision` | `envs/base.py:_begin_collision_step/_trace_pair_contact/_trace_obstacle_contact/_refresh_collisions` | 同步swept collision，记录类型/死亡；不是修改PPO |
| native越界处罚/未硬终止 | `envs/base.py:_clip_and_kill_boundary/_boundary_proximity_penalties` | 成功硬边界、-160死亡、-10 proximity；task definition |
| `MarineEnv.reset` 与native trainer生命周期 | `CoCapEnv.reset`、`VorAdjEnv.reset/step`、post-capture counters | 不同spawn分布、capture→recovery→CE、30步hold/window、3000horizon；保留RNG/reset语义 |
| `src/terl_mappo/native.py:pack_local/empty_observation` 的固定self4/19槽 | `VorAdjEnv._pack_agent_obs/get_policy_observations`；N2另实现contract-aware packing | 9/22-slot、mask与None shape；旧empty_observation不可用来填充CoCap行 |
| `arms/r1.py:RewardBridge/ring_target` | `envs/base.py:_ring_ms_target/_ring_importance_ms_reward`；`VorAdjEnv.step`原reward components | R2恢复原CR-MS/support/CE/safety与分配，不把R1重写提升为task定义 |
| `src/terl_mappo/native.py:central_state`，T1固定current/scale central | Batch02 adapter明确暴露training-only central state；复用 `src/cocap_voradj/models/small_step_ac.py:CentralValueNetwork` 接口 | **MAPPO adaptation**：actor只看CoCap local obs，V可以训练时读取global geometry/time/phase；无action输入；删掉失效current语义/新尺寸fresh V须记录 |
| Batch01 `arms/runner.py` 固定native packing/lifecycle调用 | Batch02 runner通过adapter接口收集真实task transition | **MAPPO adaptation**：on-policy rollout、GAE、active loss、ValueNorm、pre-reset bootstrap；不热改Batch01共同源码 |
| native `evaluate.py` 的capture终止指标；Core旧token evaluator汇总 | `tools/run_forward_final_bridge_20260908.py:run_episode`、`evaluation/mission_events.py`；成功IQN工作树的 `tools/evaluate_iqn_z05_independent_20260923.py` 和token evaluator删失汇总 | eval覆盖capture prefix、recovery、CE、normal/stationary、safe/collision和删失；MAPPO argmax/sample另标行为。独立IQN evaluator在Core不存在，成功源码已归档；不能误认Core旧工具整体相同 |

拟新增文件（还不存在，不是从文档猜来的旧路径）：`src/terl_mappo/batch02/cocap_env.py`、`observation.py`、`contracts.py`，及对应 `configs/experiments/terl_mappo_batch02_20261011/` profile。先实现adapter，R2/N2/V1再各建独立consumer/配置delta。若发现现有接口足够则复用，不为流程完整额外造框架。

## 顺序与最小验证

1. **当前等待下一步授权，E1生产实施尚未获准。** 收到用户对E1实施范围及首个成功参考profile的明确授权后，才进入第2步。[科学选择](BATCH02_SCIENTIFIC_CHOICES_20261011_ZH.md)中的N2路径在其consumer实施前确认；初始化/PPO/recovery训练分布留到后续训练设计，不作为E1离线工作的额外前置。原BASE独立QA状态继续保留；无需先等待T1跨规模正式成功，不新增性能门槛或训练预算。
2. 新worktree实现环境adapter，复用成功IQN配置和live消费者。先打印实际physics/current/APF/action/sensor/phase/terminal值，和本轮resolved参考比对。checkpoint初始化此时不参与。
3. 将同一fixture state、同一显式P/E action分别送入CoCap原始provider和新adapter；对observation、每agent reward/components、done/terminated/truncated、未重置next state逐项比较。native差异以本轮8个oracle为预期，adapter对CoCap需一致，不能把native≠CoCap当bug。
4. 补生命周期fixture：同一target捕获event只有一次；capture transition存pre_capture phase；recovery CE hold29→30；post-window499→500；timeout与death同step时terminal优先；reset后K10/Z/CE/PBRS/event counters清零；snapshot恢复RNG和同一next transition。补APF危险clearance/closing输入，当前8组动作相同不能证明APF语义完整。
5. 一次有限CPU集成检查后提交原始证据给独立reviewer按同state输入复验。Commander自行运行的fixture仍写 `independent_validation=false`。E1只有通过独立bridge验证才记PASS。验证不要求长训练，也不要求先在新任务学出高score。

正式训练仍须既有同candidate独立QA状态/授权得到解决，不能追认Batch01 PROVISIONAL；这些是已存在事实，不是本轮新增门禁。当前阶段没有授权启动任何Batch02训练。

## E1之后

E1共同环境代码串行发布并固定一个版本。随后 **R2、N2、V1可以并行实现**，三个ARM都引用相同环境代码。公共环境修复集中提交，所有ARM同步该commit并重放受影响fixture；独立ARM修改仅发生在各自consumer/配置。

R2恢复完整Capture reward分配及事件语义，不改环系数；N2恢复成功CoCap observation及actor适配，不加额外信息；V1实现无E的Pure Coverage正控，保留CE/PBRS/hold/safety。此时parallel指实现/离线验证，训练仍错峰且等明确预算。要做单变量训练对照时，先固定共同baseline、初始化和learner消费profile，不能拿不同任务/不同seed/best checkpoint直接互比。

M1在公共环境与三个consumer明确后集成完整Mixed/recovery/CE；M2以成功IQN生命周期为目标审计Z/evidence/释放，不从失败MAPPO的“capture即结束”逻辑扩建。Evidence、Persistent、local-friendly V3、global oracle和continuous目标动作均不自动启动。

## 两轮修复停止规则

同一问题两轮实际修复仍失败，或判断属于环境设计问题：停止该问题的继续尝试，保存 `minimal_repro`（state/config/action/source）、当前预期/实际、影响节点、修复轮数和可继续的分支。

公共环境/transition错误影响E1及依赖它的训练；仍可推进R1既有日志诊断、IQN合同/source恢复、N2的静态shape/type映射、R2组件账本和eval分析。仅N2 packing故障不阻止E1物理验证和R2组件验证；仅R2消费故障不阻止N2 shape和V1正控准备。不得无限修复、换reward或悄悄加预算来掩盖问题。
