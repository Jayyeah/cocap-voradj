# TERL vs CoCap environment contract

本轮以成功 IQN 为主要参考，恢复其代码、resolved config和既有评估原件。历史MAPPO失败版本仅用于故障背景，不作为任务定义。

## 成功参考的两层来源

**CoCap原始成功合同**：Final CR-MS/VCT-LS/CE，B0 Final IQN actor SHA `2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89`；`docs/B0_FINAL_IQN_REFERENCE_20260917_ZH.md` 保存已验证的roleful原合同。Forward-Final matched100 的mixed capture=1、CE/safe=.99，coverage CE/safe=.99；该证据是R20 sensing，不能称为NormSense V2或Z05成功。

**目标V2成功实现**：`evaluation/iqn-z05-independent-20260923` 工作树，HEAD `1a9da054ecb5652726005d00d08ab05a2188dd8e`；配置 `configs/experiments/iqn_z_unified_decay_curriculum_20260919/z05_stage1_4p1e1obs_2m.yaml`，实际 `tools.iqn_token_matched_20260919.resolved()` 开启V2。成功Stage1 checkpoint1300k SHA `d35b7984d8ca03fa7b536811794578ff402e9b8ad48589e3318f2cb1077d702b`。

独立Stage1报告 `artifacts/2026-09-23_iqn_z05_independent/stage1_native/report.json`：coverage strictCE=20/20、capture normal=20/20、mixed capture=20/20、mixed safe/recovery=18/20、各场景collision=0。Stage2 mixed safe=.20、collision=.25；2026-10-06 Stage3独立新域报告mixed capture=.95、safe=.75、collision=.05。因此以Stage1作为本轮首个桥接参考，不声称Z05全课程所有规模均已解决。

成功IQN和当前迁移Core的 `envs/base.py`、`envs/voronoi_adjacency.py`、`envs/density_sensing.py`、`dynamics/robot.py`、`control/apf.py` 五个核心文件按字节一致，[source hash证据](../../artifacts/2026-10-11_cocap_mappo_batch02/iqn_core_source_comparison.json)。本轮恢复的[完整resolved config](../../artifacts/2026-10-11_cocap_mappo_batch02/iqn_z05_reference_resolved.json)来自成功IQN配置，未使用失败Pure-Capture runner覆盖。

[恢复清单](../../artifacts/2026-10-11_cocap_mappo_batch02/iqn_reference_inventory.json)核验了四层extends配置一致、5个既有IQN checkpoint hash，以及resolved hash `6955ecf28383f7e146a0ee485ef3e33cb1d1782b197a51ceadd2ef93968d3b5a`与成功Stage1原报告完全匹配。[Stage1成功报告原件](../../artifacts/2026-10-11_cocap_mappo_batch02/iqn_z05_stage1_success_report.json)按字节归档；其它规模的既有报告摘要/hash保留在清单中。

工具整体并非全部相同：Core的`iqn_token_matched_20260919.py`缺少成功IQN后加的失败/删失时间汇总，独立IQN evaluator也尚未复制进Core。已保存[实际diff](../../artifacts/2026-10-11_cocap_mappo_batch02/diffs/iqn_reference_vs_core_evaluator.patch)与[成功参考源码](../../artifacts/2026-10-11_cocap_mappo_batch02/reference_sources/)。本轮fixture用到的`resolved/scene_config/make_env`三个函数AST一致，不受该汇总差异影响；后续eval迁移必须恢复成功版本的删失统计，不能用Core旧汇总替代。

## Environment差异表

| 轴 | TERL原始合同 | 当前TERL-MAPPO实际 | 成功IQN-CoCap V2目标 | 分类/迁移含义 |
|---|---|---|---|---|
| physics | AW9，dt=.05、N=10、decision=.5s；P vmax3/E3.5；drag=.4/vmax；先按当前velocity平移，再更新speed/theta | T0/P1/R1/N1同原生；T1改变实体数量；C0仅将AW输入改为连续物理命令 | AW9相同范围/索引，单步积分原语一致；env的运动、碰撞/死亡调用不同 | task physics；不因为都叫AW9就推定环境等价 |
| current | MarineEnv每子步`get_current_velocity(x,y)`，Stage1 4 vortex，Stage2 6，Stage3 8；current叠加seabed velocity | 当前仍使用海流；critic包含current信息 | `_move_robot()`的discrete AW路径每子步传`np.zeros(2)`，没有MarineEnv vortex动力学 | 成功任务本身是零海流；保留TERL海流将是尚未验证的新任务，不自动引入 |
| APF | vendored `thirdparty/APF.py` legacy常量和输入解释 | 原生APF，对每个活跃evader调度；T1维护逐evader APF | `control/apf.py` `v2_fixed`，self=`theta_speed`，min_clearance=.2、repulsion5000、mass500、force_range15、边界/动态closing处理 | 恢复成功APF及其59维观测格式；不是仅更换import路径 |
| collision | perception刷新时检查decision末端圆盘；可能继承collision flag | wrapper实现原trainer死亡/重置；C0仍末端检查 | `synchronized_swept_v1`：记录同步子步轨迹并检查连续接触；碰撞死亡和event类型 | task definition；native端点漏检在fixture中单列 |
| boundary | 越界处罚，env宽高120；原生不按CoCap硬边界终止 | 当前保持native，boundary penalty没有被重写成死亡 | hard boundary=true、collision death=true，按robot半径触边；boundary proximity4m罚-10；collision/death罚-160 | 成功resolved值，不能继承失败调试中的soft boundary/弱惩罚 |
| spawn | 原生生成E→P→cores/obs；E边距15、P边距2；P-E最小间距10/13/15，初始状态随机 | T0同Stage1；T1同原native对应stage | mixed map_random；P/E separation15、edge margin12、init_speed0；coverage inner_random_cluster半径10；4P1E1O→8P2E2O→12P3E3O | task distribution；同seed不保证跨env同初态，fixture必须显式注入物理state |
| terminal | trainer在所有E捕获、active<3时结束；episode判限实际3001 | wrapper分开terminated/truncated并保存pre-reset state；T1碰撞后允许剩3机继续 | Stage1 min_active4；horizon3000；捕获E后mixed继续recovery/CE；CE rms≤.05、max≤.10保持30步；postcapture window500（8/12机600/700） | CoCap生命周期目标；单任务capture terminal只用于capture scene，不能用于Mixed |
| capture | ≥3几何环，8m，maxgapπ、gapratio3；原trainer全捕获立即episode end | 同原；ring2/3仅诊断 | 原几何捕获 + 已存在stationary fallback：speed≤.2、至少2机、保持10decision；capture type区分normal/stationary/collision loss | 完整事件合同；不得删除已验证fallback或把碰撞E算capture |

源码依据：`vendor/terl/environment/env.py`、`vendor/terl/robots/robot.py`、`vendor/terl/robots/pursuer.py`、`vendor/terl/thirdparty/APF.py`；`src/terl_mappo/native.py`；CoCap `src/cocap_voradj/envs/{base,voronoi_adjacency,density_sensing}.py`、`dynamics/robot.py`、`control/apf.py`。具体旧→新函数映射见[E1计划](BATCH02_E1_IMPLEMENTATION_PLAN_20261011_ZH.md)。

## Observation与action合同

TERL/native actor：self4、friend5×7、enemy8×7、obstacle5×5、19 slots。敌人没有距离过滤，friend/obstacle走native sensor；N1没有改变信息合同。

Final原始roleful合同：self9/friend7末列为effective is_pursuing，K10 release；friend role_then_physical排序；IQN late fusion含role embedding。V2成功Z05：self9/friend7末列替换为z，include_is_pursuing=false、无role late fusion，friend physical_only；enemy8×7/obstacle5×5，22 slots。self包括robot-frame速度、可见障碍最小clearance、最近边界向量/OOB、√N归一化CE centroid向量，以及role或z。enemy仅onboard直接探测，不广播全局敌人坐标。

V2统一enemy/obstacle surface clearance `R=.8715√(A_eff/N)`，A_eff来自60×60 Final free_mask_projected，floor20m。4/8/12机参考R≈52.22/36.88/30.07m，随静态障碍mask变化，不按活跃数或reward在线修改。friendly拓扑仍使用historical global-friendly VorAdj；地图已知静态free mask可影响centroid，不能将其宣传为全链路local sensing。

Z05每decision一次：`z_next=max(direct,.5*z_prev,.5*max_neighbor(z_prev))`，小于.10硬清零；读immutable previous state，同步写入。Z、来源/lineage/update count、K10 counters/reset必须完整保留；query observation不能重复更新Z。是否N2直接恢复Z05或先roleful、M2再恢复Z，属于[待确认科学选择](BATCH02_SCIENTIFIC_CHOICES_20261011_ZH.md)，本轮没有替你选择。

成功动作：a-major AW9，a={-.4,0,.4}，w={-π/6,0,π/6}。C0 continuous AW是 **MAPPO adaptation** 独立诊断，不自动改变目标CoCap动作定义。IQN使用固定32 midpoint quantiles的expected-Q greedy；MAPPO是categorical argmax/sample，两种行为不能假设等价。

## Reward、lifecycle和evaluation恢复

| 内容 | CoCap原始成功语义 | MAPPO adaptation边界 | 尚未验证设计/未决项 |
|---|---|---|---|
| Capture reward | 原`base.py:_ring_ms_target/_ring_importance_ms_reward`，cartesian_disk occupancy；coef2、clip3、ring preferred8/outer10.5、局部direct targets | 仅将原向量reward和pre-action role metadata接入PPO/GAE | R1角度候选排除的几何重写不得带入R2；不新增系数 |
| Support | one-hop neighbor_visible target、approach_only；capture/coverage=.5/.5；role由K10 effective pre-action state决定 | metadata进入diagnostics；不得用critic global coords重构actor目标 | 失败PureCapture曾用1/0且关闭CE，与成功Final不同 |
| Coverage | CE reward scale10、PBRS κ1、phase_and_all_terminal reset；真实phase/terminal校正；严格CE与CV宽松指标分开 | terminal V=0、truncation保留物理next V、pre-reset state、active masks | reward clock承接选择待定；不得固定失败MAPPO的2M clock或关CE |
| Safety/events | collision-160、emergency/boundary proximity-10、几何goal120及实际分配/事件逻辑，captured与collision-lost分离 | episodic统计分别保存normal、stationary、collision、删失 | 不通过调低惩罚改善可训练性 |
| Lifecycle | capture action的stored phase仍为pre_capture；随后post_capture recovery；K10和Z释放、CE hold/window、全部reset状态 | PPO rollout允许跨episode；GAE对每agent死亡/任务terminal/time-limit正确处理 | 不以纯capture立即reset替代M1/M2全任务 |
| IQN学习分布 | mixed/coverage/recovery，capture snapshot pool1000、captured-state ratio.75，非capture内map_random ratio.5；replay各类64/16/32/16 | PPO on-policy不继承off-policy replay；恢复reset分布需清楚标为算法适配 | stream reset采样是否照成功IQN复用snapshot pool，须明确，不把replay批比改写为任务概率 |
| Eval | 完整horizon；coverage/capture/mixed分别报告；normal/stationary/safe/recovery/CE/碰撞、成功局time和失败删失；midpoint32固定推理；末点与selection-selected分开 | MAPPO argmax/sample同物理seed、sample RNG独立；盲final保持 | 不能以best screen替代正式评估；已有V2正控不保证新MAPPO也会成功 |

reward clock原消费者是 `trainer._set_coverage_ce_control_weights()`：IQN训练global_step在200k将speed_weight从0改为.0005。独立IQN evaluator按resolved scene实例化，读取初始weight0；报告中的task成功指标和训练时reward不是同一数值流。E1参考fixture固定新实例化clock0，本轮没有训练，因此不宣称验证200k切换。R2/V1/M1必须显式记录训练clock与evalclock，避免偷偷继承历史MAPPO重置偏移。

## 已执行transition fixture

8个CPU reference fixture均通过预先由源码推导的差异检查；[完整输入/observation/reward/done/next state](../../artifacts/2026-10-11_cocap_mappo_batch02/reference_transitions.json)，[可重放工具](../../tools/batch02_reference_transitions.py)。它调用真实native Stage2 4P1E1O与成功IQN Stage1 4P1E1O，显式注入相同robot/obstacle状态；避免3P/4P规模混淆。Native与当前MAPPO的parity另使用其真实native APF动作。

| fixture | 预期与实际 |
|---|---|
| ordinary | 相同AW命令下原海流造成最大位置差.08053m；reward/observation按各合同不同，两者未terminal |
| zero_current_motion | fixture中显式空current field，零海流运动位置完全一致；reward仍不同 |
| far_enemy_visibility | native远敌token仍可见，目标V2直接sensor不可见 |
| boundary | native越界存活/处罚；CoCap触边死亡，active<4导致任务terminal |
| swept_contact | native末端无collision；目标轨迹中途接触obstacle记录collision，episode terminal；末端位置可相同 |
| capture_to_recovery | native trainer wrapper结束；CoCap所有E被捕获但mixed dones=false，开始recovery |
| horizon | 从2999步transition：native wrapper继续，CoCap第3000步truncated |
| active_loss | native剩3机继续；CoCap Stage1要求4机，全部done |

零current只在指定diagnostic fixture中注入，并记录在输入；没有修改任何正式实验配置、海流实现或新增ARM。state包括位置、朝向、speed、velocity、active/collision、障碍、current cores、clock；观测getter的native collision副作用单列，fixture修正了一个初始已重叠的注入场景，改成非重叠初态在step内发生接触。APF在这组初态选择相同离散命令，不能因此声称legacy/v2 APF等价。

分类：`COMMANDER_OFFLINE_REFERENCE_DIAGNOSTIC`。**E1 bridge尚未实现、E1_PASS=false、independent_validation=false**。这组fixture建立预期差异和未来桥接回归oracle；不是工程self-test替代独立验证，也不是MAPPO-CoCap可训练结论。
