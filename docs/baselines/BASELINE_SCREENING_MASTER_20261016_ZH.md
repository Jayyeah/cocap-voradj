# CoCap baseline candidate screening — 总览与推荐

筛选日期：2026-10-16  
仓库事实源：`Jayyeah/cocap-voradj`，中央 `main`，筛选时 HEAD=`a5814f49fa29d869cdc3fb8d8e0df4722aa11f00`。  
本轮只做候选筛选与文档台账；没有复现、训练、实现 baseline 或修改 CoCap 主代码。

逐项来源、机制和合同见[候选方法卡](BASELINE_CANDIDATE_CARDS_20261016_ZH.md)；可供程序读取的字段见 [`baseline_registry.json`](../../artifacts/baselines/baseline_registry.json)；检索记录见 [`search_log.md`](../../artifacts/baselines/search_log.md)。

## 1. 设计目标

CoCap 当前论文主张应围绕一条连续任务链：无人机在局部感知下扩散搜索；某个 agent 发现目标后，同策略/局部证据触发邻机支援与协作围捕；一个目标处理完成后，部分 agent 恢复搜索，团队继续找下一个目标。实验要回答的不只是“能否围住一个已知目标”，还要回答“能否发现、协同捕获、恢复搜索并提升同一时限内的累计任务效率”。

仓库 `main` 目前以 CR-MS + VCT-LS + CE 为主要研究线，并包含 4v1 → 8v2 → 12v3 的规模课程。VCT-LS 文档强调友方 Voronoi 邻接通信、敌人/障碍局部感知。以下合同是 baseline screening 的目标口径，正式冻结仍应以论文 benchmark contract 与 CoCap capture 事件定义为准。

## 2. 评价维度

候选按以下维度筛选，优先任务合同匹配，然后才比较方法新颖度和开源便利性：

1. **连续任务匹配**：是否同时覆盖空间搜索、局部发现、多人协同围捕、完成后恢复搜索和后续目标处理。
2. **信息公平性**：执行策略是否只能用局部可见目标与允许的通信证据；训练期 centralized critic 单独记录，不能泄漏到执行端。
3. **协作结构**：共享策略/CTDE/独立 actor；固定角色与否；显式 assignment 或 coalition；多人对一个目标。
4. **复现证据**：论文完整度、作者官方代码、依赖/模拟器、模型与配置、训练入口是否可用。
5. **解释价值**：候选是否与 CoCap 有明确机制差异，并能支撑公平、非刻意弱化的主表比较。

## 3. 建议先冻结的 baseline 合同

### 3.1 同一任务环境

- pursuer/target 数量、初始化分布、动力学、障碍、地图范围、episode 时长、失效/碰撞定义、捕获终止条件、随机 seeds 和评估 rollout 数全部相同。
- 采用 CoCap 的正式 capture/encirclement 事件定义和目标完成计数；禁止 baseline 使用更宽松的“单机触碰即捕获”后再与 CoCap 的几何/协作捕获率直接比较。
- 每个目标要求的最少协作人数应由 CoCap 的 benchmark contract 固定，所有方法一致；不足人数时允许继续搜索/招募，不给某 baseline 预先配足人手。

### 3.2 执行时信息

- 执行 actor 不能读取未被任何 agent 局部发现的目标坐标或隐藏目标数。
- 局部发现记录只能通过 benchmark 明文规定的友方通信边传播。任何全局 assignment、地图或全局 target list 都标记为 **privileged**，单列结果，不混入 information-matched 主表。
- CTDE 训练可用 centralized critic，但 rollout actor 不可获得 critic-only state。
- 对于论文原本读取全体目标状态的方法，可选择：保留原论文接口作为 privileged upper bound；或发布“受限观测适配版”并明确它不再是原始协议复现。

### 3.3 任务分配与多轮恢复

正式传统强 baseline 应包含可解释的分层组件：

1. **搜索层**：没有已知目标时执行确定性 coverage/spread；
2. **发现/报告层**：本地检测后，按同一通信图传播有界状态摘要；
3. **分配层**：用局部可知目标集合进行稳定 assignment/coalition formation。建议按到达时间或距离报价，并施加每目标最低人数约束；分配是分布式邻居协商还是中央调度，需明确写进结果标签；
4. **几何围捕层**：按 CoCap 捕获规则构造环绕槽位/围捕控制，沿用同一障碍和碰撞约束；
5. **释放层**：目标完成后释放 coalition，重新分派仍可见目标；无剩余已知目标时回到搜索层，直到回合结束。

信息匹配主基线建议采用邻居可达的分布式 assignment。若需要中央 task-allocation 作为合理强上界，应另列 `Central-Oracle-Allocation`，并标注其看到全局发现集合，不能和局部执行结果混报。

### 3.4 共同主指标

至少报告累计完成目标数/时间、全队完整捕获率、首次发现时间、相邻目标完成间隔、捕获后恢复扩散延迟、局部区域覆盖/漏检、碰撞/失效率，以及每轮的时间分解（search、聚集、encircle、recovery）。只报单目标捕获成功率会遗漏 CoCap 的连续效率主张。

## 4. 候选总表

| ID | 候选 | 类型/结构 | 多目标围捕 | 局部搜索与再搜索 | 官方代码 | 成本 | 结论 / 优先级 |
|---|---|---|---|---|---|---|---|
| RL-01 | TERL | RL / IQN，Transformer，shared distributed policy | 强：多目标、每目标至少 3 pursuer、逐目标围捕 | 不支持任务级搜索；任意距离可见所有未围捕目标 | **有，官方仓库** | 中 | **KEEP P0**；最强 RL 围捕候选，需受限观测版或单列 oracle |
| RL-02 | Role-based MADDPG | RL / MADDPG CTDE，固定 scout + tracking pursuer | 多目标 tracking；碰撞视作攻击，不是环绕 | **有探索/scout**；无捕获后重复 cycle | **未找到官方开源** | 高 | **MAYBE P1**；最贴近 scout-vs-pursuer 论文结构，复现风险高 |
| RL-03 | OPEN | RL / MAPPO，shared actor | 单 evader，多机接触式 capture | 局部目标探测；无任务级多目标搜索 | **有，官方仓库** | 高 | **MAYBE P2**；局部单目标控制参考 |
| RL-04 | DualCL | RL / homogeneous shared policy，dual curriculum | 单 evader capture | 训练环境探索，不是任务空间搜索 | **有，官方仓库** | 高 | **MAYBE P2**；不进入最小组合 |
| RL-05 | LIPPO | RL / independent PPO-style，异构速度/能耗 | 强：多目标，每目标分 3 hunters、三角围捕 | episode 开始全局 greedy 分派；无捕获后搜索 | **未找到官方开源** | 高 | **MAYBE P1**；新异构围捕备选 |
| RL-06 | PRTDA | Hybrid / airship–UAV role structured CTDE | 3D 多目标追捕 | 有 airship coverage/分派，非 CoCap 局部角色闭环 | **未找到官方开源** | 高 | **DROP 主表**；独立全局引导平台导致合同不同 |
| RL-07 | Ex-MADDPG | RL / 分布式 assignment component | 每目标至少 2 UAV；formation 用 Boids | 局部发现/通信和动态 assignment；无完整围捕 recovery | **未找到官方开源** | 高 | **MAYBE P2**；assignment 组件参考 |
| TR-01 | MK-WTA + OAPM | 传统 k-WTA assignment + 几何围捕/避障 | 强：多目标、每目标分 k pursuer | 无显式 search/recovery | **未找到官方开源** | 高 | **KEEP P0**；最强传统任务候选，信息假设待全文核查 |
| TR-02 | Potential-game multi-target encirclement | DBSCAN + potential-game assignment + guidance | 强：多目标分组围捕/跟踪 | 无显式 search；重复捕获未知 | **未找到官方开源** | 高 | **MAYBE P1**；待拿全文确认假设 |
| TR-03 | ACC leader-follower coalition | 发现后联盟形成 | 任务联盟有；未证明几何围捕 | 未知目标可发现；可重复形成联盟 | **未找到官方开源** | 高 | **MAYBE P2**；仅作为分配组件 |
| TR-04 | Distributed RMPC encirclement | 分布式几何控制/RMPC | 强：多人对单目标 | 无多目标分配/搜索 | **未找到官方开源** | 高 | **MAYBE P2**；低层 controller 组件 |
| TR-05 | local spread + greedy coalition + geometric surround | 规范性传统启发式 | 支持多目标、多人围捕 | **完整：搜索→协作→释放→搜索** | 项目内自建，无外部官方实现 | 低至中 | **KEEP P0**；建议作为辅助下界 |

## 5. Shortlist 与推荐优先级

### P0 — 建议进入复现 preflight 的三件套

1. **TERL / RL 强围捕参考**：其方法结构与多目标围捕最匹配，且有官方代码。preflight 要并行核查原始全目标信息版本和受限观测适配版；若只跑全目标信息结果，必须作为 privileged upper bound。
2. **MK-WTA + OAPM / 传统强候选**：论文已覆盖 k 人/目标与几何环绕，适合作为不依赖学习策略的强 task-allocation/control 对照。先取到完整方法细节并确认目标坐标、通信、分配集中化，再决定复现合同。
3. **local spread + greedy coalition + geometric surround / 简单启发式**：这是透明的 project-level baseline，不是文献复现。按同一局部信息合同实现，提供累计任务效率的辅助下界。

因此，**建议正式论文目标采用 TERL + 一个传统强 baseline + 一个简单启发式 baseline**。具体正式实现应由 preflight 的合同/依赖结果决定，不在本次筛选中锁定唯一实现。

### P1/P2 — 保留但不挤进最小主表

- **Role-based MADDPG**：回答固定 scout/追踪角色能否帮助搜索—围捕切换；结构相关性高，但没有论文官方实现，且原任务是追踪/接触攻击。
- **LIPPO**：2026 年多目标异构围捕候选，几何判据有参考价值；一次性全局指派、独立 learner 和无官方代码降低优先级。
- **Potential-game multi-target encirclement**：若拿到全文后发现有可直接实现的 assignment 和多人围捕规则，可替换或补充 MK-WTA。
- **OPEN / DualCL**：适合作为单目标捕获或局部感知子任务候选，不能替代多目标连续任务 baseline。

## 6. 研究问题的明确回答

### 1. TERL 之外是否存在合适的固定角色/异构角色 RL？

**存在。**最直接的是 ICRA 2023 Role-based MADDPG：它用 scout 与 target-tracking pursuer 两种奖励角色，scout 通过 Voronoi reward 探索，追踪者协同处理已发现目标；它具备 CoCap 所需的探索—利用张力，但任务终点是 tracking/碰撞攻击，不是几何围捕后重复搜索。LIPPO (2026) 和 PRTDA (2026) 是近期 heterogeneity/role-structured 候选：LIPPO 是异构 UAV 多目标几何围捕但不分 search/pursuer 角色；PRTDA 用 airship 做全局覆盖和任务引导，平台合同不同。

### 2. 固定角色/异构角色候选里哪些有可用开源实现？

本轮**没有找到上述三类方法（Role-based MADDPG、LIPPO、PRTDA）的官方代码仓库**。DualCL 和 OPEN 有作者官方实现，但它们是共享策略/同质 UAV 追捕，不能作为“固定 scout 角色”实现。TERL 官方仓库可用，但角色不固定，且全体未围捕目标位置对 pursuer 可见。代码状态按 2026-10-16 公网检索记录，后续 preflight 前需复查。

### 3. 传统分层 baseline 的合理合同是什么？

采用 `local spread/coverage → local detection and neighbor target report → bounded-capacity coalition/task allocation → same geometric encirclement rule → coalition release → resume spread`。主表只允许局部发现集和允许的邻居消息；每目标最低人数、捕获判据、障碍/动力学、时限均与 CoCap 相同。中央全目标/全队 allocation 可另作 privileged upper bound，不得混入 information-matched 结果。

### 4. 正式论文是否采用 TERL + 一个传统强 baseline + 一个简单启发式？

**建议采用这个三层组合。**TERL 覆盖强 RL 多目标 encirclement，MK-WTA + OAPM 是当前最直接的传统强候选，局部 spread + greedy coalition + 几何围捕提供可解释下界。正式主表前需把 TERL 的全局目标可见输入处理清楚，并核实 MK-WTA 的距离矩阵/通信/assignment 假设。

### 5. 现在是否立即开始复现？

**不建议直接启动正式复现训练。建议进入下一步复现 preflight。**先冻结 benchmark contract，再检查 TERL 原始与受限观测接口、MK-WTA 全文与分配信息假设、任务事件映射、动作/动力学适配和启发式规则。preflight 只做可用性与合同核查，不启动正式训练。

## 7. 下一步建议

1. 把上面 3.1–3.4 合同映射到论文正式测试阶段（建议先对齐一个小规模多目标阶段，再做规模扩展）。
2. preflight 前获取 TERL 源码当前 commit、训练配置、许可与 checkpoint 可用性；确认远程目标状态遮罩是否能关闭且不引入目标身份泄露。
3. 对 MK-WTA 获取完整论文，核清 target–pursuer 距离矩阵来源、是否通过统一摄像/中心节点获得坐标、k 的设定、capture event、代码或补充材料。
4. 同步写下 heuristic 的 tie-break、通信延迟、每目标最低人数、角度槽位和搜索覆盖指标，避免看完 CoCap 结果后再调规则。
5. 在这些检查后锁定实际代码分支与复现计划；本文件不授权也不包含训练启动。

