# CoCap Method / Narrative 理论可行性审查 v1

> 文档性质：理论可行性分析、事实对照与供审阅的候选设计；不是下一阶段实施指令。
>
> 冻结日期：2026-09-16（Asia/Shanghai）
>
> CoCap 事实基线：分支 **experiment/density-normalized-sensing-v2-20260915**，文档起草时 HEAD **e756fb0e13d60f48ff631539cddb6c98d44ccc36**。
>
> Literature map 事实基线：**main**，HEAD **051fec3495dae40b0573c4802725767be73809c2**。
>
> 审阅纪律：本文中的 M0/M1/M2、公式和实验矩阵都必须先由项目负责人充分理解、质疑和批准，才可能转化为实现任务。本文不授权修改当前训练合同、奖励、观测、模型、环境或正在运行的实验。

---

## 0. 为什么单独写这份文档

第一阶段文献地图已经把可能的 CoCap 论文主线收窄到一个较具体的交叉点：

**局部/间歇目标信息 + 联盟可捕获性 + 覆盖—捕获—释放—恢复生命周期。**

但“文献里存在空隙”不等于“项目现在就应增加一个新网络”，也不等于“当前代码已经实现了该方法”。本文的目标是把三类内容分开：

1. 文献真正支持的结论；
2. 当前 CoCap 代码和实验真正具备的语义；
3. 从现状到候选论文方法之间，最小且可验证的理论改动是什么。

这样可以避免两种相反错误：一是把现有 reward-driven role switching 过度叙述成 capturability-aware allocation；二是为了追求方法感而过早叠加 belief network、额外 Transformer、在线 HJI、Sinkhorn、MoE、PFSP 或 CBF，破坏已经困难地建立起来的 capture 学习信号。

---

## 1. 证据范围与可信度

### 1.1 本次完整读取的 literature-map 文件

- **LITERATURE_MAP.md**
- **READING_LEDGER.md**
- **notes/synthesis/MASTER-Cocap-Novelty-Method-Design-Memo-v1.md**
- **notes/synthesis/R-PE-01.md**
- **notes/synthesis/R-LM-01.md**
- **notes/synthesis/R-LIFE-01.md**
- **notes/synthesis/R-LM-02.md**

本文不会把聊天记忆当成证据。文献结论以 literature-map 的上述 commit 和文件为准。

### 1.2 本次核对的 CoCap 事实源

- 最新 experiment/root-cause/execution ledgers；
- single-task 与 Pure-Capture 合同；
- NormSense V2 合同、runtime config 与 active run artifacts；
- 环境中的 sensing、observation、role/support、reward routing、capture、release、post-capture recovery 与 termination；
- 当前 PPO actor/central critic；
- 历史 IQN gate 代码；
- 最新已完成 checkpoint evaluation 与运行时 progress/failure artifacts。

关键代码入口包括：

- **src/cocap_voradj/envs/voronoi_adjacency.py**
  - target visibility、局部 observation；
  - capture/support/coverage role 判定；
  - reward routing；
  - capture 后 target deactivation、release delay 与 post-capture recovery。
- **src/cocap_voradj/envs/base.py**
  - loose capture、stationary capture；
  - ring/geometry shaping。
- **src/cocap_voradj/models/small_step_ac.py**
  - 当前 PPO actor 与 central critic。
- **src/cocap_voradj/models/iqn.py**
  - 历史 gate head；它不是当前 PPO active path。

### 1.3 证据等级

本文用四个词避免把不同强度的判断混在一起：

- **代码事实**：由当前实现或合同直接确认；
- **实验事实**：由 artifact 直接确认，但只对给定 checkpoint、seed 和 evaluator 有效；
- **文献归纳**：由已封闭的第一阶段检索分支支持，不是全世界不存在反例的证明；
- **设计假设**：尚未实现、尚未验证，只能作为待审候选。

---

## 2. 一页式结论

### 2.1 当前 CoCap 最准确的描述

当前系统已经有一个很有价值的骨架：

- 同质、共享策略的 UAV pool；
- agent 可随局部条件在 coverage、support、capture 之间切换；
- actor 侧目标信息不是全局广播；
- capture 有终止事件；
- FullMix 有 capture 后 release 与 coverage recovery window；
- coverage 与 capture reward 已在 support 状态中发生混合。

但其当前语义更准确地说是：

> **局部规则触发、奖励驱动的行为切换**

而不是：

> **基于联盟可捕获性与覆盖机会成本的、语义明确的资源分配。**

### 2.2 文献地图支持的候选交叉点

最值得继续审阅的闭环是：

\[
\text{persistent coverage by one fungible swarm}
\rightarrow
\text{local/intermittent target information}
\rightarrow
\text{coalition capture-feasibility estimation}
\rightarrow
\text{dynamic support recruitment}
\rightarrow
\text{target-wise resource allocation}
\rightarrow
\text{terminal capture}
\rightarrow
\text{release}
\rightarrow
\text{same-agent coverage recovery}
\rightarrow
\text{repeated arrivals}.
\]

中文含义是：

> 同一批可互换的无人机平时持续覆盖；目标仅被局部、间歇地发现；系统估计“当前这组 UAV 对该目标是否足够可捕”；仅在新增支援带来的捕获收益大于覆盖损失时借调 UAV；完成真正的终端捕获后释放联盟；原来的 UAV 回到覆盖任务；下一次目标到达时重复这一过程。

### 2.3 最重要的否定性结论

- 不应预设必须新增独立的 \(C(S,j)\) 大网络。
- 当前 gate、Q/value、ring reward、neighbor count 或 distance utility 都不能仅凭“相关”就直接称为 capturability。
- Pure-Capture 是训练阶段分解，不是论文最终问题定义。
- 当前 4v1 FullMix 尚不能证明 multi-target resource allocation。
- capture ring、2+/3+ window、no-escape 和 terminal capture 必须严格区分。
- first-stage map 没找到完全相同的闭环，只能支持“在本次封闭检索范围内未见完整组合”，不能支持泛化的 “first”。

### 2.4 候选方法的性价比排序

在理论上，最有潜力的是 **M2-lite**：

\[
U_i(S,j)=\Delta_i(S,j)-\lambda\,\mathrm{CoverageCost}(i),
\]

其中 \(M2\text{-lite}\) 的 “lite” 表示：

- \(C\) 先用现有几何与动力学构成的可解释 proxy，而不是新 Transformer；
- allocation 先用确定性 greedy/matching，而不是可微 Sinkhorn；
- PPO action policy 不改；
- 不把 proxy 冒充校准概率；
- 先证明机制语义和生命周期指标，再考虑 learned head。

但这只是理论推荐，不是当前执行建议。M0 是必要的 lifecycle 基线；M1 是辨别 capturability 是否增加价值的中间层；M2-lite 是否成为论文主线，必须由审阅和后续证据决定。

---

## 3. 术语、命名与符号

### 3.1 为什么把 \(C\) 留给 capturability

Coverage 和 Capturability 都以 C 开头，极易混淆。本文规定：

- \(C(S,j)\)：联盟 \(S\) 对目标 \(j\) 的 **capturability / capture feasibility**；
- \(E_{\mathrm{cov}}(t)\)：覆盖误差，越低越好；
- \(Q_{\mathrm{cov}}(t)\)：覆盖质量，越高越好；
- CE：当前代码已有的 centroid error，不再简称 C。

### 3.2 基础集合与状态

| 符号 | 英文名 | 中文释义 |
|---|---|---|
| \(\mathcal A\) | active agent set | 当前可用 UAV 集合 |
| \(\mathcal J\) | active target set | 当前活动目标集合 |
| \(i\in\mathcal A\) | agent index | 第 \(i\) 个 UAV |
| \(j\in\mathcal J\) | target index | 第 \(j\) 个目标 |
| \(S_j\subseteq\mathcal A\) | coalition for target \(j\) | 为目标 \(j\) 临时形成的捕获联盟 |
| \(x_i,v_i\) | agent position/velocity | UAV 的位置和速度 |
| \(x_j,v_j\) | target position/velocity | 目标的位置和速度；局部执行时未必可知 |
| \(b_j\) | target information/belief | 对目标 \(j\) 的局部、带时效的信息状态 |
| \(H\) | capture horizon | 判断“能否捕获”的有限时间窗 |
| \(z_{ij}\in\{0,1\}\) | assignment variable | UAV \(i\) 是否被分给目标 \(j\) |
| \(\lambda\ge0\) | trade-off coefficient | 覆盖机会成本权重 |

### 3.3 四个经常被混用的词

1. **Target selection / 目标选择**：一个 agent 更关注哪个目标。
2. **Assignment / 任务分配**：建立 agent—target 的约束映射。
3. **Recruitment / 支援招募**：现有联盟决定是否还要借调新的 agent。
4. **Capturability-aware recruitment / 可捕获性感知招募**：招募由新增 agent 对终端捕获可行性的边际贡献驱动。

有 target preference 不自动等于 assignment；有 assignment 不自动等于 dynamic recruitment；有 dynamic coalition 不自动等于 capturability-aware。

### 3.4 捕获语义的层级

\[
\text{approach}
\not\equiv
\text{ring visitation}
\not\equiv
\text{no-escape}
\not\equiv
\text{terminal capture}.
\]

- **approach**：追捕者接近目标；
- **ring visitation**：2 个或 3 个追捕者短时进入指定半径/角度窗口；
- **no-escape**：在动力学与控制约束下目标没有可行逃逸通道；
- **terminal capture**：满足环境规定的不可逆捕获事件，目标被移除、失能或 episode 进入明确终态。

当前 CoCap loose capture 是“一步几何条件触发的 terminal event”，不是面对策略族的鲁棒 no-escape 证书。

---

## 4. Literature map 的具体内容与结论

### 4.1 第一阶段地图覆盖了什么

第一阶段不是只找“coverage + pursuit”标题，而是沿四条链核对：

1. **Pursuit-evasion / reach-avoid 理论链**：什么条件可以叫可捕获；
2. **Learning / multi-target allocation 链**：目标偏好、容量、联盟形成怎样进入策略；
3. **Lifecycle 链**：捕获后是否解散、复用、恢复和应对重复到达；
4. **Real deployment / opponent 链**：物理部署与战略对手证据有多强。

结论不是某一组件从未出现，而是现有工作往往只覆盖闭环的一部分。

### 4.2 最强竞争者及它们真正覆盖的部分

| 竞争者 | 最强证据 | 与 CoCap 候选主线的关键差异 |
|---|---|---|
| P0044 KS-COAL | exploration、动态 capture coalition、真实 immobilizing capture、防守、捕后复用、硬件 | 固定 SCOUT/SWAT 角色池；K-serial stability 不是 \(C(S,j)\) 证书 |
| P0042 | learned target preference、硬 target capacity、subgroup control、空间捕获 | capacity 是预设常数，不由当前联盟几何/动力学的 capture feasibility 决定 |
| P0034 | literal local FOV、遮挡、丢失—重获 | 检测后 target coordinate team-wide sharing；决策时并非完全 unresolved local information |
| P0035 TERL | Transformer、规模扩展、soft target selection | active target 信息偏全局；target selection 不等于资源受限的 recruitment |
| P0048 | 同一 homogeneous pool 的 patrol→local pursuit→same-agent patrol recovery→repeated intruders | 没有 terminal capture，因而 lifecycle 少了明确完成与释放边界 |
| P0036 OPEN | 三机、CTBR、6-DoF、多 UAV 物理部署 | mocap/offboard/virtual-target 等限制；主要强在物理控制证据 |
| P0040 AgilePE | learned evader、SP/FSP/PFSP、战略对手训练 | 1v1、mocap、preprint；没有 unseen-policy exploitability 的充分证据 |

### 4.3 Pursuit-evasion 理论链给出的约束

R-PE-01 的核心不是要求在线求 HJI，而是提醒任何 \(C(S,j)\) 都要回答：

1. **终端集合是什么**：距离阈值、包围几何、失能还是持续保持？
2. **时间窗是什么**：有限时间 \(H\) 内还是最终可达？
3. **对手模型是什么**：固定 heuristic、best response，还是策略分布？
4. **动力学约束是什么**：速度、转弯率、加速度、障碍是否进入判断？
5. **联盟作用是什么**：多追捕者如何共同关闭逃逸方向？

理论上可形成如下链条：

\[
\text{reach-avoid semantics}
\rightarrow
\text{signed certificate}
\rightarrow
\text{pair/coalition feasible edge or hyperedge}
\rightarrow
\text{resource-constrained assignment}
\rightarrow
\text{speed-aware no-escape closure}
\rightarrow
\text{maintained inward contraction}
\rightarrow
\text{positive capture radius / terminal event}.
\]

它为方法命名提供底线：如果只是距离或人数 heuristic，就应叫 **capture-feasibility proxy/score**，不能叫 reachability certificate 或 guaranteed capturability。

### 4.4 Learning / allocation 链给出的约束

R-LM-01 与 R-LM-02 的核心结论是：

- attention、Transformer、GNN 是表示工具，不是 novelty 本身；
- learned target preference 可以改善协同，但不自动产生显式资源约束；
- fixed K 或 fixed capacity 解决“最多派多少”，不解决“当前状态究竟需要多少”；
- dynamic coalition size 只有在由 capture feasibility 与机会成本共同决定时，才更接近 CoCap 候选贡献；
- Q/value 是策略回报的综合预测，除非专门定义和监督，否则不能等同 \(C(S,j)\)；
- robust opponent training 是另一条贡献轴，不应在 lifecycle/capturability 尚未站稳时同时扩张。

### 4.5 Lifecycle 链给出的约束

R-LIFE-01 中与本文最相关的证据是：

- P0043：完成→coalition dissolution→reusable uncommitted pool→replacement targets；
- P0044：真实 capture、动态 coalition、post-capture reuse，但角色池固定；
- P0046：有 post-apprehension “resume patrolling” 的政策先例，但没有完整定量恢复轨迹；
- P0048：同质 pool、patrol→local pursuit→same-agent patrol recovery→repeated arrivals，但没有 terminal capture。

因此，第一阶段封闭分支没有找到同时满足以下全部条件的同一方法：

\[
\text{one fungible coverage pool}
+\text{local/intermittent information}
+\text{capturability-aware recruitment}
+\text{terminal capture}
+\text{quantified same-agent recovery}
+\text{repeated arrivals}.
\]

这是一个 **bounded search result**，不是宇宙性的不存在证明。

### 4.6 已经不宜单独声称 novelty 的方向

下列元素均有强先例，不能单独作为 “first”：

- local sensing/FOV；
- target loss/reacquisition；
- decentralized execution；
- multi-target pursuit；
- Transformer/GNN；
- target selection；
- fixed-capacity assignment；
- subgroup control；
- generic dynamic coalition；
- 广义 capturability-aware assignment；
- adaptive coalition size；
- geometric capture/no-escape；
- encirclement→physical interception；
- monitoring + capture；
- release/reuse/return/repeated arrivals；
- same-agent patrol→pursuit→patrol；
- persistent coverage + pursuit；
- real UAV / sim-to-real / CTBR；
- trajectory prediction；
- learned evader、自博弈、PFSP；
- game geometry + RL。

可以争取的不是这些词本身，而是它们在 **同一个受局部信息约束的可互换资源池生命周期里，以显式边际资源语义耦合**。

---

## 5. 当前 CoCap 的真实方法对象

### 5.1 Fungible resource pool

**已经满足的部分**

- 所有 pursuer/UAV 使用同类状态、动作空间和共享 actor；
- 没有永久固定的 coverage UAV 与 capture UAV 身份；
- 一个 agent 的有效角色可以随局部状态在 coverage、support、capture 之间变化；
- capture 完成后 FullMix 可进入 post-capture coverage recovery。

**尚未满足或未证明的部分**

- 切换是由可见性与邻接规则触发，不是显式 resource allocator；
- reward 差异可能学习出事实上的长期角色偏置，需要 role/state occupancy 与 agent-wise transition 统计确认；
- 当前 4v1 不能检验多个目标争抢同一可互换 pool。

结论：**结构上接近 fungible，行为上尚需指标证明没有 emergent fixed roles。**

### 5.2 Local information

**actor execution 侧**

- target token 受 target visibility radius 限制；
- NormSense V2 的 4v1 有效 target sensing radius 约为 52.2 m；
- global enemy broadcast 关闭；
- 直接看见目标的 agent 可成为 capture；
- 看不见目标、但与 capture friend 邻接的 agent 可成为 support；
- support agent 主要看到 friend 的相对几何与 pursuing flag，并不直接获得敌方坐标。

**重要 caveat**

- friendly VorAdj/free-mask 仍带有历史 global-friendly/static-global-geometry 色彩；
- support reward 在训练时可利用 neighbor-visible target geometry，而 actor execution 不直接收到原始 target coordinate；
- 没有显式 target belief、信息年龄、置信度或 loss/reacquisition memory；
- 当前 4v1 reset 下至少一名探测者的概率很高，局部性真实存在但难度未必强；
- sensing 主要是距离/表面半径规则，不能自动宣称具有真实遮挡 FOV。

**critic 侧**

- central critic 使用全局 pursuer、evader 与 obstacle state，这是 CTDE；
- critic 的全局信息不应被写成 actor execution 获得全局目标信息；
- critic value 是 expected return，不是 coalition capturability。

结论：**actor 的目标信息接近 local direct detection + one-hop behavioral propagation，但不是完整的 intermittent belief problem。**

### 5.3 Coalition recruitment

当前 recruitment 逻辑可概括为：

\[
\text{visible target}\Rightarrow \text{capture role},
\]

\[
\neg\text{visible target}\land
\text{adjacent capture friend}
\Rightarrow \text{support role},
\]

\[
\text{otherwise}\Rightarrow\text{coverage role}.
\]

它不是 fixed K，也不完全等同 nearest-K；它是一种 **局部邻接触发的隐式 recruitment**。但它没有询问：

- 当前 coalition 是否已经足够；
- 候选 agent 加入后终端捕获概率提高多少；
- 该 agent 离开 coverage 的代价多大；
- 多个 target 同时请求时应该服务谁。

当前 active PPO 路径没有使用历史 IQN 的 gate head。因此论文不能把 IQN gate 当成当前方法组成。

### 5.4 Capturability semantics

当前实现中与 capturability 有关、但不能直接等同它的量包括：

- pursuer count；
- agent—target distance；
- occupied angle / max angular gap；
- relative speed 与 target velocity；
- ring radius deviation；
- obstacle-aware slot geometry；
- neighbor adjacency；
- ring shaping reward；
- central value；
- 历史 gate logit。

其中 **ring importance reward** 最接近可解释几何 proxy：它综合 target velocity、preferred ring radius、teammate angular occupancy、obstacle 可行性和 toward-slot progress。它提供“几何正在变好”的 dense signal，但没有以下语义：

\[
C(S,j)=P(\text{target }j\text{ is terminally captured within }H\mid S,\text{information}).
\]

所以当前不存在经过定义、监督或校准的 \(C(S,j)\)。

### 5.5 Multi-target resource conflict

当前正式 FullMix 是 4v1。代码可以容纳多个 evader，不等于方法已解决 multi-target conflict：

- 没有显式 \(z_{ij}\)；
- 没有 \(\sum_j z_{ij}\le1\) 的 exclusivity；
- 没有 target-wise dynamic capacity；
- agent 的注意/局部几何隐式决定行为；
- 极端几何下同一 agent 可能对多个 target 的 terminal event 都有贡献。

结论：**multi-target resource allocation 是当前方法缺口，而非已有能力。**

### 5.6 Terminal capture

当前 normal/loose capture 大致要求：

- 至少 3 名 pursuer；
- 位于 capture distance（当前约 8 m）内；
- 最大角间隙不超过 \(\pi\)；
- 最大/最小角间隙比不超过 3。

另有 stationary capture：

- target speed 不高于约 0.2；
- 至少 2 名 pursuer；
- 条件持续约 10 steps。

事件触发后 target deactivation，因此在环境语义上它是 terminal capture。需要明确：

- ring2/ring3 visitation 只是过程指标；
- loose capture 是单步几何判据；
- stationary capture 是持续窗口判据；
- 两者都不是面对任意战略 evader 的数学 no-escape guarantee。

### 5.7 Release 与 coverage recovery

FullMix 中 capture 后发生：

1. target deactivation；
2. capture/support 关联清除与 release-delay transition；
3. 进入 post-capture phase；
4. 所有 agent 重新受到 coverage objective 驱动；
5. 最长约 500-step recovery window；
6. 以 CE RMS、CE max 和 hold window 判定严格恢复。

当前严格恢复目标近似是：

- CE RMS \(\le0.05\)；
- CE max \(\le0.1\)；
- 连续保持 30 steps。

这比“切回 coverage reward”更强，因为存在显式恢复成功判据。但它仍欠缺：

- 相对于事件前个体化 baseline 的恢复度；
- coverage debt 与恢复时间；
- 非参与者连续性；
- 同一 episode 的下一次到达；
- 多次事件后的 degradation。

### 5.8 Repeated arrivals

当前环境没有同一 episode 内的 arrival scheduler：

- mixed scene 的 target 在 reset 时已经存在；
- recovery snapshot pool 是跨 episode 的初始化资产；
- 它不等于 coverage→arrival→capture→recovery→next arrival。

因此 repeated arrivals 是明确的 paper-facing gap。

---

## 6. Implementation ↔ Literature Gap Audit

| Literature-derived requirement | Current implementation | Already satisfied? | Weakness | 理论上需要的变化 |
|---|---|---:|---|---|
| one fungible resource pool | homogeneous agents、shared actor、动态 role | 部分/较强 | 可能形成隐性固定角色；未在多目标验证 | agent-wise role transition/occupancy；多目标资源冲突测试 |
| local/intermittent target information | actor 无 global enemy broadcast；direct detection + neighbor pursuing flag | 部分 | 无 belief age/confidence；friendly topology 偏全局；初始可探测率较高 | 明确允许的信息边界；必要时加入最小 request/age，而非全局坐标 |
| coalition recruitment | visible→capture，adjacent-to-capture→support | 部分 | 与 coalition sufficiency、target difficulty 无关 | 用 \(\Delta_i(S,j)\) 控制是否继续招募 |
| capturability semantics | ring geometry、count、distance、speed、value 可作原料 | 否 | 没有定义、标签或校准的 \(C\) | 先构造可解释 \(C_{\mathrm{geo}}\)；需要时再学 head |
| multi-target resource conflict | 无显式 assignment；正式任务 4v1 | 否 | 无 exclusivity、无 target-wise coalition | \(z_{ij}\)、\(\sum_jz_{ij}\le1\)、target-wise \(S_j\) |
| terminal capture | loose/stationary event 后 target deactivation | 是，环境语义上 | 不等于鲁棒 no-escape | 论文准确命名；加入 opponent robustness 仅作评价 |
| release | target deactivation 后关联清除与 phase transition | 是/部分 | release latency 未正式量化 | 定义 \(t_{\mathrm{release}}-t_{\mathrm{cap}}\) |
| coverage recovery | post-capture CE recovery window 与 strict gate | 是/部分 | 非事件前相对基线；无 debt/continuity | event-relative lifecycle metrics |
| repeated arrivals | 无同 episode scheduler | 否 | 不能测恢复是否赶上下次事件 | M0 层面的 arrival process，待 capture consolidated 后 |
| coverage—capture opportunity cost | support reward 0.5 capture + 0.5 coverage | 弱 proxy | 奖励混合不等于显式 allocation cost | \(\mathrm{CoverageCost}(i)\) 进入 recruitment utility |
| decentralized execution | actor local、critic centralized | 基本是 | reward-side oracle 与 global-friendly topology 需披露 | information ablation 与清晰 CTDE 声明 |

---

## 7. Pure-Capture 阶段：训练问题与论文问题必须分开

### 7.1 Training-stage problem

Pure-Capture 的作用是隔离并稳定最困难的子能力：

- 接近与包围；
- terminal capture；
- capture reward 是否可学；
- PPO update 与 evaluator 是否可信；
- safety penalty 是否压倒 capture signal。

它可以暂时把 coverage reward 置零、捕获后结束 episode。这是训练分解，不是对 CoCap 最终任务的重新定义。

### 7.2 Paper-facing final problem

最终论文问题至少应包含：

- 持续 coverage 起始状态；
- 事件/目标到达；
- 局部发现与信息传播；
- support recruitment；
- terminal capture；
- release；
- same-agent coverage recovery；
- repeated arrivals；
- 若声称资源分配，则必须有多目标冲突。

### 7.3 写作时的实验快照

写作时当前正式执行线的事实如下：

- Pure-Capture 已训练到 100k checkpoint；
- 最近完整 evaluation artifact 是 75k；
- 75k 的 20-episode argmax/sample 均为 0 capture；
- argmax 已出现 ring2 visitation 0.75、ring3 visitation 0.15，但 collision rate 为 1.0；
- 100k checkpoint evaluation 因 pure telemetry 断言观察到 post_capture phase 而停止，留下 **STOP_IMPLEMENTATION** failure artifact；
- Original-FullMix 与 CaptureDownweight05-FullMix 均已到 50k checkpoint evaluation，进程仍存活；
- 两条 FullMix 的最新完整 artifact 仍是 25k，不能据此作最终方法结论。

解释：

- ring visitation 增加不是 terminal capture；
- 100k failure 是执行/合同不一致证据，不是学习成败结论；
- FullMix 仍在运行，本文不改变、不终止、不重启它们；
- 这些结果提示 capture consolidation 尚未完成，因此更不应立刻叠加 M1/M2。

### 7.4 理论上的恢复 gate，而非当前行动命令

在未来批准 lifecycle 扩展前，至少应有：

1. capture success 在多个 seeds/held-out scenes 下非偶然；
2. argmax 不依赖高碰撞或 evaluator loophole；
3. normal/stationary capture 语义分开报告；
4. capture time 与 ring visitation 稳定；
5. Pure-Capture telemetry/phase contract 无歧义；
6. FullMix 能出现真实 capture→post_capture 样本；
7. 加回 coverage 后 capture 能力没有完全遗忘。

这不是固定数值阈值；阈值需要负责人结合预算、baseline 和 evaluator 审核后另立合同。

---

## 8. Capturability 的数学定义与命名边界

### 8.1 理想概率定义

若有明确数据生成与校准，可以定义：

\[
C(S,j)
=
\Pr\!\left(
\tau^{\mathrm{cap}}_j\le H
\mid
b_j,\;S,\;\mathcal D,\;\pi_p,\;\pi_e
\right).
\]

逐项中文解释：

- \(S\)：目前为目标 \(j\) 服务的追捕联盟；
- \(b_j\)：在局部/间歇信息下对目标状态的可用信息；
- \(\mathcal D\)：双方动力学、障碍、边界和安全约束；
- \(\pi_p\)：pursuer policy；
- \(\pi_e\)：evader policy 或对手分布；
- \(\tau^{\mathrm{cap}}_j\)：目标 \(j\) 第一次触发 terminal capture 的时间；
- \(H\)：允许的有限捕获时间窗；
- \(C(S,j)\in[0,1]\)：在这些条件下于 \(H\) 内完成捕获的概率。

若模型输出没有经过 terminal-within-\(H\) 标签训练和 calibration，就不能叫概率。

### 8.2 面向多种对手的扩展

对手不是单一固定策略时，可写：

\[
C_{\Pi_e}(S,j)
=
\mathbb E_{\pi_e\sim\Pi_e}
\left[
\Pr(\tau^{\mathrm{cap}}_j\le H\mid \pi_e)
\right],
\]

或者更保守地使用低分位数：

\[
C^{\mathrm{rob}}_{\alpha}(S,j)
=
\operatorname{Quantile}_{\alpha,\;\pi_e\sim\Pi_e}
\left[C(S,j;\pi_e)\right].
\]

前者是平均对手分布下的可捕获性；后者关注较难对手。当前项目不应在 lifecycle 未稳时马上把它扩张成 PFSP 主线。

### 8.3 可解释几何 proxy

无需先建新网络，可以从当前已有量构造：

\[
C_{\mathrm{geo}}(S,j)
=
\sigma\!\left(
w_0
+w_n\phi_n
+w_{\theta}\phi_{\theta}
+w_r\phi_r
+w_v\phi_v
+w_o\phi_o
\right),
\]

其中 \(\sigma(a)=1/(1+e^{-a})\) 只把分数压到 \([0,1]\)。如果未校准，它仍叫 **score**，不是 probability。

各特征可定义为：

#### 人数充分度

\[
\phi_n=\min\left(\frac{|S|}{K_{\mathrm{ref}}},1\right).
\]

\(|S|\) 是 coalition size，\(K_{\mathrm{ref}}\) 是与 terminal rule 对应的参考人数。它表达“人够不够”，但单独不能表达位置和动力学。

#### 角度闭合度

令 \(\theta_{ij}\) 为 pursuer \(i\) 相对 target \(j\) 的方位角，排序后计算环上的最大角间隙 \(g_{\max}\)。可定义：

\[
\phi_\theta
=
1-\operatorname{clip}\left(\frac{g_{\max}}{2\pi},0,1\right).
\]

最大间隙越小，包围越均匀，\(\phi_\theta\) 越高。若要紧贴当前 loose rule，可另报告 \(\mathbf 1[g_{\max}\le\pi]\)，避免连续分数掩盖 terminal threshold。

#### 环半径匹配度

\[
\phi_r
=
1-
\operatorname{clip}
\left(
\frac{1}{|S|}
\sum_{i\in S}
\frac{|d_{ij}-r_\star|}{r_{\mathrm{scale}}},
0,1
\right),
\]

其中 \(d_{ij}=\|x_i-x_j\|\)，\(r_\star\) 是 preferred capture-ring radius。越接近理想环半径，得分越高。

#### 闭合速度

距离导数为：

\[
\dot d_{ij}
=
\frac{(x_i-x_j)^\top(v_i-v_j)}{\|x_i-x_j\|+\varepsilon}.
\]

定义 closing speed：

\[
c_{ij}=-\dot d_{ij}.
\]

\(c_{ij}>0\) 表示距离正在缩小。可归一化：

\[
\phi_v
=
\frac{1}{|S|}
\sum_{i\in S}
\operatorname{clip}
\left(
\frac{c_{ij}}{v_{\mathrm{scale}}},
-1,1
\right).
\]

这比只看静态距离更能表达 moving target difficulty。

#### 障碍/槽位可行度

\[
\phi_o
=
\frac{\#\{\text{free feasible capture slots occupied or reachable}\}}
{\#\{\text{desired capture slots}\}}.
\]

它表达障碍是否堵住关键包围方向。必须说明它是基于当前局部几何的可行度，不是完整轨迹规划证明。

### 8.4 边际支援收益

候选 agent \(i\notin S\) 加入联盟的边际收益：

\[
\Delta_i(S,j)
=
C(S\cup\{i\},j)-C(S,j).
\]

中文释义：

> 不是问“agent \(i\) 离目标多近”，而是问“把它加入当前这支 coalition 后，完成 terminal capture 的可行性增加多少”。

这一区别很关键：

- 若 coalition 已足够闭合，最近的额外 agent 也可能 \(\Delta_i\approx0\)；
- 若存在一个巨大逃逸角，较远但能补上该角度的 agent 可能 \(\Delta_i\) 更高；
- 不同 target 的 \(\Delta_i\) 不同，因此它自然产生 target-wise recruitment。

若 \(C\) 是 heuristic score，则 \(\Delta_i\) 也只是 score gain，不能写成“捕获概率提高了 20%”。

### 8.5 单调性不是无条件真理

可选正则：

\[
C(S\cup\{i\},j)\ge C(S,j).
\]

直觉上“多一个支援不应更差”。但在碰撞、通信拥堵和控制干扰存在时，它未必严格成立。因此：

- 作为几何 proxy，可设计成单调；
- 作为真实 learned outcome model，不应不加论证地强制严格单调；
- 负边际贡献本身也可能暴露 overcrowding。

---

## 9. Coverage opportunity cost 的数学定义

### 9.1 反事实覆盖代价

令 \(E_{\mathrm{cov}}(\mathcal A)\) 表示当前所有可用于覆盖的 agents 所产生的 coverage error，越低越好。借走 \(i\) 的瞬时反事实代价可定义为：

\[
\mathrm{CoverageCost}(i)
=
\left[
E_{\mathrm{cov}}(\mathcal A\setminus\{i\})
-
E_{\mathrm{cov}}(\mathcal A)
\right]_+,
\]

其中 \([a]_+=\max(a,0)\)。

中文释义：

> 临时从 coverage pool 移除 UAV \(i\) 后，覆盖误差比不移除它恶化多少。

该定义比“离 coverage centroid 多远”更直接，但计算可能需要反事实 Voronoi/CE 重算。它可以是分析量，不一定立即进入 actor observation。

### 9.2 时间窗版本

瞬时代价可能低估未来缺口，可定义：

\[
\mathrm{CoverageCost}_H(i)
=
\sum_{h=0}^{H_c-1}
\gamma_c^h
\left[
\widehat E_{\mathrm{cov}}^{(-i)}(t+h)
-
\widehat E_{\mathrm{cov}}(t+h)
\right]_+.
\]

它需要预测未来 coverage，复杂度更高。M2-lite 理论上优先使用瞬时或短窗近似，不先建新预测器。

### 9.3 Capturability 与 coverage cost 的耦合

\[
U_i(S,j)
=
\Delta_i(S,j)
-
\lambda\,\mathrm{CoverageCost}(i).
\]

命名逻辑：

- \(U\)：utility，支援分配的净效用；
- \(\Delta_i\)：加入 coalition 的 capture-side marginal gain；
- CoverageCost：离开覆盖岗位的 opportunity cost；
- \(\lambda\)：两类量的尺度与偏好折中。

最简单的 recruitment rule：

\[
\text{recruit }i\text{ for }j
\quad\Longleftrightarrow\quad
U_i(S,j)>\tau_U.
\]

因此“更多追捕者”不再永远更好；只有边际捕获收益超过覆盖损失时才借调。

---

## 10. Multi-target allocation 的资源约束

定义二元 assignment：

\[
z_{ij}=1
\Longleftrightarrow
\text{agent }i\text{ is committed to target }j.
\]

每个 agent 同时最多服务一个目标：

\[
\sum_{j\in\mathcal J}z_{ij}\le1,\qquad\forall i\in\mathcal A.
\]

目标 \(j\) 的 coalition 为：

\[
S_j=\{i\in\mathcal A:z_{ij}=1\}.
\]

一个直接但组合复杂的目标是：

\[
\max_{\{z_{ij}\}}
\sum_{j\in\mathcal J}C(S_j,j)
-
\lambda\sum_{i,j}z_{ij}\,\mathrm{CoverageCost}(i),
\]

受上述 exclusivity 和局部信息约束。

M2-lite 不需要精确求全局组合最优。可以按边际效用迭代：

1. 从直接 detector 构成最小 seed coalition；
2. 计算可见/可通信候选的 \(U_i(S_j,j)\)；
3. 选择最高正效用的 agent—target pair；
4. 更新对应 \(S_j\)，重新计算边际量；
5. 没有正效用时停止。

这个 greedy 过程的价值不在“全局最优”，而在 allocation 语义清楚、可消融、与现有局部 role 机制兼容。

---

## 11. 三个候选方法版本

### 11.1 M0 — Minimal lifecycle baseline

**目的**

只检验完整生命周期能否稳定工作，不引入 capturability 新方法。

**Observation**

- 沿用当前 direct target token、friend token、VorAdj 与 pursuing flag；
- 不新增 belief network 或全局 target broadcast。

**Network**

- 当前 PPO actor 与 central critic；
- 不启用历史 IQN gate；
- 不新增 Transformer/head。

**Gate / allocation**

- 沿用当前 visibility + adjacency role rule；
- 没有显式 \(z_{ij}\)，所以 M0 只应叫 lifecycle baseline。

**Reward/loss**

- 沿用 coverage、capture、support blend、terminal 与 safety；
- 重点是把 phase 与 event accounting 做清楚，不重新雕刻 reward。

**Lifecycle**

\[
\text{coverage warm-up}
\rightarrow
\text{arrival}
\rightarrow
\text{capture}
\rightarrow
\text{release}
\rightarrow
\text{recovery}
\rightarrow
\text{next arrival}.
\]

**训练信号**

- PPO 原始信号；
- capture/recovery 分阶段 curriculum 可保留；
- repeated-arrival 只在 capture consolidation 后考虑。

**能证明什么**

- 一套 shared policy 是否能完成闭环；
- terminal release 和 same-agent recovery 是否真实存在；
- repeated events 是否累积退化。

**不能证明什么**

- capturability-aware recruitment；
- coverage-cost-aware resource allocation；
- multi-target optimality。

### 11.2 M1 — Capturability-informed recruitment

**目的**

把“邻居看见 pursuing friend 就支援”升级为“只有加入能显著提高当前 coalition capture-feasibility 才支援”。

**Observation**

候选最小增加不是 target 全局坐标，而是 per-neighbor/per-request 的低维语义：

- target/coalition request id 或局部匿名 slot；
- request strength；
- 当前 coalition sufficiency score；
- 可选的信息年龄/置信度；
- agent 自身对该 request 的 \(\Delta_i\) 所需局部几何。

无法直接看见目标的 support agent 只需接收 request 语义，不必接收原始 enemy coordinate。

**Network**

- 首选无新网络：\(C_{\mathrm{geo}}\) 确定性计算；
- PPO action actor 不变；
- 若观测维度增加，采用零初始化新增列/adapter，避免破坏旧表示。

**Gate / allocation**

\[
\Delta_i(S,j)=C_{\mathrm{geo}}(S\cup\{i\},j)-C_{\mathrm{geo}}(S,j).
\]

当 \(\Delta_i>\tau_\Delta\) 才允许 support request；coalition 已足够时停止继续扩张。

**Reward/loss**

- 第一版不必把 \(C_{\mathrm{geo}}\) 直接加入 reward；
- allocation 只改变 role/target binding；
- PPO loss 保持；
- 避免同时改变 role rule 与 reward，导致无法归因。

**Training signal**

- deterministic score 无额外监督；
- 若未来学习 probability head，标签为“从当前状态、当前 coalition 出发，在 \(H\) 内是否 terminal capture”：

\[
y_{S,j}^{(t)}
=
\mathbf 1[\tau_j^{\mathrm{cap}}-t\le H].
\]

可用 binary cross entropy：

\[
\mathcal L_C
=
-y\log\widehat C
-(1-y)\log(1-\widehat C).
\]

并单独报告 Brier score、ECE 和 held-out opponent calibration。没有这些证据就仍称 score。

**稳定性**

- deterministic M1 比 learned head 更稳定；
- 风险是 \(C_{\mathrm{geo}}\) 与 terminal capture rule 过度同构，只学到手工 threshold；
- 需要对 target speed、obstacle 与 coalition geometry 做压力测试。

### 11.3 M2 — Capturability vs coverage-cost coupling

**目的**

把“能帮助捕获”进一步变成“帮助值得付出覆盖损失”。

**Observation**

在 M1 信息上加入本地可计算的 coverage criticality，例如：

- agent 的 Voronoi cell area/error contribution；
- 移除该 agent 的反事实 CE 增量；
- 当前局部 coverage deficit；
- 距离 nearest free coverage site 的代价。

**Network**

- M2-lite：无新增网络；
- \(C_{\mathrm{geo}}\) + counterfactual coverage cost + greedy allocator；
- PPO actor/critic 不改；
- 若以后需要 learned \(C\)，最多增加一个小 MLP/head，共享现有 representation，不新建大 Transformer。

**Allocation**

\[
U_i(S,j)
=
\Delta_i(S,j)
-
\lambda\,\mathrm{CoverageCost}(i),
\]

满足：

\[
\sum_j z_{ij}\le1.
\]

target capacity 不再固定为 K，而是当所有候选 marginal utility 非正时动态停止。

**Reward/loss**

- 第一版不建议再加一个 \(U\)-reward；
- \(U\) 是 allocator criterion，环境 reward 仍负责动作学习；
- 若 \(\lambda\) 固定，避免额外 Lagrangian 不稳定；
- 只有 coverage constraint 明确、固定 \(\lambda\) 失败后，才讨论 learned multiplier。

**Inference**

- detector/seed coalition 生成 request；
- 局部候选计算 capture gain 与 coverage cost；
- greedy 选正效用 pair；
- support/capture role 获得 target binding；
- terminal capture 后显式解除 \(z_{ij}\)，agent 回到 coverage。

**稳定性**

- 优点：改动小、语义强、容易与 fixed K/current gate 做消融；
- 风险：两个 proxy 尺度不一致，\(\lambda\) 可能敏感；
- 风险：若 coverage metric 计算含全局信息，必须区分 centralized training/evaluation 与 decentralized execution；
- 风险：错误的 \(C_{\mathrm{geo}}\) 会拒绝必要支援。

### 11.4 三者比较

| 版本 | 新语义 | 新网络 | 主要风险 | 论文价值 |
|---|---|---:|---|---|
| M0 | 完整 lifecycle | 0 | capture 本身尚未稳定；arrival 改变任务分布 | 必要系统基线，方法 novelty 较弱 |
| M1 | coalition sufficiency 与 marginal support gain | 0 或小 head | proxy/概率命名错误；局部信息泄漏 | 清晰中间贡献 |
| M2-lite | marginal capture gain − coverage opportunity cost + resource constraint | 0 | proxy 尺度、\(\lambda\)、多目标复杂度 | 当前理论上最高性价比主线 |
| M2-full | learned belief + learned \(C\) + differentiable assignment | 多个 | 信号耦合、训练不稳、难归因 | 暂不建议 |

---

## 12. 最低必要实验矩阵（仅理论验证设计）

### 12.1 Recruitment ablation

在相同 actor backbone、训练预算和 evaluator 下比较：

1. fixed K；
2. nearest/distance；
3. current visibility-adjacency gate；
4. capturability-informed M1；
5. capturability + coverage cost M2-lite。

必须匹配或报告：

- coalition size 分布；
- capture success 与 time-to-capture；
- coverage debt；
- compute overhead；
- local information budget。

否则 M2 可能只是“派更多人”。

### 12.2 Lifecycle ablation

- one-shot capture；
- capture→release；
- capture→release→coverage recovery；
- repeated arrivals。

每增加一层，只改变 lifecycle，不同时改变 actor architecture。

### 12.3 Information ablation

建议把 regime 明确定义为：

- **G0 Global target**：所有 agent 获得实时 target state；
- **G1 Shared-after-detection**：任一 agent 检测后广播 target coordinate/state；
- **G2 Local direct + one-hop request**：只有 detector 看见目标，support 只收到邻接 request/summary。

当前实现最接近 G2，但 friendly VorAdj 的全局性与 reward-side geometry 必须单独披露。G2 不应被写成“完全无通信”。

### 12.4 Role-pool ablation

- fixed coverage/capture partition；
- fungible same-pool swarm。

需要在同样总 UAV 数、同样 target arrival 和相近参数量下比较。该消融可以验证 fungibility 是否真正改善长期资源利用，而不是只增加训练自由度。

### 12.5 Opponent evaluation

最低层级：

- randomized/reactive heuristic evader；
- separately trained learned evader；
- 至少一个 held-out learned evader。

PFSP 不必立刻加入。只有 strategic robustness 被提升为独立 contribution 时，才需要更系统的 population/equilibrium 训练。

---

## 13. Lifecycle metrics：定义与当前 state 来源

设目标事件到达时刻为 \(t_a\)，terminal capture 为 \(t_c\)，release 完成为 \(t_r\)，恢复首次满足并保持 \(L\) 步的时刻为 \(t_{\mathrm{rec}}\)，下一次到达为 \(t_{a}^{+}\)。

### 13.1 Pre-event coverage baseline

\[
E_{\mathrm{pre}}
=
\frac{1}{W_{\mathrm{pre}}}
\sum_{t=t_a-W_{\mathrm{pre}}}^{t_a-1}
E_{\mathrm{cov}}(t).
\]

中文：目标到达前一段窗口内的平均覆盖误差。当前可由 env 已有 CE RMS/CE max history 计算。若目标在 reset 即存在，则该指标不可定义，这正说明 M0 需要 coverage warm-up/arrival event。

### 13.2 Peak coverage deficit

\[
D_{\mathrm{peak}}
=
\max_{t\in[t_a,t_{\mathrm{rec}}]}
\left[E_{\mathrm{cov}}(t)-E_{\mathrm{pre}}\right]_+.
\]

中文：整个事件期间相对事前基线最严重的一次覆盖恶化。

### 13.3 Integrated coverage debt

离散实现：

\[
D_{\mathrm{int}}
=
\Delta t
\sum_{t=t_a}^{t_{\mathrm{rec}}}
\left[E_{\mathrm{cov}}(t)-E_{\mathrm{pre}}\right]_+.
\]

中文：覆盖误差“超出基线的面积”，同时惩罚恶化幅度与持续时间。当前 evaluator 已有逐步 CE，可直接累积，无需改 policy。

### 13.4 Agent-seconds borrowed from coverage

\[
B_{\mathrm{sec}}
=
\Delta t
\sum_t
\left|
\{i:r_i(t)\in\{\mathrm{support},\mathrm{capture}\}\}
\right|.
\]

中文：从 coverage pool 借走的人数乘以时间。state 来源是 env 的 effective role/phase。

### 13.5 Capture success 与 time-to-capture

\[
\mathrm{Success}
=
\mathbf 1[t_c<\infty],
\qquad
T_{\mathrm{cap}}=t_c-t_a.
\]

必须分 normal capture、stationary capture，并同时报告 ring2/ring3 仅作 precursor。

### 13.6 Coalition size

可报告：

\[
K_{\mathrm{mean}}
=
\frac{1}{t_c-t_a}
\sum_{t=t_a}^{t_c-1}|S_j(t)|,
\]

以及 peak、terminal coalition size。当前无显式 \(S_j\) 时，可以 effective capture/support role 近似；M1/M2 后应直接读 assignment。

### 13.7 Capture→release latency

\[
T_{\mathrm{release}}=t_r-t_c.
\]

state 来源是 capture event、target active flag、role/binding clear 与 release-delay state。必须明确“target 被移除”与“所有 agent 可重新覆盖”是否同一时刻。

### 13.8 Release→coverage recovery time

\[
T_{\mathrm{recover}}=t_{\mathrm{rec}}-t_r.
\]

一个相对基线 recovery gate 可定义为：

\[
E_{\mathrm{cov}}(t:t+L-1)
\le
\rho E_{\mathrm{pre}}+\epsilon
\quad\text{连续 }L\text{ 步}.
\]

当前 strict CE gate 是绝对阈值；相对 gate 更适合不同初始场景，两者可同时报告。

### 13.9 Recovered coverage ratio

若使用越高越好的质量：

\[
R_Q
=
\frac{\overline Q_{\mathrm{post}}}
{\overline Q_{\mathrm{pre}}+\epsilon}.
\]

若直接使用越低越好的 CE，推荐报告：

\[
R_E
=
\frac{\overline E_{\mathrm{post}}}
{\overline E_{\mathrm{pre}}+\epsilon}.
\]

\(R_E=1\) 表示回到事前误差，\(R_E<1\) 表示更好，\(R_E>1\) 表示未完全恢复。不要把 \(R_E\) 命名为“越大越好”的 recovery ratio，以免方向混乱。

### 13.10 Nonparticipant coverage continuity

令 \(\mathcal N(t)\) 为从未被借调参与本次 target 的 agents。可计算：

\[
\mathrm{NPC}
=
1-
\frac{
\sum_t
\left[
E_{\mathrm{cov}}^{\mathcal N}(t)
-
E_{\mathrm{pre}}^{\mathcal N}
\right]_+
}{
(t_{\mathrm{rec}}-t_a)
(E_{\mathrm{pre}}^{\mathcal N}+\epsilon)
}.
\]

也可更稳健地直接报告 nonparticipant CE curve/debt，避免人为裁剪。需要 env/evaluator 保存 agent-wise Voronoi/centroid contribution。

### 13.11 Recovery-before-next-arrival rate

\[
\mathrm{RBNA}
=
\frac{1}{M}
\sum_{m=1}^{M}
\mathbf 1[t_{\mathrm{rec}}^{(m)}<t_a^{(m+1)}].
\]

中文：第 \(m\) 次事件造成的覆盖损失是否在下一目标到来前恢复。

### 13.12 Repeated-arrival degradation

对第 \(m\) 次事件的任意指标 \(Y_m\)，可报告：

\[
\Delta Y_m=Y_m-Y_1
\]

或对事件序号拟合 slope。关键指标包括 capture success、\(T_{\mathrm{cap}}\)、\(D_{\mathrm{int}}\)、\(T_{\mathrm{recover}}\)。

### 13.13 Role/state occupancy

\[
O_r
=
\frac{1}{N T}
\sum_{i=1}^N
\sum_{t=1}^T
\mathbf 1[r_i(t)=r].
\]

还应报告 transition matrix：

\[
P_{r\rightarrow r'}
=
\frac{\#\{r_i(t)=r,r_i(t+1)=r'\}}
{\#\{r_i(t)=r\}}.
\]

它能回答 fungible pool 是否真的发生 coverage→support/capture→coverage，而不是不同 agent 长期固定在不同状态。

### 13.14 Metric plumbing 的最低字段

每 step/event 只需可靠记录：

- episode_id、step、sim_time；
- arrival/capture/release/recovery event；
- target active/id；
- agent id、effective role、target binding；
- CE RMS、CE max、可选 agent-wise CE contribution；
- coalition size；
- normal/stationary capture reason；
- collision/boundary；
- information regime 与谁直接看见目标；
- 若启用 M1/M2，再记录 \(C,\Delta,\mathrm{CoverageCost},U,z_{ij}\)。

这是低风险 instrumentation 候选，但本文仍不授权直接改代码。

---

## 14. 对现有实验路线的理论影响

以下 NOW/NEXT/LATER/DROP 仅保留为审查词汇，不是执行排期。

### NOW — 在理论上应保持不变的对象

- 当前 formal FullMix 运行及其合同；
- Pure-Capture/FullMix 的根因追踪与 evaluator 可信性建设；
- capture terminal semantics、normal/stationary 分解；
- NormSense V2 信息边界；
- 不把 M1/M2 夹进尚未完成的 capture consolidation。

### NEXT — 经批准后最小的两个候选变化

1. **只做 lifecycle metric instrumentation/离线重算**：不改 reward、observation、action 或 network。
2. **M0 的 event-relative evaluation prototype**：先在 evaluator 或 scripted rollout 中验证 arrival/release/recovery bookkeeping，不立即训练新策略。

如果 capture gate 未满足，M1/M2 仍停留在文档层。

### LATER — 稳定后才值得研究

- deterministic \(C_{\mathrm{geo}}\) 与 \(\Delta_i\)；
- M2-lite coverage cost coupling；
- 8v2/多目标 assignment；
- learned/calibrated capturability head；
- local belief age/confidence；
- separately trained 与 held-out learned evaders；
- 物理 UAV/sim-to-real。

### DROP — 不作为 novelty 主线扩张

- 单独宣传 Transformer/GNN；
- 新增大 Transformer 只为计算 \(C\)；
- 在线完整 HJI solver；
- belief + Sinkhorn + MoE + PFSP + CBF 一次性集成；
- 把 historical IQN gate 直接包装成当前贡献；
- 把 target preference 包装成 capturability-aware assignment；
- 用固定 K 作为 dynamic feasibility；
- 把 ring2/ring3 叫 terminal capture；
- 把 post-capture reward switch 叫 quantified coverage recovery；
- 在没有多目标资源冲突时声称解决 multi-target allocation。

---

## 15. CoCap Method Decision v1（待审候选，不生效）

### 15.1 核心 method object

建议理论主对象不是一个新网络，而是：

> **对每个 target 的临时 coalition 及其边际资源决策。**

形式上由 \((S_j,C(S_j,j),\Delta_i,\mathrm{CoverageCost}(i),z_{ij})\) 组成。

### 15.2 Observation

- 保留 direct local target token；
- 非 detector 只得到 one-hop request/coalition summary；
- 不广播 raw global target state；
- 若引入 belief，只从 age/confidence 的最小状态开始；
- actor 与 critic information 明确分栏披露。

### 15.3 Gate / allocation

- M0：当前 visibility-adjacency rule；
- M1：\(\Delta_i>\tau_\Delta\)；
- M2-lite：\(U_i=\Delta_i-\lambda\,\mathrm{CoverageCost}(i)>\tau_U\)；
- 多目标时满足 \(\sum_jz_{ij}\le1\)；
- 用 greedy/matching，不先做可微 assignment。

### 15.4 Reward/loss

- 不先把 \(C\)、\(\Delta\)、CoverageCost 全塞进 reward；
- 保留已验证的 PPO action loss；
- allocation 作为上层语义机制；
- learned head 只有在 deterministic proxy 有价值后才加入 BCE/calibration；
- central value 继续做 return estimation，不冒充 \(C\)。

### 15.5 Lifecycle semantics

必须显式区分：

- detection；
- request/recruitment；
- target binding；
- terminal capture；
- coalition release；
- return-to-coverage；
- recovery achieved；
- next arrival。

### 15.6 Training stages

仅作为概念顺序：

1. capture consolidation；
2. M0 lifecycle；
3. deterministic M1；
4. M2-lite；
5. multi-target conflict；
6. optional learned \(C\)；
7. optional strategic opponent robustness。

每一层须能回退并有单独消融。

### 15.7 最小新增网络

理论首选：**零新增网络**。

若后续证据证明 heuristic \(C_{\mathrm{geo}}\) 不足，最小结构为：

- 复用现有 target/friend representation；
- 一个小 MLP capturability head；
- 输入 coalition pooled features + target feature + dynamics context；
- 输出 terminal-within-\(H\) logit；
- 独立 calibration/evaluation；
- 不复制整个 Transformer。

### 15.8 暂不采用的复杂方案

- 大型独立 \(C(S,j)\) Transformer：参数与归因成本高；
- online HJI：计算成本与当前工程阶段不匹配；
- Sinkhorn：现阶段 greedy 足以验证语义；
- learned Lagrange multiplier：固定 \(\lambda\) 尚未验证；
- PFSP：战略鲁棒性不是当前唯一瓶颈；
- CBF：安全问题可独立处理，不与 novelty 一次性耦合；
- MoE：角色本已动态，不需要先引入专家路由。

---

## 16. Paper Narrative v1（待证据支持）

### 16.1 Problem statement

候选问题表述：

> 在局部且可能间歇的目标信息下，一支同质、可互换的 UAV swarm 持续执行区域覆盖。当移动目标出现时，系统需要从同一资源池动态借调足够但不过量的 UAV，形成目标特定捕获联盟；完成 terminal capture 后及时释放资源并恢复事前覆盖能力，同时应对后续重复到达。

### 16.2 候选 contributions

若 M2-lite 与完整实验均成立，可写成三条：

1. 提出一个面向同一 fungible coverage pool 的 coalition recruitment formulation，用联盟可捕获性的边际增益而非固定 K/nearest rule 决定支援。
2. 将边际捕获增益与 agent-specific coverage opportunity cost 显式耦合，在局部信息与多目标资源约束下实现 target-wise allocation。
3. 提出并评估 terminal capture→release→same-agent coverage recovery→repeated arrival 的完整生命周期及其 coverage-debt/recovery 指标。

若只完成 M0，第一、二条不能写；论文会更偏 system/lifecycle integration。

### 16.3 Strongest competitors

主比较应优先覆盖：

- KS-COAL：overall lifecycle/real capture/hardware；
- P0042：learning-based target preference/fixed capacity allocation；
- P0034：local FOV/loss-reacquisition；
- TERL：Transformer/scaling/target selection；
- P0048：same-pool patrol-pursuit-return/repeated arrivals；
- OPEN：physical multi-UAV control；
- AgilePE：strategic learned opponent。

### 16.4 不能说的 novelty

不能说：

- first local-sensing pursuit；
- first decentralized multi-UAV pursuit；
- first dynamic coalition；
- first adaptive coalition size；
- first capturability-aware assignment；
- first coverage + pursuit/capture；
- first release/reuse/repeated arrivals；
- first same-agent patrol/pursuit/return；
- first Transformer/GNN swarm capture；
- first real UAV capture；
- first learned evader/self-play。

### 16.5 更准确的非 “first” wording

推荐模板：

> We study a previously under-examined intersection of persistent coverage, locally triggered coalition capture, and quantified post-capture recovery in a single fungible UAV pool.

中文：

> 本文研究一个以往较少被系统联合刻画的交叉问题：同一可互换 UAV 资源池中的持续覆盖、局部触发的联盟捕获，以及可量化的捕获后覆盖恢复。

若 M2 成立，可更具体：

> Our key distinction is an explicit marginal coupling between coalition capture feasibility and agent-specific coverage opportunity cost, rather than a claim that any individual component is new.

中文：

> 本文的关键区别不在于声称任一单独组件首次出现，而在于显式耦合联盟捕获可行性的边际增益与 agent-specific 覆盖机会成本。

### 16.6 最关键实验

最关键的不是最大网络对比，而是：

1. current gate vs M1 vs M2-lite；
2. fixed K/nearest baseline；
3. one-shot vs full lifecycle vs repeated arrivals；
4. global/shared/local information；
5. fixed-role pool vs fungible pool；
6. heuristic、separately trained、held-out learned evader；
7. capture 指标与 coverage debt/recovery 指标同时报告。

---

## 17. 当前 narrative 与实现冲突清单

1. **叙事说 persistent coverage 后出现 target；当前 mixed target 在 reset 已存在。**
2. **叙事说 repeated arrivals；当前没有 same-episode arrival scheduler。**
3. **叙事说 target-wise resource allocation；当前正式任务 4v1 且没有 \(z_{ij}\)。**
4. **叙事说 capturability-informed；当前只有 rule/reward/geometry ingredients，没有 \(C(S,j)\)。**
5. **叙事说 local/intermittent information；当前 actor 目标信息局部，但无 temporal belief，且初始探测相对容易。**
6. **叙事说 coverage recovery；当前有 strict post-capture CE gate，但缺少 event-relative baseline/debt 与 repeated-event evidence。**
7. **叙事说 fungible；架构同质，但需 occupancy/transition 证明行为没有固化角色。**
8. **叙事可能暗示 no-escape；当前 loose/stationary capture 是环境终端规则，不是鲁棒对抗证书。**
9. **叙事可能引用 learned gate；当前 active PPO 路径没有使用历史 IQN gate head。**
10. **叙事可能把 central critic 当 execution information；CTDE 必须明确区分。**

这些冲突不是“项目失败”，而是从训练原型走向论文问题时必须补齐或收窄声明的边界。

---

## 18. 风险、反例与待负责人裁决的问题

### 18.1 方法风险

- \(C_{\mathrm{geo}}\) 可能只是 terminal rule 的平滑复写，泛化不到不同 evader；
- \(\Delta_i\) 对 noisy/intermittent target state 敏感；
- coverage counterfactual 若依赖全局 Voronoi，可能破坏 decentralized execution 叙事；
- greedy assignment 可能在多个 target 间产生短视；
- frequent role switching 可能引发 chattering；
- terminal event 后 release 太快可能破坏安全；
- repeated arrivals 可能把训练从稀疏 capture 变成更稀疏长 horizon。

### 18.2 需要先裁决的定义

1. 最终论文最低场景是 4v1 lifecycle，还是必须 8v2/多目标？
2. \(C\) 要叫 geometry score、capture feasibility，还是经校准的 probability？
3. execution 允许传播哪些信息：request id、强度、belief summary、坐标？
4. assignment 是 centralized coordinator、distributed auction，还是局部 greedy？
5. coverage 主指标用 CE RMS、integral density error、area CV，还是组合？
6. stationary capture 是否与 normal geometric capture 同等计入主 success？
7. repeated arrival 的时间分布、目标数量和是否允许恢复完成后才到达？
8. M2-lite 是主线，还是 M0 lifecycle 已足够形成论文？
9. coverage cost 是否必须 agent-specific counterfactual，还是 local cell importance 足够？
10. 是否接受“局部执行、全局训练/评价”的 CTDE 边界？

### 18.3 证伪标准

以下结果应迫使我们放弃或收窄 M2 主线：

- M1/M2 的收益完全由平均 coalition size 增大解释；
- M2 不优于简单 nearest/fixed K；
- coverage debt 降低但 capture success 显著崩溃；
- \(C_{\mathrm{geo}}\) 在 held-out evader 上不相关或反相关；
- local information 下 allocation 不稳定，只在 global target 下有效；
- repeated arrivals 中恢复长期漂移；
- fungible pool 不优于等预算 fixed partition；
- learned \(C\) calibration 很差且不能改善 decision regret。

---

## 19. 审阅顺序与 5–10 分钟动作

本轮完成后唯一建议的 5–10 分钟动作是 **审阅，不实施**：

1. 阅读第 2 节的一页式结论；
2. 检查第 17 节十条“叙事—实现冲突”是否准确；
3. 在第 18.2 节十个问题中标出最先需要裁决的 3 个；
4. 暂不批准任何 reward、observation、network 或 lifecycle 改动。

如果第 17 节事实有误，应先修正文档；如果事实正确但 M2 的命名或数学语义不被接受，应先改 method object，而不是开始编码。

---

## 20. 未来可交给 Codex 的 implementation prompt（封存，审核通过后才使用）

> 前置条件：负责人已明确批准具体范围；未批准时不得执行本 prompt。
>
> 在 CoCap 最新实验分支上，只实施已批准的低风险 lifecycle metric plumbing，不改变 observation、action、reward、actor、critic、role rule、capture rule、训练合同或 active run。先读取最新 ledger/contract/runtime config 和现有 evaluator，保留所有当前语义。为每个 episode/event 记录 arrival、terminal capture reason、release、recovery、per-agent effective role、target binding、direct visibility、CE RMS/max、coalition size、collision/boundary，并离线计算 pre-event baseline、peak coverage deficit、integrated coverage debt、borrowed agent-seconds、capture success/time、capture→release latency、release→recovery time、recovered error ratio、nonparticipant continuity、recovery-before-next-arrival、repeated-arrival degradation、role occupancy/transition。若当前环境不存在 arrival 或 repeated arrival，则字段明确为 NA，不伪造事件。增加最小 contract tests，生成一个只读 report，对旧 artifact 尽可能向后兼容。不要实现 M1/M2，不新增 capturability network，不启动训练。完成后给出 diff、测试和未定义字段清单，等待负责人审核。

---

## 21. 最终判断

CoCap 当前最宝贵的资产不是一个尚不存在的 \(C(S,j)\) network，而是已经形成的：

- 同质 shared-policy swarm；
- 局部 target visibility；
- capture/support/coverage 动态状态；
- terminal event；
- release 与 post-capture CE recovery；
- 对执行合同和 evaluator 的严格追踪。

文献地图指出的机会，是把这些已有资产从：

> reward-driven role switching

提升为：

> semantically interpretable, resource-constrained coalition recruitment across a measurable coverage–capture–recovery lifecycle.

最小理论路径是：

\[
\text{M0 lifecycle evidence}
\rightarrow
\text{M1 marginal capturability proxy}
\rightarrow
\text{M2-lite capture gain minus coverage cost}.
\]

但当前 capture consolidation 与执行合同仍在形成证据，尤其 Pure-Capture 的 100k evaluation 暴露了 phase assertion 问题。因此，本审查的结论是：**方法方向值得保留并深入理解，但不应立即介入当前训练主线；先完成事实审核、定义裁决与低风险 metric 设计，再决定是否实施。**
