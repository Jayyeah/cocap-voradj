# CoCap 传统分层围捕定向复核（Round 2）

日期：2026-10-17。范围：文献、机制、代码与移植可行性；未实现算法、未训练、未运行评估。本文继承第一轮的12个候选和稳定ID，不另建冲突台账。程序事实源仍是 [`baseline_registry.json`](../../artifacts/baselines/baseline_registry.json)，本轮字段为 `focused_review_v2`；完整出处及访问边界见[证据表](BASELINE_RELATED_WORK_EVIDENCE_ZH.md)，代码见[开源审计](TRADITIONAL_BASELINE_OPENSOURCE_AUDIT_ZH.md)，工程判断见[可行性](TRADITIONAL_BASELINE_REPRODUCTION_FEASIBILITY_ZH.md)。

## 1. 决策

**若只允许复现一个传统方法，优先 Rao 等的 Dynamic Robust Pursuit（TR-11，JAS 2026），进入有条件的 preflight。**它有已取得的正式全文与补充推导，包含动态多目标分组、每目标多人、几何槽位和安全控制，复建依据比当前未取得全文的MK-WTA完整。尚未找到完整公开实现，不宜直接宣布可复现成功。

| 科学用途 | 首选 | 次选/储备 | 判断 |
|---|---|---|---|
| Global Strong / Privileged Comparator | TR-11：A*分组＋拍卖＋C3AVC＋CLF–CBF | TR-13：CEO-DMPC；TR-01另作补全文储备 | 前两者均 `CONDITIONAL_GO`；TR-11先做合同与动力学可行性核查，TR-01先补全文 |
| Local Distributed，任务组织最接近 | TR-07：KS-COAL的非学习骨架 | TR-10：GB-VCE局部观测/角色/环绕组件 | 没有一项被核实为完整、信息匹配的CoCap传统baseline；仅对明确命名的适配研究 `CONDITIONAL_GO` |
| 分布式多目标分配与围捕储备 | TR-06：DMK-WTA＋PPM（TSMC 2024） | — | 原文未取得；distributed不等于局部未知目标搜索，不能越级列为matched-local |
| Simple Heuristic | TR-05：Local Spread＋Greedy Coalition＋Geometric Surround | — | `GO`：只表示可进入设计preflight；属于项目自定义辅助对照 |
| RL | TERL必选，分Global与Local-Adapted | 固定Scout/Pursuer RL优先Related Work | 不增加RL训练名单 |

原先“最强传统候选＝MK-WTA”的结论更新为：**任务直接性仍强，但因全文证据缺口排在已取得完整方法的TR-11/TR-13之后**。不存在因不开源而忽略它的情况。所有GO均是下一阶段建议，不是本轮启动命令。

## 2. 恢复的事实与合同

远端 `docs/baseline-screening-20261016` 重新fetch后仍为 `94753f9fd82d137af7ff9930634788ffeba6f3e2`；历史依次为94753f9、fb70b53（候选检索）、a5814f4（历史release）、e851e17。已完整阅读第一轮master、cards、registry、search_log。新分支从94753f9建立。

另fetch `docs/20261016-iqn-z-solid-asset-audit`，HEAD为 `3038ef5cd728acdd302772ca08bf10060685af16`。已读其中Benchmark Contract、Asset Index和Readiness。基线分支的a5814f4环境不是最新Z05环境，不能从该旧分支直接跑出“Z05合同”结果。精确来源链接见证据表B01/B02。

合同目前仍标为草案 `cocap-classic-z05-draft-v1`，不能写成已由作者冻结：

- 120×120 m；原生12P/3E/3O，初始化、障碍和seed沿用同一manifest。
- 决策0.5 s，物理0.05 s；AW9 unicycle：a∈{−0.4,0,0.4}，ω∈{−π/6,0,π/6}；pursuer速度上限3 m/s、evader 3.5 m/s，evader用原v2_fixed APF。
- NormSense V2按有效面积与active人数计算半径；目标/障碍表面距离过滤。友机Voronoi邻接由全队位置计算后提供有限token，论文需披露该几何依赖；它不等于全队广播敌人坐标。
- **normal capture**：至少3架active pursuer中心距≤8 m，排序后的环形最大角间隙≤π，最大/最小间隙≤3；transition即时移除目标，无额外hold。stationary fallback单列。
- synchronized_swept_v1碰撞；3000决策步；普通回合不自动刷新目标；捕获后最多700步恢复，strict CE含人数门槛及30步hold。CE是几何均衡，不能等同地图发现完成。
- 重复到达严格A/B/C是单列评估协议；相同初始world seed和刷新RNG不保证后续波坐标逐点相同。不能让传统方法依赖CoCap专属Z变量触发到达：未来比较须冻结共同、外生到达时序，或预注册各方法通用的触发事件映射，并与历史Z05原协议结果分表。

IQN-Z审计确认最终Z05为Stage3/600k，SHA256 `8ee5c162c32883984f72aa4be4b86e82338ae1e8d8c912011d181987a476d095`。旧Local/Global 50×3仍缺corrected CE时间和运行源SHA，不拿其数字替传统方法预设结论；本轮也不补跑。

## 3. 机制逐项复核

以下“未证实”与“不支持”分开：前者表示访问或合同证据不足；后者只用于已读正文明确超出范围的功能。公式定位与PDF版本见[证据表](BASELINE_RELATED_WORK_EVIDENCE_ZH.md)。

### TR-11（新增首选）— Dynamic Robust Pursuit，JAS 2026

完整题名、六名作者、DOI与正式PDF见S11。**已发表**，13(4):989–991，2026年4月；不能因DOI含2025将年份写成2025。已读全文3页及补充2页。

1. **搜索**：没有无目标coverage、frontier或自主扩散协议。不能把避障路径规划叫任务搜索。
2. **信息**：分组要全部evader位置作固定聚类中心、全队分组人数/均值以及A*路径代价。没有局部发现/发现传播合同；适合全局信息组。理论称分布式，拍卖交换报价；实物由动捕与host workstation计算控制，不是去中心硬件证据（p.991）。地图/路径图粒度需另登记。
3. **分配**：位置固定中心的改进K-means式分组，超额组移走远离组均值的个体，设禁止立即返回的迭代窗；组内以A*长度构造收益，分布式拍卖分配f边形顶点（p.990）。人数约束有下限r，仿真r=3；动态分配迭代不等于证实“捕获移除后自动重分组”，更新频率和停止规则需preflight锁定。
4. **围捕**：目标周围多边形槽位属于真实几何围捕意图；C3AVC为概率碰撞约束Voronoi安全域，CLF/CBF约束的LP决定控制（pp.990–991，补充S2–S5）。论文成功规则仍主要是人数、距离与安全，不能直接替代CoCap角间隙判据。稳态等角f≥3在几何上可满足角度条件；瞬态、障碍、饱和时没有该合同保证。
5. **连续任务**：没有核实目标移除、新目标到达、完成后解散与搜索恢复。必须明示增加事件接口与搜索层；不得称其为原论文完整生命周期。

**代价**：补充S4给C3AVC构造关于机器人S与障碍K的O(S+K)分析，依赖低维几何条件，**不包含**分组、A*、拍卖和控制LP。若网格有V节点、E边，一次常规堆式最短路规划可按O((V+E)log V)作实现预算；全配对代价数随pursuer×evader及槽位增加，这是本评审预算推导，不是论文整框架定理。报价消息数、迭代次数和payload均需测量。无依据声称其比CoCap慢。

**证据上的优势**：近期正式发表、多目标、可设≥3机、几何目标与显式安全优化、完整方法和补充。**主要风险**：等运动能力的一阶积分器→有惯性AW9且追捕者更慢；LP可行性/量化后安全；全局路径图；仅4追捕者/2目标的实物演示不能当三机/目标实物验证。复现成本：高；移植后重新建立理论保证：很高。

### TR-01 — MK-WTA＋OAPM，RAS 2025

S01正式元数据已复核。重新打开原始出版社页面，尝试正式PDF、作者公开来源和出版社全文API；本轮**未取得完整正文**。可读材料含Introduction、章节片段与Conclusion，不能填成全文已核。

- 搜索、捕获后恢复、新目标到达：均未证实。动态环境指移动evader/未知障碍，不能据此推出多波服务。
- k-WTA将输入竞争为k个赢家；MK-WTA扩展到多目标。文章公开材料确认每目标选k名和OAPM围捕避障，但**无法据此确定MK-WTA运行在中央、各节点还是分布式网络**。
- 第一轮“全体pursuer–target距离矩阵”是待核线索，本轮未取得含该矩阵的正文公式，故降为 `UNVERIFIED_ROUND1_LEAD`，不能作为已证实事实继续传播。即使存在逻辑全局代价矩阵，也不自动意味着单中央服务器。
- OAPM完整吸引/排斥/障碍控制律、参数、控制饱和、数值积分、复杂度、通信、capture-triggered再分配都保持未知。**不补写通用APF公式冒充OAPM**。
- 文献明确多目标、k人分配和surround意图，值得保留Global Strong储备；三机角间隙是否可达尚不能判定。官方代码/软件/算法补充未找到；实物实验和CRediT Software不等于源码公开。

`CONDITIONAL_GO`的先决条件是取得§2–5及参数/补充，逐式重建分配和OAPM后再谈CoCap接口。此时不能给出一个假装完整的重建方案。

### TR-02 — Potential-game multi-target encirclement，CCC 2023

S02题名和五名作者、3778–3785、DOI已由Crossref/正式会议目录核对。不是2025/2026期刊论文。原文入口已重开，但未取得8页正文。

公开摘要只支持“DBSCAN聚类＋potential-game分配＋耦合环绕/跟踪制导”。聚类对象可能是目标群，不能直接认定每个独立目标均分配≥3机。utility、potential function、Nash迭代、人数限制、通信边、中央/分布式、全局矩阵、重求解频率及控制律均待正文。Nash equilibrium亦不自动等于全局最优。无目标搜索/释放/到达流程未知。代码 `NOT_FOUND`。`NO_GO`当前完整复现；保留related-work与补全文候选，不以摘要补公式。

### TR-03 — ACC 2018 detector-leader coalition（原ID保留）

S03作者PDF已取得。§II–III（PDF pp.2–5）关注未知任务发现、探测者leader、资源与协作信用、merge/split及目标信息上报地面站。它有发现后临时联盟，不是已证明的无人中央全生命周期；向地面站报告与由中央分配也应分别判断。未定义目标周围角度/半径控制或三机终端条件；资源人数由任务需求决定，不是k=3围捕。没有可核的coverage控制/normal capture移除→解散→再搜索协议。缺围捕层与搜索层，不能称完整baseline。`NO_GO`完整复现，分配组件参考。

### TR-04 — Distributed RMPC encirclement，RAS 2025

S04正式论文重新打开，可读§2–6片段；完整2025正文未取得。另取得同作者**2021 IROS前身**全文S04a，分别标注，不能拿前身公式假称2025公式。

- 2025明确N对1、无界二维域、从已有包围出发维持encirclement并趋近捕获；没有任务搜索、多目标分配或释放再搜索。
- 中央双线性RMPC经EGSS几何扇区内近似分解为局部Tube MPC。evader输入作为有界扰动，不要求知道未来动作；“robust”不是未来轨迹oracle。
- 前身p.2要求初始包围、使用全体状态观测；pp.4–6构建几何扇区后局部控制，通信需求不能由“distributed”一词省略。前身需至少4个扇区，不能直接承诺三机方案；2025是否放宽需全文确认。
- 前身凸包约束可写为p_e=Σλ_i p_i，λ_i≥0、Σλ_i=1；它保证包围，不保证CoCap角间隙比、8 m及至少3机同时满足。MPC开销还取决于horizon/约束/求解器，无实测排名。

适合作已包围单目标的几何控制组件；作为CoCap端到端baseline `NO_GO`。求解器依赖保持未核，不把2026其他论文的Mosek误移到此处。

### TR-05 — 项目自定义辅助启发式

原条目从“完整支持”改为**设计目标，尚未实现/验证**。无文献官方实现。至少需可测模块：有界地图coverage/spread、表面距离发现、邻接消息与TTL、多人容量约束greedy/auction与tie-break、角度槽位/运动控制、避障/碰撞、normal-capture事件释放、恢复搜索、外生新波任务。规则与调参预算先登记，validation seeds和正式评估隔离；不以一次性最近目标规则制造弱对照。见可行性文档。

### TR-06（新增）— DMK-WTA＋PPM，TSMC 2024

S06正式出版社摘要确认分布式竞争分配＋predator-pack encirclement，碰撞/障碍处理，多机器人实物；与TR-01不同论文，不把两篇算法名互换。全文未取得，不能确认局部未知目标搜索、全局代价需求、每目标人数/上限、重新分配频率和PPM公式。`CONDITIONAL_GO`仅补证preflight；不是已经合格的Local Distributed完整方法。

### TR-07（新增，关联TR-03）— KS-COAL / Accelerated KS-COAL，RA-L 2024

S07已核正式9(1):443–450，DOI含2023但卷期为2024。已读arXiv v3全文8页和v1中任务运动细节（S07a）。需要严格分开两个版本：v3把运动策略/utility细节移至补充，v1仅作可追溯早期设计，不自动填回正式算法。

1. **搜索**：v3 §III-A p.3指向区域划分、minimum-time coverage；v1 §III-A pp.3–4明确子区sweep，探索收益χ除以预计耗时τ，包含区域未访问及目标/资源密度。
2. **信息**：v3 §II-A p.2：SWAT有限距离通信、图需无限次连通；SCOUT内部通信无距离限制，TARGET同样如此。v1 p.3还写发现位置广播给全部SWAT。故它是无中央协调器的分布式任务算法，但不是CoCap同通信预算。
3. **联盟**：v3 §III-B pp.3–5：不相交coalition、均值utility最大化；通过邻接链式switch、消息buffer和最佳方案共识寻找K-serial稳定解。K=1退化为单机稳定，K增加扩大协同搜索；非学习KS-COAL可独立使用。完整Accelerated版本含HGAN监督学习加速器，不能整体贴“纯传统”标签。
4. **捕获/环绕**：v1 pp.3–4的追捕者是pure pursuit，Apollonius交集用于目标逃逸选择；资源防御是在静态资源簇外围等间隔站位。v3概括capture使用Apollonius策略，但细则在未取得的正式补充。**资源围圈不能移花接木成移动目标三机围捕**。
5. **连续任务**：新资源持续生成、捕获目标immobilized；v1资源防御定时完成释放，动态任务重新组织。没有证实新evader重复到达及CoCap normal-capture后coverage恢复。它超越“只有联盟”，有运动模块；仍缺本任务捕获几何与信息合同。

v3 §III-D p.5给每机每轮O(NΠ(2Nr+Nω))，buffer上界(NωNr)^kmax，总预算还依赖Δρ/ε与N。每轮最多NΠ条发给Nr邻居，payload含assignment与链，不是固定一个标量。可用它研究计算/通信—解质量折中，不能预设一定实时优于/劣于CoCap。

**最接近CoCap的是它的持续任务组织故事，而不是它的捕获控制。**建议非学习骨架作为Local任务分配适配的第一参考，明确名为 `KS-COAL-CoCap-Adapted`；补搜索/三机controller/事件层后是组合适配，非原论文官方复现。工程难度高至很高。

### TR-08（新增）— Distance-based multiple-target encirclement，TCST 2025（2024预印本）

S08 PDF pp.2–4 §II：N=2M，通信半径≥两倍测量半径；初始改进CBAA在观测目标中分配两机，assignment matrix可有M×N规模，局部通信不意味着固定小payload。pp.5–6 §III：距离状态估计＋Kalman滤波、antisynchronization吸引项＋经截断的互斥/障碍伪力；u_i=Γ_i−2v_i/t。物理意义是两机位于目标两侧，并非三机封闭角度几何。无持续搜索/完成释放。末轮Crossref精确题名检索确认正式发表为IEEE TCST 33(1):261–273，2025，DOI见S08；不再仅标预印本，阅读页码仍来自arXiv稿。`NO_GO`CoCap完整baseline；保留range-only估计/拍卖组件参考。

### TR-09（新增）— Triggered Aggregative Tracking Feedback，TRO 2026

S09 Crossref确认正式TRO 42:2596–2610，DOI `10.1109/TRO.2026.3686280`；机制阅读来自2024 arXiv版，页码不能冒充期刊排版。

arXiv §II–III pp.3–7：稳定低层动力学上优化Σℓ_i(x_i,σ(x))，σ为平均聚合量；使用本地梯度及两组跟踪变量，经事件触发邻居交换估计聚合项。假设固定强连通、weight-balanced图；每次向邻居交换2d个float。它是已知单目标的团队几何/多兴趣点monitoring权衡，不是未知移动目标搜索分配；收敛到stationary point不等于捕获成功。无多目标联盟和捕获移除。ROS2/Webots/ground+aerial实物很有工程参考价值，但框架开源不能当该论文完整代码。`NO_GO`完整baseline。

### TR-10（新增）— GB-VCE，Biomimetics 2026

S10正式PDF §4 pp.8–12已读：失去观测时常速度短期预测，过期回全局search center；searcher绕参考点周期搜索；leader/relay/searcher/encircler由平滑增长证据切换；环绕者追踪圆周等角槽位，叠加切向、速度前馈、连接与避碰项。它比纯tracking具备更明确的感知恢复流程。

但§4 Alg.1–2 pp.10–11显式输入全体观测和视觉邻接矩阵，计算全队可达比例Γ、argmax和TopK。论文称分布式，**没有据此核实异步、断连下这些全局reduction的消息协议**；观测有效时公式直接使用精确目标位置/速度。需登记估计误差/延迟。§7 pp.23–24明确单目标、仿真验证，多目标留待未来；没有任务完成移除/释放再搜索。代码 `NOT_FOUND`；附CSV是结果数据。

可作Local几何/角色适配第二参考，不能当完整匹配方法。其角色组织规模、排序、slot编号需要额外分布式实现，不能以集中全局数组实现后仍称local-only。CoCap移植很高难度，优先级低于KS-COAL骨架。

### TR-12（新增线索）— Hierarchical PGM，Advanced Intelligent Systems 2025

S12元数据表明为2025正式工作；local density/SPH＋层级概率图感知到动作映射值得跟踪。但全文端点403、无作者代码链接核实，不能判断是否触碰捕获、局部搜索或三机角度约束。保留线索，`NO_GO`当前复现，不因摘要“self-organized”而提升排名。

### TR-13（末轮补查，Global第二备选）— CEO-DMPC，RA-L 2024

S13作者机构完整8页PDF已取得。§III–VI，PDF pp.2–6：无目标时沿给定半径圆周PID巡逻，发现后进入追捕；这不是自由地图自主扩散。分配以两次二元MILP求解：先将每目标分到n_req名，再一对一分配正多边形槽位；实时更新，并构建leader–follower参考。CEO-DMPC对预测位置/速度、能量、输入变化和碰撞距离软惩罚作QP优化。原终端是winding-number多边形包围＋距离＋无碰撞，是实质几何围捕，但未限制CoCap角间隙比。

关键限制在p.6：追捕方假设**full real-time communication**；Alg.3还要求All information，预测融合使用队伍探测集。故原法可作为Global/通信特权比较，不归为matched-local。§II与结论所称varying numbers主要是不同初始配置，未证明在线新evader到达、捕获移除后释放与恢复巡逻。论文预测知道evader向保护区运动的意图，但不掌握其确切APF参数，不能称已知真实未来轨迹。

实现依赖可核为Python3、Quadprog QP和OR-Tools `pywraplp` MILP，Ubuntu20.04/Gazebo为物理仿真，**不是实物飞行验证**。整套源码未找到。复杂度需分开报整数分组、slot分配、各机2K维QP及预测通信；不能由“distributed”推定低通信。完整方法证据使其优先于暂缺全文的MK-WTA；但soft collision penalty无硬安全保证，double-integrator→AW9及移除/搜索扩展成本高至很高。`CONDITIONAL_GO`，作为TR-11的独立控制路线备选。

## 4. 比较设计：让传统方法有合理发挥空间

Global组允许全目标/全队/地图汇集；主要比较normal成功、完成时间、联盟利用率、碰撞与规模成本。原始信息版标 `privileged-global`；它可能优于CoCap，不作为局部发现公平性结论。TR-11虽含分布式拍卖，分组输入仍应按全局预算报告。

Local组必须逐项冻结：自身感知、邻居图、目标报告字段、hop/TTL、更新频率、丢包/延迟、缓存、任务消失确认。主要比较发现时间、最大等待接近时间、累计normal capture、三波完成/服务、捕获后恢复、计算和通信、规模与目标数。没有目标时提供有竞争力的coverage，捕获后重新分配和搜索，不用缺失模块人为削弱传统方法。

对所有方法：相同运动学、AW9动作、初始化、swept碰撞、normal capture与horizon；不把接触、stationary fallback或目标撞障碍计为normal。wall time包含感知预处理、Voronoi/路径图、分配、控制、推理；分别测单线程CPU与论文部署硬件。通信同时报条数、字节、hop、延迟和集中服务器流量，不能只报“零神经通信”。时间统计保留失败与删失，不用成功局均值证明更快。

## 5. 本轮证据边界

MK-WTA、CCC potential-game、DMK-WTA与2025 RMPC完整正文仍未取得。两篇Elsevier综述仅获得元数据和部分HTML正文，PDF页码未核；其技术判断不充当最终论文证据。已取得全文的资料均有下载URL、SHA256和页码记录。未发送作者邮件、未购买、未越过访问控制。

因此最终输出是**有边界的候选决策与可执行的preflight建议**，并非声称所有全文缺口已关闭。具体缺口、重建步骤与停止条件已经写入registry及后续合同。
