# CoCap Baseline Candidate Cards

> 历史Round 1候选卡。当前传统方法证据与排序见[2026-10-17定向复核](TRADITIONAL_BASELINE_FOCUSED_REVIEW_20261017_ZH.md)和[Shortlist V2](BASELINE_FINAL_SHORTLIST_V2_ZH.md)；原ID保留，未核线索不得作为已证实事实转引。

筛选日期：2026-10-16。论文元数据、任务假设和代码状态按公开来源记录；无法确认处明确标为“未知/推断”。`KEEP` 表示值得进入后续 preflight，不代表已经选定复现。

## RL 候选

### RL-01 — TERL

1. **论文信息**：Heng Zhang、Guoxiang Zhao、Xiaoqiang Ren，2025，*TERL: Large-Scale Multi-Target Encirclement Using Transformer-Enhanced Reinforcement Learning*，IROS 2025（作者主页列为 IROS 2025；预印本为 arXiv:2503.12395）。[arXiv](https://arxiv.org/abs/2503.12395) · [作者代码仓库](https://github.com/ApricityZ/TERL) · [IROS 2025 论文目录](https://www.proceedings.com/content/083/083023webtoc.pdf)
2. **类别**：RL；基于 IQN 的 distributional value learning，Transformer 关系编码与共享分布式执行策略。
3. **机制**：无固定 scout/pursuer 角色；有 learned target selection；没有显式 coalition formation 模块；没有独立区域探索模块。
4. **任务假设**：多目标、至少三 pursuer 可围一个目标；目标逐个围捕后，围捕者继续处理余下目标；团队和障碍感知按半径限制，但**每架 pursuer 可看到任意距离所有未围捕 evader 的状态**；执行时不做 agent-to-agent 通信，共享策略负责协同行为。
5. **开源状态**：已找到作者公开仓库 `ApricityZ/TERL`，含训练/比较脚本、配置、模型目录；仓库声明 MIT。属于官方实现。
6. **复现成本**：中。Python/PyTorch 自带模拟环境，仓库给出 Docker 入口；适配 CoCap 的动作、障碍物、传感和终止条件仍需成本。无论文专用闭源模拟器依赖。
7. **比较意义**：当前最直接的多目标 RL encirclement 对照，能检验 Transformer 关系推理与 learned target selection 的价值；比单目标 pursuit 更接近 CoCap 的围捕子任务。可列为强 RL 候选。
8. **风险/合同差异**：远距离 evader 状态等于全局目标清单，不能直接代表“局部发现后展开搜索”；2D 离散动作、目标 APF 行为、速度与捕获/围捕定义均不同。代码可用不等于能无改动公平对比。
9. **建议**：**KEEP，P0**。保留原论文输入版本作为多目标围捕参考；若加入 CoCap 主表，应先定义并命名受限观测适配版（例如 `TERL-Limited`），同时说明它是适配而非原论文设定复现。原始全目标可见版本可单列为 privileged-information 上界。

### RL-02 — Role-based MADDPG for Multi-Target Pursuit-Evasion-Exploration

1. **论文信息**：Maryam Kouzeghar、Youngbin Song、Malika Meghjani、Roland Bouffanais，2023，ICRA 2023，*Multi-Target Pursuit by a Decentralized Heterogeneous UAV Swarm using Deep Multi-Agent Reinforcement Learning*。[arXiv 全文](https://arxiv.org/abs/2303.01799) · [ICRA 2023 论文 PDF](https://applied-complexity.org/docs/ICRA-2023.pdf)
2. **类别**：RL / MADDPG，CTDE；集中 critic 仅训练期使用，执行期分布式。
3. **机制**：明确将 agent 分为**已发现目标追踪者**和**scout**；按角色采用不同 reward；scout 用 Voronoi cell reward 分散到未覆盖区域；目标发现后通过通信邻居传播位置；实验中对两目标安排 pursuer 子队。
4. **任务假设**：多目标、随机障碍、局部感知/邻居通信；假设每个 evader 至少对应一名 pursuer。论文目标是 tracking，成功攻击定义为与 evader 碰撞；未定义几何环绕。目标在被发现前可以未知，但不定义捕获后移除、恢复搜索、再分配的完整多轮协议。
5. **开源状态**：**未找到该论文的官方开源实现**。OpenAI MADDPG 是通用算法实现，不能当作这篇工作的 role reward、场景或训练管线代码。
6. **复现成本**：高。需要重建多角色 reward、MPE 场景、动态通信、学习型目标与 Crazyflie 实验代理；无官方配置/checkpoint 可对照。
7. **比较意义**：回答“固定角色能否把探索和追踪并行化”的最贴切论文候选。其 scout 角色和局部发现/邻居传播与 CoCap 的搜索—围捕切换形成有意义的结构差异。
8. **风险/合同差异**：碰撞即攻击成功，不是多人几何围捕；角色数量需预先指定；训练奖励允许角色异质；无可直接复用官方实现。
9. **建议**：**MAYBE，P1**。作为固定角色 RL 的研究参照保留；只有在较短的接口/算法 preflight 能锁定训练成本时，才考虑进入正式主表。

### RL-03 — OPEN（Online Planning for Multi-UAV Pursuit-Evasion）

1. **论文信息**：Jiayu Chen、Chao Yu、Guosheng Li、Wenhao Tang、Shilong Ji、Xinyi Yang、Botian Xu、Huazhong Yang、Yu Wang，2025，IEEE RA-L 10(8):8196–8203，*Online Planning for Multi-UAV Pursuit-Evasion in Unknown Environments Using Deep Reinforcement Learning*。[论文 DOI 页面](https://doi.org/10.1109/LRA.2025.3583620) · [官方仓库](https://github.com/thu-uav/Multi-UAV-pursuit-evasion)
2. **类别**：RL / MAPPO，参数共享 actor；部分可观测的多人协作追捕。
3. **机制**：无固定搜索角色；单目标追捕，无目标分配或 coalition module；Evader Prediction-Enhanced Network 预测已发现 evader；Adaptive Environment Generator 的“探索”是训练分布/课程生成，不是任务中的空间搜索策略。
4. **任务假设**：局部 evader detection 与局部障碍观测；多 pursuer 对**一个** evader；任一 drone 达到 capture radius 即全队得 capture 奖励；不要求几何环绕，不支持论文任务中的多目标/捕获后继续搜索。
5. **开源状态**：已找到作者组织 `thu-uav` 的官方公开 MIT 仓库。仓库含 OmniDrones/任务代码与训练脚本，但 issue 记录了 checkpoint、场景文件和实体部署脚本的缺项；官方代码存在但可直接复现实验仍需核查。
6. **复现成本**：高。依赖 Isaac Sim 2022.2.0、OmniDrones、特定 TorchRL/TensorDict 版本、子模块和 GPU；论文训练规模大。
7. **比较意义**：很好的局部感知、连续控制单目标追捕参照；可用于 CoCap 捕获前段，不足以单独衡量多目标连续任务效率。
8. **风险/合同差异**：一次只追一个目标；touch capture 容易低估几何合作要求；无人机六自由度 CTBR 动作与 CoCap 2D 动作不一致。
9. **建议**：**MAYBE，P2**。保留为单目标局部感知 RL 辅助候选，不列入最小正式 baseline 组合。

### RL-04 — DualCL

1. **论文信息**：Jiayu Chen、Guosheng Li、Chao Yu、Xinyi Yang、Botian Xu、Huazhong Yang、Yu Wang，2023/2024，arXiv:2312.12255（未见正式 venue），*A Dual Curriculum Learning Framework for Multi-UAV Pursuit-Evasion in Diverse Environments*。[arXiv](https://arxiv.org/abs/2312.12255) · [作者仓库](https://github.com/jiayu-ch15/DualCL)
2. **类别**：RL / 同质 UAV 共享策略；方法贡献重点是 dual curriculum，而不是任务分配。
3. **机制**：无固定角色；没有多目标 task allocation/coalition；训练期 intrinsic parameter curriculum 与 external environment generator 探索环境难度；任务策略追捕一个 evader。
4. **任务假设**：多 UAV 协同单目标捕获，障碍场景；论文以到达 capture radius 为成功，不是多人几何环绕；未建立多目标重复指派与捕获后恢复搜索。
5. **开源状态**：已找到作者公开 MIT 实现 `jiayu-ch15/DualCL`，基于 OmniDrones。
6. **复现成本**：高。仓库要求 Isaac Sim 2022.2.0、Python 3.7、OmniDrones 及固定版本第三方 submodules；与 CoCap 现有模拟器接口差异大。
7. **比较意义**：可对比单目标捕获和训练课程泛化，但任务覆盖不如 TERL。
8. **风险/合同差异**：把“环境采样探索”误称为“任务空间搜索”会误导；单目标及点捕获合同与 CoCap 不同；模拟器迁移成本很高。
9. **建议**：**MAYBE，P2**。保留参考，不建议当前进入正式最小组合。

### RL-05 — LIPPO（heterogeneous multi-target active interception）

1. **论文信息**：Yuxin Song、Hanning Chen，2026，*Designs* 10(2):26，*Active Interception for Multi-Target Encirclement by Heterogeneous UAVs: An LSTM-Enhanced Independent PPO Algorithm*。[开放论文](https://www.mdpi.com/2411-9660/10/2/26) · [DOI](https://doi.org/10.3390/designs10020026)
2. **类别**：RL / independent PPO-style actor–critic；各 UAV 独立学习，共享经验池；异构的是速度/能耗能力，不是 scout/pursuer 功能角色。
3. **机制**：每回合初始时用全局距离的 greedy 规则将每个目标分给 3 架 hunter；LSTM 预测目标未来位置供主动拦截；没有在线捕获后重分配或专门搜索角色。
4. **任务假设**：2D 多 hunter—多 target；几何成功条件为目标同时在三架 UAV 的范围内且落在三角形内；actor 只拿自身、近邻与自己被分配的 target，但 episode-start greedy assignment 要求全体 hunter 到未捕获 target 的距离矩阵。未明确捕获后重新派单/搜索。
5. **开源状态**：开放论文可得；精确论文/作者 GitHub 搜索未找到官方代码。不能把“论文开放获取”写成“代码开源”。
6. **复现成本**：高。2D 任务本身可实现，但需要重建异构能耗/速度、LSTM replay 与独立 PPO；没有官方仓库/checkpoint。
7. **比较意义**：较新的多人围捕/异构动力学对照，明确每目标三名 hunter，且有真实几何围捕标准；可与 CoCap 的几何捕获指标对齐。
8. **风险/合同差异**：目标在回合开始时全局已知且固定 assignment；异构资源能力会影响比较；没有 CoCap 的“未知目标—局部发现—捕获后扩散”闭环。
9. **建议**：**MAYBE，P1**。列为较新的异构 encirclement candidate；代码缺失和一次性全局指派限制了优先级。

### RL-06 — PRTDA（role-structured airship–UAV pursuit）

1. **论文信息**：Kejie Yang、Ming Zhu、Yifei Zhang，2026，*Drones* 10(4):248，*Role-Structured Multi-Agent Pursuit–Evasion with Potential Game Constraints for Heterogeneous Airship–UAV Systems*。[开放论文](https://www.mdpi.com/2504-446X/10/4/248) · [DOI](https://doi.org/10.3390/drones10040248)
2. **类别**：RL / hybrid CTDE；明确 airship guidance 与 UAV interceptor 的异构角色，带 assignment-aware centralized critic 和动态分配 curriculum。
3. **机制**：有角色与 assignment；有 pursuer team potential-game constraint；airship 提供区域覆盖引导/任务分配；没有 CoCap 式 scout agent 发现后在同一局部 swarm 内转成围捕。
4. **任务假设**：3D、多目标、airship 对 UAV swarm 有高层 guidance/assignment 支持；局部 UAV 执行；每目标需多 UAV capture。文中含 dynamic reassignment curriculum，但 agent 平台/信息架构与 CoCap 不同。
5. **开源状态**：开放期刊文章；本轮未找到作者官方代码仓库。
6. **复现成本**：高。需要 airship supervisor、3D dynamics、自定义 critic/role pipeline 和 Gazebo 验证链。
7. **比较意义**：2026 年角色结构与动态分配的近期学习型研究，值得追踪。
8. **风险/合同差异**：额外 airship 是强大的全局协调/coverage 资源，形成 CoCap 不具备的传感和决策能力；角色分层与局部 scout/pursuer 角色不是同一任务合同。
9. **建议**：**DROP（正式主表）/MAYBE（相关工作跟踪）**。特殊平台依赖导致直接比较不公平；保持文献跟踪即可。

### RL-07 — Ex-MADDPG（task-assignment component）

1. **论文信息**：Bo Liu、Shulei Wang、Qinghua Li、Xinyang Zhao、Yunqing Pan、Changhong Wang，2023，*Drones* 7(5):297，*Task Assignment of UAV Swarms Based on Deep Reinforcement Learning*。[期刊全文](https://www.mdpi.com/2504-446X/7/5/297) · [DOI](https://doi.org/10.3390/drones7050297)
2. **类别**：RL / Ex-MADDPG；局部通信、mean simulation observation、同步参数训练与多决策机制。
3. **机制**：没有固定搜索/追捕角色；显式 dynamic task assignment；目标要求至少两 UAV；目标检测后可通信；实验 formation 使用 Boids，不是论文明确给出的几何 encirclement controller。
4. **任务假设**：未知目标数、局部目标检测和通信范围；目标任务要求多人；支持动态 assignment/扩展任务规模，但不等价于完成一个目标后恢复搜索并再次围捕的完整 benchmark。
5. **开源状态**：本轮精确作者/论文仓库检索未找到官方代码；论文报告 MPE 与九机三目标实物实验。
6. **复现成本**：高。分配算法和 MPE 可实现，但需重建 mean observation、同步训练和任务环境；缺少直接可用代码。
7. **比较意义**：适合作为局部通信与多人 task allocation 的 RL 组件参照，任务信息合同比 TERL 更接近未知目标发现。
8. **风险/合同差异**：作者明确指出 formation 算法使用 Boids 且未详细呈现；它解决 assignment 而非 CoCap 的学习型围捕—搜索闭环。
9. **建议**：**MAYBE，P2（组件）**。除非正式论点突出 assignment 学习，否则不建议当完整端到端 baseline。

## 传统分层 / coalition / geometric 候选

### TR-01 — MK-WTA + obstacle-aware particle model (OAPM)

1. **论文信息**：Ning Tan、Zhenghui Cui、Yang Liu、Ruikun Hu、Bo Zhu、Tianjiang Hu，2025，*Robotics and Autonomous Systems* 191:104990，*A competition-based scheme using k-WTA networks for multi-target pursuit and encirclement in dynamic environments*。[期刊页](https://www.sciencedirect.com/science/article/pii/S0921889025000764) · [DOI](https://doi.org/10.1016/j.robot.2025.104990)
2. **类别**：传统分层控制：multi-target k-WTA task allocation + OAPM pursuit/encirclement and obstacle avoidance。
3. **机制**：无固定 scout/pursuer 角色；显式多目标 assignment；为每目标选择足额 pursuer 子集（论文结论说明可选出 k 名）；属于任务 coalition，但不是长期身份角色；无显式搜索策略。Assignment centralized/distributed：**公开可访问材料未能确认**；方法需处理全体 pursuer-target 距离矩阵，暂按全局/共享信息风险登记。
4. **任务假设**：多个 moving targets、每目标多个 pursuers、障碍可未知；论文用 pursuer-target 全部距离构造分配矩阵，assignment 所需全局/共享目标状态假设**未从可访问材料确认**，因此按疑似 privileged 信息登记。支持多目标并行追捕/环绕；未描述捕获后解散并持续搜索再分配。
5. **开源状态**：**未找到官方开源代码**；作者报道仿真与 differential-drive physical experiments。
6. **复现成本**：高。需要实现多目标 k-WTA、竞争通信/任务切换、OAPM 几何 controller、捕获和障碍动态；文章代码未公开，完整参数/接口可复现性有限。
7. **比较意义**：当前最强传统候选：直接同时覆盖多目标、每目标多人和 encirclement，且有物理实验；与端到端 RL 结构差异明确。
8. **风险/合同差异**：全局距离输入、任务发现模型、通信组织尚需查全文；论文强调并行 encircle，不证明未知目标空间扩散或捕获后搜索。
9. **建议**：**KEEP，P0（强传统候选）**。preflight 前必须查清全局信息/分配网络假设；主实验若需 matched information，应实现局部已知目标集合上的 assignment 变体，并把原始全状态版本作为 upper bound。

### TR-02 — DBSCAN + potential-game assignment + encirclement/tracking guidance

1. **论文信息**：Jiangbo Jia、Xin Chen、Weizhen Wang、Hongjin Liao、Guangyuan Zhu，2023，42nd Chinese Control Conference (CCC)，*Cooperative Control of Multi-UAV for Multi-Targets Encirclement and Tracking Based on Potential Game*，pp. 3778–3785。[CCC 2023 论文目录](https://www.proceedings.com/content/070/070536webtoc.pdf) · [DOI](https://doi.org/10.23919/CCC58697.2023.10240696)
2. **类别**：传统分层/博弈式 assignment + coupled encirclement and tracking guidance。
3. **机制**：无固定角色；DBSCAN 目标聚类；potential game task allocation；有目标到 UAV group 的协作分配和几何/耦合围捕跟踪；无显式 search 模块。Assignment centralized/distributed：**公开摘要未说明**；DBSCAN 至少需要共享的聚类数据范围，需拿全文确认。
4. **任务假设**：文章摘要明确 multi-UAV/multi-target；多个 UAV 协作追踪/围捕目标。DBSCAN 和潜在博弈需要的 target map、通信与 assignment 是否集中尚无法从公开摘要确定。支持目标动态追踪；capture 后再次搜索/重分配未知。
5. **开源状态**：**未找到官方开源实现**；本轮可得材料主要为摘要和会议索引。
6. **复现成本**：高。关键公式与 assignment detail 需取得全文；需实现聚类、潜在博弈与耦合制导。
7. **比较意义**：另一条真正“多目标分配 + 几何围捕”的传统路线，可与 k-WTA 形成独立备选。
8. **风险/合同差异**：缺少可核验的观测与重分配合同；没有 search/recovery 模块；会议论文和无代码增加复现不确定性。
9. **建议**：**MAYBE，P1**。在获得全文并确认信息假设后再与 TR-01 竞争传统 baseline 名额。

### TR-03 — Leader-follower coalition formation for UAV task allocation

1. **论文信息**：Fatemeh Afghah、Mohammad Zaeri-Amirani、Abolfazl Razi、Jacob Chakareski、Elizabeth S. Bentley，2018，American Control Conference (ACC)，*A Coalition Formation Approach to Coordinated Task Allocation in Heterogeneous UAV Networks*。[ACC 论文 PDF](https://fafghah.people.clemson.edu/Papers/ACC18.pdf) · [DOI](https://doi.org/10.23919/ACC.2018.8431278)
2. **类别**：传统 coalition/task-allocation component。
3. **机制**：有 leader/follower 临时角色；target first-discovery UAV 成为 leader 并挑选 follower 形成 coalition；使用 merge/split 与 cooperation credit；是 detector-leader 触发的分布式 coalition 形成，不是中央 assignment；不含 geometric encirclement controller，也没有显式的区域扩散 planner。
4. **任务假设**：目标位置/所需资源事先未知，首次发现触发 coalition；分布式 cooperative relay 将目标消息发给地面站。处理 adversary-target tasks，但论文没有证明每个 target 必须多人几何围捕或目标捕获判据。重复 coalition formation 可用于多个 task，但捕获后再搜索协议未给出。
5. **开源状态**：**未找到官方开源实现**；有作者公开论文 PDF。
6. **复现成本**：高。coalition logic 易实现但需定义资源需求、通信拓扑与稳定条件；还需外接 CoCap 几何围捕控制器才能成为端到端 baseline。
7. **比较意义**：作为“发现者发起的局部联盟”分配组件，和 CoCap 的共享策略自动召集支援形成有价值结构对照。
8. **风险/合同差异**：原任务偏检测、通信与资源任务完成，不是对抗 pursuit/encirclement；直接作为完整 baseline 会欠缺低层控制和 search performance 可比性。
9. **建议**：**MAYBE，P2（组件）**。可以借鉴作传统 coalition layer，但不能声称这是原论文完整复现。

### TR-04 — Distributed RMPC with encirclement guarantee

1. **论文信息**：Chen Wang、Hua Chen、Jia Pan、Wei Zhang，2025，*Robotics and Autonomous Systems* 192:105019，*Distributed cooperative pursuit with encirclement guarantee via robust model predictive control*。[期刊页](https://www.sciencedirect.com/science/article/pii/S0921889025001058) · [DOI](https://doi.org/10.1016/j.robot.2025.105019)
2. **类别**：传统分布式几何/优化 pursuit controller。
3. **机制**：无角色与 target assignment，也无 coalition allocation 层；不做多目标 assignment；显式在 RMPC 目标/约束中处理 encirclement 与 capture；无探索模块。
4. **任务假设**：多个 pursuer 对单个 target，论文重视通信约束并把问题分解到 pursuer；没有多目标匹配、捕获后恢复搜索或多轮 capture。
5. **开源状态**：**未找到官方实现代码**。
6. **复现成本**：高。RMPC 约束/求解器、在线优化和 communication decomposition 需要独立实现；本文未核实是否必须商业优化器。
7. **比较意义**：可作单目标几何控制的强低层参照，也可帮助传统 baseline 定义环绕约束。
8. **风险/合同差异**：本身不覆盖 CoCap 的搜索、多目标 assignment 与连续任务效率，因此不是完整端到端对照。
9. **建议**：**MAYBE，P2（控制组件）**。

### TR-05 — CoCap-matched local spread + greedy assignment + geometric encirclement (heuristic)

1. **论文信息**：项目定义的规范性启发式，不对应单篇论文。设计参考应明确引用 TR-01 的多人目标分配、TR-02 的分簇/围捕制导和 CoCap main 的局部感知约束；不得称为这些论文的复现。
2. **类别**：传统分层启发式。
3. **机制**：无永久角色；按本地可见/经邻居收到的目标状态做稳定的 nearest-target/auction greedy 分配；建议由邻居协商分布式执行，不用全局中央 assignment；每目标最低人数达到后选择角度槽位并执行几何环绕/APF 避障；无目标时执行确定性 coverage/spread；capture 后释放 coalition 并回到 spread，继续处理后续目标。若增加中央 assignment，只能作为独立 privileged 变体报告。
4. **任务假设**：不读取全局隐藏目标坐标；只使用 CoCap 合同允许的本地 sensing 与邻居消息；支持多目标、每目标多人和连续多轮；不使用学习型策略或 privileged critic。
5. **开源状态**：不存在单篇“官方代码”；需在冻结 baseline contract 后实现项目级 heuristic。其代码将由 CoCap 项目维护，不属于外部论文官方实现。
6. **复现成本**：低。规则和 assignment 需确定 tie-break、邻居同步、slot 数与搜索覆盖评价；无需训练或专有模拟器。
7. **比较意义**：低成本可解释下界，能区分 CoCap 是否只是靠一般几何环绕获益；匹配了持续任务闭环，是合适的辅助 baseline。
8. **风险/合同差异**：若规则调参过少会弱，调参过多则可能形成强工程基线；必须使用与 CoCap 相同训练/评估场景分布，并在独立 validation seeds 调参。
9. **建议**：**KEEP，P0（辅助下界）**。作为第三条辅助对照优先级高，但不替代强传统方法。

