# 全局分层方法：文献与综述证据 V4

核查日期：2026-10-17。PDF 页码从实际取得版本第 1 页起计；印刷页另列。下文是阅读摘要与推论，非原文引语。仅有元数据/摘要的资料不承担公式、通信或动态生命周期的技术证明。此前逐篇元数据继续见 [V2 证据](BASELINE_RELATED_WORK_EVIDENCE_ZH.md)和 [V3 综述与新方法](TRADITIONAL_SURVEYS_AND_RECENT_METHODS_V3_ZH.md)。

## 1. 新增/重读资料的正式身份

| ID | 完整标题与作者 | 正式发表及 DOI | 实际阅读/代码入口 |
|---|---|---|---|
| V4-M01 / TR-14 | **Adaptive Force Ratio Allocation for Multi-UAV Cooperative Multi-Target Encirclement**；Qiting Liu, Meixuan Li, Xianqiang Zhu | **Drones 10(6):406，2026-05-25**；[10.3390/drones10060406](https://doi.org/10.3390/drones10060406) | [正式 PDF，21 页](https://mdpi-res.com/d_attachment/drones/drones-10-00406/article_deploy/drones-10-00406.pdf)，重点 pp.5–6、9–11、13–14；[作者代码](https://gitee.com/lqt20/multi-uav) |
| V4-M02 | **Model Predictive Control for Cooperative Hunting in Obstacle Rich and Dynamic Environments**；Jacky Liao, Che Liu, Hugh H. T. Liu | **ICRA 2021:5089–5095**；[10.1109/ICRA48506.2021.9561054](https://doi.org/10.1109/ICRA48506.2021.9561054) | [作者 PDF，7 页](https://flight.utias.utoronto.ca/fsc/wp-content/uploads/2021/05/icra2021jl.pdf)，pp.2–5、7；完整官方代码未找到 |
| V4-M03 | **Cooperative Pursuit With Multi-Pursuer and One Faster Free-Moving Evader**；Xu Fang, Chen Wang, Lihua Xie, Jie Chen | **IEEE Transactions on Cybernetics 52(3):1405–1414，2022**；[10.1109/TCYB.2019.2958548](https://doi.org/10.1109/TCYB.2019.2958548)；online/预印本在 2020，不能用 DOI 的 2019 代替卷期年 | [arXiv:2001.04731v2，10 页](https://arxiv.org/pdf/2001.04731v2)，pp.2–4、6–8；正式版差异未逐式核；作者项目地址当前 404，代码未核 |
| V4-M04 | **Multi-Stage Multi-Robot Cooperative Hunting with Dynamic Alliance**；Jingbo Yang, Haoyu Wei, Lulu Pan, Haibin Shao, Dewei Li, Yuanlong Li, Liangren Shi | **ICUS 2024:1769–1774**；[10.1109/ICUS61736.2024.10840019](https://doi.org/10.1109/ICUS61736.2024.10840019) | [实验室出版记录](https://english.csc-lab.com/en/publication/yang-2024-multi/)及 Crossref；IEEE PDF 返回 418，全文未取得，无可核页码/公式；代码未找到 |
| V4-S01 | **A survey of the pursuit–evasion problem in swarm intelligence**；Zhenxin Mu, Jie Pan, Ziye Zhou, Junzhi Yu, Lu Cao | **Frontiers of Information Technology & Electronic Engineering 24(8):1093–1116，2023**；[10.1631/FITEE.2200590](https://doi.org/10.1631/FITEE.2200590) | [正式 PDF，24 页](https://www.fitee.zjujournals.com/rc-pub/front/front-article/download/37437903/lowqualitypdf/A%20survey%20of%20the%20pursuit%E2%80%93evasion%20problem%20in%20swarm%20intelligence.pdf)，pp.2–3、14、16–18；综述，不是可执行 baseline |

2024–2026 是重点，2021/2022 原论文用于核实集中控制、角几何和速度比，而非按年份排除经典理论。下载状态、原文件 SHA256 与代码定位见 [V4 manifest](../../artifacts/baselines/global_review_source_manifest_v4_20261017.json)。

## 2. TR-11：最佳完整论文强基线，中央变体需另标

Rao、Yan、Huang、Xu、Yang、Lv，**Dynamic Robust Pursuit of Multiple Evaders Under State Measurement Uncertainty in Obstacle Environments**，JAS **13(4):989–991，2026**，[DOI](https://doi.org/10.1109/JAS.2025.125684)。[正式正文 3 页](https://www.ieee-jas.net/article/exportPdf?id=9ab91428-5ba2-44cc-ae76-e5917191bfe1&language=en)、[补充 2 页](https://www.ieee-jas.net/fileZDHXBEN/journal/article/file/d5633e16-678c-4d80-8337-3da4303ecd23.pdf)已取得，本轮重读印刷 pp.990–991。

- **搜索/信息**：目标作为全局已知分组中心；A* 使用障碍信息。未核到未知目标持续搜索、捕获后恢复搜索或新目标到达实验。GE 仅给敌人真值并不自动满足其地图需求；原生已知地图实现应记作 GEM。
- **分组**：目标固定为中心，A* 距离构造 pursuer–target 代价；均衡每组 `floor(m/n)` 或多一人，资源满足每目标所需人数。过载组按距离调整成员并避免回移振荡。p.990 的 grouping 与 distributed auction 应分开：全局代价不等于整个原算法是单节点集中式。
- **槽位/控制**：每组围目标布置正多边形，以拍卖分配顶点。C3AVC 将机器人互避超平面、障碍分隔面和 Gaussian 不确定性 buffer 组成概率安全域，局部参考投影入域；CLF 引导参考收敛，CBF 约束安全边界，p.991 式(4)为带松弛/权重的线性规划。用一般意义表达：`dot(V)+c_l V` 控制下降，`dot(h)+c_b h` 控制安全裕度；实际重建须按补充 S2–S5 的不确定性、符号和输入变换，不能只套通用 CBF-QP。
- **验证/复杂度**：正文仿真含 7P2E、9P3E，每目标至少三机；实验为 4P2E，使用 host/mocap 计算与下发，不能声称硬件已验证每目标三机。原生求解用 Mosek。补充 S4 的安全域复杂度不包括 A*、分组、拍卖和 LP；不能作为全系统常数时延。
- **连续任务**：动态分配机制值得保留，但未据此确认目标移除、联盟释放和新目标进入的完整循环。CoCap 角间隙比、AW9 与生命周期均为适配接口，必须单列验证。

**为何仍排名最高**：同一篇正式论文整合多目标多人资源组织、几何目标、障碍与测量不确定性控制，比仅有联盟或单目标环绕更完整。完整作者代码 `NOT_FOUND` 不构成忽略其学术价值的理由；但原生控制和 AW9 重建难度高。`CONDITIONAL_GO`。

## 3. TR-13：最清楚的集中分组/槽位重建依据

Chen、Wu、Yang、Cao、Wang、Liu，**A Distributed Pipeline for Collaborative Pursuit in the Target Guarding Problem**，RA-L **9(3):2064–2071，2024**，[DOI](https://doi.org/10.1109/LRA.2024.3349977)，[作者 PDF，8 页](https://april.zju.edu.cn/wp-content/papercite-data/pdf/chen2024adp.pdf)。

- p.3 §III-B/IV：patrol/detect 后分组。式(7)最小化距离，`Σ_j y_ij≤1`、每目标 `Σ_i y_ij=n_req`，原文由所有 pursuer 求解；这依赖全体相关状态，并非仅邻域通信。p.6 §VI 明确 full real-time communication。
- p.4 式(8)：目标中心 `r_pol≤d_pursuit` 的正 `n_req` 边形，二元位置指派一对一。Algorithm 1 选择 leader、其他机器人参考 leader，构造位置/速度预测。
- pp.4–5 §V：位置/速度二阶离散预测 `x[k+1]=Ax[k]+Bu[k]`；式(13)代价包含位置、速度、控制消耗、控制变化与避碰项，采用 CEO-DMPC。软避碰代价不等于硬碰撞保证；预测也不是偷读目标未来真实动作。
- pp.6–7 实验/仿真：Python3，Quadprog、OR-Tools `pywraplp`，Ubuntu/Gazebo；有不同初始 evader 数与 3/4/5 人编队。不同初始数不等于新目标在线加入，Gazebo 不是实机测试。未核捕获后释放再巡逻的完整任务循环。

**本轮推论**：固定人数式(7)具有 min-cost-flow 结构，式(8)具有 LSAP 结构，因此分配层可低风险重建；中央只算一次是新执行架构。完整 CEO-DMPC 保留原 QP、模型与预测才是论文方法重建；只借用式(7)–(8)、换 AW9 控制的是 G-CAP。完整官方代码 `NOT_FOUND`；完整系统高至很高，分配层低至中。

## 4. TR-14 AFRA：真正的全局人数优化，新但有明确缺口

### 原论文实现的部分

pp.5–6 §3 的二维二阶动力学与 APF：`u_i=k_att(p_T−p_i)+Σu_sep+Σu_obs`；目标吸引收拢，机间排斥避免聚集，障碍排斥抵御接触。近场机间项随平方距离衰减，远场 spacing 随距离衰减，障碍项随表面间距平方衰减。原式(5)–(7)参数、饱和与积分应一起重建，不能只抄一个“APF”标签。

原生捕获：**至少四机进入 200 m 半径 + 最大角间隙严格小于 π**；没有 CoCap 的最大/最小间隙比。地图 5000 m 方形，边界反弹，UAV 最大速度 30 m/s，dt=0.5 s；不能直接缩放半径就当成相同动力学和碰撞合同。

pp.9–10 §4.1：用单目标 Monte Carlo 建立成功率经验库，按人数/速度比插值，采样点在高曲率/不确定区域加密。pp.10–11 §4.2：每目标从 `{0,4,…,15}` 选人数，预算约束下最大化预期捕获数。以本报告记号写递推：

`D(t+1,w)=max_{k∈K_(t+1), k≤w}[D(t,w−k)+p_(t+1)(k)]`。

`O(MN|K|)` 时间、`O(MN)` 原生内存。这个 DP 处理**全局人数预算**，不是机器人身份分组或几何槽位优化。正文明确目标间仅通过预算耦合；未显式建模跨群碰撞、目标协作或相关成败。预期捕获数不要求独立事件，但经验库的单目标边际概率在多人多目标相互干扰下可能失配。

pp.13–14 §5.2 是十目标、60→45 预算、部分目标速度比变化的三个阶段分配示例，允许弃服务目标。**不能将阶段重规划解读成已验证捕获移除、新目标到达及联盟释放。** 相同 CoCap 速度上限的目标未必需要按速度比分层，须验证收益是否存在。

### 官方代码与正文的差异

官方 commit `72440fd…` 的多目标路径没有执行角度判据、没有实际查询经验库 CSV 插值；另有参数/维度与速度比方向不一致，以及余量再分配可能破坏选项集的问题。详见[固定行号审计](GLOBAL_HIERARCHICAL_OPENSOURCE_AUDIT_V4_ZH.md)。因此代码为 `OFFICIAL_PARTIAL`；论文依然是有效的已发表分配候选，直接称“完整可复现 AFRA 基线”则 **NO_GO**。

## 5. 回溯原论文后应保留的几何边界

**V4-M02（ICRA 2021）**：p.2 §II 是单目标，已知目标/障碍的完整状态与行为；角位置用 Hungarian 指派。p.3 式(21)惩罚 `Σ_iΣ_j(r_i−r_j)²`，将本来局部径向/角向 LQR 控制耦合为集中问题。pp.3–5 使用 spacetime WHCA* 及 virtual agent，p.5 Algorithm 3 切换避碰、路径和 LQR；不是一套普通 QP 就完成全部环节。Crazyflie 实验可核，但多目标分组、释放和新任务不存在于已读方法中。可借鉴同步接近/障碍绕行，不能作为现成完整多目标候选。

**V4-M03（TCyb 2022；阅读 arXiv v2）**：p.3 式(5)给 occupied angle `θ_i=2 arcsin(V_i/V_e)`，p.4 式(12)用相邻角区差调节角度；Assumption 1 要目标位置和两侧最近方位邻居。p.3 Definition 1 的 capture 是**至少一台**到达 `d_c`，并非 CoCap 多人同时几何终端。p.7 Theorem 2 要求目标在初始 pursuer 多边形内和距离维护条件；连续速度控制与非完整 AW9 不同。

p.8 三机例要求每机 occupied angle ≥2π/3，对应速度比 ≥sin(π/3)≈0.866。CoCap 最大速度比 3/3.5≈0.857 低于此充分构造条件，故**不能引用其定理为 CoCap 三机任意初始状态捕获背书**；这也不证明 CoCap 三机捕获不可能，因为目标策略、有限空间和捕获定义不同。四机/更多人数及初始角度值得验证，不应机械固定三机。

## 6. 综述如何支撑架构，而非代替方法核查

| 来源 | 本轮使用的原文位置 | 可支持的限定表述 | 不能推出 |
|---|---|---|---|
| **V4-S01 Mu 等，FITEE 2023** | PDF pp.2–3 分类；p.14 §4.1；pp.16–17 §4.3/4.4；pp.17–18 §5.1–5.5 | 追逃有博弈、控制/AI、生物启发等路线；速度与转向共同影响能力；初始几何、缺口闭合与接近的权衡、未知速度与泛化值得评估 | 任意群体策略均适用 AW9；传统控制一定慢；单一成功判据可跨论文直接比较 |
| **V3-S01，TAMP survey，AIR 59:198，2026**，[PDF](https://link.springer.com/content/pdf/10.1007/s10462-026-11588-5.pdf) | pp.3–9 §§2–3，pp.24–27 §§5.3–6，沿用 V3 已读证据 | 离散任务层、连续运动层、执行反馈可以分层协调；架构与求解信息条件需分别登记 | 层级分解后各层最优就保证整体最优；动态移动围捕闭环已解决 |
| **V3-S03，A Tutorial on Distributed Optimization for Cooperative Robotics: From Setups and Algorithms to Toolboxes and Research Directions**，Andrea Testa、Guido Carnevale、Giuseppe Notarstefano；Proc. IEEE 113(1):40–65，2025，[DOI](https://doi.org/10.1109/JPROC.2025.3557698)，阅读 [arXiv v2](https://arxiv.org/pdf/2309.04257v2) | p.4 §II、pp.5–7 §III/IV；p.10 Alg.1；p.12 Alg.2；pp.15–16 §VIII | 中央与分布式优化可从状态需求、消息图和同步条件比较；重复单机抵达任务不等于多人围捕 | 分布式自动低通信/低 wall-clock，或中央自动不可扩展 |
| **Optimizing Coalition Formation Strategies for Scalable Multi-Robot Task Allocation: A Comprehensive Survey of Methods and Mechanisms**，Krishna Arjun、David Parlevliet、Hai Wang、Amirmehdi Yazdani；Robotics 14(7):93，2025，[DOI](https://doi.org/10.3390/robotics14070093)，[PDF](https://mdpi-res.com/d_attachment/robotics/robotics-14-00093/article_deploy/robotics-14-00093.pdf) | pp.3–4 §2.1、pp.7–8 §2.2、pp.19–20 §3.5/Table8、p.35 §5/6；沿用 V2 已读版本 | 联盟组织与执行控制是不同模块，规模/通信取决于机制与假设 | 普通 coalition 代码已实现围捕；所有固定需求指派都 NP-hard |

指定的 **A survey on target encirclement algorithms with UAVs, USVs and AUVs**（Neurocomputing 672:132687，2026，[DOI](https://doi.org/10.1016/j.neucom.2026.132687)）完整 PDF 仍未取得；**A Review of Reinforcement Learning Approaches for Pursuit-Evasion Games**（CJA 39(6):103940，2026，[DOI](https://doi.org/10.1016/j.cja.2025.103940)）已有 HTML 阅读但 PDF 页码未核。本轮保留 V2/V3 的缺口，不把它们写成已逐页核验。典型控制结构和通信条件用上表及完整原论文支撑。

综述 p.16 对 Fang 方法的初始角度表述较简化；本报告采用原论文 Theorem 2 的多边形和距离条件，不将综述归纳直接替换定理。相关工作应分别说清：人数组织、几何闭合、动态控制与任务循环，并按共同协议实测学习/传统方法的优势。

## 7. 原候选及新线索为何不提升

TR-01 MK-WTA/OAPM（RAS 2025，DOI [10.1016/j.robot.2025.104990](https://doi.org/10.1016/j.robot.2025.104990)）本轮重查作者页面/OpenAlex，未获得全文；全局距离矩阵、具体集中性、OAPM 完整律及完成后自动重分配仍未核。TR-02 potential game、TR-06 DMK-WTA、TR-12 PGM 保留 V2 的证据门槛；不能为满足本轮集中式主题而把它们改判为完整强基线。

TR-03/KS-COAL 是任务组织依据，TR-04 RMPC 是单目标几何/安全控制模块，TR-05 是项目自定义辅助启发式。V3 的 2026 whole-group encirclement、Ocean Engineering 2026 elliptical encircling 等新线索继续保留；围住整个目标群不自动满足逐目标至少三机 normal capture。V4-M04 即使标题含 Dynamic Alliance，也须先取得正文确认多人/多目标、中央性及控制，而不是用摘要填补。
