# V3：综述、近期方法与逐层证据

核查日：2026-10-17。PDF 页码均指实际取得文件从第一页开始的页码，不混同期刊印刷页。正式发表状态、阅读版本、代码能力分别核查。缓存及 SHA256 见 [V3 manifest](../../artifacts/baselines/hierarchical_review_source_manifest_v3_20261017.json)；Round 2 来源继续使用[上一轮证据表](BASELINE_RELATED_WORK_EVIDENCE_ZH.md)。

## 1. 从综述中提取什么，不能提取什么

本轮按架构需要组织检索：未知目标搜索→最低人数 coalition→任务与运动耦合→执行安全→连续任务。重点读取四类可获取全文的综述/教程，再沿引用回到原论文。没有以“综述称 distributed”替代通信审计，也不把某种一般理论保证转用为 CoCap 的实验结论。

### V3-S01：2026 TAMP 正式综述

Xi Tao, José-Fernán Martínez-Ortega, Néstor Lucas-Martínez, Mario San-Emeterio-de-la-Parte，**Multi-agent task and motion planning trends analysis: a survey**，Artificial Intelligence Review **59:198, 2026**，2026-05-23 online；[DOI 与正式 PDF](https://doi.org/10.1007/s10462-026-11588-5)。PDF 40 页；已读相关 §2、§3、§4 学习方法讨论、§5.3、§6。

| 原文位置 | 可采用的有限结论 | 对 CoCap 设计的影响（本报告推论） |
|---|---|---|
| PDF p.3 §2.1；pp.7–9 §3.1–3.3 | 任务与运动可 task-first、motion-first 或 interleaved；通信结构和适应性是另外两个维度 | 联盟分配要接收槽位不可达/路径代价反馈；层级结构本身不意味着中央控制 |
| p.8 §3.2 | 综述为简化分类，把含 decentralized 部分的混合系统也归进 decentralized 类 | 不能用该综述表格给完整方法贴“无中央”标签；须回看 base station、共享目录、全局归约 |
| p.9 §3.3 | adaptive 包括在执行中改部分或全部任务/运动计划，依赖明确重规划触发 | 本项目分别记录目标丢失、队员减少、超时、捕获和新任务事件，不能只写“dynamic” |
| pp.20–21 §4 中 RL/CTDE 讨论；pp.24–26 §5.3 | 学习、网络流、博弈与控制提供不同建模/求解路径；真实部署与模型误差仍是问题 | 传统可显式处理最低人数和约束，但 AW9/感知改变会影响保证；RL 的模拟经验不是局部信息或泛化的证明 |
| p.27 §6 | 该综述样本是 2021–2025 的 60 篇 TAMP，不是穷尽所有围捕研究 | 作为系统结构依据，不作为“没有传统方法解决 CoCap”的证明 |

审查意见：原文对“集中式一般保证全局最优”等表述较宽泛。实际近似求解、时限终止、非凸性都可能使其不成立；本报告不采用这种普遍化描述，更不据此推断 CoCap 延迟优劣。

### V3-S02：2026 CPP 最新预印本综述

Zongyuan Shen, Shalabh Gupta, Shancheng Zhao, Dehua Zhou, Gao Wang, Zhongqiang Ren, Yaming Ou, Yikui Zhai, C. L. Philip Chen，**Coverage Path Planning: Classical Foundations, Recent Advances, and Future Directions**，[arXiv:2607.10649v1](https://arxiv.org/pdf/2607.10649v1)，2026-07-12，21 页。**本轮未核到该标题正式发表的 venue/DOI**；arXiv DOI 为 [10.48550/arXiv.2607.10649](https://doi.org/10.48550/arXiv.2607.10649)。已读分类、multi-robot、learning 与 future directions 相关章节。

| 原文位置 | 核查内容 | 对当前任务的使用边界 |
|---|---|---|
| PDF p.2 §I-A、Fig.2 | 区分 offline 已知环境与 online 增量环境；按单/多机、几何和平台约束分类 | “online”描述地图/路径更新，不能自动解读为动态移动敌人或重复捕获 |
| pp.7–8 §III-A、Table III | STC、子区分解、路径均衡及 turn cost；列 LS-MCPP 等原论文 | 已知全图规划是很合理的 privileged 搜索模块，不能伪装局部地图 |
| pp.9–10 §III-B、Table IV | grid-based 与 subarea-based online 方法；覆盖重分配和区域层级 | 指向 C*、Multi-CAP、APF-CPP 等原论文；“connectivity-aware”要确认究竟是空间可达性还是无线网络 |
| p.16 §VI、Table VII | 学习增强可发生在分区、waypoint、路径或协调等不同层 | “不用 RL”不要求放弃层级；也不能把某一层为 learned 的混合算法叫纯传统 |
| pp.17–18 §VIII | 分区协调、在线非局部决策、通信受限/异质性与动态环境仍待研究 | 激励持续搜索与通信成本记录；不作为本系统必然更好/更慢的定量依据 |

按它的引用继续取得了 C* 的 2026 TRO 对应 arXiv v3、Multi-CAP v2 和 LS-MCPP TRO 作者全文，避免停在综述摘要。

### V3-S03：分布式优化 tutorial（2025 正式发表）

Andrea Testa, Guido Carnevale, Giuseppe Notarstefano，**A Tutorial on Distributed Optimization for Cooperative Robotics: From Setups and Algorithms to Toolboxes and Research Directions**，Proceedings of the IEEE **113(1):40–65, 2025**，[DOI](https://doi.org/10.1109/JPROC.2025.3557698)。正式元数据与作者机构记录核对；实际读取 [arXiv:2309.04257v2, 2025-04-02](https://arxiv.org/pdf/2309.04257v2)，24 页。作者机构 AAM 下载被拒，不能把 arXiv 页码写成正式页码。

| 原文位置 | 可核机制/假设 | 对分层设计的意义 |
|---|---|---|
| PDF p.4 §II；pp.5–7 §III/IV | 本地变量与耦合约束；§IV-A 例子是一对一 assignment | 普通线性 assignment 能转成 LP；不应一概写传统分配 NP-hard；也不能把一对一例子当三机联盟 |
| p.10 Alg.1、Assumptions VI.2–VI.6 | 凸性、Slater、步长、权重及跨时间强连通等是 dual decomposition 的条件 | 断连/短时限/移动目标代价改变时，需要另外论证；“分布式”不是零轮通信 |
| p.12 Alg.2、Assumptions VI.9–VI.12 | MILP primal decomposition 结果依赖静态无向连通图、收缩与可行性条件 | 不把它直接套用到动态 Voronoi 拓扑或每 0.5 s 重置的任务 |
| p.15 §VIII、Fig.7 | DISROPT、ChoiRbot、CrazyChoir 将优化、规划、低层控制分离；实验为静态无向图、WiFi | 支持分层接口思想；ROS2 中间件不是完整围捕方法 |
| p.16 §VIII-A | 四机器人/四任务例子使用共享任务内存；单机抵达算完成，完成后出现新任务再求解 | 确实展示重复任务，但没有未知目标搜索或三机几何捕获；不能拼成我们的全生命周期 |

特别核对：论文脚注指向的 [multirobot-tutorial](https://github.com/OPT4SMART/multirobot-tutorial) 固定 HEAD 只有 README，没有实现。论文说代码可用与本轮实际仓库内容存在差距，审计标 `OFFICIAL_PARTIAL`。独立 DISROPT/ROS2 包是否足够复现实验还需另审，不在这里当已核事实。

### V3-S04：2025 联盟形成正式综述（沿用已有全文证据）

Krishna Arjun, David Parlevliet, Hai Wang, Amirmehdi Yazdani，**Optimizing Coalition Formation Strategies for Scalable Multi-Robot Task Allocation: A Comprehensive Survey of Methods and Mechanisms**，Robotics **14(7):93, 2025**，[DOI](https://doi.org/10.3390/robotics14070093)，[41 页正式 PDF](https://mdpi-res.com/d_attachment/robotics/robotics-14-00093/article_deploy/robotics-14-00093.pdf)。Round 2 已读取相关章节，本轮沿用该版本。

PDF pp.3–4 §2.1、pp.7–8 §2.2 用于说明 coalition 的组合空间、任务约束和通信组织；pp.19–20 §3.5/Table8 用于查算法族；p.35 §5/6 用于动态任务、部分观测、通信和适应性的挑战。其 §4 自定义 coalition 仿真不是围捕实验，Table8 的一般优劣分类也不是 CoCap 的速度或成功率证据。

### 其余近期综述：纳入资料池，保留获取缺口

| ID | 已验证正式元数据 | 阅读状态与用途 |
|---|---|---|
| V3-S05 | **A survey on theories and applications for multi-robot cooperative hunting**；Jianjun Ni, Yonghao Zhao, Ziru Zhang, Chunyan Ke, Simon X. Yang；RAS **197:105296, March 2026**；[10.1016/j.robot.2025.105296](https://doi.org/10.1016/j.robot.2025.105296) | Crossref/出版社元数据可核；未得完整 PDF；章节/公式结论不采用 |
| V3-S06 | **Advancing Multirobot Task Allocation: A Comprehensive Review of Emerging Trends and Challenges**；Saddam I. D. Bakhat, Inam Ullah, Xiaoshan Bai, Weidong Zhang, Jianqiang Li, Shuzhi Sam Ge；IoTJ **13(19):44021–44058, October 2026**；[10.1109/JIOT.2026.3699696](https://doi.org/10.1109/JIOT.2026.3699696) | 正式卷期可核；全文未得；仅最新资料池，不为系统能力作证 |
| 沿用 V01 | **A survey on target encirclement algorithms with UAVs, USVs and AUVs**；Wenyu Cai, Jiahan Wang, Meiyan Zhang, Tianyuan Zhang；Neurocomputing **672:132687, 2026**；[DOI](https://doi.org/10.1016/j.neucom.2026.132687) | Round 2 PDF 获取缺口延续；不假称已读控制/通信章节 |
| 沿用 V03 | **A review of reinforcement learning approaches for pursuit-evasion games**；Kun Yang, Ao Shen, Nengwei Xu, Fang Deng, Maobin Lu, Chen Chen；CJA **39(6):103940, 2026**；[DOI](https://doi.org/10.1016/j.cja.2025.103940) | Round 2 读到部分 HTML 正文，但 PDF 页码未核；不重复扩张 RL 候选 |

## 2. 分配层：回到 GRAPE 原论文的关键修正

**V3-M01**：Inmo Jang, Hyo-Sang Shin, Antonios Tsourdos，**Anonymous Hedonic Game for Task Allocation in a Large-Scale Multiple Agent System**，TRO **34(6):1534–1548, 2018**，[DOI](https://doi.org/10.1109/TRO.2018.2858292)。阅读 [arXiv:1711.06871](https://arxiv.org/pdf/1711.06871)，15 页；该经典来源不因年份旧而排除。

- p.3 §III-A Assumptions 1–4：本机每次最多一任务、同质机器人、强连通通信、每机知道对各项任务的效用；只知道自己的效用不代表不知道其他目标位置/目录。
- pp.4–6 §III-B/Alg.1–2：SPAO 偏好、Nash partition、local deviation 与 distributed mutex。不是中央服务器分配，但会传播覆盖全队的 partition。
- p.7 §IV-A：O(d_G N²) 轮、每机每轮 O(M) 评价与 O(degree_i·N) 消息量级；这些是固定实例理论量级。
- **p.9 §V、式(23)–(28)、Condition 1、Proposition 1**：明确最低人数 R_j，用 dummy utility 将不足人数任务置于招募优先，同时保留 SPAO 以使用稳定性结论。总资源足够的条件不能省略。
- p.10 Proposition 2：最低人数分支的真实效益要用原效用评估；不能随意沿用基础方法的 1/2 近似口号。

代码：MATLAB Fig.7 官方部分实现与作者 Python equivalent；两者所查效用分支没有 §V 公式。建议复用 partition/mutex 结构，按原式重建最小人数评分，再单独处理局部发现、任务集变化、会合承诺。完整 H-L 系统仍是项目组合，而非该论文已完成的移动目标围捕。

**V3-M02**：Xiaoyong Zhang, Wei Yue, Liyuan Wang, Lili Li, Lei Zhu, Zhongchang Liu，**Coalition Formation Game-Based Task Allocation With Minimum Requirements in Dynamic Environments**，IoTJ **13(6):12501–12518, 2026-03-15**，online 2026-01-12，[DOI](https://doi.org/10.1109/JIOT.2026.3651713)。出版社摘要/目录表明确 MTR、动态任务数量/要求、CFG-BAP bilateral altruism；**全文、具体效用、通信图、完整复杂度及代码未取得**。因此它很值得补全文，却不能取代已读 GRAPE 成为马上重建的首选。`CONDITIONAL_GO（分配层补证）`，非完整围捕候选。

其近年相关来源 **Task Allocation With Minimum Requirements for Multiple Mobile Robot Systems: A Game-Theoretical Approach**，Xin Huo, Hao Zhang, Chao Huang, Zhuping Wang, Huaicheng Yan，TNSE **11(1):1202–1213, 2024**，[10.1109/TNSE.2023.3321605](https://doi.org/10.1109/TNSE.2023.3321605)。正式卷期不是 2023；同样保留全文/代码缺口，不据摘要复写其 potential-game 证明。

## 3. 搜索层：从最新综述回查的方法

### V3-M03 — C*：正式 TRO 2026，未知环境搜索依据较强

Zongyuan Shen, James P. Wilson, Shalabh Gupta，**C*: A Coverage Path Planning Algorithm for Unknown Environments Using Rapidly Covering Graphs**，TRO **42:1233–1253, 2026**，[DOI](https://doi.org/10.1109/TRO.2026.3661719)。实际读取 [arXiv:2505.13782v3, 2026-03-06](https://arxiv.org/pdf/2505.13782v3)，23 页。

PDF pp.5–10 §II/III：增量 Rapidly Covering Graph、采样/剪枝、back-and-forth、retreat nodes 和 coverage-hole TSP；pp.11–13 §IV 为复杂度/覆盖证明；**p.18 §V-C.2 明确多机发送传感器数据到中央服务器、构建统一地图和单一 RCG**，四机均分区域，最佳分区/任务完成和失效后的资源重分配不在范围。p.20 §VI 把 decentralized coverage 列入未来方向。

贡献是未知静态环境覆盖；没有多目标 coalition、目标移除、追逃动力学和捕获后循环。未找到可核作者实现，`NOT_FOUND`。原公式结构清楚，搜索层重建难度中至高；若加入年龄重访是另一个项目扩展。推荐作较强搜索层升级/Related Work，不作为现成 local distributed 完整方法。

### V3-M04 — Multi-CAP：层级结构贴近，但通信是集中式

Zongyuan Shen, Burhanuddin Shirose, Prasanna Sriganesh, Bhaskar Vundurthy, Howie Choset, Matthew Travers，**Multi-CAP: A Multi-Robot Connectivity-Aware Hierarchical Coverage Path Planning Algorithm for Unknown Environments**，[arXiv:2509.14941v2, 2025-09-26](https://arxiv.org/pdf/2509.14941v2)，8 页；[arXiv DOI](https://doi.org/10.48550/arXiv.2509.14941)。**本轮未核到该标题正式 venue/DOI**，不能把相关单机 CAP 的 IROS 2025 DOI 填给 Multi-CAP。

PDF p.2 §III 使用 base station 收集全队数据；pp.3–4 §IV：空间连通子区图、Open MDVRP、nearest-neighbor + 2-opt、局部 back-and-forth/TSP。p.8 §VII 把通信受限、非完整机器人和动态障碍列为未来方向。“connectivity”主要是空间子区连通性，不能解释成无线连通性约束。未找到作者源码，`NOT_FOUND`。适合 global search/partition 架构借鉴，移植中至高；不能直接进入局部信息主表。

### V3-M05 — LS-MCPP：可复用性最清楚的已知图覆盖组件

Jingtao Tang, Zining Mao, Hang Ma，**Large-Scale Multirobot Coverage Path Planning on Grids With Path Deconfliction**，TRO **41:3348–3367, 2025**，[DOI](https://doi.org/10.1109/TRO.2025.3567476)。读 [作者 20 页 PDF](https://www.cs.sfu.ca/~hangma/pub/tro25.pdf)；与 Tang、Ma 的 AAAI 2024 **Large-Scale Multi-Robot Coverage Path Planning via Local Search** 是前后版本，不混写结果或分支。

PDF pp.4–7 §III/IV：已知四邻接图、ESTC、覆盖/复杂度结果；pp.8–11 §V：grow/deduplicate/exchange 局部搜索；**pp.11–12 §VI** 明确基础 MCPP 不自动无碰撞，另外用 PBS 和低层连续时间规划消解。PBS 在一般实例不完整，well-formed 条件不能省略。

官方 [LS-MCPP](https://github.com/reso1/LS-MCPP) 有算法、实例与 notebook，GPLv3。依赖及 gurobipy 声明见代码审计。它的 local search 指优化邻域搜索，不是机器人局部信息决策。适合已知图搜索模块；地图全知、任务暂停/恢复和 AW9 仍需重建接口。

### V3-M06 — HEDAC-maze：2026 最新正式且有代码，原场景差异较大

Bojan Crnković, Stefan Ivić, Mila Zovko，**Fast Algorithm for Centralized Multi-Agent Maze Exploration**，Mathematics **14(19):3613, 2026-10-05**，[DOI](https://doi.org/10.3390/math14193613)，[31 页正式 PDF](https://mdpi-res.com/d_attachment/mathematics/mathematics-14-03613/article_deploy/mathematics-14-03613.pdf)。另有 2023 arXiv 早期版；这里结论基于 2026 正式版。

PDF pp.9–11 §3.1/3.2：未知正交迷宫、一步邻居感知、源项 s=max(0,1−c)S，势场解 `−Δu+α²u=s`；p.12 Theorem 1 的求解/探索条件；**pp.13–14 Alg.1** 顺序决定占位、立即共享下一位置、找到出口即停止。p.28 §4.3 明确现系统依赖中央控制，decentralized 为建议改造。

官方 [GitLab](https://gitlab.com/sivic/hedac-maze-exploration) 有 SOR/Krylov/直接求解、迷宫和批次脚本；pyproject/uv.lock 存在，但 README 的 requirements.txt 路径不存在，许可证未见。单步计算涉及已发现图的迭代求解，不能根据简单梯度推断总成本 O(1)。它没有移动敌人、三机围捕、释放或持续重访；本轮作为集中搜索模块后备，非优先 local baseline。

### V3-M07 — Sensor-based coverage：局部避障不等于未知目标自主搜索

Xinyi Wang, Jiwen Xu, Chuanxiang Gao, Yizhou Chen, Jihan Zhang, Chenggang Wang, Yulong Ding, Ben M. Chen，**Sensor-based Multi-Robot Coverage Control with Spatial Separation in Unstructured Environments**，ICRA **2024:10623–10629**，[DOI](https://doi.org/10.1109/ICRA57147.2024.10611020)。实际读 [arXiv:2403.01710v3](https://arxiv.org/pdf/2403.01710v3)，7 页；arXiv 标题用 Multi-agent，正式题名用 Multi-Robot，作者数按正式元数据和实际 v3 校核。

PDF p.2 §III 的 single-integrator 与局部点云；pp.3–4 §IV 空间分解、spherical mirroring 和 buffered Voronoi safe region；**p.5 式(8)/Lemma 3** 使用信息分布、cost-to-go guided map，并收敛到静态部署。它能支持局部避障与部署思想，但不能因此宣称持续未知移动目标发现。论文 ROS/C++/Python、MATLAB 对照配置可核，官方方法代码未找到，`NOT_FOUND`。只作为搜索/安全子层理论来源。

### V3-M08 — SPACE：动态任务框架的官方代码，不是围捕论文

Inmo Jang，**SPACE: A Python-based Simulator for Evaluating Decentralized Multi-Robot Task Allocation Algorithms**，[arXiv:2409.04230](https://arxiv.org/pdf/2409.04230)，2024，8 页；[arXiv DOI](https://doi.org/10.48550/arXiv.2409.04230)。未核到该标题正式 venue；作者仓库也按 arXiv 引用。

PDF pp.3–4 §III：局部感知、通信、行为树、任务工作量；明确省略障碍和机间避碰。p.5 §IV：原 GRAPE/CBBA 之外的动态任务适配；p.6 式(1)含 social inhibition，实测任务至少一人可做并可多机加速，**不是 ≥3 才有效**。新任务分三次加入且完成后另做其他任务是真实实现的生命周期部分；围捕/捕获移除与 APF 追逃不是。

官方代码值得参考，原任务对象共享、全任务 ID 初始化与未冻结依赖均需处理，不能只改距离阈值就变成完整 CoCap baseline。

## 4. 很新但暂不提升排序的线索

- **Dynamic Task Allocation and Whole-Group Encirclement of Multiple Targets in Multi-Agent Swarms via Voronoi Partitioning**；Chenyang Li, Yonghui Yang, Yilin Han, Xue-Bo Chen, Tianxin Zhao；Mathematics and Computers in Simulation，2026-08-26 online，出版社显示 In Press / Journal Pre-proof；[10.1016/j.matcom.2026.08.028](https://doi.org/10.1016/j.matcom.2026.08.028)。已见出版页面，未得全文，whole-group 表述不能推断逐目标三机；与 V2 已排除的目标群包围问题相同风险。代码未找到，保留线索。
- **Finite-time encircling control of underwater targets for multiple AUVs based on Elliptical Orbit Optimization**；Jiahan Wang, Wenyu Cai, Meiyan Zhang, Ben Niu；Ocean Engineering **366 Part 1:127487, 2026-10-15**；[10.1016/j.oceaneng.2026.127487](https://doi.org/10.1016/j.oceaneng.2026.127487)。仅出版社摘要/引言可读，完整控制和分配未核；不能据 finite-time 标题声称 CoCap 三机捕获。
- 中文 AUV distributed sliding MPC 新文虽声称局部状态观测和多目标分组，本轮页面只返回摘要，未核完整卷期、公式和配置；[原刊入口](https://www.juestc.uestc.edu.cn/article/doi/10.12178/1001-0548.2025094)。不提升为已验证候选。

## 5. 能写入论文的证据句及禁区

| Claim ID | 建议限定表述 | 原始页码支持 |
|---|---|---|
| V3-E01 | 传统系统可将离散资源组织和连续运动可行性分层，并通过执行反馈重规划 | S01 pp.3、7–9；TR-11 p.990 与 TR-13 §III–V（沿用 V2） |
| V3-E02 | 最低参与人数可用约束或特殊 coalition utility 表达；普通一对一任务分配不足以保证多人捕获 | M01 p.9 §V；S03 p.6 §IV-A；M08 p.6 §V |
| V3-E03 | 局部传感与去中心化决策应分别检查；未知地图覆盖也可使用全队中央融合 | M03 p.18 §V-C.2；M04 p.2 §III；M06 p.28 §4.3 |
| V3-E04 | 固定密度 CVT 部署、一次完整覆盖与持续移动目标搜索是不同目标 | 经典 Cortés arXiv pp.3–5；M07 p.5 Lemma3；M06 p.14 Alg.1；持续年龄搜索为本项目设计 |
| V3-E05 | 动态任务代码可以已有完成与新增事件，同时仍缺最低人数、几何和安全模块 | M08 pp.3–6 及固定源码审计 |
| V3-E06 | 计算与通信复杂度必须按算法条件和组件边界给出，不能用于未经实验的 CoCap 快慢排序 | M01 p.7；S03 pp.10–12；M05 pp.11–12；本项目统一预算为实验设计 |

文献中的 CPU 时间、覆盖时间百分比或稳定性界均对应其自身地图、动力学、硬件和信息条件。本报告不搬用这些数值作为 CoCap 对照成绩。
