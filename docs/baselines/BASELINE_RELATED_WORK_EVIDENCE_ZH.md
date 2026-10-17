# Related Work证据、元数据与引用边界

核查日期：2026-10-17。PDF页码均从所列实际取得的版本第1页起计，期刊印刷页另列；未取得PDF一律写“未核”，绝不从摘要猜页码。正式发表与阅读版本分开。可供复查的URL、文件大小、SHA256、访问失败和代码搜索结果索引见 [`focused_review_source_manifest_20261017.json`](../../artifacts/baselines/focused_review_source_manifest_20261017.json)。只提交元数据/研究笔记，不上传论文PDF或大模型资产。

## 1. 恢复的仓库事实源

| ID | 原始资料 | 固定版本与用途 |
|---|---|---|
| B01 | [Round 1 Baseline Screening](https://github.com/Jayyeah/cocap-voradj/tree/94753f9fd82d137af7ff9930634788ffeba6f3e2/docs/baselines)及[原registry](https://github.com/Jayyeah/cocap-voradj/blob/94753f9fd82d137af7ff9930634788ffeba6f3e2/artifacts/baselines/baseline_registry.json) | 94753f9；完整恢复master/cards/registry/search_log；不是重新筛选RL |
| B02 | [Z05 Benchmark Contract](https://github.com/Jayyeah/cocap-voradj/blob/3038ef5cd728acdd302772ca08bf10060685af16/docs/paper/PAPER_ASSETS_20261015/PAPER_BENCHMARK_CONTRACT_DRAFT_ZH.md) | 审计分支3038ef5；§1–5：事实源、AW9、normal capture、到达与信息合同；仍为draft |
| B03 | [IQN-Z Asset Index](https://github.com/Jayyeah/cocap-voradj/blob/3038ef5cd728acdd302772ca08bf10060685af16/docs/paper/PAPER_ASSETS_20261015/PAPER_EXPERIMENT_ASSET_INDEX_20261015_ZH.md)与[Readiness](https://github.com/Jayyeah/cocap-voradj/blob/3038ef5cd728acdd302772ca08bf10060685af16/docs/paper/PAPER_ASSETS_20261015/PAPER_EXPERIMENT_READINESS_ZH.md) | 同版本；selected checkpoint/正式评估与Evidence缺口；本轮阅读，未重新校验二进制或运行 |

## 2. 传统候选正式元数据与可读版本

| ID | 完整标题；作者 | 正式发表状态 / DOI | 原始入口与实际访问 |
|---|---|---|---|
| S01 / TR-01 | **A competition-based scheme using k-WTA networks for multi-target pursuit and encirclement in dynamic environments**；Ning Tan, Zhenghui Cui, Yang Liu, Ruikun Hu, Bo Zhu, Tianjiang Hu | Robotics and Autonomous Systems **191:104990, 2025**；[10.1016/j.robot.2025.104990](https://doi.org/10.1016/j.robot.2025.104990) | [出版社](https://www.sciencedirect.com/science/article/pii/S0921889025000764)；Crossref元数据；Introduction/章节片段/Conclusion可读；PDF及完整§2–5未取得 |
| S02 / TR-02 | **Cooperative Control of Multi-UAV for Multi-Targets Encirclement and Tracking Based on Potential Game**；Jiangbo Jia, Xin Chen, Weizhen Wang, Hongjin Liao, Guangyuan Zhu | 42nd Chinese Control Conference，**3778–3785, 2023**；[10.23919/CCC58697.2023.10240696](https://doi.org/10.23919/CCC58697.2023.10240696) | [IEEE](https://ieeexplore.ieee.org/document/10240696)、[正式目录](https://www.proceedings.com/content/070/070536webtoc.pdf)及Crossref；全文/PDF页码未核 |
| S03 / TR-03 | **A Coalition Formation Approach to Coordinated Task Allocation in Heterogeneous UAV Networks**；Fatemeh Afghah, Mohammad Zaeri-Amirani, Abolfazl Razi, Jacob Chakareski, Elizabeth S. Bentley | American Control Conference，**5968–5975, 2018**；[10.23919/ACC.2018.8431278](https://doi.org/10.23919/ACC.2018.8431278) | [作者PDF](https://fafghah.people.clemson.edu/Papers/ACC18.pdf)已取得；§II–III，PDF pp.2–5；未知任务、探测者leader与联盟，非几何围捕控制 |
| S04 / TR-04 | **Distributed cooperative pursuit with encirclement guarantee via robust model predictive control**；Chen Wang, Hua Chen, Jia Pan, Wei Zhang | Robotics and Autonomous Systems **192:105019, 2025**；[10.1016/j.robot.2025.105019](https://doi.org/10.1016/j.robot.2025.105019) | [出版社](https://www.sciencedirect.com/science/article/pii/S0921889025001058)；§2–6片段，2025完整PDF未得 |
| S04a / 前身 | **Encirclement Guaranteed Cooperative Pursuit with Robust Model Predictive Control**；Chen Wang, Hua Chen, Jia Pan, Wei Zhang | IROS **2021**，pp.1159–1165；会议DOI本轮未核，不倒填2025 DOI | [arXiv:2108.07445v1 PDF](https://arxiv.org/pdf/2108.07445v1)，7页已读；§II–V；这是前身，不是2025排版 |
| S06 / TR-06 | **Multitarget Pursuit-Evasion Based on Distributed and Competitive Mechanisms**；Ning Tan, Yang Liu, Ruikun Hu, Hui Cheng | IEEE Transactions on Systems, Man, and Cybernetics: Systems **54(10):5989–6000, 2024**；[10.1109/TSMC.2024.3410693](https://doi.org/10.1109/TSMC.2024.3410693) | [出版社](https://ieeexplore.ieee.org/document/10586228/)＋Crossref；原文未取得，仅确认distributed DMK-WTA/PPM和验证概况 |
| S07 / TR-07 | **Accelerated K-Serial Stable Coalition for Dynamic Capture and Resource Defense**；Junfeng Chen, Zili Tang, Meng Guo | IEEE Robotics and Automation Letters **9(1):443–450, 2024**；[10.1109/LRA.2023.3334560](https://doi.org/10.1109/LRA.2023.3334560)；2023为online/预印本线，不取代卷期年 | [arXiv v3 PDF](https://arxiv.org/pdf/2305.15729v3)，8页已读；正式卷期由Crossref核对。运动细则在v3指向补充，正式补充未取得 |
| S07a / 早期细则 | 同题同作者 | **2023-05-25 v1预印本**；不是第二篇正式论文 | [v1 PDF](https://arxiv.org/pdf/2305.15729v1)；§III-A PDF pp.3–4用于早期sweep/pure-pursuit/资源防御公式；不混成v3最终公式 |
| S08 / TR-08 | **Distance-Based Multiple Noncooperative Ground Target Encirclement for Complex Environments**；Fen Liu, Shenghai Yuan, Kun Cao, Wei Meng, Lihua Xie | **IEEE Transactions on Control Systems Technology 33(1):261–273, January 2025**；[10.1109/TCST.2024.3469032](https://doi.org/10.1109/TCST.2024.3469032)，末轮Crossref精确题名核实；另有[2024预印本](https://arxiv.org/abs/2409.15840) | 阅读[arXiv PDF](https://arxiv.org/pdf/2409.15840)，13页；§II–III，PDF pp.2–6。正式排版/版本差异未核；未把TCST页眉单独当发表证明 |
| S09 / TR-09 | **Multirobot Target Monitoring and Encirclement via Triggered Distributed Feedback Optimization**；Lorenzo Pichierri, Guido Carnevale, Lorenzo Sforni, Giuseppe Notarstefano | IEEE Transactions on Robotics **42:2596–2610, 2026**；[10.1109/TRO.2026.3686280](https://doi.org/10.1109/TRO.2026.3686280)，Crossref已核 | 阅读[2024 arXiv:2409.20399 PDF](https://arxiv.org/pdf/2409.20399)，标题写Multi-Robot；§II–III，PDF pp.3–7；正式版方法差异/排版未核 |
| S10 / TR-10 | **Wolf-Pack-Inspired Distributed Encirclement for UAV–USV Systems via Visual Connectivity Preservation**；Jinheng Xiao, Shihan Kong, Jinan Sun, Yingnan Li, Fang Wu, Junzhi Yu | Biomimetics **11(7):513, 2026-07-21**；[10.3390/biomimetics11070513](https://doi.org/10.3390/biomimetics11070513) | [出版社全文](https://www.mdpi.com/2313-7673/11/7/513)、[正式PDF](https://mdpi-res.com/d_attachment/biomimetics/biomimetics-11-00513/article_deploy/biomimetics-11-00513.pdf)，实际下载27页；§4 pp.8–12、§7 pp.23–24 |
| S11 / TR-11 | **Dynamic Robust Pursuit of Multiple Evaders Under State Measurement Uncertainty in Obstacle Environments**；Kai Rao, Huaicheng Yan, Zhihao Huang, Tiantian Xu, Penghui Yang, Yunkai Lv | IEEE/CAA Journal of Automatica Sinica **13(4):989–991, 2026**；[10.1109/JAS.2025.125684](https://doi.org/10.1109/JAS.2025.125684) | [出版社](https://www.ieee-jas.net/article/doi/10.1109/JAS.2025.125684?pageType=en&viewType=HTML)、[正式3页PDF](https://www.ieee-jas.net/article/exportPdf?id=9ab91428-5ba2-44cc-ae76-e5917191bfe1&language=en)、[补充2页](https://www.ieee-jas.net/fileZDHXBEN/journal/article/file/d5633e16-678c-4d80-8337-3da4303ecd23.pdf)均已取得 |
| S12 / TR-12线索 | **Self-Organized Multitarget Pursuit in Multi-Unmanned Aerial Vehicle Systems via Hierarchical Probabilistic Graphical Models**；Yixin Huang, Xiaojia Xiang, Chao Yan, Heda Xu, Tianjiang Hu, Han Zhou | Advanced Intelligent Systems **7(11):2401110, 2025**；[10.1002/aisy.202401110](https://doi.org/10.1002/aisy.202401110) | [出版社PDF入口](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/aisy.202401110)403；元数据已定位；全文机制不下结论 |
| S13 / TR-13 | **A Distributed Pipeline for Collaborative Pursuit in the Target Guarding Problem**；Yansong Chen, Yuchen Wu, Helei Yang, Junjie Cao, Qinqin Wang, Yong Liu | IEEE Robotics and Automation Letters **9(3):2064–2071, March 2024**；[10.1109/LRA.2024.3349977](https://doi.org/10.1109/LRA.2024.3349977) | [作者机构PDF](https://april.zju.edu.cn/wp-content/papercite-data/pdf/chen2024adp.pdf)，8页；§III–VI pp.2–6/印刷2065–2069；全文已取得 |

TR-05没有论文元数据，不能制造DOI/作者。全部traditional repo/commit/许可/文件和依赖逐项列在[代码审计](TRADITIONAL_BASELINE_OPENSOURCE_AUDIT_ZH.md)。

## 3. 可写入Related Work的技术证据

| Claim ID | 严格限定的技术表述 | 可核原文位置 | 不能推出的结论 |
|---|---|---|---|
| E01 | 几何包围可通过目标处于pursuer凸包表达，再通过扇区内近似构造局部MPC问题 | S04a PDF p.2定义/约束、pp.3–5分区与TMPC；S04 2025仅作版本关系 | 凸包包围不自动满足CoCap半径/间隙比；不能说2025源码已复现 |
| E02 | 显式分组和目标周围槽位分配可与安全控制组合，区分高层资源组织和低层运动 | S11印刷p.990 “Pursuers dynamic allocation”、p.991式(4)；补充S2–S5 | 不保证未知目标发现，也不保证AW9量化后仍安全 |
| E03 | 任务coalition可在邻接链上以多步switch提升utility；K增大扩大搜索但也扩大buffer搜索空间 | S07 v3 PDF pp.3–5，Definitions 1–5、Alg.1、§III-D | 一般KSS不是全局最优；不能说任意动态断连图上有限时间共识 |
| E04 | 分布式控制应逐层核查通信：KS-COAL的SWAT有限范围，SCOUT队内无范围限制 | S07 v3 §II-A PDF p.2；S07a p.3发现广播 | “无中央”不等于“只用CoCap邻接消息” |
| E05 | event-triggered aggregative feedback通过两组跟踪变量估计全局聚合量，邻居交换2d个float | S09 arXiv PDF p.3 Assumption2.1、p.5式(6)、pp.5–7触发机制 | fixed strongly-connected/weight-balanced假设不能直接套动态Voronoi图；stationary point不是normal capture |
| E06 | 两机对向antisynchronization是围绕/感知几何，但与至少三机闭合捕获不同 | S08 PDF p.2 Assumptions1–2、pp.5–6式(12)–(15) | 不能仅增加第三台而宣称原稳定性/可观测性证明保持 |
| E07 | 局部感知论文仍需检查全局聚合操作的实现；GB-VCE伪代码使用Γ、argmax与TopK | S10 PDF pp.9–11式(14)–(23)、Alg.1–2；p.12 Alg.3；pp.23–24局限 | 不据“distributed”标签推断消息复杂度或断连下的实现；不把失联搜索等于多目标持续服务 |
| E08 | 全局安全域构造与全框架复杂度必须分开，低维几何子问题可有线性约束数开销 | S11补充PDF p.2 S4，p.1 S2–S3 | O(S+K)仅为C3AVC，不含A*/拍卖/LP；不能据此比较CoCap端到端时延 |
| E09 | 已发表分布式MPC框架可同时有巡逻/发现、实时分组和几何捕获，而仍采用全实时通信 | S13 PDF p.1 Fig.1、p.3 §III-B/IV式(3)–(8)、p.5式(23)、p.6 §VI明确通信假设 | varying initial conditions不是在线新目标/捕获后释放；soft collision罚项无硬安全证明；Gazebo是仿真 |

可采用的综述性写法：传统方法常把任务组织、几何目标与安全控制分层建模（E01–E03）；分布式算法依赖的状态与通信条件各异（E04–E07）。在共同normal-capture、动力学和到达合同下比较，才能解释学习策略与显式控制的差异。以上是本评审对已读原文的归纳，不是任何单篇论文已经证明CoCap更优。

## 4. 三类指定综述：元数据及阅读边界

### V01 — Target encirclement survey

Wenyu Cai, Jiahan Wang, Meiyan Zhang, Tianyuan Zhang，**A survey on target encirclement algorithms with UAVs, USVs and AUVs**，Neurocomputing **672:132687, April 2026**，DOI [10.1016/j.neucom.2026.132687](https://doi.org/10.1016/j.neucom.2026.132687)。Crossref作者/卷/文章号已核，[出版社](https://www.sciencedirect.com/science/article/pii/S0925231226000846)。正式发表，不是预印本。

可读Introduction给出localization/encirclement-controller结构及径向、相位和角速度目标，并列出§2协同、§3定位、§4控制、§5任务、§7未来方向的组织。**完整§2–5/7和PDF未取得**；publisher full view需要授权，API元数据响应没有正文。这里只把它纳入资料池与补证队列，不为通信/动态多任务难点提供未读章节结论。PDF页码：`UNVERIFIED`。该部分未满足最终论文逐页引用条件，不能写成“已读综述证明”。典型控制结构改用E01/E02的原文页码支持。

### V02 — Coalition survey

Krishna Arjun, David Parlevliet, Hai Wang, Amirmehdi Yazdani，**Optimizing Coalition Formation Strategies for Scalable Multi-Robot Task Allocation: A Comprehensive Survey of Methods and Mechanisms**，Robotics **14(7):93, 2 July 2025**，DOI [10.3390/robotics14070093](https://doi.org/10.3390/robotics14070093)。[正式PDF](https://mdpi-res.com/d_attachment/robotics/robotics-14-00093/article_deploy/robotics-14-00093.pdf)41页已取得。已读§2.1/2.2、§3相关分类与§3.5、§5/6；引用正式版，不引用其早期preprint。

| 对应问题 | 页码/小节 | 可采用的限定结论 |
|---|---|---|
| 联盟与任务分配复杂度 | PDF pp.3–4 §2.1；pp.7–8 §2.2 | 一般组合联盟空间与动态约束带来挑战；**不将全部MRTA子类一律称NP-hard**，线性一对一assignment可多项式求解 |
| 典型结构与通信 | p.7 §2.2；pp.19–20 §3.5/Table8 | CNP/auction/consensus/merge-split具有不同信息组织，算法名不能单独决定central/distributed |
| 动态/重复任务难点 | p.8 §2.2；p.35 §5 | 部分观测、non-stationarity、通信中断、任务重叠与重新协调；原文讨论不是CoCap任务的实测结论 |
| 学习与传统的局限 | pp.19–20 §3.5、p.35 §5/§6 | 训练成本、通信与任务约束都需共同预算。Table8存在高度概括的优劣描述；不把其RL/传统排序转写为普遍定理 |

§4的仿真是该综述自定义coalition任务（p.20起给Python/库环境），不是围捕几何/安全性benchmark。其CPU/收敛数值不能用来断言传统方法比CoCap慢；搜索动作、探索RL训练与真实任务自主发现也需区分。

### V03 — RL pursuit-evasion review

Kun Yang, Ao Shen, Nengwei Xu, Fang Deng, Maobin Lu, Chen Chen，**A review of reinforcement learning approaches for pursuit-evasion games**，Chinese Journal of Aeronautics **39(6):103940, June 2026**，DOI [10.1016/j.cja.2025.103940](https://doi.org/10.1016/j.cja.2025.103940)。正式卷期2026，DOI中的2025不是卷期年。[出版社全文入口](https://www.sciencedirect.com/science/article/pii/S1000936125005461)。

已读可访问HTML §2.1、§3开头、§3.1、Table2附近及§6，超出摘要：其按单边/双边交替/双边同时学习组织方法，并在结论讨论解释、规模与迁移问题。**PDF下载403/全文API401，PDF页码未核，完整§5未读取**。因此不把该综述的“学习可适应/集中训练全局视角”泛论当成执行端局部性证明，也不采用其集中训练保证全局最优之类宽泛语句。正式RA-L写作引用前必须补页码；当前仅元数据/资料池用途。学习/传统优劣由已读原始方法的具体合同与未来共同实验回答。

## 5. RL与组件的固定证据

- C01：[TERL官方repo](https://github.com/ApricityZ/TERL/tree/143359b2722d49c29b4fecc0ad1fd8d46326e45a)，作者Heng Zhang、Guoxiang Zhao、Xiaoqiang Ren；**TERL: Large-Scale Multi-Target Encirclement Using Transformer-Enhanced Reinforcement Learning**，IROS 2025，pp.13438–13445；末轮Crossref确认[10.1109/IROS60139.2025.11246832](https://doi.org/10.1109/IROS60139.2025.11246832)，另有[arXiv:2503.12395](https://arxiv.org/abs/2503.12395)。README自认本文实现；输入局部性用固定revision源码L276–345作直接证据，而非宣传摘要。
- C02：[SDQP固定revision](https://github.com/ZJU-FAST-Lab/SDQP/tree/78e9860508cc7c191870dea79e63b1e7c45e2296)，Zhepei Wang、Fei Gao；header MIT，Eigen3；S11补充p.1明确链接。它只支持独立低维严格凸QP，不等于整套JAS源码。
- RL-02：Maryam Kouzeghar、Youngbin Song、Malika Meghjani、Roland Bouffanais，ICRA 2023，**Multi-Target Pursuit by a Decentralized Heterogeneous UAV Swarm using Deep Multi-Agent Reinforcement Learning**，[arXiv:2303.01799](https://arxiv.org/abs/2303.01799)、[作者PDF](https://applied-complexity.org/docs/ICRA-2023.pdf)。沿用第一轮角色/追踪合同，不新增大训练承诺。

## 6. 论文撰写时禁止越界的表述

末轮额外排除两条可读全文线索：[Chaib等RA-L 2026](https://zaguan.unizar.es/record/170982/files/texto_completo.pdf)，**Safe and Adaptive Multi-Target Encirclement With Socially Aware Multi-Robot Control**，11(6):6608–6615，DOI10.1109/LRA.2026.3682584，PDF p.3 §III-C用QuickHull定义目标群表面，解决整个群体包围，不满足逐目标多人分配；[Zhou等JAS 2024](https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2023.124041)，**Multi-Robot Collaborative Hunting in Cluttered Environments With Obstacle-Avoiding Voronoi Cells**，11(7):1643–1655，DOI10.1109/JAS.2023.124041，PDF p.3的假设要求追捕者不慢于目标、全机器人实时位置；其单evader捕获点Hungarian匹配/SVM安全域可作几何来源，不能代替多目标任务层。完整作者及下载hash列manifest的`excluded_late_leads`；它们未被当成完整候选，不因标题新而挤占shortlist。

“传统方法有在线优化”不等于“传统方法一定比CoCap慢”；“distributed”不等于局部未知目标发现；“encirclement”不等于满足CoCap三机角度终端；“重复资源生成”不等于重复敌人到达；“recovery”不等于捕获后区域搜索；“有源码组件”不等于整套方法开源。以上区别应同时体现在主表标签、实现命名和结果讨论中。
