# CoCap Baseline最终候选排序V2

> 第四轮补充：允许全局敌人信息与集中式分层分配，最新决策见 [V4 全局分层基线](GLOBAL_HIERARCHICAL_BASELINE_REVIEW_V4_20261017_ZH.md)。本文既有结论保留作历史/局部方案依据。

> 历史 Round 2 排序。第三轮新增传统分层替代方案、GRAPE 最低人数原文核查及逐层代码审计；当前决策入口为 [V3 分层研究结果](BASELINE_HIERARCHICAL_REVIEW_V3_20261017_ZH.md)。完整论文强基线首选仍是 TR-11；局部组合方案与单篇论文复现分别列出。

2026-10-17。替代第一轮推荐排序，保留其候选历史。详细证据见[传统复核](TRADITIONAL_BASELINE_FOCUSED_REVIEW_20261017_ZH.md)、[可行性](TRADITIONAL_BASELINE_REPRODUCTION_FEASIBILITY_ZH.md)、[代码审计](TRADITIONAL_BASELINE_OPENSOURCE_AUDIT_ZH.md)与[出处](BASELINE_RELATED_WORK_EVIDENCE_ZH.md)。本轮未开始正式复现。

## 1. RL：固定决策

| 排序 | 方法 | 决策 | 信息与实施边界 |
|---|---|---|---|
| 必选 | TERL，IROS 2025 | 原始协议preflight GO；局部适配CONDITIONAL_GO | 官方MIT，commit143359b；Global版本保留全距离目标输入；Local-Adapted单独命名、重新核对训练/selection，不把遮罩当忠实复现 |
| Related Work | Role-based MADDPG，ICRA 2023及固定角色方法 | 大规模训练NO_GO当前投入 | scout/tracker故事相关；未找到可靠作者代码/第三方合同，且跟踪/碰撞攻击不同于多人normal capture |

其余第一轮RL候选保留registry历史，不为凑数量启动新检索或训练。

## 2. Traditional：推荐结构

| 类别/序位 | 方法、正式发表 | 代码 | 已有任务与控制 | 对CoCap的主要差异 | 决策/难度 |
|---|---|---|---|---|---|
| Global Strong #1 | **TR-11 Rao等 Dynamic Robust Pursuit，IEEE/CAA JAS 2026，13(4):989–991** | 整套NOT_FOUND；独立SDQP组件可复用 | 动态分组、A*代价、slot拍卖、C3AVC＋CLF–CBF LP；≥3机/目标仿真 | 全目标/团队汇集；无自主发现/释放搜索；等能力积分器→AW9且追捕者更慢 | **CONDITIONAL_GO，高** |
| Global Strong #2 | TR-13 Chen等 CEO-DMPC，RA-L 2024，9(3):2064–2071 | NOT_FOUND | 两层MILP分组/槽位、leader–follower、QP形式DMPC；圆周巡逻与发现切换 | full real-time通信；原模型double-integrator；无已证实完成释放/新波；碰撞软惩罚 | CONDITIONAL_GO，高至很高 |
| Global补全文储备 | TR-01 Tan等 MK-WTA＋OAPM，RAS 2025，191:104990 | NOT_FOUND | 多目标k人竞争分配、围捕避障、实物报道 | 完整正文、中央/分布式、矩阵/OAPM公式和生命周期待核 | CONDITIONAL_GO，高至很高 |
| Local Distributed #1（任务骨架） | **TR-07 非学习KS-COAL，RA-L 2024，9(1):443–450** | NOT_FOUND | 区域探索、动态coalition、邻接chain switch/共识 | scout无限通信；动态target捕获不等于三机ring；Accelerated版本含HGAN | **CONDITIONAL_GO适配，高至很高；完整忠实matched-local复现尚不成立** |
| Local Distributed #2（几何/角色组件） | TR-10 GB-VCE，Biomimetics 2026，11(7):513 | NOT_FOUND，只有结果数据 | 视觉发现/失联搜索、角色增长、圆周slot控制 | 只验证单目标；全局Γ/argmax/TopK协议缺口；无多目标/移除释放 | CONDITIONAL_GO组件；完整多目标NO_GO，很高 |
| 分配/控制储备 | TR-06 DMK-WTA＋PPM，TSMC 2024，54(10):5989–6000 | NOT_FOUND | 出版社确认分布式竞争分配＋predator-pack几何控制 | 全文未得；未核局部未知发现、人数与循环 | CONDITIONAL_GO补证 |
| Simple Heuristic | TR-05 Local Spread＋Greedy Coalition＋Geometric Surround | 项目待实现，无外部官方代码 | 拟实现完整局部任务链 | 项目自定义；不能称上述论文复现；需给合理搜索/再分配与validation调参 | GO设计preflight，低至中 |

**没有核实到一个同时满足局部未知发现、有限通信、多目标每目标≥3机几何围捕和完成后再搜索、且有完整官方开源的传统方法。**Local名单是最接近的文献骨架/组件排序，不把缺失环节算成论文贡献。若正式论文要求忠实完整复现，当前Local文献栏应留空并说明证据边界。

## 3. 如果只复现一个传统方法

选 **TR-11**。理由是学术任务匹配、近期正式发表、明确多目标/多人几何组织、全文＋补充可逐式重建；它提供有竞争力的信息特权对照，有能力在已知目标子任务上优于CoCap。相对MK-WTA，不是断言控制性能更好，而是目前重建证据更完整。

先进入不训练的preflight：冻结共同合同→验证原生公式→验证AW9/相对速度可行性→登记新增capture/释放/coverage接口→再决定正式评估。若量化动作导致不可行，不能改变动力学迁就它；保留风险并转核CEO-DMPC或回到MK-WTA补证。

## 4. 科学比较与主要结论边界

- Global组：normal capture、时间、联盟利用、安全、计算/通信、规模。标Privileged Comparator，不与局部主表混成信息公平排名；不假设其一定优于CoCap，因此也不是数学意义的性能上界。
- Local组：发现效率、最大等待接近、累计捕获、三波服务、捕获后恢复、成本和泛化。相同normal事件、AW9、初始化与到达合同；为传统方法提供合理搜索和重新分配。
- 目前只能说CoCap已具备同策略局部任务链的现有评估资产；不能从文献缺模块或复杂度符号推出CoCap更快、更安全或更强。收益必须由共同合同下实测回答。

## 5. 尚未关闭的证据缺口

TR-01/02/06与2025 TR-04正文未取得；两篇Elsevier综述的PDF页码未核。正式related-work措辞只用已验证证据；缺口清单已写入主registry。结论为**值得进入有条件复现preflight，尚不值得直接启动正式复现/训练**。
