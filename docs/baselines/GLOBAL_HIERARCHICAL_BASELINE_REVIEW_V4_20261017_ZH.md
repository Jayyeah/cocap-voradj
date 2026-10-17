# CoCap 第四轮：全局信息传统分层协同围捕基线

日期：2026-10-17。继承 V3 远端 `a3726025352a3cf0a688ea79d04b097e68c4b8b0`。本轮完成文献、代码静态审计和设计；未实现算法、运行 preflight、训练或测量性能。

## 1. 本轮建议

**值得新增一个“全局敌人信息 + 集中式分配 + 几何围捕 + 捕获后重分配”的传统基线。** 建议采用下表中的 G-CAP 作为可实施的项目组合方案，同时保留 TR-11 作为最值得复现的完整已发表方法。两者身份分别登记，不能把自己接入的 AW9 控制器称为文献官方实现。

| 选择目的 | 首选 | 理论与控制 | 代码和难度 | 决策 |
|---|---|---|---|---|
| 新增一个集中式、贴合 CoCap 接口的传统系统 | **G-CAP：Global Quota & Slot Assignment + AW9 Geometric Control，TR-15** | 固定人数配额；最小费用流分组 + 线性槽位指派，或一次全局槽位指派；目标中心多边形；AW9 前向预测；完成事件触发重分配 | 分配/路径组件有开源；完整系统需自建，中至高 | **CONDITIONAL_GO**，推荐下一步 preflight |
| 只复现一篇完整已发表传统围捕论文 | **TR-11 Dynamic Robust Pursuit**，JAS **2026** | A* 路径代价、平衡分组、多边形槽位拍卖、概率安全 Voronoi 域 C3AVC、CLF/CBF 线性规划 | 完整官方代码未找到；高 | **CONDITIONAL_GO**，先原生公式再 AW9 |
| 集中分组/槽位层最直接的论文依据；完整控制备选 | **TR-13 CEO-DMPC**，RA-L **2024** | 人数约束二元分组、正多边形位置指派、leader–follower、预测控制 QP | 完整官方代码未找到；原生全框架高至很高 | **CONDITIONAL_GO** |
| 最新全局人数自适应分配备选 | **TR-14 AFRA**，Drones **2026** | 蒙特卡洛经验库 + 多选择背包 DP + APF | 作者代码存在但关键行为与正文不符，`OFFICIAL_PARTIAL`；分配中、完整适配高 | **CONDITIONAL_GO（研究备选）**；直接照搬代码 **NO_GO** |

TR-11、TR-13 原文的分配/控制并非都由一个中央节点执行。它们使用全局信息/全实时队内通信，可以支撑本轮比较，但“中央节点只求解一次”的版本应明确写作适配版本。[TR-11 正文与补充](https://www.ieee-jas.net/article/doi/10.1109/JAS.2025.125684?pageType=en#Supplements)，[TR-13 作者 PDF，pp.3–4、6](https://april.zju.edu.cn/wp-content/papercite-data/pdf/chen2024adp.pdf)。

## 2. 为什么这比继续增加候选更有价值

当前需要检验的是：**如果直接给传统系统全部当前敌人状态，并允许中央协调，它能否通过合理的资源分配和几何控制，高效完成相同 normal capture？** 因此主场景不应要求它先寻找已知敌人，也不应把每目标人数锁死为三。主合同 4/1、8/2、12/3 均允许每目标四机；三机是最低捕获要求。四机、均衡配额及重分配频率应通过共同验证预算确定，不能为了让 CoCap 占优而省略。

G-CAP 的分组和几何槽位有 TR-13 式(7)–(8)的直接依据；A* 代价与障碍安全域有 TR-11 依据。将其改为中央求解、接上 AW9 和移除/释放状态机，是本项目的组合设计。**它有明确文献来源，但不是任何一篇论文的完整复现。** 完整蓝图见[分层方案](GLOBAL_HIERARCHICAL_BASELINE_BLUEPRINT_V4_ZH.md)。

最新 AFRA 确实把多目标人数分配写成 DP，而不是只做目标跟踪。它优化的是**预期捕获目标数**，允许给部分目标分配零机；这与“全部目标正常捕获”的主指标并不等价。其经验库也来自不同尺度、速度、动力学和捕获规则，不能迁移后直接使用。[正式 PDF，pp.9–11 §4、pp.13–14 §5.2](https://mdpi-res.com/d_attachment/drones/drones-10-00406/article_deploy/drones-10-00406.pdf)。

## 3. 特权信息必须分档

| 配置 | 可用信息 | 不自动授予的内容 | 实验身份 |
|---|---|---|---|
| **G-CAP-GE（首选）** | 所有活动敌人的当前位置/速度、全队当前状态；集中计算与下发；障碍按允许感知再融合 | 完整障碍地图、隐藏未来轨迹、APF 内部动作、未到达目标 | Global-enemy Privileged Comparator |
| **G-CAP-GEM（可选上界）** | GE + 完整障碍地图 | 目标未来真实动作/轨迹 | 额外 map-privileged comparator，单独列结果 |
| **H-L（沿用 V3）** | 局部发现、明确预算的邻居消息与记忆 | 全局敌人列表/地图 oracle | 局部传统系统，独立比较组 |

用户本轮授权全局敌人信息，不应据此无说明增加障碍地图和未来目标策略。全局版的新到达目标由 oracle 当即通知，发现延迟为协议设定；不能把这一数值当作自主搜索能力。无目标时可执行部署/CE 恢复，有目标时直接分配。V3 的搜索层及局部系统继续保留，不被本轮替代。

## 4. 本轮新增证据与负面结果

- AFRA（Liu、Li、Zhu，Drones 10(6):406，2026-05-25，DOI [10.3390/drones10060406](https://doi.org/10.3390/drones10060406)）全文已读重点章节；官方 Gitee 固定 commit 已审计。多目标代码仅用四机入半径判捕获，未执行正文角条件；经验库查询路径使用手写 logistic；动态重分配脚本只有 `.pyc`，没有对应源文件。详见[开源审计](GLOBAL_HIERARCHICAL_OPENSOURCE_AUDIT_V4_ZH.md)。
- 新读 Mu 等的正式 pursuit–evasion 综述（FITEE 2023），并回查更快目标追捕的经典原论文；人数、速度比、初始几何和最终捕获定义必须一起审查。CoCap 的最大速度比 3/3.5 不能直接套用别的连续控制定理。
- 重读 ICRA 2021 的 MPC cooperative hunting 全文：其集中耦合控制、Hungarian 角度指派和 WHCA* 值得参考，但只研究单目标，且假定目标/障碍状态和行为已知，不能提升为现成多目标全局基线。
- 2024 Dynamic Alliance 会议论文已核正式元数据，全文未取得；TR-01 MK-WTA/OAPM 与 TR-12 PGM 的全文证据缺口仍在。本轮不以摘要补全其控制律或生命周期。

全部页码、公式和正式发表状态见[文献证据](GLOBAL_HIERARCHICAL_LITERATURE_EVIDENCE_V4_ZH.md)；横向及逐层选项见 [CSV](GLOBAL_HIERARCHICAL_METHOD_COMPARISON_V4.csv)。

## 5. 相对 CoCap 的科学价值与边界

全局中央系统消除了未知目标发现和消息传播的困难，资源分配明确，适合分析多人联盟与槽位效率；它有可能在捕获成功率、完成时间和等待接近时间上优于 CoCap。其代价是全局遥测、中央决策与显式模型依赖，且连续控制理论不自动覆盖 AW9、swept 碰撞和移动敌人的速度优势。谁更快、更安全必须测量，不能由“传统算法复杂”推出。

全部方法共享 AW9、normal capture、初始化、目标策略、碰撞与停止合同。按信息条件分别报告；同时测人数使用、切换次数、每目标等待、计算分层耗时、全队 wall-clock 和上下行字节数。重复到达另冻结外生 arrival manifest；不复用依赖 CoCap 的 Z 归零触发协议。详见[可行性与 preflight 门槛](GLOBAL_HIERARCHICAL_PREFLIGHT_FEASIBILITY_V4_ZH.md)。

## 6. 文档与事实源

唯一候选台账仍为 [baseline_registry.json](../../artifacts/baselines/baseline_registry.json)，本轮增加 TR-14、TR-15，原 20 个 ID 及历史字段保留；追加 [search_log.md](../../artifacts/baselines/search_log.md)。来源版本、文件哈希和访问失败在 [V4 manifest](../../artifacts/baselines/global_review_source_manifest_v4_20261017.json)，验证结果在 [validation](../../artifacts/baselines/global_review_validation_v4_20261017.json)。

启动时重新 fetch 并核验 V1 `94753f9`、V2 `ec15f16`、V3 `a372602` 和 IQN-Z 审计 `3038ef5` 的远端 HEAD 与提交历史；从 V3 创建 `docs/baseline-global-hierarchical-review-20261017`。基准依据为同审计版本的 [Benchmark Contract DRAFT](https://github.com/Jayyeah/cocap-voradj/blob/3038ef5cd728acdd302772ca08bf10060685af16/docs/paper/PAPER_ASSETS_20261015/PAPER_BENCHMARK_CONTRACT_DRAFT_ZH.md)，草案身份不变。TERL 必选和固定角色 RL 优先 Related Work 的决定保持不变。
