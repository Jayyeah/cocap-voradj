# CoCap 第三轮：传统分层替代方案与逐层文献、代码筛选

日期：2026-10-17。检索截止同日。状态：设计与证据审查完成；尚未实施或运行 reproduction preflight。

## 1. 决策先行

**如果不使用 CoCap 的 RL，建议构建“持续搜索—局部跟踪—最小人数联盟—几何围捕—动力学执行—释放再搜索”的闭环。** 对当前问题，分配层必须处理每目标至少三机，搜索层必须处理目标尚未发现以及捕获后的再次出发；只换一个连续围捕控制器不足以替代现有方法。

推荐保留三种清楚区分的实验对象：

| 对象 | 推荐构成 | 科学用途 | 本轮决定 |
|---|---|---|---|
| **H-G：完整论文强基线** | TR-11 Dynamic Robust Pursuit，JAS 2026；备选 TR-13 CEO-DMPC，RA-L 2024 | 已知目标/全局信息下的传统强比较；单独列 Privileged Comparator | **CONDITIONAL_GO**：先做原生公式与 AW9 可行性 preflight |
| **H-L：局部传统分层组合** | 持续年龄搜索 + GRAPE 第 V 节最小人数分配 + 环形槽位/角间隔反馈 + AW9 前向预测 + 释放状态机 | 最贴近未知目标、多轮服务的科学问题；是文献支撑的项目组合方法 | **CONDITIONAL_GO**：推荐进入小规模 preflight；不能称为 GRAPE 官方围捕复现 |
| **H-S：简单透明辅助对照** | 同一搜索、执行、生命周期接口，联盟层用带人数承诺的局部贪心招募 | 分离复杂联盟算法是否带来收益；沿用 TR-05 项目启发式身份 | **GO（设计就绪）**；尚未实现，不能列已有成绩 |

**若只能复现一篇完整传统围捕论文，仍选 TR-11。若只能做一个贴近 CoCap 场景的非 RL 系统，优先 H-L 的 preflight。** 两个问题的答案不同：前者强调论文方法完整性，后者强调任务生命周期和信息条件匹配。

本轮没有找到经核实、可直接运行且同时满足“未知多目标发现、有限通信、每目标三机几何捕获、移除释放、再次搜索、AW9”的完整官方实现。此结论限于记录的检索范围，不等于公开实现绝对不存在。

## 2. 第三轮真正增加的证据

1. **GRAPE 的最小人数机制来自原论文，而不是我们发明的补丁。** Jang、Shin、Tsourdos，TRO 2018，34(6):1534–1548，[DOI](https://doi.org/10.1109/TRO.2018.2858292)。已读 [arXiv 全文](https://arxiv.org/pdf/1711.06871) PDF p.9 §V、式(23)–(28)、Condition 1、Proposition 1。辅助效用使未满足人数要求的任务获得招募优先级。其结论依赖总人数足够、偏好条件、信息传播与稳定任务实例；不是有限带宽、移动目标、不断变化任务下的无条件保证。
2. 作者 [pyGRAPE](https://github.com/inmo-jang/pyGRAPE) 有可定位的 Python 分配核心，作者 [SPACE](https://github.com/inmo-jang/space-simulator) 有局部感知、邻居消息、随机探索、任务完成及新增任务框架。但所查示例没有 GRAPE §V 的最小人数辅助效用，也没有围捕控制；SPACE 直接传递任务对象和初始化全任务 ID，不能原样认定满足 CoCap 的未知目标合同。
3. **2026 新覆盖方法值得采用为搜索层候选，但不应误写为局部分布式。** C* 已正式发表于 TRO 42:1233–1253，[DOI](https://doi.org/10.1109/TRO.2026.3661719)；多机扩展使用中央服务器共享地图和 RCG，重分配不在其范围内。Multi-CAP 的空间连通性也不是通信连通性；原文使用 base station，当前核实到的该标题版本是 2025 预印本。
4. **HEDAC 迷宫探索已于 2026-10-05 正式发表**，Mathematics 14(19):3613，[DOI](https://doi.org/10.3390/math14193613)。官方代码可读，但未见许可证和 README 所指 requirements 文件；原文 p.28 §4.3 把去中心化列作可改造方向，实际算法是集中式，不能把方向写成完成能力。
5. 新增阅读 2026 TAMP 正式综述、2026 CPP 预印本综述及 2025 分布式优化 tutorial，结合上一轮已读的 2025 联盟形成综述。逐项页码和审查意见在[综述与近期方法证据](TRADITIONAL_SURVEYS_AND_RECENT_METHODS_V3_ZH.md)。综述提供分类和检索线索；实际通信、保证和代码能力仍回查原论文与源文件。

## 3. 每层如何选

| 层 | 建议先做 | 可替换方法 | 主要代价/边界 |
|---|---|---|---|
| 搜索 | 局部年龄记忆 + 周期巡扫/空间分散 + 死锁恢复 | 动态密度 Voronoi；C*；Multi-CAP；LS-MCPP；HEDAC | 静态 Lloyd 仅收敛到分布，不等于持续搜索；后三类常需全图/全队信息 |
| 发现与信息 | 本机检测 + α–β/KF 跟踪 + 带时间戳邻居转发 | 分布式观测器/共识估计 | 目标消失不等于已捕获；不得读环境隐藏目标列表或真实 APF 未来动作 |
| 联盟 | GRAPE §V，最低人数取 3；先在冻结快照验证 | 局部贪心招募；KS-COAL 非学习骨架；集中式最小费用流/MILP | GRAPE 需要一致任务集/偏好条件；原代码不含最小人数分支；不能简单把一个目标拆成三个无承诺独立任务 |
| 几何 | 正多边形槽位 + 保序分配 + 半径/角间隔反馈 | TR-11 polygon auction；TR-13 CEO-DMPC；bearing/circumnavigation/RMPC 模块 | 围住目标群、两机反相、保持跟踪距离都不等于每目标三机 normal capture |
| 执行与安全 | 使用环境相同动力学预测 AW9；筛选动作并记录不可行事件 | 小规模 MPC；CBF-QP/Robotarium；SDQP 求解组件 | 连续速度截断/量化不能自动保留原安全或收敛证明 |
| 生命周期 | SEARCH→RECRUIT→RENDEZVOUS→ENCIRCLE→RELEASE→SEARCH；另有 LOST/超时分支 | SPACE 行为树作为结构参考 | 任务完成/目标移除反馈、租约、搜索保留人数均需显式实现 |

每层的公式、接口、替代条件及成本分解在[完整蓝图](TRADITIONAL_HIERARCHICAL_BLUEPRINT_V3_ZH.md)；可排序表在[逐层选项 CSV](TRADITIONAL_LAYER_OPTIONS_V3.csv)。

## 4. 开源与移植判断

| 资产 | 核实范围 | 许可/主要依赖 | CoCap 移植难度 |
|---|---|---|---|
| TR-11 / TR-13 | 全文与公式；未找到完整官方代码 | TR-11 原文 Mosek；TR-13 Python/Quadprog/OR-Tools | 高 / 高至很高；主要风险是 AW9、速度差和信息条件 |
| GRAPE / pyGRAPE | 官方分配示例；§V 最小人数分支未见实现 | MATLAB 版 MIT；Python 版 GPLv3，NumPy/Matplotlib/ImageIO | 分配核心低至中；整个 H-L 系统中至高 |
| SPACE | 官方局部任务分配模拟与动态任务框架 | GPLv3，pygame/py-trees/PyYAML 等 | 结构借鉴中；直接移植整个模拟器收益低 |
| LS-MCPP TRO 2025 | 官方 ESTC、局部搜索、PBS/低层冲突消解与实例 | GPLv3；setup 声明 gurobipy，核心 .py 未检到其调用 | 已知图搜索组件中；转成局部持续服务高 |
| Robotarium | 官方 SI/unicycle 控制与 barrier 示例 | MIT；Python≥3.10、CVXOPT | 参考接口低至中；AW9 安全适配中至高 |
| HEDAC 2026 | 官方探索代码与批次脚本；pyproject/uv.lock 存在，README 安装路径不一致，许可未见 | NumPy/SciPy/Matplotlib 等；版本可获取性尚未验证 | 原生复现中；局部持续搜索改造高 |

详见[七个仓库的固定 commit 审计](TRADITIONAL_COMPONENT_CODE_AUDIT_V3_ZH.md)。这些是组件可用性判断，没有运行它们，也没有把演示 GIF 当作本地复现实验。

## 5. 比较 CoCap 时必须保留的边界

H-G 允许全局目标状态，可能在捕获成功率与完成时间上优于 CoCap；它回答全局信息下的分配/控制能力。H-L 回答局部发现、多目标累计捕获、最大等待接近时间、释放再搜索和重复服务效率。两组均共享 normal capture、AW9、swept 碰撞与初始化。

H-L 即使只通过邻居通信，若转发额外目标坐标、完整 partition 或长期地图，也不能直接标作与 CoCap token 输入完全公平。需要分别登记图、频率、字节数、TTL、记忆和每次决策内允许的通信轮数。历史重复到达 B/C 依赖全队 Z 归零，不能直接用于没有 Z 的传统方法；建议另冻结所有方法共享的外生 arrival manifest。

不预测谁更快，不用“传统方法计算复杂”推导 CoCap 实测延迟优势。算法阶段耗时、整队 wall-clock、消息成本和不可行率要实测。没有运行数据时不报告成功率或秒数优势。

## 6. 建议的下一步

进入[分阶段 preflight](TRADITIONAL_PREFLIGHT_RECOMMENDATION_V3_ZH.md)：先冻结信息/反馈合同，验证 GRAPE 最低人数快照和三机 AW9 单目标几何可达性，再接入 4/1/1、8/2/2、12/3/3 的搜索、释放闭环；同时为 TR-11 做原生论文小例重建。预先规定失败原因与调参预算。**本轮只完成这一可审查方案，没有自动启动实现、preflight 或训练。**

## 7. 追溯与阅读入口

- 继承 Round 2 远端 HEAD：`ec15f16eca08f9071f35ba4cbcccfcf1542a5d2a`；其父 Round 1 HEAD：`94753f9fd82d137af7ff9930634788ffeba6f3e2`。
- IQN-Z 资产审计远端 HEAD：`3038ef5cd728acdd302772ca08bf10060685af16`。本轮重新核对其 [Benchmark Contract 草案](https://github.com/Jayyeah/cocap-voradj/blob/3038ef5cd728acdd302772ca08bf10060685af16/docs/paper/PAPER_ASSETS_20261015/PAPER_BENCHMARK_CONTRACT_DRAFT_ZH.md)，没有将草案写成已冻结标准。
- 沿用 [唯一 registry](../../artifacts/baselines/baseline_registry.json) 与[搜索日志](../../artifacts/baselines/search_log.md)。新增 literature_sources、components、architecture_review_v3，原候选及历史快照继续保留。
- 新证据版本、PDF SHA256、固定代码 commit 和搜索状态见 [V3 source manifest](../../artifacts/baselines/hierarchical_review_source_manifest_v3_20261017.json)。PDF/外部代码缓存位于 `/tmp/cocap-round3-sources`，不上传第三方全文或源码。
- RL 决策沿用 V2：TERL 必选；固定角色 RL 优先 Related Work。本轮没有扩张 RL 候选或开始训练。
