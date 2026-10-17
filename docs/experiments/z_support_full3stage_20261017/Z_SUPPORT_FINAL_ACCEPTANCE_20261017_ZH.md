# Z-Guided Support v0 最终验收（2026-10-17）

**验收结果：CONDITIONAL_PASS。工程阻断已解决，允许并自动执行完整三阶段Treatment课程。** 条件来自尚未证实的性能泛化与效率，不包含待解决的实现或科学差异阻断。用户本轮授权覆盖旧文档的pilot预算限制，不再询问是否开始Stage1/2/3。

## 最新事实与隔离

启动重新`git fetch origin --prune`，真实Support远端HEAD与本地均为`4d7005d2bc80c21d1798b04ba711d258be2b8956`；不是直接假定用户给出的SHA。原工作树干净，无相同CoCap训练或残留pilot进程。新建独立分支`experiment/iqn-z-guided-support-full3stage-20261017`和同名工作树，原Z05及100k pilot全部只读。

已读取AGENTS.md、Git工作流、Support指南/审计/测试/warm-start回执、原Z05训练与恢复协议、独立评估器及`/home/yjq/rl/cocap-voradj/docs/paper/PAPER_ASSETS_20261015/`。原训练器、replay、IQN和基础物理代码与`f0c0dfca8a508233f30ddbb5158699400d92a00f`逐字节一致，SHA见`engineering_provenance.json`。有效当前授权允许完整课程与Git里程碑同步；无须依旧任务限制停在offline。

服务器super，两张A6000各49GB。启动检查GPU0约38GB、GPU1约34GB空闲，已有compute进程属于其他用户，全部保留。系统盘约85GB空闲，可用RAM约82GB。按四个原1M replay类别上限、三份latest完整状态、原子写入和新资产预算约48GiB，保留20GiB底线及额外备份空间；训练GPU0，评估/GIF强制CPU。资源实际值与进程记录在工程来源JSON，不把GPU显示0%误作无人占用。

## 数学与信息合同逐项判断

| 验收项目 | 实际核验 | 结果 |
|---|---|---|
| Z与原主体 | 同步previous-state传播，lambda=eta=.5、hard-zero .1；原捕获/ring、CE/PBRS、normal与stationary、碰撞失活/post-capture、Transformer-IQN共享策略、AW9和语义replay | PASS |
| 候选集合 | self加动作前实际有效friend tokens，保留原physical_only排序和8槽cap；隐藏截断邻居不加入 | PASS |
| 最大Z选择 | Z容差1e-6；self近似相等优先，alpha0；否则最高Z中最近友机；实际token浮点距离同值按可观测几何确定性破平，完全不可分才ID；不平均位置 | PASS |
| 奖励 | self完整CE；friend alpha=z_j，before友机固定锚点，仅自身移动进展clip3，approach系数1；alpha ally+(1-alpha)完整CE | PASS |
| scaling与PBRS | static_capture_scale实际为1，吸引权重仅乘一次；完整CE仅乘一次1-alpha；同一实际权重传给phase/terminal PBRS correction，零权重不消不存在项，同步最多消一次 | PASS |
| 新增信息边界 | 选择器API没有敌人坐标、隐藏目标ID或全局支援目标；privileged enemy distance只用于原日志诊断，不进入policy输入、TD计算或新Support奖励 | PASS |
| 原角色语义 | 保留effective coverage且一跳effective capture邻居才Support；非Support的非零Z机器人仍原capture/coverage路由；多跳Z进入observation不自动激活Support | PASS |
| 失活与终止 | 双敌情来源移除一个不清另一来源；失活friend不入候选；不强清目标失活后的Z；全部捕获后下步完整CE路由 | PASS |
| Control历史相等 | 2 seeds×3 scenes×最多80固定AW9 transition逐obs/token/mask/reward/terminal/Z/replay key比较原Z05 | PASS |
| 科学差异可核对 | 两臂配置仅run_name和Support开关不同，25k权重相同、后续权重不同，全部launch/config/full-resume合同与模型真实一致 | PASS |

不是“全体机器人基于Z连续切换三种奖励”。原CE与拓扑派生量、奖励角色和捕获事件仍有历史环境信息；本轮只限制新Support目标，不宣称整个系统完全局部化。

## 独立工程测试

没有沿用以前“64 tests passed”的结论。本轮关键组合**91 passed**；另运行原历史删失效率测试**2 passed**。JUnit与逐用例矩阵均归档。最后新增保存/队列代码变更后，6个流水线端到端用例再次通过。

覆盖Control逐transition回归、obs/token/mask、同步传播/自然衰减、多最大Z/self tie/最近最大Z、8-token截断/失活、两个真实敌情来源失活其一、固定锚点、实际normal捕获和PBRS边界/terminal消项、真实stationary阈值、checkpoint精确恢复、finite reward/loss/梯度。

新执行器额外验证：first episode尚未结束时12决策内产生两个快照，即使两个评价任务仍pending，训练照常结束；与原训练器的model/target/optimizer/RNG/Z完全逐位一致；6+6分段中途恢复与12步连续训练逐位一致，实际policy token映射恢复一致；只读快照拒绝同名改权重；死worker认领、dedup及3次尝试上限；原生sample与model.act一致；复用原episode runner的轨迹相等、完整缓存恢复汇总一致、GIF解码通过。

正式结构GPU scratch数值诊断独立完成16+1决策、15次更新并恢复后再更新，loss有限，梯度在optimizer.step前检查；峰值reserved约202MiB。此诊断的小batch/replay不用于正式合同，不加载历史权重，也不计入Stage1。

## 100k配对数据重审

19个登记checkpoint全部重算SHA一致；两臂25k/50k/100k分别0/4225/16725次真实optimizer更新；100k full-resume的online模型与final权重逐tensor一致，配置contract hash及launch hash全部复核。

三个窗口×两臂×三场景×两模式×50=1800次执行；900对seed与初态fingerprint一致。相同seed在窗口/模式复用，不能叫1800个独立初态。全部36个scene/mode摘要格及1800个逐局文件存在，统计从逐局重算。25k没有学习，不能作方法改善证据。

| 100k模式 | 场景 | Normal C/T | Collision C/T | CE或post-capture恢复 C/T | 完成条件下捕获均时(s) C/T |
|---|---|---:|---:|---:|---:|
| argmax | capture | 50/50，50/50 | 0/50，0/50 | — | 31.96，34.28 |
| argmax | mixed | 48/50，49/50 | 2/50，2/50 | 0/50，46/50 | 32.1771，34.9388 |
| argmax | coverage | — | 0/50，0/50 | 0/50，47/50 | — |
| sample | capture | 49/50，50/50 | 1/50，0/50 | — | 32.8980，33.05 |
| sample | mixed | 48/50，49/50 | 2/50，2/50 | 9/50，47/50 | 32.3646，35.5918 |
| sample | coverage | — | 0/50，0/50 | 38/50，49/50 | — |

Treatment保持normal capture；各窗口没有持续性严重捕获退化或碰撞恶化。50k argmax mixed/coverage的碰撞最多增加4pp，100k没有延续到同等幅度；不能忽略单局风险。100k捕获均时略慢：argmax纯捕获约+7.3%、mixed约+8.6%；sample mixed约+10.0%。这些时间基于成功episode且分母不同，不能称无条件速度变慢的因果估计；更不能称围捕加速。

**没有触发10个百分点工程预警，不是统计非劣性证明。** 单训练seed、每场景50局和warm-start不证明scratch或全课程优越。正面工程信号是Coverage与post-capture恢复保持，足以进入原尺度完整课程。

## Control CE退化归因与设备数值边界

已完成18个完整horizon归因episode：Control50k/100k及Treatment100k各3个原归档seed，分别使用Control与Treatment环境开关。GPU原推理路径下9个归档结果精确复现；相同checkpoint交换Support开关的纯Coverage结果和轨迹统计相等。Control50k→100k的模型权重确有变化。现有证据排除纯Coverage评估配置开关伪差异，不能进一步把优化遗忘的具体原因说成已识别。

CPU首次严格末值复现未通过，失败记录保留。额外同checkpoint/同输入CPU-CUDA诊断发现最大Q差1.5259e-5，第2742步两动作近乎相等时argmax不同；这是明确数值敏感性证据，不能把旧GPU轨迹要求为CPU逐位重现。后续selection与final均独立CPU，同一最终seed对历史模型也统一CPU复评；主推理模式仍固定midpoint32，不按更好的设备或sample结果选主表。

## 披露的限制

100k Treatment全部capture/mixed两模式共10,214个Support转移：10,191次alpha1（99.775%）、20次alpha.5、3次self tie alpha0。原门控邻接effective Capture且常有直接敌情Z1，最大Z选择自然饱和；不是漏乘CE或alpha实现错误。alpha1确实去除Support的Coverage分量，不修改公式。此方法变化应归因于友机目标和动态权重的联合最小改动，不能单独声称Z传播带来收益。

多跳Z不改变原角色门控；动态加权CE/PBRS一般不再全程固定potential望远镜，已有边界残差需继续报告；单seed尚无方法优越性或统计非劣性结论。以上均已披露，不构成工程阻断。

## 执行签收与下一步

`acceptance_gate.json`签收为CONDITIONAL_PASS且`engineering_blockers_resolved=true`。自动执行Stage1 scratch 2M → 本线selected → Stage2 700k → 本线selected → Stage3 700k。按原BalancedFloor选模，不预填历史最佳步数。正式主评价argmax；独立seed和完整20-rollout/10-GIF每场景合同冻结。

原trainer每个环境决策推进global_step，episode成功与否不影响预算；训练只保存不可变快照并入SQLite队列，不执行评价。独立CPU评估和渲染不会修改训练RNG、replay、optimizer、Z或episode。阶段交接签收后可立即进入下一Stage，GIF后台继续。科学错误由队列通知训练在安全决策边界保存后暂停，有限重试与失败记录持久化。

部署与执行细节、配置/seed/资源/恢复命令见`Z_SUPPORT_FULL3STAGE_TRAINING_CONTRACT_ZH.md`。各阶段报告和comparison会随真实结果自动刷新；训练正在运行不等于完成。大型资产留服务器，代码、验收报告、registry和轻量指标按里程碑同步GitHub并核验远端HEAD。
