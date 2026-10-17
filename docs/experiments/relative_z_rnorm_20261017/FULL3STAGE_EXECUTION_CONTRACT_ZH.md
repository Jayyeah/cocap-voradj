# 完整三阶段执行与验收合同

旧线已P0安全暂停；新工作树/home/yjq/rl/CoCap1/iqn-zsupport-relative-gate-rnorm-20261017，新分支experiment/iqn-zsupport-relative-gate-rnorm-20261017，新runtime同工作树路径加-runtime；checkpoint、evaluation、trajectory/event/GIF均在新runtime，不覆盖旧线。

| 阶段 | 原规模 | 环境决策预算 | 初始化 |
|---|---|---:|---|
| Stage1 | 4P1E1O | 2000000 | 随机scratch；禁止旧Z05/pilot/800k/任意训练权重 |
| Stage2 | 8P2E2O | 700000 | 本线Stage1 selected，重建optimizer/replay/env |
| Stage3 | 12P3E3O | 700000 | 本线Stage2 selected，重建optimizer/replay/env |

沿用旧线2026101701/02/03种子以免额外变量。AW9、Transformer-IQN hidden256/8heads/4layers、原IQN目标/优化器、episode交替/recovery pool、四类Replay配额64/16/32/16及min3000、epsilon/LR schedule、CE控制权重schedule、100k checkpoint、25k rolling exact-resume、BalancedFloor及全部selection候选保持。每次真实env.step才global_step+1，成功或episode结束不改变步数推进。

严格顺序：主Agent内部静态/动态→独立gpt-6.1-sol xhigh APPPROVE→正式CPU/CUDA/Replay/optimizer/checkpoint/exact-resume/evaluation/renderer/supervisor smoke全部通过→自动Stage1 scratch。不因代码完成或启动而结束；主Agent必须实际监督到Stage1>=300000并完成健康签收。若错误则安全决策边界保存暂停、修复、独立重审与smoke，必要时重scratch。不得禁用检测。若无法安全暂停/指定审核能力不可用/科学错误/资源风险，报告阻断，不降标准。

正式smoke必须有真实优化更新和四类非空Replay，不以20个无学习决策宣称成功。验证解析/科学断言、Relevant tests、CPU真实转移、CUDA前反向、四类写入采样、真实短训、checkpoint、exact-resume逐位对照、低Z(.5/.25/.125)触发/alpha分布、CE/loss/Q/gradient有限、evaluation/renderer、监督器暂停恢复和队列持久化。

训练在物理GPU0；所有独立selection/formal/reference/render强制CPU独立队列，不占训练主循环；两selection worker和一formal/render worker。仅阶段边界等待全部原候选评价后BalancedFloor选模与selected SHA签收；GIF不阻塞下阶段。三阶段各argmax midpoint32主评价和sample uniform32 IQN mean-Q argmax次评价；每模式三场景各20完整rollout、全部trajectory/event/token logs；每场景固定偶数index10个GIF，共180。沿用原renderer300抽帧、trail70、100ms；不缩短正式episode。selection seed2026091900，final2026102700，scene偏移100000。历史Z05另native统一CPU复评，保持其旧输入归一化，初态配对但不同训练seed不作严格因果结论。

新的pause_request同时阻止worker claim和Supervisor重试/跨阶段交接，等待多GB完整保存时没有60秒杀训练器的超时。数值/科学失败不普通重试；普通崩溃最多3次exact-resume。operator pause保留marker直到显式修复/恢复。任意科学/数值重启都不得借新scratch丢状态。

监督清单：Supervisor/PID/GPU显存；global_step/updates/episode进度；真实吞吐；四类Replay数量和采样；CE物理与R统计；Support总率、self/neighbor-Z分层及alpha；normal/stationary capture、collision/Coverage；loss/Q/gradient finite；checkpoint SHA；SQLite状态；Git同步；磁盘/RAM。不能单靠Support次数证明好转。

25k/100k/200k/300k自动报告并Git同步。吞吐至少两个不重叠50k窗口后才给稳定ETA；25k仅暂估。每次ETA保存当前绝对时间/Asia-Shanghai、step、窗口、steps/s、剩余小时、当地预计完成时间、不确定性与是否含评估渲染。Stage2/3未知速率明确外推。训练与evaluation/GIF ETA分开；后者在真实service未测得前记未知，之后披露外推。

300k签收条件：真实步>=300k、充分真实更新、无未解决合同/NaNInf、低Z路由正确、实际日志R生效、checkpoint正常、队列正常或持久等待、后台可自动完成三阶段、资源安全、300k报告已GitHub核验。300k只允许结束主动监督，不代表三阶段完成；Supervisor继续运行直到所有预算和资产队列闭合。

只同步代码、配置、报告、轻量指标和registry；不上传大型checkpoint/原始GIF。每阶段commit/push并ls-remote核验。最终回复仅在300k健康签收后提交成功验收，或在用户指定硬阻断时如实说明阻断。
