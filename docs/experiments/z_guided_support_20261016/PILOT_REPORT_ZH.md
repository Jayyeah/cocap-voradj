# Paired pilot 登记与运行回执

本文件在训练前登记；最终运行快照与训练/评价结果由 artifacts/2026-10-16_z_guided_support/pilot_status.json、training_curves.csv、checkpoint_registry.csv、rollout_index.csv 提供。running 状态不是完成结果。

## 已完成的启动 Gate

- 64项单元/合同/逐transition回归通过；optimizer smoke两臂各16步/15次更新，断点映射恢复通过。
- 15条归档固定动作轨迹，9,636 agent-transitions账本与物理状态对照通过；326有效Support，self tie=0，alpha全1。
- 全角色PBRS审计9,576相邻转移通过，额外权重残差15处、maxabs .70597。
- T2固定动作下9/10带敌场景normal capture，1/10发生碰撞（另有5个纯coverage episodes）；物理轨迹在两种reward下相同，不能将其解释为训练后Treatment对比结果。历史Z05正式结果仍来自原独立评价分支。

## 训练与评估预登记

- 共同Stage1 selected step_1300000，完整hash见registry；fresh optimizer/replay/recovery/RNG/env/Z，不是full-state resume。
- train seed=2026101601。两臂4P1E1O，LR=3e-5，epsilon .28→.05/200k，原CE schedule及replay/网络/AW9保留。
- 上限100k/arm，先25k，完成匹配评价与工程告警判断后才到50k/100k；不自动200k、scratch或Stage2/3。
- 每窗口capture/mixed/coverage各50独立episodes ×argmax/sample；seed_base=2026102600（scene offset1000），初态fingerprint配对。sample是native IQN随机分位推理，不是categorical。
- capture下降>10pp或collision上升>10pp、有效attraction<100则暂停；未告警仅工程通过，不是统计非劣性证明。两臂均保留全部预定checkpoint，不挑best screen点。

## 证据解释与下一步

目前只能确认实现、信息来源与工程链路可运行，尚无训练后50episode matched结论。alpha全1意味着旧role gate下新Support暂表现为纯ally attraction，未来若捕获或安全退化，应优先检查失去CE约束、跟随方向和群聚，而不先改Z递推或角色门控。self-tie在首批样本未大量出现，因此不把friend-first混入v0。

完整课程建议：等待matched pilot完成。若警报触发，停止扩展并定位；若积极且可比，再登记fresh-init对照（学习预算应参照2M尺度），之后才安排8P2E2O与12P3E3O。

## 运行与资源规则

worktree=/home/yjq/rl/CoCap1/iqn-z-guided-support-20261016；独立runtime同名加-runtime。Control物理GPU0、Treatment物理GPU1，各进程cuda:0。launch检查两GPU无compute进程及root至少50GiB；运行中低于20GiB停止本线并保留已保存checkpoint，不停止任何他人进程。

按继承AGENTS.md的运行协议，独立CPU工作完成后登记PID、step、checkpoint、throughput、ETA和next wake-up并交付；不无限等待长训。当前用户明确允许本轮有限训练，覆盖旧离线-only限制。当前会话没有核实本实验分支有效push授权，采用本地提交，GitHub待发布。

## 工程修正记录

首轮历史模块加载缺少包名、初版日志不能直接JSON序列化NumPy数组、tiny单任务smoke误用mixed-only recovery字段，均在训练启动前修复。保留的initial_boundary_expectation_failed回执来自零CE权重却要求reset的测试预期；最终unit_tests.xml为64/64通过。未更改Z05科学合同来迁就测试。

## Live 启动与有效学习速度

源码启动commit=5b885c7718071a355fd9916e4d6bfabfeecab674；train seed=2026101601。Control PID3330786 / GPU0，Treatment PID3330787 / GPU1，起始两臂初始化权重hash一致，fresh optimizer/replay/recovery pool和Z全部通过。初始约42 env steps/s/arm（尚未开始反向更新，不能用其估算完整训练速度）。

截至5,000步两臂updates=0；4,000步 replay pursuing=2370、pre_capture_cover=402、post_capture_real=5760、recovery_pure=7468，原最小样本阈值3000使稀少类别成为等待项。25k若仍无更新，matched评价主要是能力保持/评价链路证据，不能声称新奖励已经适应；100k后若学习量不足才单独记录扩至200k理由，不能改原采样合同加速。

50episode训练后matched捕获/碰撞结果：**PENDING**。当前训练episode的比例是带探索的相关样本，不升级为正式评价。完整课程建议仍为等待；scratch/Stage2/Stage3均未启动。NEXT WAKE-UP=2026-10-16 22:15+08或首个checkpoint/错误事件，以实时状态为准。

交付时最新持久化状态：两臂均已采集 13000 env steps，optimizer updates均0；最新rare-class数量、PID、checkpoints与预计25k到达时间见 runtime_snapshot.json。当前没有trained checkpoint或completed paired Gate；init.pt是共同初始权重副本。GitHub状态=LOCAL_ONLY_PENDING_AUTHORIZATION。
