# Z-Guided Support Stage1 执行结果

状态：**训练中**。未完成任务保留待完成状态，不把预算或运行中状态当作结果。

规模：4 Pursuers / 1 Evaders / 1 Obstacles。预算：2,000,000 environment decision steps。
已持久化进度：500,000环境决策；124,137次optimizer更新；670个episode。三者独立计数。

初始化：scratch随机初始化，禁止历史或pilot权重。
配置：`/home/yjq/rl/CoCap1/iqn-z-guided-support-full3stage-20261017-runtime/stages/stage1/resolved.yaml`；训练日志：`/home/yjq/rl/CoCap1/iqn-z-guided-support-full3stage-20261017-runtime/stages/stage1/training`。

## 所选模型与交接

待全部100k selection checkpoint评价完成后，按原BalancedFloor及原tie-break顺序选择；没有预定旧Z05最佳步数。

交接Gate：`/home/yjq/rl/CoCap1/iqn-z-guided-support-full3stage-20261017-runtime/stages/stage1/handoff_gate.json`；下一阶段：stage2。
训练完成凭据：`/home/yjq/rl/CoCap1/iqn-z-guided-support-full3stage-20261017-runtime/stages/stage1/training_complete.json`。GIF渲染不作为下一阶段启动前置条件。

## 正式独立评价与可视化

主模式：`/home/yjq/rl/CoCap1/iqn-z-guided-support-full3stage-20261017-runtime/stages/stage1/independent/argmax/report.json`；次模式：`/home/yjq/rl/CoCap1/iqn-z-guided-support-full3stage-20261017-runtime/stages/stage1/independent/sample/report.json`。
每模式三场景各20固定seed、全部episode纳入统计；60 trajectory、60 event日志；每场景固定偶数index共10 GIF，选取与成功或失败无关。每份GIF登记seed、源轨迹SHA、renderer参数和命令。失败任务最多3次尝试，未成功项目保留在failure_retry_registry和队列中。

正式结果尚未产生；不填写推测性成功率。

## 产物登记

服务器原始目录：`/home/yjq/rl/CoCap1/iqn-z-guided-support-full3stage-20261017-runtime/stages/stage1`。代码库轻量registry见`artifacts/2026-10-17_z_support_full3stage/`。
