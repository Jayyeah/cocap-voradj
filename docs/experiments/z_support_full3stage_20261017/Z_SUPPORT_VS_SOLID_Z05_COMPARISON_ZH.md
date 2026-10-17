# Z-Guided Support 与 Solid IQN-Z05 对照

当前执行状态：`PREPARED`；阶段：`待启动`；过程：`准备`。

主结果冻结为 argmax / fixed-midpoint32；sample 是32个随机 IQN 分位均值 Q 的 argmax，只作独立次要结果。各行使用原生4P1E1O、8P2E2O、12P3E3O合同。旧正式结果引用既有native批次；新最终评价seed族2026102700与selection的2026091900分离。新旧训练seed不同，不能描述为严格matched训练因果实验。队列另用新最终seed对历史正式selected做统一argmax复评；复评完成时表中使用该结果，补全逐局存活人数并验证新旧评价初态配对；此前引用既有报告并明示。

| 阶段 | 模型 | selected step | Capture normal | Capture完成均时(s) | Coverage CE | Mixed capture | Post-capture CE | Collision(CE/Cap/Mix) | Mixed平均存活 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| stage1 | 历史Z05正式selected（既有报告） | 1300000 | 1.0000 | 35.5500 | 1.0000 | 1.0000 | 0.9000 | 0.0000/0.0000/0.0000 | 旧报告未登记 |
| stage1 | 新Z-Guided Support | 待选择 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |
| stage2 | 历史Z05正式selected（既有报告） | 100000 | 1.0000 | 40.0750 | 0.9000 | 0.9500 | 0.2000 | 0.0500/0.0000/0.2500 | 旧报告未登记 |
| stage2 | 新Z-Guided Support | 待选择 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |
| stage3 | 历史Z05正式selected（既有报告） | 600000 | 1.0000 | 37.8000 | 1.0000 | 1.0000 | 0.8500 | 0.0000/0.0000/0.1000 | 旧报告未登记 |
| stage3 | 新Z-Guided Support | 待选择 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |

完成时间仅在达成事件的episode上统计，同时登记成功分母、失败与删失，不能解释为无条件速度优势。旧全量报告没有逐局存活人数，明确记为未登记；不从GIF子样本或碰撞率反推。

Repeated-arrival适应性：下一步在本线Stage3 selected冻结后，复用既有strict repeated-arrival协议与held-out seed检验。当前完整课程不自动扩展为多波研究；没有数据时不声称已改善。

限制：当前旧门控使Support常选Z=1友机，Coverage混合权重常为0；多跳Z只影响policy observation，不自动改变旧角色；动态CE权重保留原PBRS项但不具有全程固定potential望远镜保证。单训练seed不能证明方法优越。

历史报告：`/home/yjq/rl/CoCap1/iqn-z05-independent-20260923/artifacts/2026-09-23_iqn_z05_independent`；所有失败阶段及全曲线保留。
