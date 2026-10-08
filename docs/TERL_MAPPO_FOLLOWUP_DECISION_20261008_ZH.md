# TERL-MAPPO Stage1 后续决定与诊断

更新时间：2026-10-08T23:58:55.365574+08:00。
当前分类：`TERL_MAPPO_STAGE1_PARTIAL`；实际预算进度：100,000/100,000。

分类和best保持启动前预声明的supervisor规则；诊断不参与重选，不改训练合同或超参。

## Checkpoint诊断

| step | domain | mode | n | normal | collision | ring2/ring3/strict | entropy / log9 | boundary驻留占比下界 |
|---:|---|---|---:|---:|---:|---|---:|---:|
| 100000 | final | argmax | 50 | 1 | 49 | 27/2/1 | 0.9662 | 0.00% |
| 100000 | final | sample | 50 | 1 | 49 | 18/4/1 | 0.9540 | 0.00% |
| 0 | screen | argmax | 20 | 0 | 1 | 0/0/0 | 1.0000 | 98.37% |
| 0 | screen | sample | 20 | 0 | 2 | 0/0/0 | 1.0000 | 49.07% |
| 25000 | screen | argmax | 20 | 0 | 1 | 0/0/0 | 0.9951 | 98.38% |
| 25000 | screen | sample | 20 | 0 | 3 | 0/0/0 | 0.9949 | 57.12% |
| 50000 | screen | argmax | 20 | 0 | 0 | 0/0/0 | 0.9926 | 16.31% |
| 50000 | screen | sample | 20 | 0 | 9 | 0/0/0 | 0.9926 | 39.51% |
| 75000 | screen | argmax | 10 | 0 | 0 | 0/0/0 | 0.9490 | 19.56% |
| 75000 | screen | sample | 10 | 0 | 1 | 0/0/0 | 0.9494 | 15.29% |
| 100000 | screen | argmax | 20 | 1 | 19 | 9/1/1 | 0.9656 | 0.00% |
| 100000 | screen | sample | 20 | 0 | 20 | 5/0/0 | 0.9536 | 0.00% |
| 100000 | selection | argmax | 20 | 0 | 20 | 11/0/0 | 0.9662 | 0.00% |
| 100000 | selection | sample | 20 | 0 | 20 | 9/1/0 | 0.9546 | 0.00% |

boundary下界由原生Stage1 global reward推导：合作项非负、越界每agent每decision罚-5。它统计agent时间占比下界，不等同于越界episode率或碰撞；未增加终止或安全shaping。

## Reward / PPO / critic / exploration / representation / safety

Reward：逐episode分量与原reward总和审计一致。distance是距离驻留收益，不是进度差；global负值可由反复越界主导。完整每decision分量在training_diagnostics.json。不能用reward上升替代capture/ring证据。
PPO：记录KL、更新后ratio范围和clip统计；actor_loss含entropy项，更新是否发生需结合actor_update_l2与梯度。梯度norm是裁剪前的minibatch值取平均，大于0.5本身不意味着裁剪失效。
Critic：按0–25k/25–50k/50–75k/75–100k窗口报告EV、value_loss和ValueNorm尺度的min/median/mean/max。EV只解释采样rollout的return，不能证明长期围捕几何已学会。
Exploration：独立sample/argmax分开，entropy/log9给出相对均匀AW9的随机程度。高entropy仍可能缺少联合几何探索；若argmax和sample行为差异大，应先诊断动作分布与轨迹，不能直接称探索已解决。
Representation：保留公开TERL未mask的max pooling与完整evader tokens；必要finite mask修复已测试。本轮没有pooling或backbone消融，不能将表现因果归于representation。
Safety：collision与boundary分开。低collision可能伴随高越界驻留，不能据此宣称安全或有效追逃。

训练完成episode 59；capture 0；collision 36。这些是探索训练记录，不能并入独立评估成功率；episode日志ring标志仅为终止时刻。

完整窗口统计和图见本分支artifact：`training_diagnostics.json`、`training_diagnostics.png`；每checkpoint相同步数PPO指标与完整capture-time/censor/type/reward证据见 `checkpoint_evidence_audit.json`。

## 最佳checkpoint与下一步

best：`/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/runs/terl_mappo_stage1_seed9_100k/checkpoints/step_000100000.pt`；SHA256：`781c391a175e51dfabead66bac645a6efb1bc230747276768d998ccab1f9153e`。
最终隔离测试argmax/sample各50局，未参与selection；不得将最优screen成绩作为最终测试成功率。
建议保留同合同、同超参，下一轮先扩至500k bounded Stage1 gate，再据ring与capture稳定性决定是否向2M延长。扩步需建立清楚的resume/config lineage；当前不自动启动，不加入新shaping，也不把partial写成可靠捕获。

未解决：单训练seed与有限预算；公开APF/current/padding的设计意图；没有backbone/pooling消融；高entropy和critic EV无法单独归因围捕瓶颈。上述公开行为已保留，不静默修正任务难度。

原始合同、迁移及43项测试见Migration/Implementation报告；中央DAG只读，handoff由中央owner接收。
