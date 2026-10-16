# CoCap MAPPO 是否继续：最终决策（2026-10-15）

## 决策

**STOP / ARCHIVE。**暂停MAPPO新训练，保存既有结果和checkpoint。本轮没有启动训练，也没有改动运行进程。MAPPO仍有一个可证伪的critic假说，但不足以保留成与IQN-Z并行的长期训练主线。

| 选项 | 证据 | 成本与机会成本 | 本轮 |
|---|---|---|---|
| STOP / ARCHIVE | Batch02 mixed full-task success为0；R2 selected capture伴随高碰撞、last退化；R3只增加telemetry；N3/V3无正向selection且final未闭合；低target KL没有CoCap late-window收益。BC→PPO 5k CE退化值得保存，但critic因果未识别。 | 放弃近期获得MAPPO成功结果的机会；减少GPU与论文切换成本，保留负结果、数据和checkpoint。 | **选择** |
| BOUNDED_DIAGNOSTIC | P2 early-recovery value正偏差；IQN Q可作动作排序参照。 | 低预算offline分析也会占数据审核时间；正结果不保证actor改善。 | 现在不执行；IQN-Z若需要PG baseline再重开。 |
| CONTINUE_PARALLEL | 更好的central credit理论上可改善capture、安全和效率。 | 需要多seed、matched controls、完整任务评估和独立论文定位，实际会成为第二主项目；目前正向证据不足。 | 不选；空闲GPU不是启动理由。 |

## 最值得保留的假说和优先级

**假说：**action-free central V可能把几何接近但phase/recovery clock不同的状态合并，使capture transition和early recovery出现return calibration/advantage排序偏差。P2正bias和teacher-Q审计提供测试线索；P3 advantage sign大体正确、pure coverage early误差也高、双Value Head未改善恢复校准，因此不能称为MAPPO失败根因。

优先级：**IQN-Z自主扩散搜索与协同围捕主论文 P0 > 成功IQN/BC完整任务证据 P1 > MAPPO Critic假说 P4/deferred**。MAPPO仍有有限研究价值：记录TERL Stage1能力、CoCap子技能与full-task差距、PPO/Critic负结果。它当前不值得作为独立并行训练线，也没有已确立的新critic论文贡献。

## 具体下一步

1. 五份文档留在当前本地delivery worktree；原Oct12状态快照、manifest与训练产物不覆盖。
2. 不续跑/补跑MAPPO、不改target KL、不加Value Head、不为使用GPU而启动实验。
3. 用户可将结论交gpt-6.1 Sol High复核，重点审阅R2/R3配对late-window统计、N3/V3 final缺失对结论的限制、P2是否足以支持离线假说。复核不会自动成为训练授权。
4. 只有IQN-Z主线后续明确需要policy-gradient baseline，且phase-stratified calibration与teacher-Q ranking离线门槛通过，才重新讨论单项matched continuation。

**最终判断：**MAPPO路线仍留下科学信息和一个低优先级可检验假说，但相对IQN-Z主线优先级显著更低。现在应停止新训练并归档。
