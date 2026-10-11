# Batch02 云端交付快照 2026-10-11

分支：`delivery/batch02-cocap-mappo-20261011`；观测时间：2026-10-11T21:59:16.575916+08:00。

科学代码固定为 `3575b742548f8ce5ceba68f3db776533fb5d83ba`。本提交仅归档交付报告、pilot/恢复证据、中央 DAG 快照和已完成 screen 数据；不改变 reward、observation、网络、action、terminal 或实验配置。

中央 DAG 历史快照来源 `9ead7633b5d659d28dca09aea62175b0cdaf0dd5`，保留原发布时状态。当前执行状态以本目录 `runtime_snapshot.json` 为准；运行中的本地 `progress.json` 才是实时来源。旧交付报告保留其原始观测时间。

现有 N2/R2 使用 8ce5593，N3/R3/V3 使用 3575b74，各线固定 1M decisions / 25k checkpoint 与评估。训练仍在各原工作区运行；本交付分支不作为新的训练/恢复来源。

| 线 | GPU | PID | 进程存活 | phase | decisions | last checkpoint |
|---|---:|---:|---|---|---:|---:|
| N2 | 0 | 179017 | True | TRAINING | 125768 | 125000 |
| R2 | 1 | 179417 | True | TRAINING | 104352 | 100000 |
| N3 | 0 | 214952 | True | TRAINING | 1536 | 0 |
| R3 | 1 | 216982 | True | TRAINING | 16896 | 0 |
| V3 | 1 | 219016 | True | initial_evaluation | 0 | 0 |

SHA/config/seed/evaluator、checkpoint 路径/hash/清单、吞吐/ETA 和下一次查看时间见 `runtime_snapshot.json`。权重 `.pt` 保留本地。

保持 **PROVISIONAL**：工程和短 pilot PASS 不代表学习成功。R3 是 reward decomposition，reward 数值 delta 为零；V3 与 N3 的差异为 telemetry，不能据此声称独立的科学增益。

NEXT WAKE-UP：2026-10-11T22:29:16.575916+08:00，或新线首个 25k checkpoint；此次发布不启动新的训练任务。
