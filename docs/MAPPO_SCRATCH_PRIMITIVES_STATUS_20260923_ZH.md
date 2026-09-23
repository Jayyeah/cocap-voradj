# MAPPO Scratch Primitives 运行状态

观测时间：2026-09-23 22:51 CST  
分支：`experiment/mappo-scratch-primitives-20260923`  
代码 HEAD：`dce2583`  
上游同步：已通过 `127.0.0.1:17892` fetch `origin/ops/ac-master-dag-20260921`；上游未新增 commit，本分支相对该上游为 `ahead 1`。

## 运行状态

| 线 | 状态 | tmux / PID | GPU 映射 | 当前 step | formal |
|---|---|---|---|---:|---|
| M-CAP | `RUNNING` | `cocap_mappo_scratch_20260923_mcap2` / `1208694` | physical GPU0 (`CUDA_VISIBLE_DEVICES=0`, process `cuda:0`) | 59,300 / 500,000 | 0/25k/50k 已完成 |
| M-COV | `RUNNING` | `cocap_mappo_scratch_20260923_mcov2` / `1213702` | physical GPU1 (`CUDA_VISIBLE_DEVICES=1`, process `cuda:0`) | 90,300 / 200,000 | 0/25k/50k/75k 已完成 |

两条线都使用 latest-only checkpoint；没有停止训练或因早期指标提前判死。

## M-CAP：CAPABILITY_FIRST_CAPTURE_BASELINE

50k formal：

- argmax：capture `0/20`，normal `0/20`，ring2 `1/20`，ring3 `0/20`，collision `20/20`；capture time 无观测值。
- sample：capture `1/20`（5%），normal `1/20`，ring2/ring3 各 `1/20`，collision `19/20`；唯一成功 capture time 为 `924` steps。
- censoring：argmax `20/20` censored；sample `19/20` censored。

结论：已经出现第一个弱探索信号，但还不是稳定 Capture primitive；继续执行到 500k。

## M-COV：Pure Coverage positive control

75k formal：

- argmax CE success `16/20`（80%），sample CE success `20/20`（100%）。
- argmax/sample collision 与 boundary 均为 0。
- argmax CE RMS `0.0381`，sample CE RMS `0.0350`。

这与历史 positive control 的学习方向一致；训练继续到 200k。

## Learner telemetry

最近已持久化的 learner 记录显示：

- M-CAP：entropy `2.1241`，approx KL `1.95e-4`，explained variance `0.014`，ValueNorm mean/std `-24.74 / 34.69`。
- M-COV：entropy `1.9667`，approx KL `3.56e-3`，explained variance `0.954`，ValueNorm mean/std `-40.75 / 43.66`。

## 资源与 artifact

观测快照：GPU0 `2131/49140 MiB`、3% utilization；GPU1 `3556/49140 MiB`、11% utilization。根分区剩余约 `25GB`（98% used），inode 使用率 7%。由于磁盘余量下降，保持 latest-only，不复制历史 checkpoint。

主要 artifact：

- M-CAP：[progress.json](../artifacts/2026-09-23_mappo_scratch_primitives/m-cap-full/progress.json)、[formal_step_050000.json](../artifacts/2026-09-23_mappo_scratch_primitives/m-cap-full/formal_step_050000.json)、[resolved_capture_sensing.json](../artifacts/2026-09-23_mappo_scratch_primitives/m-cap-full/resolved_capture_sensing.json)
- M-COV：[progress.json](../artifacts/2026-09-23_mappo_scratch_primitives/m-cov-run/progress.json)、[formal_step_075000.json](../artifacts/2026-09-23_mappo_scratch_primitives/m-cov-run/formal_step_075000.json)
- 合同审计：[MAPPO_SCRATCH_PRIMITIVES_CONTRACT_AUDIT_20260923_ZH.md](MAPPO_SCRATCH_PRIMITIVES_CONTRACT_AUDIT_20260923_ZH.md)

早期启动保护/设备映射失败目录 `m-cap/`、`m-cov-full/` 保留为诊断记录；正式运行目录为 `m-cap-full/` 与 `m-cov-run/`。

