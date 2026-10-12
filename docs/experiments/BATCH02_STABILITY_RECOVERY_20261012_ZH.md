# Batch02 MAPPO stability recovery

本实验保持 PROVISIONAL。目标是检验已有 Batch01 P1 的 KL 控制能否改善 CoCap-MAPPO 后期保持，不寻找最终 best，不修改任务、网络或 reward。

历史来源已恢复：P1-control 实际运行 SHA `0e5b03b3c39fe4615b91060d7225b3034d820952`，P1-treatment 为 `5b53c3f4f56265dbb8039c9b9ba8bcf53ed75892`；treatment 的首次实现为 `0847dc71ee4174843d68e28d678729558b134ed2`。配置是 `configs/experiments/terl_mappo_batch01_20261009/p1-treatment_delta.json`。运行 manifest、1M checkpoint 元数据、sidecar 和 fork receipt 共同确认唯一 PPO 变化为 `target_kl: 0.02 → 0.01`。完整前后配置与源码 diff 已归档在 [历史审计](../../artifacts/2026-10-12_batch02_stability/historical_audit.json) 和同目录 `history/`。

| 参数 | baseline | stability |
|---|---:|---:|
| actor LR | 3e-5 | 3e-5 |
| critic LR | 1e-4 | 1e-4 |
| LR scheduler | 无 | 无 |
| entropy coefficient | .01 | .01 |
| clip range | .2 | .2 |
| PPO epochs | 3 | 3 |
| minibatches | 2 | 2 |
| target KL | .02 | .01 |
| gamma / GAE lambda | .99 / .95 | .99 / .95 |
| max gradient norm / value coefficient | .5 / 1 | .5 / 1 |
| ValueNorm beta / epsilon | .99999 / 1e-5 | .99999 / 1e-5 |

Adam epsilon 1e-5、rollout 256 和其余配置不变。复用原 `MAPPOTrainer.update`；KL early-stop 沿用原消费语义，不增加 KL penalty 或重新实现 PPO。

三条线均固定用户预先指定的 `500k` checkpoint，而不是 screen best。这样在 550k 观察峰值前分叉，避免按后验成绩选择起点；每条只再训练 500k，绝对步数到 1M。

| 线 | baseline SHA | 实际原合同 | delta |
|---|---|---|---|
| R2-stability | `8ce5593` | CoCap environment/reward + TERL observation/backbone | target_kl .02 → .01 |
| R3-stability | `3575b74` | R2 合同 + reward decomposition telemetry | target_kl .02 → .01 |
| V3-stability | `3575b74` | CoCap observation/backbone/reward + 原 central critic + telemetry | target_kl .02 → .01 |

R3/V3 的上述含义以实际 manifest/config/source 为准。当前代码没有 V3 专属 critic/value 修改；不把 R3 标签误当 CoCap backbone 线，也不把两条关联线当独立 seed。

成功 IQN-CoCap `1a9da05`、resolved hash `6955ecf2…` 继续是任务合同唯一来源；P1 提供的只是 MAPPO algorithm adaptation。新的科学配置与各自 baseline JSON 仅有 `learner.ppo.target_kl` 一处 diff。运行目录、line_id、父 checkpoint 和代码 SHA 写在独立 registry，不混入环境或 learner 配置。

原始 baseline 代码和运行目录保持不变。续训在隔离分支 `experiment/batch02-mappo-stability-20261012` 执行，三条线保留各自原物理 GPU1，以 process-local `cuda:0` 恢复父 CUDA RNG。

工程检查限定为已有需求：原 strict loader 检查 checkpoint/config/source；完整 actor/V/Adam/ValueNorm/env/APF/RNG 与父状态 hash 一致；覆盖 KL 前后完整状态一致；16-decision 原 rollout 与 baseline 完全一致；16-decision PPO continuation 在新 Python 进程恢复后完全一致；原 evaluator 两局与父500k screen 相同。诊断 rollout 状态丢弃，正式作业再次从父500k加载。

原 checkpoint schema 与 task/source manifest 保留，故 evaluator 源文件逐字节不变。新增 continuation 代码的 hashes、experiment SHA、父 SHA、config hash 和运行元数据单独绑定。每25k保存所有 checkpoint并做原 screen：argmax/sample 各10局、相同 reset/action seed、相同原 terminal。500k原20局 screen作为两条件公共起点保留，并显式记录源路径/hash；525k起逐25k执行新 screen。

同时报告每个 checkpoint 的 capture/collision/reward/length/capture time/failure type，以及 entropy、approx KL/max、policy/value loss、explained variance、finite gradient 和 KL early-stop。按同一个绝对 step 配对，不用 best 替代 last，也不在本实验新增选优或 formal promotion。

报告生成 `paired_screen_curve.csv`、`optimization_curve.csv`、`paired_screen_curves.png` 和 `stability_report.json`。关注550k、800k、1M，以及800k→1M保持窗口；窗口只做描述性分析，不是新增门禁。

科学判断在完整 continuation 后 review：A 为有效 KL 控制伴随一致的后期改善；C 为两模式/两线分歧、capture-safety tradeoff或信号不足。若无改善，B 只能表明该历史设置没有修复，不能单独证明 reward/network 为原因；特别须检查 target KL 是否实际触发。单 seed、screen 抽样不允许升级为正式成功。

不自动重试/延长预算/启动下一阶段。阻断 bug 最多两轮修复；非阻断问题记录 warning。作业启动后记录 PID/step/checkpoint/吞吐/ETA/NEXT WAKE-UP，不等待 GPU 长任务完成来冒充已得到稳定性结论。
