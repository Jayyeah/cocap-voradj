# Critic Identifiability Audit：A0 → Gate → A1（2026-09-17/18）

## 结论

本轮严格止于 A1，最终分类为 **`LOCAL_SUFFICIENT`**。在同一冻结 Final IQN trajectory bank 上，LQ 的 heldout MC explained variance 为 `0.5933`，满足预注册 healthy 条件；NQ 虽进一步达到 `0.6431`，但 RMSE 只改善 `6.44%`，未达到 `10%` 显著改善门槛；CQ/V 都更差。因此当前证据不支持为解决 critic identifiability 直接引入 central critic。

这不是“动作排序已经可靠”的结论。三类 Q 都能稳定区分真实成功捕获 transition 与真实碰撞 transition，但两个独立 seed 对 AW9 反事实动作的 top-1 一致率仅 `0.185/0.208/0.283`。故 A1 提供了进入 A2 native-bootstrap audit 的表示层证据，但本轮没有执行 A2、A3、A4，也没有 online RL、bootstrap、actor update、reward 或 sensing 变更。

## 恢复与范围

- 从远端 `experiment/td3-local-aw-stage1-20260916` 最新基线 `1da49cdb11a64918fe6bd6830899babd634de67c` 创建独立 worktree/分支 `experiment/critic-identifiability-audit-20260917`；没有修改、停止或复用其他正式实验进程。
- 已核对双线规划、TD3 Stage-1 tracker/final artifacts、Final IQN/BC 合同、Final MAPPO/BC parity 与 NormSense-V2 合同。Stage-1 C1–C4 均已到 `100k` 且没有产生可晋级策略；本轮不续训它们。
- 唯一 data generator 为 Final IQN `stage1_4v1_step_2000000.pt`，checkpoint SHA-256 `2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89`，固定 `epsilon=0.05` 行为分布；模型 state hash 前后均为 `242cc13f4c40b69afa670ca2991aca079bc66df7449e15a5224c072fdbde73cc`。

## A0：冻结 trajectory bank 与 Gate

Matched contract 为 NormSense-V2 4v1。资格集每场景 8 回合：Pure Capture `8/8`、Pure Coverage `8/8`、Mixed/post-capture lifecycle `6/8`，均为 `0` collision，三项均通过预注册 `>=75% success、<=25% collision` 门槛。

为避免碰撞只落在 heldout，额外在同一冻结策略下扫描 89 个 mixed seed，得到 4 个独立自然碰撞 seed：`2026096170/6217/6251/6258`，固定分配为 train 1、validation 1、test 2。最终 bank：

| 项 | 数值 |
|---|---:|
| episodes | 60（49 success / 11 failure） |
| transitions | 16,728 |
| active-agent rows | 66,912 |
| split transitions | train 11,572 / validation 1,284 / test 3,872 |
| bank SHA-256 | `ad5607c82ba077c1cbde99e852c8f8201bbf13b7dbfef816691136b206e3a5ee` |

事件数按 active-agent rows 统计：

| pure coverage | coverage steady | ordinary pursuit | ring2 | ring3 | normal capture |
|---:|---:|---:|---:|---:|---:|
| 29,764 | 21,300 | 8,288 | 1,620 | 188 | 164 |

| stationary capture | collision | capture terminal | early recovery | late recovery | pure coverage restart |
|---:|---:|---:|---:|---:|---:|
| 0 | 16 | 72 | 3,440 | 23,612 | 1,440 |

collision 在 train/validation/test 分别为 `4/4/8` rows。`stationary_capture=0` 是真实未出现，不填造样本。`G_t` 来自完整真实轨迹、`gamma=.99`，learned target 未使用；逐 episode 复算最大绝对误差 `1.5232e-5`。所有 episode 以 terminated 或 truncated 结束、两者重叠为 0；episode/seed split 无交集。A0 Gate 全部通过。

## A1：公平 MC 监督拟合

LQ/NQ/CQ/V 均为 hidden `256`、`4` layers、`8` heads、Adam `1e-4`、batch `256`、`1,200` updates、两个 seed `2026091711/12`，共享 train/validation/test、target normalization 与 event-balanced sampler。参数量为 LQ `4.030M`、NQ `4.228M`、CQ `3.238M`、V `3.237M`，最大/最小比 `1.306`。LQ 结果同时代表输入/架构等价的 TD3/SAC local twin-Q；没有重复按算法名训练。

### Heldout（15,488 agent rows）

| critic | RMSE | MAE | EV | Pearson | Spearman | calib slope/intercept | decile ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| LQ | 34.835 | 12.414 | 0.593 | 0.775 | 0.398 | 0.905 / 2.851 | 3.141 |
| NQ | 32.592 | 11.350 | 0.643 | 0.802 | 0.445 | 0.989 / -0.778 | 1.636 |
| CQ | 42.393 | 12.844 | 0.398 | 0.670 | 0.426 | 0.747 / 0.924 | 5.717 |
| V | 43.277 | 13.109 | 0.375 | 0.651 | 0.406 | 0.747 / -0.183 | 5.429 |

### Event-conditioned RMSE

| event | LQ | NQ | CQ | V |
|---|---:|---:|---:|---:|
| pure coverage | 4.04 | 3.63 | 2.29 | 2.46 |
| ordinary pursuit | 86.30 | 84.00 | 102.75 | 105.95 |
| ring2 | 79.08 | 72.34 | 121.14 | 119.34 |
| ring3 / near-capture | 102.29 | 100.21 | 147.12 | 149.92 |
| normal capture | 102.29 | 100.21 | 147.12 | 149.92 |
| collision | 66.30 | 96.03 | 74.55 | 76.93 |
| early recovery | 65.79 | 49.46 | 66.84 | 68.47 |
| late recovery | 3.82 | 2.64 | 2.67 | 2.08 |

关键异常是 return 尺度与事件占比的强异质性：总体指标主要受大量 coverage/late-recovery 低误差行支撑，而 ring3/capture、ordinary pursuit 和 early recovery 仍是高误差区。NQ 对 early recovery 与 ring2 改善最明显；CQ/V 在 ring2/ring3/capture 上反而显著退化，不能解释为 central information 必需。

### Ordering 与反事实动作稳定性

成功/碰撞 ordering 严格比较 heldout 成功 normal-capture transition（20 rows）与实际 collision transition（8 rows）。真实 MC 均值为 `134.68 vs -42.14`，pairwise `success>collision=0.956`。

| Q | predicted success / collision | pairwise success>collision | near-capture pairwise | twin rank rho mean / p10 | twin top-1 |
|---|---:|---:|---:|---:|---:|
| LQ | 124.14 / -53.51 | 1.000 | 0.600 | 0.045 / -0.867 | 0.185 |
| NQ | 120.42 / -87.51 | 0.988 | 0.594 | 0.107 / -0.817 | 0.208 |
| CQ | 128.43 / -39.00 | 0.988 | 0.125 | 0.360 / -0.222 | 0.283 |

真实 near-capture pairwise 为 `0.713`。LQ/NQ 可部分保持 near-capture outcome ordering，CQ 明显失败；所有 Q 的 twin counterfactual AW9 排序都不稳定。这里的 action ranking 只有双 seed 一致性，没有反事实 ground-truth return，不能被解释为策略改进证明。

## 判决与边界

- **A1 classification：`LOCAL_SUFFICIENT`**。LQ 达到 `EV>=.50` 且 `RMSE<=.75×heldout target std`；NQ 未同时达到相对 LQ `RMSE<=.90×` 与 `EV gain>=.05`；CQ/V 不优于 local。
- **有 A2 表示层证据：是**。一个 local Q 已能拟合 realized MC return，因此后续若单独授权 A2，可隔离 native bootstrap/target dynamics；本轮没有执行 A2。
- `LOCAL_SUFFICIENT` 仅说明 realized-policy MC identifiability，不说明 counterfactual action ranking 稳定。A2 若启动，必须继续把 rare collision、ring3/capture、early recovery 与 twin action ranking 作为硬诊断。

验证：A0/A1、NormSense-V2 与 MAPPO 合同测试共 `23 passed`；config/bank/8 个 fitted checkpoint SHA、split seed 互斥、Gate、classification 与禁止项自洽校验通过。历史 `test_forward_final_two_head_value_20260910.py` 依赖未版本化的旧 `fixed_bank.npz`，在本独立 worktree 中为 fixture setup error，未将其伪报为本轮断言失败或通过。

机器证据：`artifacts/2026-09-17_critic_identifiability_audit/{trajectory_bank_manifest.json,a0_gate.json,critic_comparison.json,predicted_vs_mc_scatter.png}`。`critic_comparison.json` SHA-256 为 `170d79de4014971036afb59757d1365acb2f890fe9ad0b24e293e7456552436b`；raw bank、heldout predictions 与 fitted weights 只保留本地并由 manifest/checkpoint SHA 固定。
