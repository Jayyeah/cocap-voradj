# AC MASTER DAG（2026-09-21）

本文件与 `artifacts/2026-09-21_ac_master_dag/state.json` 是跨 session 的唯一中央状态。只有 MASTER 可以写这两个文件；child 只能在各自 branch/worktree/rundir 工作，并以结构化 handoff 返回结果。禁止 nested subagents。

## 当前事实源

- MASTER branch：`ops/ac-master-dag-20260921`，HEAD `668d5f7`，基于 `ops/training-performance-sync-20260921`。
- remote：`https://github.com/Jayyeah/cocap-voradj.git`；同步使用命令级 proxy `127.0.0.1:17892`，未修改 global git、`.bashrc` 或 system proxy。
- 训练同步事实：`docs/ops/TRAINING_PERFORMANCE_SYNC_20260921_ZH.md` 及其 `artifacts/2026-09-21_training_performance_sync/`。
- IQN 恢复/存储事实：`docs/ops/Z05_Z07_RECOVERY_STATUS_20260921_ZH.md`、`docs/ops/Z05_Z07_LATEST_ONLY_FULL_RESUME_AUDIT_20260920_ZH.md`。
- BC lineage/qualification：`AC0_BC_CRITIC_LINEAGE_RECOVERY_20260920_ZH.md`、`artifacts/2026-09-20_ac1/AC1_CANONICAL_BC_RUNTIME_QUALIFICATION.json`。
- AC-3/4/5 ranking、Bellman、support 证据：`docs/audits/AC3*`、`AC4*`、`AC5*`。

## 第一次 reconciliation

### IQN Z05 / Z07

Z05 当前 PID `17097`、tmux `iqn_z05_recovery_20260921`、物理 GPU0；Z07 当前 PID `19555`、tmux `iqn_z07_recovery_20260921`、物理 GPU1（其进程内仍使用 `cuda:0`，由 `CUDA_VISIBLE_DEVICES=1` 映射）。两条 heartbeat 均活跃，无 unexpected stop、dead PID、broken tmux 或 resume-needed 证据；因此本轮不停止、不迁移、不重启、不改科学合同。

历史 formal 事实：Z05/Z07 Stage2 最新同步点分别为 100k/200k；Z05 strict CE 0.9、CE RMS 0.0489383，Z07 strict CE 1、CE RMS 0.0402727。后续正式评估必须新增：Pure Capture 的成功 episode `capture_steps/time`（mean/median/p90/success_n，failure/censored 分开）、Pure Coverage 的 strict CE 时间、Mixed 的 capture/recovery/mission 时间及 post-capture CE、safe_complete、collision。

### 旧 AC 线

- AC-COV：已有 500k formal；strict CE 0、分类 `NO_CLEAR_SUSTAINED_SIGNAL`。不重启原失败 run。
- AC-CAP：300k formal artifact 已存在；其后 recovery log 记录 `actor_grad_norm=inf` 与 `finite=0`，当前无 live PID。分类为 `NONFINITE_AFTER_300K_FORMAL`，只保留 forensic summary，不追加旧预算。
- AC-MIX：当前 PID `19542`、tmux `ac_mix_recovery_20260921`、物理 GPU1，实时约 293k/300k；只允许完成 300k formal closeout，禁止扩 500k。
- AC-COV/CAP/MIX 的旧 resume 连续性不宣称 bit-exact：COV provenance incomplete，CAP/MIX replay reset 后 functional only。不能把这些旧线当作新 DAG 的 exact baseline。

## GPU lease 记录

采样窗口 `21:15:09`--`21:15:19`，每 1 秒一次：

| 物理 GPU | active compute | util mean / peak | min free VRAM | 温度 | 结论 |
|---|---|---:|---:|---:|---|
| 0 | PID 17097 / Z05 | 8.8% / 16% | 48327 MiB | 65 C | IQN 保留，未发放叠加 lease |
| 1 | PID 19542 / AC-MIX；19555 / Z07 | 11.5% / 26% | 47579 MiB | 77--79 C | 已有任务保留；新任务必须重新采样、smoke、显存审计并由 MASTER grant |

两卡满足 `<60% mean`、`<90% peak` 的利用率门槛，但这不是自动授权；新 GPU child 必须提交 `GPU_LEASE_REQUEST`，MASTER 先审 disk/heartbeat/VRAM，再返回 `GPU_LEASE_GRANTED` 或 `GPU_LEASE_WAIT`。启动后再次 10 秒采样；若 OOM、NaN/Inf、已有 heartbeat stall 或显著 saturation，只停止刚由 MASTER 新开的任务并标记 `GPU_LEASE_REVOKED_OVERLOAD`。

根盘当前约 78 GiB free、92% used、inode 6%。任何长训前必须重新执行 `df -h /`、`df -i /`、planned run root `du -sh`，并估算 checkpoint/replay/resume/evaluation/telemetry；优先 latest-only，禁止重复大 replay/resume 累积。

## DAG gate

| task | 当前状态 | branch / worktree | 下一 gate |
|---|---|---|---|
| A0 | READY_TO_SPAWN | `experiment/ac-mappo-cov-repro-20260921` / `/home/yjq/rl/CoCap1/ac-mappo-cov-repro-20260921` | exact 0--200k MAPPO reproduction |
| A1 | READY_TO_SPAWN | `experiment/ac-entropy-cov-20260921` / `/home/yjq/rl/CoCap1/ac-entropy-cov-20260921` | 0/25/50/75/100k entropy causal gate |
| B1 | READY_TO_SPAWN | `experiment/ac-bc-exploratory-critic-20260921` / `/home/yjq/rl/CoCap1/ac-bc-exploratory-critic-20260921` | support + ranking handoff |
| B2 | READY_TO_SPAWN | `experiment/ac-bc-counterfactual-critic-20260921` / `/home/yjq/rl/CoCap1/ac-bc-counterfactual-critic-20260921` | R0；R1/R2 only under MASTER gate |
| A2 | LOCKED_CHILD_SLOT | later isolated worktree | implementation/tests/smoke now; formal only after A0 learnable + A1 insufficient |
| A3 | LOCKED_DESIGN_ONLY | later isolated worktree | proposal then `CAPTURE_CURRICULUM_AWAITING_USER_APPROVAL` |
| B3 | LOCKED_DEPENDENCIES | MASTER-only comparison | select `BEST_PRETRAINED_CRITIC` only if ranking-first evidence supports it |
| B4 | LOCKED_DEPENDENCIES | later isolated worktree | step0 retention before any long joint RL |
| OLD-MIX-CLOSEOUT | LIVE_CLOSEOUT_ONLY | existing live worktree | 300k formal then release |
| IQN-METRIC-AUGMENT | QUEUED_READ_ONLY | future isolated branch | evaluation-only changes; no live training impact |
| A4 | BLOCKED | none | no action until learner + user-approved initialization curriculum gates |

第一波最多四个 child：A0、A1、B1、B2。A2/A3 只能在 slot 释放后派生；A3 任何时候都停在用户审核 gate，不得自动实现或训练。

## Child handoff contract

child 返回必须包含：

```text
TASK_ID
branch
HEAD
worktree
scientific_question
exact_changes
tests
artifact_paths
runtime_status
GPU/PID/tmux if any
current_step
formal_results
classification
blockers
next_gate
whether_user_approval_is_required
```

child 不得写中央 DAG/state，不得抢 GPU，不得启动额外长训，不得停止其他任务，不得 spawn child。MASTER 收到 handoff 后才更新中央 JSON/Markdown、reconcile process/GPU、决定下一 gate。

## 当前自动转移

1. 先派生 A0/A1/B1/B2；A0/A1 若需 GPU，必须先获得 lease。B1/B2 默认 CPU/offline，不因已有 GPU 低利用率而强占。
2. A2 只在 slot 释放后做 implementation/tests/smoke/config；其 formal 100k 仍锁在 A0/A1 gate。
3. A3 只做 design，完成后写 proposal 并停在 `CAPTURE_CURRICULUM_AWAITING_USER_APPROVAL`。
4. A0 未得到 reproduction classification 前，MASTER 不宣称当前环境 drift 或 no-drift；A1 100k 前不宣称 entropy causal answer。
5. 只有 `A0 indicates learnable` 且 `A1=ENTROPY_NOT_SUFFICIENT` 才解锁 A2 formal；只有 BC qualified + B3 selected critic 才解锁 B4。
