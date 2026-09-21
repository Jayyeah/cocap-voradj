# AC MASTER DAG（2026-09-21）

本文件与 `artifacts/2026-09-21_ac_master_dag/state.json` 是跨 session 的唯一中央状态。只有 MASTER 可以写这两个文件；child 只能在各自 branch/worktree/rundir 工作，并以结构化 handoff 返回结果。禁止 nested subagents。

## 当前事实源

- MASTER branch：`ops/ac-master-dag-20260921`，本地 HEAD `b14217b`，基于 `ops/training-performance-sync-20260921`。已完成两次成功 push；按本 DAG 的 push 上限，后续中央更新保留在共享本地 worktree，状态标记为 `REMOTE_SYNC_PENDING`。
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
- AC-MIX：已完成并保留 `eval_step_000300000.json`；由于从 200k resume 后 additional-step 语义使 telemetry 越过边界到 global step 375k，MASTER 已停止 PID `19542`，tmux 已消失，GPU1 已释放；禁止扩 500k。该线只能做 closeout forensic，不把 375k 后窗口当作新增正式预算。
- AC-COV/CAP/MIX 的旧 resume 连续性不宣称 bit-exact：COV provenance incomplete，CAP/MIX replay reset 后 functional only。不能把这些旧线当作新 DAG 的 exact baseline。

## GPU lease 记录

采样窗口 `21:15:09`--`21:15:19`，每 1 秒一次：

| 物理 GPU | active compute | util mean / peak | min free VRAM | 温度 | 结论 |
|---|---|---:|---:|---:|---|
| 0 | PID 17097 / Z05 + 224643 / A0 | 4.8% / 5%（22:18 sample） | 47173 MiB | 66--70 C | A0 formal 已获 lease；IQN Z05 heartbeat 正常 |
| 1 | PID 19555 / Z07 + 229509 / A1 | 17.4% / 29%（22:18 sample） | 47173 MiB | 77--78 C | A1 formal 已获 lease；IQN Z07 heartbeat 正常 |

两卡满足 `<60% mean`、`<90% peak` 的利用率门槛，但这不是自动授权；新 GPU child 必须提交 `GPU_LEASE_REQUEST`，MASTER 先审 disk/heartbeat/VRAM，再返回 `GPU_LEASE_GRANTED` 或 `GPU_LEASE_WAIT`。启动后再次 10 秒采样；若 OOM、NaN/Inf、已有 heartbeat stall 或显著 saturation，只停止刚由 MASTER 新开的任务并标记 `GPU_LEASE_REVOKED_OVERLOAD`。

根盘当前约 75 GiB free、92% used、inode 6%。A0/A1 formal 已分别使用 latest-only/无 replay 累积合同；任何后续长训前仍必须重新执行 `df -h /`、`df -i /`、planned run root `du -sh`，并估算 checkpoint/replay/resume/evaluation/telemetry。

## DAG gate

| task | 当前状态 | branch / worktree | 下一 gate |
|---|---|---|---|
| A0 | FORMAL_RUNNING | `experiment/ac-mappo-cov-repro-20260921` / `/home/yjq/rl/CoCap1/ac-mappo-cov-repro-20260921` | PID 224643 / tmux `a0_mappo_cov_formal_20260921`；step 12600，25k next |
| A1 | FORMAL_RUNNING_100K_CAUSAL_GATE | `experiment/ac-entropy-cov-20260921` / `/home/yjq/rl/CoCap1/ac-entropy-cov-20260921` | PID 229509 / tmux `a1_entropy_localq_formal_20260921`；0 gate初始化，向100k causal gate推进 |
| B1 | ACTIVE_CPU_BOUNDED | `experiment/ac-bc-exploratory-critic-20260921` / `/home/yjq/rl/CoCap1/ac-bc-exploratory-critic-20260921` | support + ranking handoff |
| B2 | R0_COMPLETE_R1_GATE_PENDING | `experiment/ac-bc-counterfactual-critic-20260921` / `/home/yjq/rl/CoCap1/ac-bc-counterfactual-critic-20260921` | 16 anchors/144 branches；MASTER 判断 R1，R2 仍锁定 |
| A2 | PREFLIGHT_PASS_FORMAL_LOCKED | `experiment/ac-discrete-sac-preflight-20260921` / `/home/yjq/rl/CoCap1/ac-discrete-sac-preflight-20260921` | commit `9231c25`；9 tests + CPU smoke passed；formal仍锁定 |
| A3 | CAPTURE_CURRICULUM_AWAITING_USER_APPROVAL | `design/ac-capture-curriculum-20260921` / `/home/yjq/rl/CoCap1/ac-capture-curriculum-20260921` | proposal commit `377a6afd`；必须用户审核批准后才能实现/训练 |
| B3 | LOCKED_DEPENDENCIES | MASTER-only comparison | select `BEST_PRETRAINED_CRITIC` only if ranking-first evidence supports it |
| B4 | LOCKED_DEPENDENCIES | later isolated worktree | step0 retention before any long joint RL |
| OLD-MIX-CLOSEOUT | CLOSED_300K_FORMAL | existing live worktree | released; no 500k extension |
| IQN-METRIC-AUGMENT | QUEUED_READ_ONLY | future isolated branch | evaluation-only changes; no live training impact |
| A4 | BLOCKED | none | no action until learner + user-approved initialization curriculum gates |

第一波最多四个 child：A0、A1、B1、B2。A0/A1 formal 现在由 MASTER 直接持有已审计的 tmux/PID/GPU lease；B1 仍 bounded CPU，A2/A3 已完成并释放 child slot。A3 已停在用户审核 gate，不得自动实现或训练。

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

1. A0 CPU preflight 与 fresh-output CUDA smoke 均已通过；formal 已启动到 step 12600，0--200k 合同不变。
2. A1 已完成 py_compile、18 个 contract tests、1200-step CPU smoke；formal 100k 已启动，不能在 100k 前宣称 causal answer。
3. B2 R0 已完成：16 anchors、144 AW9 branches；D_CF_R0 全 AW9 support、0 branch transition collision、ranking gate 正；R1/R2 不自动启动。
4. A2 只做 implementation/tests/smoke/config；其 formal 100k 仍锁在 A0/A1 gate。
5. A3 proposal 已完成并停在 `CAPTURE_CURRICULUM_AWAITING_USER_APPROVAL`；任何实现/训练都暂停到用户明确批准。
6. A0 未得到 reproduction classification 前，MASTER 不宣称当前环境 drift 或 no-drift；A1 100k 前不宣称 entropy causal answer。
7. 只有 `A0 indicates learnable` 且 `A1=ENTROPY_NOT_SUFFICIENT` 才解锁 A2 formal；只有 BC qualified + B3 selected critic 才解锁 B4。
