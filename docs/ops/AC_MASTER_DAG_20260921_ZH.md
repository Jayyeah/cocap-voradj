# AC MASTER DAG（2026-09-21）

本文件与 `artifacts/2026-09-21_ac_master_dag/state.json` 是跨 session 的唯一中央状态。只有 MASTER 可以写这两个文件；child 只能在各自 branch/worktree/rundir 工作，并以结构化 handoff 返回结果。禁止 nested subagents。

## 当前事实源

- MASTER branch：`ops/ac-master-dag-20260921`，本轮 reconciliation base HEAD `f682c78`，基于 `ops/training-performance-sync-20260921`。`origin/ops/ac-master-dag-20260921` 为 `7238ffa`，两者不一致；后续同步轮次最多两次，并必须显式使用 mihomo `127.0.0.1:17892` 的 `http_proxy/https_proxy/all_proxy` 与 `git -c http.proxy/-c https.proxy`，禁止 stale `17891`。
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
| 0 | PID 17097 / Z05 + 224643 / A0 | 13.3% / 58%（22:29 sample） | 47173 MiB | 67--68 C | A0 formal live；IQN Z05 heartbeat 正常 |
| 1 | PID 19555 / Z07 | 10.7% / 15%（22:29 sample） | 47579 MiB | 77--78 C | A1 lease 已因 device-side assert 撤销；IQN Z07 正常 |

两卡满足 `<60% mean`、`<90% peak` 的利用率门槛，但这不是自动授权；新 GPU child 必须提交 `GPU_LEASE_REQUEST`，MASTER 先审 disk/heartbeat/VRAM，再返回 `GPU_LEASE_GRANTED` 或 `GPU_LEASE_WAIT`。启动后再次 10 秒采样；若 OOM、NaN/Inf、已有 heartbeat stall 或显著 saturation，只停止刚由 MASTER 新开的任务并标记 `GPU_LEASE_REVOKED_OVERLOAD`。

根盘当前约 74 GiB free、92% used、inode 6%。A0/A1 formal 已分别使用 latest-only/无 replay 累积合同；任何后续长训前仍必须重新执行 `df -h /`、`df -i /`、planned run root `du -sh`，并估算 checkpoint/replay/resume/evaluation/telemetry。

## DAG gate

| task | 当前状态 | branch / worktree | 下一 gate |
|---|---|---|---|
| A0 | FORMAL_RUNNING | `experiment/ac-mappo-cov-repro-20260921` / `/home/yjq/rl/CoCap1/ac-mappo-cov-repro-20260921` | PID 224643 / tmux `a0_mappo_cov_formal_20260921`；step 118500；100k argmax/sample 均 20/20；继续 200k |
| A1 | CUDA_STEP0_DIAGNOSTIC_ACTIVE | `experiment/ac-entropy-cov-20260921` / `/home/yjq/rl/CoCap1/ac-entropy-cov-20260921` | Aquinas；GPU1 diagnostic lease 已授予；完成后 CPU regression + CUDA smoke；通过才 formal |
| B1 | B1_COMPLETE | `experiment/ac-bc-exploratory-critic-20260921` / `/home/yjq/rl/CoCap1/ac-bc-exploratory-critic-20260921` | handoff 完成；support 扩大但无 global ranking stability；等待 B2-R0-FULL 后进入 B3 |
| B2 | R0_FULL_ACTIVE | `experiment/ac-bc-counterfactual-critic-20260921` / `/home/yjq/rl/CoCap1/ac-bc-counterfactual-critic-20260921` | Carson CPU PID `285669`；run `artifacts/2026-09-22_b2_r0`；106/128 anchors、954/1152 branches；目标 32 anchors/phase；R1/R2 锁定 |
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

1. A0 CPU preflight 与 fresh-output CUDA smoke 均已通过；formal 已到 step 118500。100k argmax/sample strict CE 均为 20/20、无 collision，说明当前 MAPPO Coverage 可学；仍必须跑满注册的 200k reproduction contract。
2. A1 已获得用户批准。GPU1 10x1s pre-sample 通过，Aquinas 正在执行一次 CUDA step-0 diagnostic；通过后自动执行 CPU regression、CUDA smoke，再按 entropy-only 合同重启 formal。
3. B1 handoff 已完成：epsilon 0.05/0.10 扩展到全 AW9 support，但 overall ranking 未改善，B3 选择仍锁定。
4. B2 preliminary R0 只有 16 anchors/144 AW9 branches；因样本过小不得解锁 R1。当前由 Carson 扩展 B2-R0-FULL，保持 one-step AW9→BC continuation 定义不变。
5. A2 只做 implementation/tests/smoke/config；其 formal 100k 仍锁在 A0/A1 gate。
6. A3 proposal 已完成并停在 `CAPTURE_CURRICULUM_AWAITING_USER_APPROVAL`；任何实现/训练都暂停到用户明确批准。
7. A0 未得到 reproduction classification 前，MASTER 不宣称当前环境 drift 或 no-drift；A1 100k 前不宣称 entropy causal answer。
8. 只有 `A0 indicates learnable` 且 `A1=ENTROPY_NOT_SUFFICIENT` 才解锁 A2 formal；只有 BC qualified + B3 selected critic 才解锁 B4。

## 2026-09-22 00:09 reconciliation handoff

- remote 只读 fetch：local `8ff81f8`，origin `7238ffa`；状态 `REMOTE_SYNC_PENDING`，不再 push。
- IQN Z05 PID `17097` / GPU0 与 Z07 PID `19555` / GPU1 均 alive，tmux、heartbeat、GPU health 正常；没有 resume-needed，不干扰。
- A0 PID `224643` / GPU0 / tmux `a0_mappo_cov_formal_20260921` alive，step `81700`。75k formal：sample strict CE `1.0/20`，argmax strict CE `0/20`，sample CE RMS mean `0.034989`，area CV mean `0.088407`；只登记 `EARLY_REPRO_SIGNAL_SAMPLE_ONLY`，继续到 200k。
- A1 审计已 handoff：CPU regression/static checks 通过；CUDA 异步 assert 的 first invalid tensor 仍需一次经用户批准的 bounded CUDA diagnostic 才能定位。无 retry/resume、无科学结论。
- B1 handoff：BC-only support=8，epsilon 0.05/0.10 support=9；三档 quality PASS，但 overall top1/Spearman 没有形成全局稳定提升，不能直接注册 pretrained critic。
- B2 preliminary R0：16 anchors/144 branches，仅作样本量不足的 preliminary。Carson 已在独立 B2 worktree 启动 CPU-only R0-FULL，PID `285669`，实际 run `artifacts/2026-09-22_b2_r0`，当前 25/128 anchors、225/1152 branches；目标每 phase 32 anchors（至少 128 total），不含 multi-deviation；完成后 MASTER 再决定 B3 比较或是否满足 R1 条件。
- A2 仍 `PREFLIGHT_PASS/FORMAL_LOCKED`；A3 仍 `CAPTURE_CURRICULUM_AWAITING_USER_APPROVAL`；B4 仍 `BLOCKED_BY_B3`。

## 2026-09-22 user-approved continuation

- A1 diagnostic lease：physical GPU1，窗口 `00:34:30`--`00:34:39`，平均/峰值 util `15%/15%`，minimum free VRAM `47726 MiB`，IQN-Z07 PID `19555` heartbeat 正常。GPU0 同时出现过 94% 单点峰值，但没有给新任务叠加 GPU0。
- A0 100k formal：argmax strict CE `20/20`，sample strict CE `20/20`，collision `0`；argmax CE RMS mean `0.025129`，area CV mean `0.061531`。该证据满足“可学”方向 gate，但不能替代 200k exact reproduction classification。
- B2-R0-FULL 当前已到 `106/128 anchors`、`954/1152 branches`；完成后立即由 MASTER 启动 B3 ranking gate。若 B3 选出明确优于 historical naive critic 的候选，自动进行 B4 step0 BC retention gate，gate 通过后申请并启动 overnight joint RL。
- 若 A1 完成有效 formal 科学测试后确认 `ENTROPY_NOT_SUFFICIENT`，A0 已提供 MAPPO learnability evidence，MASTER 自动解锁 A2 formal；A3 仍禁止实现/训练。
