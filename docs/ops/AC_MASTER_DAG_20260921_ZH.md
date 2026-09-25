# AC MASTER DAG（2026-09-21）

本文件与 `artifacts/2026-09-21_ac_master_dag/state.json` 是跨 session 的唯一中央状态。只有 MASTER 可以写这两个文件；child 只能在各自 branch/worktree/rundir 工作，并以结构化 handoff 返回结果。禁止 nested subagents。

## 当前事实源

- MASTER branch：`ops/ac-master-dag-20260921`，当前 runtime reconciliation HEAD 为 `ab7cbd52`，已用命令级 proxy `127.0.0.1:17892` fetch，中央 worktree 干净。基于 `ops/training-performance-sync-20260921`；禁止 stale `17891`。
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
| 0 | PID 17097 / Z05 | A0 完成后释放；当前 IQN-only | 45616 MiB observed | 65--67 C | A0 200k `REPRO_PASS`；IQN Z05 heartbeat 正常 |
| 1 | PID 19555 / Z07 + 317964 / A1 | pre 9.1% / 15%；post 16.9% / 27% | pre 47726；post 47320 MiB | 77--79 C | A1 CUDA smoke 32/32 通过，formal live；IQN Z07 正常 |

两卡满足 `<60% mean`、`<90% peak` 的利用率门槛，但这不是自动授权；新 GPU child 必须提交 `GPU_LEASE_REQUEST`，MASTER 先审 disk/heartbeat/VRAM，再返回 `GPU_LEASE_GRANTED` 或 `GPU_LEASE_WAIT`。启动后再次 10 秒采样；若 OOM、NaN/Inf、已有 heartbeat stall 或显著 saturation，只停止刚由 MASTER 新开的任务并标记 `GPU_LEASE_REVOKED_OVERLOAD`。

根盘清理后约 49 GiB free、95% used、inode 7%。已删除 10 个完成历史阶段/关闭 bounded diagnostic 的 resume，共 26110547530 bytes（约 24.31 GiB）；保留最新 IQN stage3、正式 A1/A2 resume、普通 checkpoint 与报告。清理台账：`artifacts/2026-09-21_ac_master_dag/storage_audit_20260925.json`。任何后续长训前仍必须重新执行 `df -h /`、`df -i /`、planned run root `du -sh`，并估算 checkpoint/replay/resume/evaluation/telemetry；当前 storage gate 仍等待 A3 planned peak + 15 GiB margin 复核。runtime snapshot：`artifacts/2026-09-21_ac_master_dag/runtime_snapshot_20260925.json`。

## DAG gate

| task | 当前状态 | branch / worktree | 下一 gate |
|---|---|---|---|
| A0 | REPRO_PASS | `experiment/ac-mappo-cov-repro-20260921` / `/home/yjq/rl/CoCap1/ac-mappo-cov-repro-20260921` | 200k complete；最终 argmax/sample strict CE 均 20/20、collision 0；GPU0 released；当前 MAPPO Coverage 可学 |
| A1 | A1_SUSTAINED_PARTIAL_LEARNING_REOPENED | `control/a1-no-entropy-20260925` @ `411afed`；clean worktree `/home/yjq/rl/CoCap1/a1-no-entropy-control-20260925` | matched no-entropy control 已提交；唯一科学变量 `actor_entropy_alpha=0.0`；7 focused tests、CPU/CUDA smoke 通过；正式 0→100k 仍等待 storage peak/GPU lease |
| B1 | B1_COMPLETE | `experiment/ac-bc-exploratory-critic-20260921` / `/home/yjq/rl/CoCap1/ac-bc-exploratory-critic-20260921` | handoff 完成；support 扩大但无 global ranking stability；等待 B2-R0-FULL 后进入 B3 |
| B2 | B2_COMPLETE | `experiment/ac-bc-counterfactual-critic-20260921` / `/home/yjq/rl/CoCap1/ac-bc-counterfactual-critic-20260921` | 128/128 anchors、1152/1152 branches；alternative top-1 0.7734；R1 draft eligible 但未自动解锁 |
| A2 | A2_TRANSIENT_LEARNING_CLASSIFIED | `experiment/ac-discrete-sac-preflight-20260921` @ `59faae3` / `/home/yjq/rl/CoCap1/ac-discrete-sac-preflight-20260921` | fixed-seed 结果不变；bounded semantic audit 已完成（4 tests passed），未发现公式/符号/终止掩码错误；`0.98*log(9)` 使 alpha≈0.979，仍不扩 300k、不启动新训练 |
| A3 | A3_USER_APPROVED_FOR_IMPLEMENTATION_PREFLIGHT_AND_GATED_TRAINING | `design/ac-capture-curriculum-20260921` / `/home/yjq/rl/CoCap1/ac-capture-curriculum-20260921` | 用户已批准；只允许 initialization-only 实现/preflight；M-CAP failure decomposition 已完成，下一步是 implementation contract、smoke、storage/GPU gates |
| M-COV | COMPLETE_BUDGET | `experiment/mappo-scratch-primitives-20260923` / `/home/yjq/rl/CoCap1/cocap-voradj-mappo-scratch-20260923` | contemporaneous Pure Coverage positive control 200k；argmax/sample CE 80%/100%；不改写 A0 |
| M-CAP | COMPLETE_BUDGET | `experiment/mappo-scratch-primitives-20260923` / `/home/yjq/rl/CoCap1/cocap-voradj-mappo-scratch-20260923` | NormSense V2 Pure Capture 500k；终点评估 4/40 capture、36/40 collision；12 次 collision before ring2、19 次 during ring2、5 次 during ring3、0 次 detection 前；分解工件已交接 |
| B3 | B3_COMPLETE_NO_RANKING_QUALIFIED_CRITIC | MASTER-only comparison / [summary.json](/home/yjq/rl/CoCap1/ac-master-dag-20260921/artifacts/2026-09-22_b3_critic_ranking_gate/summary.json) | B1 exploratory candidates 未优于 historical naive；B2 raw Q 不是 learned checkpoint；[root-cause summary](/home/yjq/rl/CoCap1/ac-master-dag-20260921/artifacts/2026-09-22_b3_critic_ranking_gate/root_cause_summary.json)；不注册 BEST_PRETRAINED_CRITIC |
| B4 | BLOCKED_BY_B3_NO_QUALIFIED_CRITIC | no launch | 不启动 B4；保留 ranking artifact |
| OLD-MIX-CLOSEOUT | CLOSED_300K_FORMAL | existing live worktree | released; no 500k extension |
| IQN-METRIC-AUGMENT | IMPLEMENTED_TESTED_PENDING_INTEGRATION | `evaluation/iqn-metric-augment-20260922` / `/home/yjq/rl/CoCap1/iqn-metric-augment-20260922` | HEAD `5c146f7`；CPU compile、指标回归与 IQN contract tests 通过；只在下一次 formal evaluation 接入，不改 live training |
| IQN-Z05-INDEPENDENT-EVIDENCE | EXTERNAL_EVIDENCE_ONLY | `evaluation/iqn-z05-independent-20260923` @ `2eeec7e` / `/home/yjq/rl/CoCap1/iqn-z05-independent-20260923` | selected 600k；Native 12p3e Coverage/Capture/Mixed 100/100/85；记录 friend-token truncation 与 fixed-map Mixed collision 上升；不启动新 IQN training |
| A4 | BLOCKED | none | no action until learner + user-approved initialization curriculum gates |

child 数量不再使用静态上限；MASTER 按 scientific dependency、GPU/VRAM、CPU/RAM、disk、worktree 冲突和平台实际限制动态并行。保持单层 `MASTER -> child`，child 不得 spawn nested child、不写中央 DAG/state、不停止他人进程。当前所有 formal job 均不 live；根盘约 49 GiB free、95% used，新的 long run 在 planned concurrent peak + 15 GiB safety margin 清晰前保持 blocked。用户已明确批准 A1 control、A3 initialization-only preflight/gated training、A2 bounded audit 和 M-CAP failure decomposition；不需要再次请求上述授权。

## 2026-09-25 最新 reconciliation

- Resume 清理台账：`artifacts/2026-09-21_ac_master_dag/storage_audit_20260925.json`。删除范围限定为完成的 IQN stage1/stage2 历史 full resume 与关闭的 A2 diagnostic full resume；最新 IQN stage3 和正式/验证证据均保留。
- A1 clean control handoff：`control/a1-no-entropy-20260925` @ `411afed`；`artifacts/2026-09-25_a1_control_preflight/cpu/smoke_report.json` 与 `cuda/smoke_report.json` 均 PASS。长跑未授权，等待并发峰值估算、15 GiB margin 与 GPU lease。
- A2 semantic audit：`artifacts/2026-09-25_a2_semantic_audit/a2_semantic_audit_20260925.json`。实现语义正确；近最大熵目标是有界假设，不构成新训练授权。
- M-CAP failure decomposition：`artifacts/2026-09-25_mcap_failure_decomposition/mcap_failure_decomposition_20260925.json`。检测前碰撞未见证据；主要瓶颈是 geometry/ring3 state visitation，collision 与 critic credit/reward scale 为放大器。

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

1. A0 CPU preflight、fresh-output CUDA smoke 与 200k formal 均已完成；最终 argmax/sample strict CE 均 20/20、collision 0，分类 `REPRO_PASS`。这证明当前 MAPPO Coverage 可学；A2 仍等待 A1 有效科学 gate。
2. A1 test-only harness 修复、CPU 7项回归与缩小 smoke 均通过（32/32 steps、finite、checkpoint save/load pass）。fresh formal-width run 曾在 step0 报 production CUDA probability assert，但随后 CPU/GPU exact `_gate_evaluation(0) → _collect_step()` 均通过；真实 runner 1-step smoke 仅因第二次 deterministic eval 在 120s 内未完成。结论仍是 engineering blocker、没有 entropy 科学结果；不重启 formal、不解锁 A2。
3. B1 handoff 已完成：epsilon 0.05/0.10 扩展到全 AW9 support，但 overall ranking 未改善，B3 选择仍锁定。
4. B2-R0-FULL 已完成 128 anchors/1152 AW9 branches；MASTER 已完成 B3 ranking gate，结论 `NO_RANKING_QUALIFIED_CRITIC`，因此 B4 不启动，R1 不自动解锁。
5. A2 已在 A0 `REPRO_PASS` 且 A1 deferred 后获得 formal lease；CPU/CUDA smoke 通过，但 formal 首次尝试在 step 808 出现 CUDA actor target logits 非有限，六次 bounded diagnostic 未得到稳定修复，当前 `A2_ENGINEERING_BLOCKED_DEFERRED`，不留 overnight 进程。
6. A3 proposal 已完成并停在 `CAPTURE_CURRICULUM_AWAITING_USER_APPROVAL`；任何实现/训练都暂停到用户明确批准。
7. A0 未得到 reproduction classification 前，MASTER 不宣称当前环境 drift 或 no-drift；A1 100k 前不宣称 entropy causal answer。A1 CUDA smoke 通过不是科学结果。
8. A0 `REPRO_PASS` 已证明 MAPPO Coverage 可学；按当前用户 gate，A1 engineering deferred 后 A2 可作为独立 parallel capability experiment 解锁，但 A2 结果不得解释为 entropy-only 失败的因果证据。B4 仍必须等待 qualified pretrained critic。

## 2026-09-22 current bounded closeout

- A1：用户批准的最后 bounded 工程窗口已收口。CPU/GPU exact `_gate_evaluation(0) -> _collect_step()`、CUDA staged finite checks 均通过；原 formal-width probability assert 没有稳定 production reproduction，因此写入 `A1_ENGINEERING_BLOCKED_DEFERRED`，无 formal retry、无科学结论。
- A2：在 A0 `REPRO_PASS` 后独立解锁。新增 formal runner/config commit `135eca3`，后续 finite diagnostics/数值 guard commit `e383c03`；CPU 1200-step smoke、CUDA 1200-step smoke 与 lease post-sample 通过。正式 PID `354627`、tmux `a2_discrete_sac_cov_formal_20260922`、物理 GPU1、run root `/home/yjq/rl/CoCap1/ac-discrete-sac-preflight-20260921/artifacts/2026-09-22_discrete_sac_cov/ac_discrete_sac_cov_formal_20260922` 在 step 808 退出，未到 25k formal。bounded 诊断显示有限 next observation 输入下 actor `target_policy_logits` 仅 1143/1152 finite；关闭 TF32/SDP 与 MHA fastpath 仍未形成稳定运行，故延期，不把它标为科学失败。
- GPU/IQN：A2 失败后 GPU1 释放。最终 Z05 PID `17097` / tmux `iqn_z05_recovery_20260921` / GPU0 与 Z07 PID `19555` / tmux `iqn_z07_recovery_20260921` / GPU1 均健康；本轮没有停止、迁移、重启或修改 IQN。
- B3：已写 bounded root-cause summary。证据支持“state coverage 不足 + phase-dependent ranking”为主、collision/target noise 为次；critic fitting 单独主因未被证明。B4 继续 `BLOCKED_BY_B3_NO_QUALIFIED_CRITIC`。
- A3：仍为 `CAPTURE_CURRICULUM_AWAITING_USER_APPROVAL`，不实现、不训练。

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

## B2-FULL → B3 handoff（2026-09-22）

- B2-R0-FULL：128 anchors，四个 phase 各 32；1152 AW9 branches；D_CF_R0 183728 rows；alternative top-1 `0.7734375`，mean best-minus-BC `13.2970`。
- B2 仍不是 learned critic：raw `Q^BC(s,a)` 仅作 direct simulator ranking diagnostic。D_CF successful-state fraction 为 `0`，27/1152 branches collision；state support 虽 unique ratio `6.2718`，nearest-D_BC p90 `0.1822` 且 100% 在 radius `0.75` 内，说明 off-BC state 扩展仍有限。
- B3 已统一比较 historical naive/B1 LocalQ candidates。B1 BC-only：top1 `0.1719`、Spearman `0.1536`、regret `15.0065`；epsilon=.05/.10 均使主要 global ranking 指标退化。结论：`NO_RANKING_QUALIFIED_CRITIC`，B4 不启动。

## A1 修复、smoke 与 formal handoff（2026-09-22 01:10）

- 根因分类：原始 CUDA device-side assert 在 bounded diagnostic 中未复现；真实 logits/softmax/behavior mixture/`torch.multinomial` 路径均通过。失败点是 test-only harness 把合法 raw `(B,1)` 输出误判为必须 `(B,)`；已做最小 test-only squeeze/range 修复，未改 actor、critic、entropy alpha、replay、reward、epsilon、LR、环境或 observation。
- CPU regression：7 tests passed，clean harness PASS。CUDA smoke：physical GPU1、32/32 steps、telemetry 全 finite、checkpoint save/load PASS、3.63s；没有 formal/resume 科学结论。
- formal lease：pre-sample `01:01:29--01:01:39`，GPU1 util mean/peak `9.1%/15%`、free VRAM `47726 MiB`；post-sample `01:08:44--01:08:54`，GPU1 `16.9%/27%`、free `47320 MiB`，GPU0 `3.4%/4%`、free `45616 MiB`。IQN Z05 PID `17097`、Z07 PID `19555` 与 A0 PID `224643` 全部保持 heartbeat。
- A1 formal 启动时记录：branch `experiment/ac-entropy-cov-20260921`，HEAD `dc14560e1f2761ee92e36ee8ce769fe0caa8eb11`，PID `317964`，tmux `a1_entropy_localq_cov_formal_20260922`，physical GPU1 / logical `cuda:0`，run `/home/yjq/rl/CoCap1/ac-entropy-cov-20260921/artifacts/2026-09-22_entropy_localq_cov/ac_entropy_localq_cov_20260922`，预算 `300000`；随后在 step0 engineering failure 退出。
- A0 当时已到 `175000/200000`；150k argmax/sample strict CE 均 `20/20`，collision `0`；argmax CE RMS/area CV `0.024722/0.051225`，sample `0.025452/0.056678`，随后继续并完成 200k `REPRO_PASS`。

## A1 formal-width failure handoff（2026-09-22 01:18）

- Fresh A1 formal 使用同一 entropy-only 合同、300k budget、physical GPU1，PID `317964` / tmux `a1_entropy_localq_cov_formal_20260922`；仅完成 step-0 initial evaluation 后退出，未产生任何有效 formal scientific result。
- 日志显示 formal-width production forward 重现 `probability tensor contains either inf, nan or element < 0` 的 CUDA assert；由于未设置 `CUDA_LAUNCH_BLOCKING=1`，stack 后续显现在 actor `decision_feature` 的 linear/CUBLAS 调用。该结果确认 formal-width engineering blocker，不能标为 `ENTROPY_CONTROL_FAILED` 或 `ENTROPY_NOT_SUFFICIENT`。
- GPU1 已释放；退出后 10×1 秒审计 util mean/peak `10%/15%`、free VRAM `47726 MiB`，IQN-Z07 PID `19555` 全程健康。A0、IQN Z05/Z07 未受影响。
- 当时下一步是 Aquinas bounded `CUDA_LAUNCH_BLOCKING=1` formal-width diagnostic；该 diagnostic 后续已完成并通过，但 root cause 仍未复现。未启动 formal retry，未解锁 A2；A2 仍需 A1 完成有效科学测试后确认 `ENTROPY_NOT_SUFFICIENT`。

## A0 final reproduction handoff（2026-09-22 01:27）

- A0 已自然完成 `200000/200000` exact corrected Scratch MAPPO Pure-Coverage contract；PID/tmux 已结束，physical GPU0 released，IQN-Z05 保持正常。
- final 20-episode evaluation：argmax strict CE `20/20`、sample strict CE `20/20`，两者 collision `0`；argmax CE RMS mean/p50/p90 `0.020757/0.021022/0.0338117`，area CV `0.048594/0.035882/0.100254`，time-to-CE `83.35/45.75/157.35`；sample CE RMS `0.024586/0.021842/0.034675`，area CV `0.067607/0.053551/0.106792`，time-to-CE `48.75/42.5/75.85`。
- classification：`REPRO_PASS`。A0 提供了当前 MAPPO Coverage 可学证据，但 A2 仍不能仅凭 A1 formal engineering crash 解锁；必须先完成有效 entropy-only formal 科学测试并得到 `ENTROPY_NOT_SUFFICIENT`。

## A1 exact sequence closeout（2026-09-22 01:53）

- MASTER 直接执行了与 production 相同顺序的 bounded exact diagnostic：`_gate_evaluation(0)` 先完成 3000 deterministic steps，再 `_collect_step()` 到 `global_step=1`。CPU PASS；GPU1 + `CUDA_LAUNCH_BLOCKING=1` 也 PASS，alpha `0.05`，无 invalid tensor/probability/index。
- 真实 runner 1-step smoke 只生成了初始 `eval_step_000000000.json`，在 120 秒上限内因末尾第二次 deterministic evaluation 未完成而 timeout；没有新的 CUDA traceback。该 timeout 不是科学结果，也不证明 formal 已修复。
- 最终分类：`ENGINEERING_BLOCKED_UNREPRODUCED_FORMAL_ASSERT`。原 PID `317964` 的 formal-width assert 仍保留为 engineering evidence，但没有可复现 root cause，也没有合理最小 production fix；A1 不重启 formal，A2 不解锁。

## 2026-09-22 22:00 MASTER reconciliation / IQN-METRIC-AUGMENT

- IQN 正常且未受干扰：Z05 PID `17097` / tmux `iqn_z05_recovery_20260921` / physical GPU0；heartbeat `400000/400000`，最新 heartbeat `22:00:39`。Z07 PID `19555` / tmux `iqn_z07_recovery_20260921` / physical GPU1（`CUDA_VISIBLE_DEVICES=1` 映射）；heartbeat `400000/400000`，最新 heartbeat `22:00:43`，处于 formal-evaluation 状态。两条均无 dead PID、broken tmux 或 resume-needed 证据，不恢复、不迁移、不改科学合同。
- 只读资源审计：GPU0 `17%` util、`30722 MiB` free、`54 C`，另有不属于 MASTER 的 external LightNav PID `754220` 占用显存；GPU1 `0%` util、`47726 MiB` free、`73 C`，仅有 IQN-Z07。根盘约 `37 GiB` free、`96%` used；本轮没有申请新 GPU lease，也没有启动长训。
- `IQN-METRIC-AUGMENT` 已在独立 worktree `/home/yjq/rl/CoCap1/iqn-metric-augment-20260922`、branch `evaluation/iqn-metric-augment-20260922` 完成实现并提交 HEAD `5c146f7`，没有修改 live IQN worktree/runtime。评估层新增：Pure Capture 的 `capture_steps/capture_seconds`；Pure Coverage 的 `time_to_strict_CE_steps/time_to_strict_CE_seconds`；Mixed 的 `capture/recovery/mission steps/time` 与 `post_capture_CE`。所有时间统计保留 `mean/median/p90/success_n`，并显式分开 `failure_n/censored_n`，失败或 censored episode 不进入成功时间均值。
- 验证：`py_compile` 通过；`tests/test_iqn_efficiency_metrics.py` `2 passed`；`tests/test_iqn_z_unified_decay_curriculum_contract.py` `14 passed`。没有进行 live GPU evaluation overlay；现有 Z05/Z07 formal reports 保持原样，下一 gate 是在下一次 formal evaluation 前由 MASTER 选择/接入该隔离分支。
- AC gate 不变：A0 `REPRO_PASS`；A1/A2 均 deferred engineering blocker，无科学重试；B3 `NO_RANKING_QUALIFIED_CRITIC`，B4 blocked；A3 继续 `CAPTURE_CURRICULUM_AWAITING_USER_APPROVAL`。当前没有可合法启动的 AC overnight 长训线。

## 2026-09-23 A1/A2 Astra 根因修复后续

- Astra 分支 `audit/a1-a2-cuda-root-cause-20260922`（HEAD `8fd2401`）完成了 A1/A2 共同根因审计，分类为 `SHARED_ROOT_CAUSE_CONFIRMED_FIXED`。原始 A1 失败发生在 target actor，A2 失败发生在 online actor target-policy；两者都由 fully-masked terminal placeholder row 进入 Transformer encoder 的 masked softmax 产生 NaN。
- 最小 production fix 为 commit `4e210c0`：仅在 encoder 前为全 mask terminal row 打开 placeholder token，保留有效 row bit-exact；不改 entropy、SAC、reward、replay、epsilon、LR、gamma、tau、observation 或 environment 合同。Astra 的 exact failing-row replay、CPU/CUDA regression、14 项 shared tests、A1 7 项 tests，以及 A1/A2 post-fix finite production smokes 均通过。
- 当前 A1/A2 只表示 `ROOT_CAUSE_CONFIRMED_FIXED_NO_SCIENTIFIC_RESULT`。下一门禁是把 `4e210c0` cherry-pick 到两条实验 branch，分别完成 CPU/CUDA regression；A1 完成 bounded production path，A2 必须超过 step 808 并验证 replay、actor、target、Q1/Q2、target、update 全部 finite。通过后才启动全新 0/25k/50k/75k/100k formal；100k 后是否追加 300k 由 registered positive signal 决定。
- 正式 run 采用 latest-only 输出根目录与独立新 run 名称；启动前重做 `df -h /`、`df -i /`、planned root `du -sh` 及 IQN heartbeat/GPU 10×1s 资源审计。现有 IQN 进程保持原 PID、GPU、tmux 与科学合同，不作停止、迁移或修改。

## 2026-09-23 A1/A2 formal 已启动

- A1 branch HEAD `78adf78` 的真实 CUDA bounded production path 完成 1000 steps / 63 updates，actor、critic、target 均发生参数更新且 telemetry 全 finite。A2 branch HEAD `59faae3` 的真实 CUDA 900-step path 越过历史 step 808，replay `3600`、38 次 twin-Q SAC update，actor/critic1/critic2 均更新且 telemetry 全 finite。
- A1 fresh formal：PID `963907`，tmux `a1_entropy_localq_cov_formal_20260923`，physical GPU0（`CUDA_VISIBLE_DEVICES=0`），run root `/home/yjq/rl/CoCap1/ac-entropy-cov-20260921/artifacts/2026-09-23_entropy_localq_cov/a1_entropy_localq_cov_formal_20260923`，预算 `100000`。
- A2 fresh formal：PID `963912`，tmux `a2_discrete_sac_cov_formal_20260923`，physical GPU1（`CUDA_VISIBLE_DEVICES=1`），run root `/home/yjq/rl/CoCap1/ac-discrete-sac-preflight-20260921/artifacts/2026-09-23_discrete_sac_cov/a2_discrete_sac_cov_formal_20260923`，预算 `100000`。
- 启动前根盘为 `34 GiB` free、`97%` used、inode `7%`；启动后 10×1s 采样为 GPU0 util `4–6%`、free `47320 MiB`，GPU1 util `5–7%`、free `47304 MiB`。Z05 PID `17097` 与 Z07 PID `19555` 仍 live，未停止、迁移或修改。正式检查点为 `25k/50k/75k/100k`；100k 后是否追加 300k 只按 registered positive signal 决定。
- 本地中央记录 commit `4b54394` 已包含本轮所有事实；随后已通过显式 proxy 推送，origin 已同步到 `4b54394`。

## 2026-09-23 13:33 final reconciliation：DAG 各线与 IQN Z 线

- **A0**：`200000/200000`，`REPRO_PASS`；最终 argmax/sample strict CE 均 `20/20`，collision `0`，已释放 GPU0。
- **A1**：fresh `100000/100000` complete；0/25k/50k/75k/100k checkpoints、actor/critic/target 更新和 telemetry finite 均通过。75k 单 deterministic eval strict CE 成功，100k 单 eval strict CE 失败并发生 collision；当前分类 `A1_FORMAL_100K_COMPLETE_NO_SUSTAINED_ENTROPY_SIGNAL`，不追加 300k。
- **A2**：exact categorical AW9 SAC fresh `100000/100000` complete；replay `400000`、`24813` updates、telemetry 全 finite、alpha `0.9786`。最终 20-episode deterministic eval strict CE `0/20`、collision `2/20`、CE RMS mean `0.2165`、area CV mean `0.4671`；当前分类 `A2_FORMAL_100K_COMPLETE_NO_STRICT_CE_SIGNAL`，不追加 300k。
- **B1/B2**：均为 `COMPLETE`，保留既有 handoff；B1 没有全局 ranking 稳定提升，B2 R0 完成但 state support 有限。
- **B3/B4**：B3 为 `NO_RANKING_QUALIFIED_CRITIC`，B4 保持 blocked。
- **A3/A4**：A3 仍 `CAPTURE_CURRICULUM_AWAITING_USER_APPROVAL`；A4 保持 `BLOCKED`，没有实现或训练。
- **旧 AC-COV/CAP/MIX**：分别保持旧线 closeout、300k nonfinite closeout、300k closeout，不恢复、不扩预算。
- **IQN Z05**：PID `17097` 已 clean complete，training target `700000`；alpha `0.5`，stage3 selected checkpoint `600000`，final report 的 stage3 指标为 pure capture `1.0`、pure coverage strict CE `1.0`、mixed post-capture CE/safe-complete `0.85/0.85`、worst collision `0.10`。
- **IQN Z07**：PID `19555` 已 clean complete，training target `700000`；alpha `0.7`，stage3 selected checkpoint `300000`，final report 的 stage3 指标为 pure capture `0.95`、pure coverage strict CE `0.85`、mixed post-capture CE/safe-complete `0.65/0.65`、worst collision `0.05`。
- Z05/Z07 final reports 已写入各 runtime 目录，PID/tmux 均已释放；当前 GPU0/GPU1 idle，各约 `48524 MiB` free。根盘 `33 GiB` free、`97%` used、inode `7%`。
## 2026-09-23 15:30 repository sync

- 用户授权的命令级 proxy push 已成功：`4538aca..4b54394` 推送到 `origin/ops/ac-master-dag-20260921`。
- 推送后校验：本地 HEAD 与 origin 均为 `4b54394af011515f40ca26ba316c6755579119e7`，工作树干净；中央 JSON 与本文件的远端同步状态已更新为 `PUSHED_SYNCED`。

## 2026-09-25 RECOVER → RECONCILE → RECLASSIFY

本轮先恢复本地事实，再更新中央状态；没有启动新长训。`git fetch origin` 使用命令级 mihomo `127.0.0.1:17892` 完成，未使用 `17891`，未修改 `.bashrc`、global Git 或 system proxy。远端快照不是本轮事实优先级；本地 verified artifacts/runtime 优先。

### CENTRAL

- branch：`ops/ac-master-dag-20260921`；开始 reconciliation 时 HEAD/origin 均为 `7dbd6945`；本轮中央写入尚未提交，提交后再做 proxy push。
- 中央事实源仍只有本文件和 `artifacts/2026-09-21_ac_master_dag/state.json`。
- 已删除旧的 `child_concurrency_limit=4` 语义，改为按 scientific dependency、GPU/VRAM、CPU/RAM、disk 和 worktree 冲突动态并行；结构保持单层 `MASTER -> child`。
- 根盘：约 26 GiB free、98% used、inode 7%；新的 long run 需要 latest-only、planned-root `du -sh` 和安全 margin 复核后才可申请 GPU lease。
- 当前 GPU0/GPU1 均 idle，未发现本轮恢复的 live formal PID/tmux。

### RECOVERED CHILD HANDOFFS

#### A1：旧负结论撤销并正式 reopen

`STATE-A1` 在 `/home/yjq/rl/CoCap1/a1-conclusion-push-20260923` @ `ad8c6b5` 核对了原始 0/25k/50k/75k/100k artifact 与 independent evaluator。固定 reset seeds 为 `2026097101..2026097120`，每个 checkpoint 20 argmax + 20 sample；评估期间 `parameter_updates=0`，这只表示 inference evaluation 不更新参数，训练本身确实更新过 actor/critic/target。

| checkpoint | argmax strict CE | sample strict CE | sample collision |
|---|---:|---:|---:|
| 50k | 0/20 | 12/20 | 5/20 |
| 75k | 9/20 | 12/20 | 8/20 |
| 100k | 13/20 | 20/20 | 0/20 |

旧分类：`A1_FORMAL_100K_COMPLETE_NO_SUSTAINED_ENTROPY_SIGNAL`。新证据：跨 checkpoint 的 40-rollout 双模式结果。新分类：`A1_SUSTAINED_PARTIAL_LEARNING_REOPENED`。改变原因：旧结论主要依赖 100k 单次评估；新结果显示持续改善，但 100k argmax 仍有 5/20 collision、2/20 censored，不能宣称 deterministic/full success。Astra fix provenance 统一为完整 commit `4e210c090ec696bda73ff36877f8e33270ef8a86`；训练 branch `78adf78` 含 patch-equivalent fix。

下一步注册两个相互独立的 child：

- `A1-CONTROL`：corrected matched Local-Q no-entropy，唯一科学变量 entropy coefficient=0，fresh 0→100k；不得用旧 pre-Astra run 替代。
- `A1-EXTEND`：先审计 actor/critic/target/optimizer/replay/RNG/environment/counters 的 resume provenance；只有 exact/scientifically valid 才能称 `100k continuation`，否则另立 warm continuation。

#### MAPPO：M-COV 与 M-CAP 不改写 A0

`STATE-MAPPO` 在 `/home/yjq/rl/CoCap1/cocap-voradj-mappo-scratch-20260923` @ `61b1d17` 核对本地正式 artifact；旧 59.3k/90.3k remote/status snapshot 已判定 stale。

- `M-COV`：`COMPLETE_BUDGET`，真实 step `200000`；milestones `0..200k` 每 25k；terminal latest-only checkpoint `m-cov-run/latest.pt`，SHA256 `3adfdbe90970cfc1333a4a5cba82ca496393d170c56b789a842f0c4ba06fe68f`；最终 argmax/sample CE `80%/100%`，collision `5%/0%`。最佳观测为 150k 的 100%/100%，但无独立 best checkpoint。
- `M-CAP`：`COMPLETE_BUDGET`，真实 step `500000`；milestones `0..500k` 每 25k；terminal latest-only checkpoint `m-cap-full/latest.pt`，SHA256 `01eeb67c689dd323c2e8e680549ed8e5bd871b7b0a75b8b989e19aa02e953197`；最终 argmax/sample capture `10%/10%`，collision/censoring `90%/90%`。最佳观测也只是 argmax 25%（150k/350k）和 sample 15%（225k）。

旧分类：中央尚未登记，远端仅显示中途进度。新证据：本地 formal progress/report 完成。新分类：M-COV `CONTEMPORANEOUS_POSITIVE_CONTROL_COMPLETE`，M-CAP `WEAK_CAPTURE_FAILED_OR_COLLISION_LIMITED`。改变原因：本地正式 artifact 完整度高于 remote snapshot；A0 `REPRO_PASS` 保持不变，M-COV 不替代 A0。

M-CAP 失败触发 `MAPPO-CAP-ROOTCAUSE` read-only bounded gate；不得自动实现/训练 A3。只有 root cause 支持 `STATE_VISITATION_LIMITED` 或 successful geometry 稀疏时，A3 才进入 `JUSTIFIED_CANDIDATE` 用户审核。

#### A2：fixed-seed 双模式复评完成，分类为 transient learning

旧分类：`A2_100K_FORMAL_COMPLETE_NO_STRICT_CE_SIGNAL`，随后因只有单一 100k argmax 而暂置 `A2_REEVALUATION_INCOMPLETE / A2_PENDING_FIXED_SEED_REEVALUATION`。新证据已补齐五个 checkpoint（`0/25k/50k/75k/100k`）的固定 env/policy seed、20 episode argmax+sample pair；四个 artifact root 位于 `/home/yjq/rl/CoCap1/ac-discrete-sac-preflight-20260921/artifacts/2026-09-25_a2_reeval_fixedseed_v3`、`..._v4_step50000`、`..._v4_step75000`、`..._v4_step100000`，四份 seed manifest SHA256 相同：`4d17ddbeb5c2355272cebae18f1bef7524d38f136139e4706c76d4468c8cdcf8`。

结果（每格均为 strict CE 成功数/20；括号为 collision 数/20）：

| checkpoint | argmax | sample |
|---|---:|---:|
| 0 | `0/20 (20/20)` | `0/20 (20/20)` |
| 25k | `0/20 (0/20)` | `6/20 (5/20)` |
| 50k | `0/20 (0/20)` | `5/20 (3/20)` |
| 75k | `0/20 (4/20)` | `0/20 (1/20)` |
| 100k | `0/20 (1/20)` | `0/20 (2/20)` |

所有 10 个 pair 均 `episodes=20`、`parameter_updates=0`、`trainer_update_calls=0`、evaluation tensors finite。旧分类 → 新证据 → 新分类 → 原因：单一 100k 结论不足；完整序列显示 strict CE 只在 sample 的 25k/50k 短暂出现，argmax 从未出现，75k/100k 双模式均回到 0/20，因此最终为 `A2_TRANSIENT_LEARNING`，不是 sustained partial/strong learning。下一步仅允许一次 bounded semantic audit；不扩 300k，不启动新 A2 training。

#### IQN：外部证据，不扩训练

`STATE-IQN` 在 `/home/yjq/rl/CoCap1/iqn-z05-independent-20260923` @ `2eeec7e` 核对 selected checkpoint `600000` 与 60/60 complete artifacts：Native 12p3e `Coverage/Capture/Mixed=100/100/85`；4p `65/100/50`；8p `95/95/85`；fixed-map 16p4e `100/100/75`、20p5e `100/100/60`、24p6e `95/100/65`。friend-token truncation 在 fixed-map 增加，Mixed collision 从 Native 10% 增至 15%/25%/25%。

新分类：`EXTERNAL_EVIDENCE_ONLY`。不改变 live training、不解锁新 IQN training；Z05/Z07 原已完成的训练状态保留。

### BLOCKERS / NEXT AUTOMATIC GATES

- Engineering：A2 fixed-seed evaluator 已完成，下一步为 bounded semantic audit；A1 continuation resume provenance 尚未审计；A1 corrected no-entropy control 尚未启动。
- Resource：根盘 98% used；任何新 long run 在 storage margin 未清除前 blocked。评测/CPU analysis 优先。
- Scientific：A3 只在 Capture root-cause 支持 visitation/geometry 稀疏时进入用户审核；B1/B2/B3/B4 不推进；A4 保持 blocked。
- Automatic next gates：`A2 fixed-seed argmax+sample complete -> A2_TRANSIENT_LEARNING -> one bounded semantic audit; no 300k extension`; `M-CAP weak complete -> MAPPO-CAP-ROOTCAUSE`; `A1 partial reopened + storage/GPU lease cleared -> A1-CONTROL and A1-EXTEND may run independently`; `root-cause supports visitation limitation -> A3_JUSTIFIED_CANDIDATE + explicit user approval`; no condition currently authorizes A3 implementation/training.

### A1 continuation / control bounded handoffs

- `A1-EXTEND-PROVENANCE` 已完成：`model_step_000100000.pt` 是 `MODEL_ONLY_WARMSTART`；`full_resume.pt` 是 `FUNCTIONAL_NONEXACT_RESUME`。虽然 actor/critic/target、optimizer、global RNG 和 counters 部分存在，但 replay/recovery pool、runner 私有 RNG、replay RNG、environment/scenario state、episode/scene/scheduler state、source/manifest hash 缺失；loader 会 `rewarm_without_replay` 并 reset environment。因此 A1 100k overall=`INSUFFICIENT_FOR_CONTINUATION`，不得命名为 100k→300k exact continuation。
- `A1-CONTROL-PREFLIGHT` 已完成：matched contract PASS；唯一科学变量为 `actor_entropy_alpha=0.0`，复用 seed `2026092101`，fresh run root 为 `/home/yjq/rl/CoCap1/ac-entropy-cov-20260921/artifacts/2026-09-25_a1_matched_no_entropy_control/a1_matched_no_entropy_localq_cov_20260925`，估算约 382 MB。当前 launch blocked：配置尚未提交、runner/trainer worktree dirty，且完整 Astra regression collection 缺 `cocap_voradj.training.discrete_sac`；不得以旧 pre-Astra run 替代。

### MAPPO Capture root-cause bounded handoff

`MAPPO-CAP-ROOTCAUSE` 已完成只读分析，未实现/训练 A3。M-CAP 的 enemy visible fraction 为 `98.76%/99.16%`，first detection 通常 latency=1，因此 `DETECTION_LIMITED` 排除；ring3 visitation 仅 `15%/30%`，ring3 max hold `0.85/1.65` steps，支持 `GEOMETRY_LIMITED` 主因及 `STATE_VISITATION_LIMITED` 的 ring2→ring3 transition 瓶颈。终点 collision `90%/90%`，其中 agent-agent collision 占主要部分，故 `COLLISION_LIMITED` 是强放大器；M-CAP explained variance `0.356` 对比 M-COV `0.792`，GAE/return 方差与 safety-dominated negative return 支持 `CRITIC_CREDIT_LIMITED`/`REWARD_SCALE_LIMITED` 次级放大。Entropy、KL、clip fraction 没有显示 PPO collapse/early-stop 饥饿。精确的 geometry vs collision vs critic/reward 因果排序仍为 `UNRESOLVED`。

因此 A3 已升级为 `A3_JUSTIFIED_CANDIDATE_USER_APPROVAL_REQUIRED`：root-cause 足以支持 geometry/state-visitation 方向，但这只是用户审核候选；未经明确批准不得实现或训练 curriculum。

### A2 fixed-seed reevaluation closeout

A2 的两个早期 bounded child 曾因响应窗口未产出完整结果而关闭；随后 MASTER 在不训练、不恢复 optimizer/replay 的 weight-only evaluator 中完成了全部五 checkpoint×两模式×20 episodes。evaluator 明确禁止 `trainer.update`，最终 10 个 pair 都报告 `parameter_updates=0`、`trainer_update_calls=0`，并保留 checkpoint SHA、loaded state hash、固定 seed manifest 与 finite tensor 审计。

这组结果已把 A2 从 `UNDETERMINED` 收敛为 `A2_TRANSIENT_LEARNING`：只有 sample 在 25k/50k 有短暂 strict-CE 成功，不能支持 sustained partial/strong learning。下一步是一次独立 bounded semantic audit；在 audit 完成前不启动 300k extension、不做新的 A2 training。A3 的实现/训练仍需用户明确批准。
