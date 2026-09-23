# AC MASTER DAG（2026-09-21）

本文件与 `artifacts/2026-09-21_ac_master_dag/state.json` 是跨 session 的唯一中央状态。只有 MASTER 可以写这两个文件；child 只能在各自 branch/worktree/rundir 工作，并以结构化 handoff 返回结果。禁止 nested subagents。

## 当前事实源

- MASTER branch：`ops/ac-master-dag-20260921`，本地 HEAD `beea8a4`，origin 基线 `4538aca`；显式 `127.0.0.1:17892` push 被自动 review 拒绝，未尝试任何 workaround，当前保持 local-ahead。基于 `ops/training-performance-sync-20260921`；后续 push 每次必须显式使用 mihomo `127.0.0.1:17892` 的 `http_proxy/https_proxy/all_proxy` 与 `git -c http.proxy/-c https.proxy`，禁止 stale `17891`。
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

根盘当前约 37 GiB free、96% used、inode 6%。A0/A1 formal 已分别使用 latest-only/无 replay 累积合同；任何后续长训前仍必须重新执行 `df -h /`、`df -i /`、planned run root `du -sh`，并估算 checkpoint/replay/resume/evaluation/telemetry。

## DAG gate

| task | 当前状态 | branch / worktree | 下一 gate |
|---|---|---|---|
| A0 | REPRO_PASS | `experiment/ac-mappo-cov-repro-20260921` / `/home/yjq/rl/CoCap1/ac-mappo-cov-repro-20260921` | 200k complete；最终 argmax/sample strict CE 均 20/20、collision 0；GPU0 released；当前 MAPPO Coverage 可学 |
| A1 | A1_FORMAL_RUNNING_AFTER_ROOT_CAUSE_FIX | `experiment/ac-entropy-cov-20260921` @ `78adf78` / `/home/yjq/rl/CoCap1/ac-entropy-cov-20260921` | PID `963907` / tmux `a1_entropy_localq_cov_formal_20260923` / physical GPU0；fresh 0→100k，25k/50k/75k/100k checkpoints；尚无科学分类 |
| B1 | B1_COMPLETE | `experiment/ac-bc-exploratory-critic-20260921` / `/home/yjq/rl/CoCap1/ac-bc-exploratory-critic-20260921` | handoff 完成；support 扩大但无 global ranking stability；等待 B2-R0-FULL 后进入 B3 |
| B2 | B2_COMPLETE | `experiment/ac-bc-counterfactual-critic-20260921` / `/home/yjq/rl/CoCap1/ac-bc-counterfactual-critic-20260921` | 128/128 anchors、1152/1152 branches；alternative top-1 0.7734；R1 draft eligible 但未自动解锁 |
| A2 | A2_FORMAL_RUNNING_AFTER_ROOT_CAUSE_FIX | `experiment/ac-discrete-sac-preflight-20260921` @ `59faae3` / `/home/yjq/rl/CoCap1/ac-discrete-sac-preflight-20260921` | PID `963912` / tmux `a2_discrete_sac_cov_formal_20260923` / physical GPU1；fresh 0→100k，25k/50k/75k/100k checkpoints；尚无科学分类 |
| A3 | CAPTURE_CURRICULUM_AWAITING_USER_APPROVAL | `design/ac-capture-curriculum-20260921` / `/home/yjq/rl/CoCap1/ac-capture-curriculum-20260921` | proposal commit `377a6afd`；必须用户审核批准后才能实现/训练 |
| B3 | B3_COMPLETE_NO_RANKING_QUALIFIED_CRITIC | MASTER-only comparison / [summary.json](/home/yjq/rl/CoCap1/ac-master-dag-20260921/artifacts/2026-09-22_b3_critic_ranking_gate/summary.json) | B1 exploratory candidates 未优于 historical naive；B2 raw Q 不是 learned checkpoint；[root-cause summary](/home/yjq/rl/CoCap1/ac-master-dag-20260921/artifacts/2026-09-22_b3_critic_ranking_gate/root_cause_summary.json)；不注册 BEST_PRETRAINED_CRITIC |
| B4 | BLOCKED_BY_B3_NO_QUALIFIED_CRITIC | no launch | 不启动 B4；保留 ranking artifact |
| OLD-MIX-CLOSEOUT | CLOSED_300K_FORMAL | existing live worktree | released; no 500k extension |
| IQN-METRIC-AUGMENT | IMPLEMENTED_TESTED_PENDING_INTEGRATION | `evaluation/iqn-metric-augment-20260922` / `/home/yjq/rl/CoCap1/iqn-metric-augment-20260922` | HEAD `5c146f7`；CPU compile、指标回归与 IQN contract tests 通过；只在下一次 formal evaluation 接入，不改 live training |
| A4 | BLOCKED | none | no action until learner + user-approved initialization curriculum gates |

第一波最多四个 child：A0、A1、B1、B2。A0/A1 formal 现在由 MASTER 直接持有已审计的 tmux/PID/GPU lease；B1/B2 已完成并释放 child slot，A2/A3 已完成 preflight/design。A3 已停在用户审核 gate，不得自动实现或训练。

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
- 本地中央记录 commit `beea8a4` 已包含本轮所有事实；origin 仍为 `4538aca`，因为显式 proxy push 被自动 review 拒绝，未通过其他路径外传。
