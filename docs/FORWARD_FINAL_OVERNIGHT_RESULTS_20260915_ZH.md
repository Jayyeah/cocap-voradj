# Forward Final overnight 结果与历史性能完整性核验（2026-09-15）

本页记录 2026-09-14 夜间两条 bounded diagnostic 的最终结果。两条均为 `EXPLORATORY_NON_GATE`，不改变 P1/P2/P3 或 formal Gate：P1 transition/return 在四类修复后 PASS，P1 central-V state aliasing `UNRESOLVED`，P2 `FAIL/HOLD`，P3 `INCONCLUSIVE/HOLD`，formal PPO 与 formal Scratch 均 `HOLD`。所有运行锁定 `terminal-priority-truncation-bootstrap-weighted-ce-v2`。

完整逐 episode、phase、reset-source、reward、occupancy、return、entropy、KL、clip、EV、gradient 与 SHA 数据保存在 [MASTER_SUMMARY.json](../artifacts/2026-09-14_overnight/MASTER_SUMMARY.json) 及各 run 目录；本页给出可读的关键数值和裁决。大 checkpoint 仍按仓库规则只保存在本地。

## 后台完成状态

| 线 | GPU / PID / tmux | 最终状态 | 实际预算 |
|---|---|---|---:|
| BC→PPO | GPU1 / 622686 / `cocap_overnight_bc_ppo_20260914` | `STOP_BC_PPO_AT_5K`，`REGRESSED` | 5,000；未续 10,000 |
| Scratch MAPPO | GPU0 / 622682 / `cocap_overnight_scratch_20260914` | `COMPLETE_BUDGET_STOP`，`NEGATIVE_DIAGNOSTIC_NOT_ALGORITHM_FAILURE` | 100,000 |

启动时两张 A6000 均无项目外计算进程，两个 run 使用独立 output、checkpoint、RNG、online capture pool 和 seed。CPU/CUDA 两组 smoke、zero-update log-prob、finite metrics、resume transition/RNG/optimizer exact round-trip 全部通过；回归测试为 38 passed（仅既有 protobuf deprecation warnings）。Scratch provenance guard 报告 `teacher_dependency=0`，没有读取 IQN、BC actor/dataset、teacher-Q 或已训练 tensor。

## BC→PPO：step0 → 5k

step0 和 5k 都完成固定 seed 的 argmax/sample × mixed/coverage、每组 20 局（总 80 局）完整 evaluator。step0 是 immutable `step_000000.pt` 与 `eval_step_000000.json`；5k 保存 `step_005000.pt`、`eval_step_005000.json`、`matched_step_005000.json` 和 `gate_005000.json`。

| eval 组 | step0 capture / CE / safe / collision | 5k capture / CE / safe / collision | matched safe Δ | matched return Δ | 裁决要点 |
|---|---:|---:|---:|---:|---|
| argmax mixed | 20 / 20 / 20 / 0 | 20 / 0 / 0 / 0 | −1.00 | +8.027 | post-CE/safe collapse |
| sample mixed | 20 / 20 / 20 / 0 | 20 / 0 / 0 / 0 | −1.00 | +6.079 | post-CE/safe collapse |
| argmax coverage | 20 / 20 / 20 / 0 | 0 / 0 / 0 / 0 | −1.00 | −2.364 | safe decline；return consistent decline |
| sample coverage | 0 / 20 / 20 / 0 | 0 / 3 / 3 / 0 | −0.85 | −2.471 | safe decline；仅 3 个共同安全时间样本 |

四组均触发 safe 下降超过 5 percentage points；mixed 同时触发 post-recovery collapse；两 coverage 组触发一致 return 恶化。共同安全 mission time 的配对样本为 0、0、0、3，故时间项按合同是不可判读的；这不影响其他硬失败项。操作结论为 `REGRESSED`，按预声明规则停止在 5k，不调 LR/entropy/clip，不补跑 10k，也不启动 PPO 25k。该 screen 不是统计显著性或 formal Gate PASS。

5k 最后一次实际更新指标（训练 checkpoint 在 step 5000，最后完整 rollout update 在 step 4864）：value loss `0.001364`，EV `−0.5913`，entropy `0.4292`，clip fraction `0.1063`，exact full-batch KL `0.05308`，actor/value grad norm `7.681/0.807`，zero-update log-prob error `1.14e−5`。训练流共 5,000 env steps、20,000 active-agent rows、6 个真实 capture events、6 个 recovery snapshots。

## Scratch MAPPO：0 → 100k

每个节点均完成 argmax/sample × mixed/coverage × 20 局（80 局），完整 native horizon；25k、50k、75k、100k 没有 zero-capture 早停。

| 节点 | mixed capture（argmax/sample） | mixed post action / CE success | pure CE success（argmax/sample） | 备注 |
|---:|---:|---:|---:|---|
| 0 | 0 / 0 | 0 / 0 | 0 / 0 | random-init；mixed collision 20/20，sample coverage collision 20/20 |
| 25k | 0 / 0 | 0 / 0 | 0 / 0 | argmax 两场景 collision 0；sample mixed/coverage 18/19 |
| 50k | 0 / 0 | 0 / 0 | 0 / 0 | argmax 两场景 collision 0；sample mixed/coverage 9/17 |
| 75k | 0 / 0 | 0 / 0 | 0 / 0 | argmax 两场景 collision 0；sample mixed/coverage 11/15 |
| 100k | 0 / 0 | 0 / 0 | 0 / 0 | argmax 两场景 collision 0；sample mixed/coverage 19/20 |

所有 Scratch checkpoint 的 mixed 都没有真实 capture、post action、recovery snapshot、2+ 或 3+ visitation；训练流 100k 共有 400,000 active-agent rows、support rows `42,518`，phase steps `pre_capture=45,801`、`pure_coverage=54,199`，真实 capture pool size `0`。固定 eval 的 pure/mixed safe completion 没有持续正向变化，配对 mission/recovery time 在无共同成功时正确记为 `n=0`，不把删失 episode 当短任务。

| 训练 checkpoint | 最后完整 update step | value loss | EV | entropy | exact KL | clip | actor/value grad norm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25k | 24,832 | 1.1971 | 0.0040 | 2.1660 | 3.53e−5 | 0 | 0.545 / 4.590 |
| 50k | 49,920 | 0.00846 | 0.1813 | 2.1358 | 1.06e−4 | 0 | 0.433 / 0.328 |
| 75k | 74,752 | 0.00796 | 0.9847 | 2.1044 | 7.58e−5 | 0 | 0.544 / 0.471 |
| 100k | 99,840 | 0.02595 | 0.9509 | 2.1177 | 7.14e−4 | 0 | 0.552 / 1.290 |

按预声明分类，100k 结果为 `NEGATIVE_DIAGNOSTIC_NOT_ALGORITHM_FAILURE`：没有连续前驱改善，也没有 capture/post；这只说明本 single-seed bounded diagnostic 没有得到 Full-Task learnability signal。它不证明 Scratch MAPPO 算法不可行。禁止自动 200k、400k 或 3 seeds。

## 历史性能测试完整性核验与补测决定

逐项读取并解析以下关键历史性能 artifact，均存在且 JSON 可读：

| 历史链条 | 必要证据 | 状态 |
|---|---|---|
| C3 Forward bridge | `c3_formal100/comparison.json`、BC argmax/sample 与 IQN reports、C3 CPU smoke | 完整 |
| Direct PPO 工程 screen | Direct512/Warm-up512 task eval20 与 comparison | 完整，旧结果已记录 HOLD 边界 |
| Actor LR diagnostic | original/low-LR reports、`lr_probe/comparison.json` | 完整，旧结果已记录 HOLD |
| critic / return calibration | fixed-MC、4096 critic-only、heldout stratified reports | 完整，失败样本边界已记录 |
| P2/P3 standalone B | 独立文档、诊断源码、report、raw NPZ、repository verification | 完整，P2 FAIL / P3 INCONCLUSIVE |
| 本次 BC→PPO | immutable step0、5k eval/matched/gate、finite learning log | 完整，5k 停止 |
| 本次 Scratch | immutable step0、25/50/75/100k eval/matched/checkpoint、learning log | 完整，100k 停止 |

因此没有“大量或关键性能测试”缺失，不启动补测。5k BC 的共同安全时间样本不足、Scratch 的共同安全时间为零，已经作为数据中的 `common_safe_n`/`null` 明确记录；重新随机抽样只为制造共同成功会破坏预声明 matched seed 设计，不能作为补测。若未来要补充时间效应，应由 MASTER 明确新的独立确认 seed/episode 计划，而不是修改本夜结果。

## 交接与 gate 保持

机器可读交接：[MASTER_SUMMARY.json](../artifacts/2026-09-14_overnight/MASTER_SUMMARY.json)、[SESSION_HANDOFF.json](../artifacts/2026-09-14_overnight/SESSION_HANDOFF.json)、[stage_gate.json](../artifacts/2026-09-09_root_cause/stage_gate.json)。实现与协议：[overnight diagnostic contract](FORWARD_FINAL_OVERNIGHT_DIAGNOSTIC_20260914_ZH.md)。本夜结果只作为下一 Gate 的候选输入；formal PPO、formal Scratch、P2/P3、PPO 25k 均继续 HOLD。
