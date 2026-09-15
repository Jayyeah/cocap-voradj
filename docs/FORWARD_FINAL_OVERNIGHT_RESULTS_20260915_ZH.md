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

## BC→PPO 退化的逐项数值复核

下面的数字直接来自 [matched_step_005000.json](../artifacts/2026-09-14_overnight/bc_ppo_seed1/matched_step_005000.json)、[eval_step_000000.json](../artifacts/2026-09-14_overnight/bc_ppo_seed1/eval_step_000000.json) 和 [eval_step_005000.json](../artifacts/2026-09-14_overnight/bc_ppo_seed1/eval_step_005000.json)。每个格子都是固定 matched seed 的 20 局；百分数是完成局数/20，return 是每局 discounted return 均值。

| 组别 | 指标 | step0 → 5k | 变化 |
|---|---|---:|---:|
| argmax/mixed | capture | 100% → 100% | 0 pp |
| argmax/mixed | CE / safe / post-CE | 100% / 100% / 100% → 0% / 0% / 0% | 各 −100 pp |
| argmax/mixed | collision | 0% → 0% | 0 pp |
| argmax/mixed | discounted return | 57.4148 → 65.4421 | +8.0274 |
| sample/mixed | capture | 100% → 100% | 0 pp |
| sample/mixed | CE / safe / post-CE | 100% / 100% / 100% → 0% / 0% / 0% | 各 −100 pp |
| sample/mixed | collision | 0% → 0% | 0 pp |
| sample/mixed | discounted return | 51.9287 → 58.0078 | +6.0791 |
| argmax/coverage | CE / safe | 100% / 100% → 0% / 0% | 各 −100 pp |
| argmax/coverage | collision | 0% → 0% | 0 pp |
| argmax/coverage | discounted return | −17.0345 → −19.3983 | −2.3638 |
| sample/coverage | CE / safe | 100% / 100% → 15% / 15% | 各 −85 pp |
| sample/coverage | collision | 0% → 0% | 0 pp |
| sample/coverage | discounted return | −17.6949 → −20.1662 | −2.4713 |
| sample/coverage | common-safe mission time | 85.1667 s → 593.6667 s（共同样本 n=3） | +508.5000 s，+597.06% |

mixed 两组的 capture 仍是 100%，退化发生在 capture 之后：post phase 各 20 局均未完成 CE，因此 safe completion 从 20/20 变为 0/20。coverage 的 argmax 为 20/20→0/20，sample 为 20/20→3/20；coverage 的共同成功时间只有 3 对，故只能报告这个极小样本的实际值，不能当作稳定时间效应。return 变好不抵消任务完成链条崩溃，因为 Gate 同时检查 post-CE/safe。所有组 collision 都保持 0%，不存在“退化来自碰撞上升”的解释。

5k 触发的实际硬失败为四组 safe 下降超过 5 pp；mixed 另有 post-recovery collapse，两 coverage 组另有一致 return 恶化。因而 `continue_to_10k=false`，没有 10k 或 25k 结果；这条线的结论是 bounded diagnostic `REGRESSED`，不是 formal PPO Gate 结论。

## Scratch MAPPO 的历史最佳与 Final 对比

### 历史最佳是哪一次

在当前仓库可审计的 scratch MAPPO 历史记录中，最佳相关测试是 **2026-08-30 的 MAPPO-9-v2 三 seed 正式桥接运行**，而不是本次 2026-09-14 Final Scratch。配置和原始产物分别在 [mappo9_v2 common.yaml](../configs/experiments/mappo9_v2_20260830/common.yaml) 与 [2026-08-30_mappo9_v2](../artifacts/2026-08-30_mappo9_v2/)；完整审计见 [AC/CTDE gap audit §10–11](AC_CTDE_GAP_AUDIT_20260830_ZH.md)。它是 3 个 random-init seed、每 seed 400k env steps、每 25k checkpoint、deterministic20 + stochastic20，并以 `PASS_TO_MAPPO_AW_V2` 通过了“离散 MAPPO 桥接最低门槛”。这个 PASS 不是强策略成功结论。

| 历史 seed | 最佳 checkpoint | deterministic capture | 同点 collision | 2+ / 3+ visitation | 最长 3+ hold |
|---|---:|---:|---:|---:|---:|
| 2026083001 | 250k | 2/20 = 10% | 18/20 = 90% | 8/20 = 40% / 0/20 = 0% | 0 |
| 2026083002 | **250k（最佳单点）** | **10/20 = 50%** | **10/20 = 50%** | 16/20 = 80% / 2/20 = 10% | 3 steps |
| 2026083003 | 400k | 1/20 = 5% | 19/20 = 95% | 9/20 = 45% / 1/20 = 5% | 1 |
| 三 seed best 的均值 | — | **21.67%** | **78.33%** | — | — |

因此，“历史最佳测试”有两个口径：若指一整条历史实验链，是 MAPPO-9-v2 的 3×400k gate；若指单个最好 checkpoint，是 seed2@250k 的 50% capture / 50% collision。该历史 run 的 terminal checkpoint capture 为 seed1/2/3=`0%/5%/5%`，不能只看最佳点宣称持续性能。

### 与本次 Final Scratch 的配置差异

| 维度 | 历史 MAPPO-9-v2 | 本次 Final Scratch | 是否只是 bug 修复 |
|---|---|---|---|
| 任务/场景 | corrected **Pure-Capture**；`scene_cycle=[capture]`；capture 是任务终止；没有 pure-coverage、post-capture transition | Final **Full-Task**；`cycle=[mixed, coverage]`；capture 后继续 500-step recovery/coverage，另有纯 coverage episode，horizon 3000 | 否，任务合同变化 |
| 目标与奖励 | pure-capture direct/informed reward；旧合同明确禁止 coverage reward/PBRS 路径（resolved YAML 中的继承 coverage 字段在该场景不可达） | CR-MS ring reward、support capture/coverage blend、CE PBRS、pure/post CE；`reward_clock_offset=2,000,000`，CE speed weight `.0005` | 否，奖励/信用分配变化 |
| reset / recovery | capture-only reset；没有 Final 的 mixed/coverage 交替和真实 capture snapshot 暴露 | pool 初始 0、capacity 1000、capture snapshot ratio `.75`、非 capture map-random `.5`；本夜 100k 实际 pool size 仍为 0 | 否，训练分布变化 |
| transition / collision | 运行在旧 terminal/truncation/reward contract；collision 为 synchronized swept v1 | 固定 `terminal-priority-truncation-bootstrap-weighted-ce-v2`；四类 transition/return bug 已修复，collision 仍为 synchronized swept v1 | transition 部分是已知 bug 修复 |
| actor | categorical AW9、Legacy-VorAdj decision-feature backbone；256 hidden / 8 heads / 4 layers / self 9 / pursuing embed 8，正交 head gain `.01` | 同为 categorical AW9 和同一 backbone 形状/正交 head gain `.01`；canonical scratch 明确 dropout `.1` | 主要相同 |
| critic | centralized action-free V，但历史 resolved central schema 使用 `max_agents=12` padding | fresh centralized V，`max_agents=4`（当前 4v1 Final contract） | 否，schema 也有差异 |
| PPO/GAE/ValueNorm | rollout256；γ=.99、λ=.95、clip=.2、3 epochs/2 minibatches、actor 3e−5、critic 1e−4、entropy .01、value 1、grad .5、target-KL .02、ValueNorm β=.99999 | 完全相同 | 否，基本不是差异来源 |
| 预算与评估 | 3 seeds × 400k；每 25k；pure-capture deterministic/stochastic 20 | 1 seed × 100k bounded diagnostic；0/25/50/75/100k；Full-Task argmax/sample × mixed/coverage | 否，证据量与 metric 不同 |
| teacher / 初始化 | random actor、fresh V/Adam/ValueNorm；没有 BC warm-start | random actor、fresh V/Adam/ValueNorm；`teacher_dependency=0`，禁止 IQN/BC/dataset/teacher-Q/tensor 读取 | 相同的 scratch 初始化 |

### “最大差异是不是只有 pure-capture？”

**Pure-Capture 是最大的一项单一差异，但不是唯一差异。** 更准确地说，最大变化是完整的 **任务目标/终止方式/训练分布**：历史策略只需要在 capture 终止前学会围捕；Final 策略必须先捕获，再在 post phase 完成 CE/recovery，同时还要在没有 evader 的 pure-coverage 分布上完成 CE。这个变化同时带来 episode horizon、phase credit、coverage/support reward、reset/recovery pool 和可见 visitation 的改变。

所以历史 50% capture 不能直接与本次 Final 的 0% Full-Task capture/post 结果作同一 Gate 的数值比较：前者是在 capture-only、terminal-on-capture 的任务上测得，后者是在 mixed/coverage、capture 后继续任务的任务上测得。历史结果能证明“该 backbone + MAPPO 配方在旧 Pure-Capture 上曾产生过有限 scratch 信号”，不能证明 Final Full-Task 的 coverage/post-capture learnability。当前 Final Scratch 100k 的真实 pool size 为 0、mixed post action/CE 为 0，这正是本次 Full-Task 的 phase-visitation bottleneck 证据，而不是把算法写成失败。

## 交接与 gate 保持

机器可读交接：[MASTER_SUMMARY.json](../artifacts/2026-09-14_overnight/MASTER_SUMMARY.json)、[SESSION_HANDOFF.json](../artifacts/2026-09-14_overnight/SESSION_HANDOFF.json)、[stage_gate.json](../artifacts/2026-09-09_root_cause/stage_gate.json)。实现与协议：[overnight diagnostic contract](FORWARD_FINAL_OVERNIGHT_DIAGNOSTIC_20260914_ZH.md)。本夜结果只作为下一 Gate 的候选输入；formal PPO、formal Scratch、P2/P3、PPO 25k 均继续 HOLD。
