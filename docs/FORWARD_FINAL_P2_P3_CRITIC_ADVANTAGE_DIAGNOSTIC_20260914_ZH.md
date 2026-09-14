# Forward Final P2/P3 Critic–Advantage Diagnostic（2026-09-14）

## 结论先行

- `P2 VALUE CALIBRATION: FAIL`：固定 BC policy 下，centralized V 在 early recovery 有明确的正偏（V 把负回报预测得过高），且其时间轮廓解释力明显弱于 late recovery。
- `P3 ADVANTAGE SEMANTICS: INCONCLUSIVE`：关键 capture/early-recovery transition 的 advantage 符号和 MC−V 大体一致，没有证据表明它系统性奖励拖慢 capture/recovery；但同状态 counterfactual action 的 empirical ranking 不存在，不能把整个 P3 宣布 PASS。
- early recovery 是相对 late recovery 的真实校准热点，但不是唯一的高难度早期段；pure coverage early 的误差和 target 方差同样很高。
- 不允许进入 Direct PPO 5k。下一项最小诊断是固定 policy 的 matched-state counterfactual action-ranking probe，不训练 Actor/Critic。

## 1. 运行边界、版本与冻结保证

仓库分支为 `experiment/small-step-ac-migration-20260828`，同步远端后 HEAD 为 `75e3361`。本诊断只读取当前仓库的 Forward Final runtime、冻结 BC categorical Actor 和已有 geometry fixed-MC critic checkpoint；没有 optimizer step、Actor update、reward 修改或新 critic architecture。

运行身份：

- 有效报告：[report.json](../artifacts/2026-09-14_p2_p3_critic_advantage_v2/report.json)；首轮事件抽取错误的目录 `artifacts/2026-09-14_p2_p3_critic_advantage` 不纳入结论。
- Actor parent SHA256：`7916a970226a0452a8f71a22235524f6ba4511aa4c9bda907a9c8368a60bf2cd`。
- fixed-MC critic SHA256：`f80d505e7a748a6b119b7ced1c33c0cef065792f45593b786da1263c50d88457`；运行时 `actor_bit_exact=true`，`optimizer_steps=0`。
- transition contract：`terminal-priority-truncation-bootstrap-weighted-ce-v2`；使用生产 GAE：`δ=r+γ(1-terminated)V_next−V`，并以 `episode_end` 阻断反向 GAE。
- γ=`.99`，λ=`.95`，PPO rollout window=`256`；ValueNorm 在 GAE 前还原为 raw V。
- held-out seed=`2027099201`，10 episodes；5 个 mixed capture/recovery，5 个 pure coverage；全部 safe/CE success，无 failure 样本。

phase 分桶按 capture event 划分：capture 前最后 10 个 pre 行为 `capture_near`，触发 capture 的动作行为 `capture_transition`，其后 20 个 action steps 为 `early_recovery`，其余为 `late_recovery`；pure coverage 前 20 行和其余行分别为 `pure_coverage_early/steady`。

## 2. 样本覆盖

| bucket | active rows | episodes |
|---|---:|---:|
| pre_capture | 1320 | 5 |
| capture_near | 200 | 5 |
| capture_transition | 20 | 5 |
| early_recovery | 400 | 5 |
| late_recovery | 2820 | 5 |
| pure_coverage_early | 400 | 5 |
| pure_coverage_steady | 2224 | 5 |

这些是 agent-active rows，不是独立 episode 数；同一 episode 内存在相关性。特别是 `capture_transition` 只有 20 行，不能单凭该桶估计总体分布。

## 3. P2：V、MC return 与 GAE/λ-return

下表的 bias 定义为 `V−target`；`EV` 是按桶计算的 explained variance。GAE target 是 PPO 实际 target，MC target 是同一冻结 BC rollout 的完整 episode return。

| bucket | MC target mean±std | V mean | MC bias / RMSE / MAE / EV | GAE RMSE / MAE / EV |
|---|---:|---:|---:|---:|
| pre_capture (1320) | 74.24±53.55 | 65.78 | −8.46 / 51.55 / 41.63 / .098 | 22.91 / 16.28 / .555 |
| capture_near (200) | 90.31±64.82 | 102.51 | +12.20 / 54.98 / 44.00 / .316 | 42.10 / 33.19 / .434 |
| capture_transition (20) | 94.82±67.97 | 104.33 | +9.51 / 53.37 / 41.59 / .403 | 51.50 / 40.40 / .434 |
| early_recovery (400) | −14.00±10.95 | −3.21 | **+10.79 / 14.99 / 11.05 / .097** | **10.33 / 7.84 / .161** |
| late_recovery (2820) | −1.75±2.40 | −2.88 | −1.14 / 2.37 / 1.78 / .245 | 1.49 / 1.00 / .564 |
| pure_coverage_early (400) | −11.18±11.37 | −4.82 | +6.36 / 12.93 / 7.09 / .021 | 8.44 / 4.94 / .115 |
| pure_coverage_steady (2224) | −1.52±1.92 | −3.23 | −1.72 / 2.64 / 2.13 / −.091 | 1.64 / 1.12 / .193 |

overall 7384 rows 的 MC RMSE/MAE/EV 为 `24.29/11.05/.658`，GAE 为 `12.66/5.33/.884`；这个 overall 被 pre/capture 的大尺度 capture reward 主导，不能替代 phase-conditioned 结论。

### 3.1 early recovery 是否是真热点

是，但需要限定比较对象：

- 对 late recovery，early recovery 的 MC RMSE `14.99 vs 2.37`（约 6.3×），target std `10.95 vs 2.40`（约 4.6×）；GAE RMSE `10.33 vs 1.49`（约 6.9×）。这是明确的 early-recovery 校准热点。
- 对 pure coverage early，early recovery 的 MC RMSE `14.99 vs 12.93`、target std `10.95 vs 11.37`，并没有显示出同等强度的唯一性。因此更准确的表述是“capture 后早期时间段与所有 early/reset 段共同困难”，不是“只有 capture recovery 出错”。
- early recovery target 本身具有稳定的负回报形状（MC mean `−14.00`，GAE mean `−11.01`），并非空 target 或全为正的明显构造错误；V mean `−3.21` 且 std `3.12`，相对 target std `10.95` 被明显压扁，说明主要问题是 V 未学出 early temporal profile/context-conditioned dip。
- 最新 P1 transition audit 已验证 capture 后和 pure MC target 逐 bit 不变；所以本轮不能把 early-recovery 偏差归因于已修复的 timeout/terminal/correction/inactive-reset bug。

### 3.2 原因区分

当前最可信的原因按优先级为：

1. **V 表征/校准欠拟合**：early recovery 的 prediction quantile 仍整体高于真实负 target；历史 context-V 未击败时间/phase baseline，且 early 状态的相似观测可能对应不同剩余回报。
2. **return variance 与 phase/time imbalance**：early recovery 和 pure coverage early 都有约 11 的 MC target std，而 late/steady 约 2；capture 前的大尺度 reward 还会主导 value loss。不能把高方差误读为单纯 V bug。
3. **有限样本及共享参数优化压力**：这里只覆盖 5 个 mixed episodes；既有 fixed-bank 梯度审计也观察到 pre/recovery 的共享 trunk 竞争。该证据支持“需要更多分层样本/更合适 loss 权重”的假设，但没有证明应当新增 critic architecture。

因此，early recovery 是真实异常热点；证据不支持“target 已坏”作为首因，也不足以在当前 5 个 mixed episodes 上把 value 欠拟合与方差/不平衡完全分离。

## 4. P3：实际 PPO advantage 的符号与排序

下表的参照是同一行的 `MC−V`。`positive-bad` 表示 `A>0` 但 `MC−V<0`；`negative-good` 表示 `A<0` 但 `MC−V>0`。

| bucket | raw A mean±std | raw positive | sign agreement | positive-bad | negative-good | Spearman(A, MC−V) |
|---|---:|---:|---:|---:|---:|---:|
| pre_capture (1320) | 6.57±21.95 | .614 | .819 | .088 | .093 | .761 |
| capture_near (200) | −3.87±41.92 | .520 | .905 | .095 | .000 | .986 |
| capture_transition (20) | −2.00±51.46 | .550 | .950 | .050 | .000 | .982 |
| early_recovery (400) | **−7.80±6.78** | **.063** | **.913** | **.025** | .063 | **.963** |
| late_recovery (2820) | .10±1.49 | .604 | .756 | .002 | .242 | .802 |
| pure_coverage_early (400) | −4.81±6.93 | .100 | .853 | .000 | .148 | .957 |
| pure_coverage_steady (2224) | .22±1.62 | .606 | .701 | .000 | .299 | .796 |
| overall (7384) | .49±12.65 | .547 | .769 | .021 | .211 | .802 |

结论：

- early recovery 的 advantage 主要是负的，且与 MC−V 的排序很强；`A>0` 却对应更差 MC 的比例只有 2.5%。因此没有证据表明 PPO advantage 会系统性鼓励拖慢 capture/recovery。相反，V 高估负回报会让多数 early-recovery action 得到负 advantage，这是 value calibration 问题的表现。
- capture transition 的方向检查通过但只有 20 行；capture-near 的 sign agreement 也较高，不能把大尺度 advantage 直接判作符号颠倒。
- late/steady 有较多 `negative-good`，反映 baseline/时间轮廓仍不完美；这阻止了“全 phase advantage 已正确”的 PASS，但它没有转化为明显的 `positive-bad` 系统偏置。
- per-256 rollout normalization 的 overall raw→normalized sign flip 为 `.241`，这是按窗口减均值/除标准差造成的跨窗口 sign 变化；窗口内排序理论上保持。early recovery 自身 flip 仅 `.045`。因此 normalization 没有显示出关键 transition 的额外排序破坏。

fast/slow episode 极端组也没有形成“拖慢就普遍更优”的证据：fast Q25 平均长度 `96.7`、normalized A `−.146`；slow Q75 平均长度 `276.7`、normalized A `+.041`。样本只有各 3 episodes 且混合了 capture/pure phase，最多说明需要更多 episode-level stratification，不能作为系统性结论。

## 5. Teacher-Q / action-ranking diagnostic

当前 C1 dataset 的 contract 为 `forward-final-aw9-4v1-swept-v1`，与本次 Forward Final contract 匹配，因此 teacher-Q 查询**合法**。146,612 rows 上：

- frozen BC/PPO actor 的高概率 action 与 teacher greedy action 不同的比例为 `10.82%`；
- `teacher-Q(best) − teacher-Q(BC action)` 的全体中位数为 `0`，p90 仅约 `2.08e−4`，大部分不一致接近 tie，不能解释成经常选到明显更差动作；
- 但是 C1 每个 state 只有一条 realized action/reward/trajectory，没有同状态 BC/PPO alternative action 的 matched rollout。因此 empirical short-horizon/MC action ranking **不足以支持**，不能声称 advantage 已通过 counterfactual action test。

这项结果支持“没有明显的 teacher-Q gross inversion”，但不支持“PPO advantage 完全正确”。

## 6. Gate 与下一项最小诊断

`P2 VALUE CALIBRATION: FAIL`

`P3 ADVANTAGE SEMANTICS: INCONCLUSIVE`

不允许进入 Direct PPO 5k；也不启动 25k+ 正式 PPO、Actor 训练、reward 修改、teacher-KL 或新 critic architecture。

下一项最小诊断：在当前 Forward Final contract 下，冻结同一 BC Actor，在代表性的 capture-near/capture-transition/early-recovery states 做 matched-state action permutation probe，收集 BC action、当前 PPO actor 高概率 action、teacher-Q rank 以及 1–5 step short-horizon/MC outcome。该 probe 只增加诊断数据，不更新任何模型；它同时补齐 P3 缺失的 empirical alternative ranking，并可检查 early-recovery 的 action-level signal 是否真的与 value bias 分离。

## 7. 可复现入口与限制

诊断脚本为 [audit_forward_final_p2_p3_critic_advantage_20260914.py](../tools/audit_forward_final_p2_p3_critic_advantage_20260914.py)，原始数组为 [rollout_diagnostics.npz](../artifacts/2026-09-14_p2_p3_critic_advantage_v2/rollout_diagnostics.npz)。本结果是 10 个 held-out episodes 的固定策略诊断；rows 之间相关，全部 safe 导致 failure calibration 缺失。任何后续 gate 都应先补充按 mixed/pure、capture/recovery 和 failure-tail 分层的 held-out episodes，再重新运行同一报告，不应把本报告的 overall mean 外推成任务级定律。
