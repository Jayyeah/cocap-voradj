# Z05 casualty-tolerance / continuation diagnostic

## 结论与范围

本报告记录独立的 relaxed-survival-stop diagnostic，不替代原 A/B/C full-team service formal 结果。原正式结果仍保存在 `/home/yjq/rl/CoCap1/z05-repeated-arrival-formal-20261007-heldout20x3`；本轮结果保存在 `/home/yjq/rl/CoCap1/z05-repeated-arrival-casualty-robustness-20261006/{a_timeout_continue,b_survival_relaxed,c_survival_relaxed}`。

三组各 20 局，配对使用 initial-world seed `2026100701+i` 与 target-refresh RNG seed `2026800701+i`；C 使用独立 delay RNG seed `2026900701+i`。checkpoint 为 Z05 Stage3 `step_600000`，SHA256 `8ee5c162c32883984f72aa4be4b86e82338ae1e8d8c912011d181987a476d095`。

判定结论：

- 三组 **60/60** 完成 evaluator contract diagnostics；Wave4=0、global-horizon stop=0、zero-active stop=0、evaluator parameter updates=0；checkpoint SHA 前后相同。
- 原来 14 个 `active<12` 终止 episode 全部继续运行，并全部完成 Wave1–3 capture。11/14 在旧 evaluator 停止时还没有完成三波 capture。
- 这 14 局后来均因 Wave3 strict-CE recovery window 到期结束。它们的 casualty 使 active 数低于12，而 coverage criterion 仍要求12，因此这是“capture继续成功、原 coverage endpoint 不可满足”的直接分离。
- B/C 的 strict `persistent_service_complete` 仍分别为13/20、14/20；放宽 survival stop 没有改变其 full-team service 数。A′ 为13/20，较原 A 的9/20增加4局；增加来自 timeout 后接受后续任务，不应归因于 survival stop 放宽。

## 实现和核验

Stage3 的两个配置门槛保持12：`min_active_pursuers=12`、`coverage_ce_min_active_pursuers=12`。evaluation-only survival 函数明确使用 `active_count == 0` 作为 zero-only terminal；没有把配置门槛改成0。普通训练和原 evaluator 路径仍走原配置阈值。

relaxed evaluator 在 casualty 后忽略单个已失活 pursuer 自身的 terminated 标志，并在 Wave3 不因 pursuer collision event 本身停机。Zero active、环境中的 target loss / terminal failure、global horizon 和 final recovery window 仍按各自语义终止。A′ 另行 defer 中间 wave 的环境 post-capture timeout，由 evaluator 在700步到期时记 `recovery_window_expired=true`、`safe_complete=false`，并在同一到期边界刷新下一 wave；Wave3 不刷新 Wave4，仍最多等待700步 final recovery。

逐局额外记录 wave start、capture、arrival trigger、spawn、recovery success/timeout、wave/episode end 的 active count，以及每波 minimum、失活 step/ID/type、累计 casualty、first casualty 和 post-casualty captures。2个 survival helper unit tests passed；Python compile 与 `git diff --check` 通过。

总 rollout 使用3个独立进程池、每池2个单线程 worker。运行期间 Evidence Global/Local Stage1 training process 均保持运行并更新 status；没有修改或重启训练任务。

## Endpoint 对照

final recovery time 单位为秒，仅在 recovery success episode 统计；`n` 是观测成功数，未观测数列于括号。

| Regime | persistent service ↑ | all 3 captured ↑ | final recovery success ↑ | final recovery time mean / median / p90 ↓（n；未观测） | final safe complete ↑ | collision events ↓（episode） |
|---|---:|---:|---:|---|---:|---:|
| A′ timeout-continue | 13/20 | 20/20 | 13/20 | 144.3 / 96.0 / 269.2（13；7） | 13/20 | 5（4） |
| B Z-clear immediate | 13/20 | 20/20 | 13/20 | 117.4 / 106.0 / 198.3（13；7） | 13/20 | 5（5） |
| C Z-clear + delay | 14/20 | 20/20 | 14/20 | 141.1 / 104.25 / 253.2（14；6） | 14/20 | 6（5） |

`all_3_safe_complete` 对 B/C 仍只作诊断；Wave1/2 会被 arrival 主动中断。A′中间波可能 timeout 后带 coverage debt 接单，也不把该指标作为主排序数。

## Per-wave capture 与 recovery

Capture time 为 wave start 到最后一个 target capture，单位秒，格式 mean / median / p90（n）。

| Regime | Wave1 capture | Wave2 capture | Wave3 capture | Wave1/2 CE recovery | Wave3 CE recovery |
|---|---|---|---|---:|---:|
| A′ | 40.73 / 36.50 / 60.35（20） | 27.35 / 25.00 / 36.35（20） | 29.45 / 28.75 / 37.50（20） | 14/20；15/20 | 13/20 |
| B | 40.73 / 36.50 / 60.35（20） | 41.33 / 39.00 / 59.00（20） | 40.28 / 41.75 / 49.10（20） | arrival-interrupted | 13/20 |
| C | 40.73 / 36.50 / 60.35（20） | 39.53 / 36.75 / 62.55（20） | 39.03 / 41.25 / 53.40（20） | arrival-interrupted | 14/20 |

B/C Wave1/2 recovery interruption 符合 arrival policy，不能记作 policy recovery failure。三组每个 wave capture 均为20/20。

## Survival 与 casualty 后能力

“到达 active count”按 trajectory 中确实出现过的**精确 active 数**统计；因此一次2架 agent-agent collision 可从12直接降到10而没有到达11。

| Regime | Final / minimum active mean ↑ | Mean pursuers lost ↓ | casualty episodes ↓ | 曾到达11 / 10 / 9 / 0 active | casualty 后继续 | casualty 后 target captures / waves captured | W2 / W3 capture with <12 | casualty 后 all-3 capture |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| A′ | 11.70 / 11.70 | 0.30 | 4 | 3 / 2 / 0 / 0 | 4 | 21 / 8 | 3 / 3 | 4 |
| B | 11.65 / 11.65 | 0.35 | 5 | 3 / 2 / 0 / 0 | 5 | 27 / 10 | 4 / 4 | 5 |
| C | 11.65 / 11.65 | 0.35 | 5 | 4 / 1 / 1 / 0 | 5 | 25 / 10 | 4 / 4 | 5 |

14个 old active-stop 局全部在 relaxed run 中继续并完成三波 capture：A′ 4/4、B 5/5、C 5/5。新终止点均是 Wave3 recovery timeout，而不是 collision stop 或 zero-active。其 casualty episode 的 minimum active 数为：A′ 11、10、10、11；B 11、11、10、10、11；C 11、11、9、11、11。

Collision continuation 后只有两局出现了额外 collision event：A′ 的 seed `2026100706` 与 C 的 seed `2026100707`；B 没有第二个事件。三组共14个 casualty episode 中没有零存活，也没有 global horizon stop。

低于12架后仍能捕获的 episode 内 capture time 对照样本较小，且受波次到达和先前 survivor selection 影响：A′ Wave2/Wave3 分别有3/3局，B 为4/4局，C 为4/4局仍完成 capture。Wave2 低于12架时均值分别为25.3、45.3、54.5秒；12架组均值为27.7、40.3、35.8秒。结果显示 capture 能继续完成，但不能据此声称低编队规模下 latency 不变。

## 与原 formal 的 paired comparison

| Paired arms | 原 formal all-3 | relaxed all-3 | 原 / relaxed final recovery | 原 / relaxed persistent service |
|---|---:|---:|---:|---:|
| strict A / A′ | 12/20 → 20/20 | +8 局 | 9/20 → 13/20 | 9/20 → 13/20 |
| original B / relaxed B | 16/20 → 20/20 | +4 局 | 13/20 → 13/20 | 13/20 → 13/20 |
| original C / relaxed C | 16/20 → 20/20 | +4 局 | 14/20 → 14/20 | 14/20 → 14/20 |

14个原 `insufficient_active_pursuers` case 的配对结果：

| Case | 原停止 wave | 新 run minimum active | Lost | 新 run 完成结果 |
|---|---:|---:|---:|---|
| A-04 / A-05 / A-10 / A-16 | W1 / W1 / W3 / W1 | 11 / 10 / 10 / 11 | 1 / 2 / 2 / 1 | 四局均 all-3 capture，W3 recovery timeout |
| B-04 / B-05 / B-08 / B-12 / B-16 | W1 / W2 / W2 / W3 / W1 | 11 / 11 / 10 / 10 / 11 | 1 / 1 / 2 / 2 / 1 | 五局均 all-3 capture，W3 recovery timeout |
| C-04 / C-05 / C-06 / C-10 / C-16 | W1 / W1 / W2 / W3 / W1 | 11 / 11 / 9 / 11 / 11 | 1 / 1 / 3 / 1 / 1 | 五局均 all-3 capture，W3 recovery timeout |

旧A另有5局因中间 recovery timeout 结束。A′下五局均继续到 Wave3、完成 all-3 capture；A-13、A-15、A-17、A-18 后来完成 strict final recovery 和 persistent service，A-06 最后在 Wave3 timeout。因此 A′ 的 service 提升主要来自**timeout-continue policy**，不能算作 casualty-stop 对 strict persistent service 的提升。

解释旧reported success时需要分清 endpoint：

- 14个碰撞失败局里，11局在旧停止点尚未达到 all-3 capture；放开 survival stop 后这些任务完成了三波 capture。直接 all-3 capture 的提升为A组 casualty component 3/20（15pp）、B/C各4/20（20pp）。A′另外改变了中间timeout策略，五局都获得新任务机会，使 strict A 到 A′ 的 all-3 总数增加8局。
- 若“persistent success”指原 full-team `persistent_service_complete`，survival cutoff 本身压低的数字是 **0pp**：任何 collision/casualty 仍是 disqualifying failure；B/C 的正式 service 数完全不变。A′的9→13是另一项 arrival timeout policy 的 paired 改变。

## Failure taxonomy 与科学判断

互斥 episode end taxonomy：

| 最终结果 | A′ | B | C |
|---|---:|---:|---:|
| Full persistent service complete | 13 | 13 | 14 |
| Wave3 recovery timeout | 7 | 7 | 6 |
| Capture not achieved by Wave3 | 0 | 0 | 0 |
| Zero active pursuers | 0 | 0 | 0 |
| Global horizon | 0 | 0 | 0 |

Collision 不再自动等同于 episode failure：4/4、5/5、5/5 casualty episodes 都继续执行。14/14 casualty episodes 完成 all-3 capture，但最后 strict recovery 均未成功；少于12 active 后 canonical coverage success 资格已不可满足，所以不能把这些 recovery failures 单独解释为 policy 不会 recovery。

**Q1.** 旧 active-stop 使11个 episode 的三波 capture 结果被截断（原来14个失败中另外3个在 collision 前已完成 all-3 capture）。它没有压低按原 collision-free、full-team 定义计算的 B/C `persistent_service_complete`。A′ service 增加4局是继续到达任务后的策略结果。

**Q2.** 有证据表明 Z05 在11架和10架时仍能继续并捕获；C 中还有一局到9架后完成剩余 capture。最低active case不是稳定性保证，只是本配对 cohort 的观察结果。

**Q3.** 有明显 separation：14个 casualty episodes 仍完成三波 capture，但都未完成 strict final recovery。捕获能力和12机 coverage service 是不同能力结论。

**Q4.** 结果足以支持把 fault tolerance / degraded-team operation 作为后续单独研究方向；本轮60局只构成 continuation diagnostic，不建立成功率保证或 degraded-team service contract。

## Behavior reuse 与合同检查

Support→capture transition counts：A′ 93、B 120、C 118；coverage→support：4、1、2；capture→coverage：5、4、5；coverage→capture：2、1、2。多数 wave 的 capture/support participants 在后续 wave 继续参与相关行为。它们是 event-derived reuse/transition diagnostics，不解释为 emergent roles。

60/60 episodes 的 contract booleans 全部通过：3 waves、slot reuse、无 ghost/stale target、Z 更新与 reactivation、pursuer/obstacle continuity、dead agent不复活、timer reset、有限数值、无Wave4、参数更新0。全部 worker 的 policy-state hash 一致，checkpoint SHA 前后均为预期值。B/C `all_3_safe_complete` 仍只作诊断。

完整 summary、逐局 event、trajectory 与每组5个 representative GIF 均保存在上述独立输出目录。原 formal 目录未覆盖。

## 2026-10-08 artifact 核验与 recovery / collision 分解

原 strict 60 局与 relaxed 60 局的 summary、原始 events 字段及 trajectory endpoint 均一致；summary 中后加的 contract diagnostics / worker hash 保留。没有模型推理、env.step、重新 rollout 或训练。脚本：[audit_persistent_artifacts_20261008.py](../../tools/audit_persistent_artifacts_20261008.py)；完整 provenance / GIF SHA / 逐局源文件 hash：[verification.json](../../artifacts/2026-10-08_persistent_audit/verification.json)。

relaxed 的20个 Wave3 recovery timeout 分为14个 casualty 后 strict coverage 不可满足、6个满编 hold miss。按 regime-episode 计数，A′为4+3，B为5+2，C为5+1；不同 regime 使用配对 seed，不是20个独立世界。

6个满编 case 从既有700步 post-capture snapshot 重新计算 canonical Voronoi CE，阈值 RMS≤0.05、max≤0.10、连续 hold≥30。下表 best RMS/max 来自同一个联合阈值最优 snapshot；连续 geometry hold 忽略其它资格门槛，是实际有效 hold 的上界。六局均曾同时达到数值阈值，但没有连续30步，不能解释为从未到达 CE 几何区域。

| regime / episode | seed | best joint CE RMS / max ↓ | qualifying snapshots / 700 ↑ | maximum geometry hold / 30 ↑ |
|---|---:|---|---:|---:|
| PERSIST-A-TIMEOUT-CONTINUE / 01 | 2026100702 | 0.045000 / 0.088343 | 92 | 26 / 30 |
| PERSIST-A-TIMEOUT-CONTINUE / 02 | 2026100703 | 0.045444 / 0.087912 | 57 | 25 / 30 |
| PERSIST-A-TIMEOUT-CONTINUE / 06 | 2026100707 | 0.043303 / 0.077941 | 152 | 25 / 30 |
| PERSIST-B / 01 | 2026100702 | 0.042353 / 0.084964 | 96 | 19 / 30 |
| PERSIST-B / 07 | 2026100708 | 0.042015 / 0.085915 | 116 | 25 / 30 |
| PERSIST-C / 08 | 2026100709 | 0.047007 / 0.082338 | 81 | 26 / 30 |

Collision 按 trajectory event 去重计数；pre/post capture 互斥，capture-region 与 early-refresh 可重叠。capture-region 定义为事件涉及 pursuer 在前一 snapshot 距 active target≤8；early-refresh 为W2/W3 spawn 后首20 decision steps（10秒）。它们是本次明确口径的离线诊断，不改变 formal endpoint。

| arm | obstacle | boundary | agent-agent | evader_contact | capture-region | early refresh | post-capture recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
| strict PERSIST-A | 1 | 1 | 1 | 1 | 2 | 0 | 2 |
| strict PERSIST-B | 1 | 1 | 2 | 1 | 2 | 2 | 1 |
| strict PERSIST-C | 1 | 2 | 1 | 1 | 3 | 0 | 2 |
| relaxed PERSIST-A-TIMEOUT-CONTINUE | 1 | 2 | 1 | 1 | 2 | 0 | 3 |
| relaxed PERSIST-B | 1 | 1 | 2 | 1 | 2 | 2 | 1 |
| relaxed PERSIST-C | 1 | 2 | 1 | 2 | 4 | 0 | 2 |

Strict 共14 events；relaxed 共16 events。relaxed 的14个 casualty episodes 全部继续并完成 all-3 capture；11/10/9 active 的精确到达统计与后续 capture 数均与原报告一致。最低9 active 仅是C的一局观察，不构成稳定性保证。

6个 representative e00 GIF（strict A/B/C、relaxed A′/B/C）存在、非零、逐帧完整解码；gif/event 的 seed与trajectory清单一致，采样帧确实覆盖W1/W2/W3并完成三波capture。GIF采用稀疏采样，末帧可比trajectory终点早若干步；核验JSON同时保留两者step，避免把动画末帧当作严格endpoint。

Repeated capture capability 已较强；full-team persistent service 仍受 safety/collision 和 recovery hold stability 限制。casualty-induced recovery failure 是12-active合同不可满足，不能单独归因于恢复策略能力。本轮仍保持T0 / NO_EXTRA_TRAINING_YET，等待 Evidence 完整正式比较后再联合决策。
