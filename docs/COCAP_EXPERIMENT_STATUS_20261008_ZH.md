# CoCap 当前实验阶段总结（2026-10-08）

事实快照：`2026-10-08T23:18:42.735153+08:00`（Asia/Shanghai）。角色：MASTER / central integration。中央恢复起点 `3940d87e3a12f8c3d51e8c68aa720e2370f7c1c0`；本轮结论来自 checkpoint、运行配置、logs、summary、events、trajectory 和 process，未启动新训练、ablation 或 formal experiment。

## 1. 当前阶段摘要

**EXP-EVIDENCE-01：`TRAINING_COMPLETE / SELECTION_PENDING / FINAL_EVAL_PENDING`。** Local 与 Global 均真正训练完成，各3.4M steps；Global三阶段与Local前两阶段selection完成。Local Stage3仍缺700k完整磁盘report和selection登记，三臂修正后50局/场景held-out尚未完成。不能称Evidence实验COMPLETE。

**EXP-PERSIST-01：strict formal COMPLETE / relaxed continuation diagnostic COMPLETE。** strict A/B/C persistent service分别9/20、13/20、14/20；relaxed A′/B/C均20/20三波capture，但full-team service分别13/20、13/20、14/20。14个casualty后coverage资格不可满足与6个满编hold miss已从artifact核验。

**联合决策：N0 / NO_EXTRA_TRAINING_YET。** Persistent T0保持；Evidence未有完整正式比较，不能以筛选结果解锁representation、recovery或safety新训练。

## 2. Evidence三臂合同

| 项目 | Local-Binary | Z05 | Global-Oracle |
|---|---|---|---|
| self / friend evidence | 自己、邻居当前直接可见active target的二值位 | 既有Z传播：λ=η=0.5，hard-zero=0.10 | 所有active target state全局可见，active-target scalar |
| memory / recursive / decay | 均无；不恢复旧is_pursuing | 同步递归与时间decay保持原合同 | 无Z记忆；敌情token全局可见 |
| 训练 | S1 scratch；同表示selected warm-start到S2/S3 | 使用既有selected，未重训 | S1 scratch；同表示selected warm-start到S2/S3 |
| 信息解释 | decentralized local baseline | temporal / propagated evidence | 信息上界条件，非公平decentralized baseline |

三臂网络、reward/internal roles、NormSense V2、AW9、synchronized_swept_v1、physical_only friend排序和token容量一致：self/friend维9/7，friend/enemy/obstacle容量8/8/5。Global reward的global_enemy_flag仍为false，不把信息改变混成reward改变。

## 3. Training / selection核验

Local/Global训练预算S1 4p1e1obs 2M、S2 8p2e2obs 700k、S3 12p3e3obs 700k；seed依次2026091901、2026091902、2026091903。六份final模型payload实际step、finite参数、每1000步连续metrics、final_checkpoint日志和runtime配置通过。未发现异常中断/resume或科学变量drift；只存在注册的路径/run-name/warm-start差异。

| representation | stage | 实际训练终点 | candidate checkpoints | screening完整数（20局/scene） | selected step |
|---|---|---:|---|---|---:|
| local_binary | stage1 | 2,000,000 | 100k间隔至2M | 20/20 | 1500000 |
| local_binary | stage2 | 700,000 | 100k间隔至700k | 7/7 | 300000 |
| local_binary | stage3 | 700,000 | 100k间隔至700k | 6/7 | PENDING |
| global_oracle | stage1 | 2,000,000 | 100k间隔至2M | 20/20 | 1700000 |
| global_oracle | stage2 | 700,000 | 100k间隔至700k | 7/7 | 700000 |
| global_oracle | stage3 | 700,000 | 100k间隔至700k | 7/7 | 300000 |

已登记选择全部为balanced_floor、fallback=false。主分数为min(min(pure normal capture,mixed capture),strict coverage)，按mixed safe-complete、post-capture CE、harmonic、collision、CE RMS、area CV、capture time、step顺序tie-break。没有使用final held-out选择checkpoint。

| representation / stage | selected step | selected checkpoint SHA256 |
|---|---:|---|
| local_binary / stage1 | 1500000 | `18de0e289efc0fb07bd4b890c505c21da92f4092f6d2be155faa4265ffa938c6` |
| local_binary / stage2 | 300000 | `f6e0cf2172fc66e43a18bd1a77d6559a6c3ce6797c296f533703c3717e1ead5c` |
| local_binary / stage3 | PENDING | 未登记，不能推测 |
| global_oracle / stage1 | 1700000 | `f16b90fedd377e496f6a02e43e7eb6068d48446f6dc84af67845184107d74640` |
| global_oracle / stage2 | 700000 | `97e5561b0938b71c4698be3d7dff701fd332a3f2caa5e5da5cfd030e71309ca4` |
| global_oracle / stage3 | 300000 | `b4e6566c4ac7860cdd802320f28c103b36b083e2651e1e335527c2782e16fbe0` |
| z05 / stage1 | 1300000 | `d35b7984d8ca03fa7b536811794578ff402e9b8ad48589e3318f2cb1077d702b` |
| z05 / stage2 | 100000 | `f137f4fcd302c5ff8b7282609344a02e4c1702c84d15cefff9deda7c23d56f8c` |
| z05 / stage3 | 600000 | `8ee5c162c32883984f72aa4be4b86e82338ae1e8d8c912011d181987a476d095` |

Z05 Stage3 hash与合同给定`8ee5c162...476d095`逐字相同。Local当前S3 600k只是已完成候选中的最佳balanced_floor=0.85，尚非最终selected。Global最终S3 selected=300k；其筛选20局Coverage/Capture/Mixed均20/20且collision0，仅是selection cohort，不是held-out性能保证。

## 4. 统一final held-out主表与缺口

正式合同为每表示每场景50局，共450局；seed base2026100601，Coverage/Capture/Mixed分别再加0/100000/200000。审计已保存150个**expected** seeds；actual三臂manifest与initial-world fingerprint配对尚未验证。三份`final_heldout_50_corrected/{local_binary,z05,global_oracle}/report.json`均缺失。下表PENDING表示没有正式结果，不是0成功或n=0。

| Coverage | strict CE success ↑ | CE RMS ↓ | CE max ↓ | time-to-CE ↓ | collision ↓ | censored ↓ |
|---|---|---|---|---|---|---|
| Local | PENDING | — | — | — | — | — |
| Z05 | PENDING | — | — | — | — | — |
| Global | PENDING | — | — | — | — | — |

| Capture | capture success ↑ | normal capture ↑ | capture time ↓ | collision ↓ | stationary fallback（诊断） | censored ↓ |
|---|---|---|---|---|---|---|
| Local | PENDING | — | — | — | — | — |
| Z05 | PENDING | — | — | — | — | — |
| Global | PENDING | — | — | — | — | — |

| Mixed | safe-complete ↑ | capture success ↑ | normal capture ↑ | post-capture CE / recovery ↑ | recovery time ↓ | mission time ↓ | collision ↓ | censored ↓ |
|---|---|---|---|---|---|---|---|---|
| Local | PENDING | — | — | — | — | — | — | — |
| Z05 | PENDING | — | — | — | — | — | — | — |
| Global | PENDING | — | — | — | — | — | — | — |

指标定义：strict CE为canonical Voronoi centroid-energy RMS≤0.05、max≤0.10且30-step hold；capture success为该场景所需targets全部捕获；normal capture排除stationary fallback；safe-complete为capture后strict CE恢复且无disqualifying failure；collision为出现碰撞的episode比例；censored为未达到该场景成功endpoint，需再分collision/terminal failure/timeout。Coverage CE RMS/max为所有episode终态几何误差，不是成功时间。

所有时间单位秒，分别报告**mean / median / p90 / n / censoring**。Coverage time-to-CE只纳入CE成功；Capture time只纳入capture成功；Mixed recovery只纳入post-capture CE成功；mission time只纳入safe-complete。失败或删失局不混入成功时间均值。

既有Z05最新native独立评测（20局/scene，commit1a9da05）为Coverage20/20、Capture20/20、Mixed safe-complete15/20，Mixed capture19/20、collision1/20；它使用不同evaluator且只有20局，不能填入上面50局三臂表。旧registry的Mixed17/20是另一套20局，不替换最新独立结果。

## 5. Evidence diagnostics与收尾工程状态

| 表示 | 最终必须核对的诊断 | 当前正式状态 |
|---|---|---|
| Local | direct-visible、one-hop-informed count、occupancy / transitions | PENDING |
| Z05 | direct、z>0、direct/self_decay/neighbor source、lineage hops、age、all-Z-zero / release | PENDING |
| Global | target availability、enemy-token occupancy、shortfall、capacity overflow、entity truncation | PENDING |

Global统计修正已经在commit7e7371b进入文件：token occupancy使用可见active-target token / 累积capacity slots；逐agent-observation统计shortfall与overflow。当前旧supervisor PID14306在修正前已import evaluator，所以其未来final输出不能证明已应用修正；最终必须用fresh current evaluator并检查implementation marker。

本轮发现并修复：①正在运行的筛选候选可重复入队，随后删除已完成report；②Coverage time-to-CE读取了不存在的episode_record字段。现从first strict-CE-success transition记录时间，成功过滤保留。旧screening未重写，balanced_floor不使用该时间字段，因此不会因此改变既有selection。当前live进程没有热替换；磁盘report优先于stale heartbeat的complete列表。

新增`--check-final-readiness`和`--final-only`：仅读取完整训练/selection/hash，不走trainer/resource lease/resume路径；当前readiness按预期拒绝Local S3 pending。待现有screening结束并登记selected，再执行fresh final-only完成原合同；旧输出留作审计，不重训、不改seed、不加ablation。

## 6. Evidence科学结论

| 问题 | 当前判断 | 证据等级 |
|---|---|---|
| Z相对Local存在稳定优势？ | 正式50局比较未完成，不能回答 | 尚不能下结论 |
| 优势在Coverage/Capture/Mixed/recovery/collision/mission哪一环？ | 只能看到Local各候选coverage/recovery不稳定；筛选不支持正式环节归因 | 暂时趋势 |
| Global形成明显经验上界？ | Global selected筛选很强；信息上界定义成立，经验排序未确立 | 暂时趋势 |
| 若Z>Global如何解释？ | 当前无此正式结果；需检查optimization、distribution shift、regularization、token截断；不能说Z信息更多 | 尚不能下结论 |
| propagation足够成为论文主机制因果贡献？ | propagation诊断证明机制被使用，不证明性能收益由其因果产生；本轮正式primary comparison仍缺失 | 尚不能下结论 |
| 后续多训练seed？ | 稳定性/机制主要贡献仍需要独立training-seed重复，不能把50个eval seeds当训练seed重复 | 后续验证需要；本轮不启动 |

## 7. Persistent strict与relaxed结果

Strict formal：20局×A/B/C，三个到达regime配对initial-world seed2026100701+i与refresh RNG2026800701+i，C delay RNG2026900701+i。每wave3targets、共3waves、12p/3obs/120×120、全局3000步、post-capture recovery window700步。Regime paired RNG不保证validity rejection后W2/W3坐标相同。

Relaxed为evaluation-only continuation diagnostic：survival耗尽仅active_count==0终止，coverage_ce_min_active_pursuers与min_active_pursuers仍12；A′额外timeout-continue。其service提升不能全归因于survival-stop。

| arm | all_3_captured ↑ | persistent service / final recovery ↑ | final recovery seconds ↓ mean / median / p90 | n；未观测/失败 | collision events ↓（episode） |
|---|---:|---:|---|---|---|
| strict PERSIST-A | 12/20 | 9/20 | 166.00 / 153.50 / 280.80 | 9；11 | 4（4） |
| strict PERSIST-B | 16/20 | 13/20 | 117.38 / 106.00 / 198.30 | 13；7 | 5（5） |
| strict PERSIST-C | 16/20 | 14/20 | 141.11 / 104.25 / 253.20 | 14；6 | 5（5） |
| relaxed PERSIST-A-TIMEOUT-CONTINUE | 20/20 | 13/20 | 144.27 / 96.00 / 269.20 | 13；7 | 5（4） |
| relaxed PERSIST-B | 20/20 | 13/20 | 117.38 / 106.00 / 198.30 | 13；7 | 5（5） |
| relaxed PERSIST-C | 20/20 | 14/20 | 141.11 / 104.25 / 253.20 | 14；6 | 6（5） |

Strict all_3_safe_complete：A9/20；B/C W1/W2会被next arrival中断，0/20只作diagnostic，不作为service主指标。strict A/B/C service分别45%/65%/70%，20局配对样本不支持regime强排序或可靠性保证。Relaxed三组每wave capture均20/20，60/60 all-3；B/C service不变，A9→A′13主要来自中间timeout后继续接任务。

## 8. Casualty continuation、recovery与collision分解

| relaxed arm | casualty episodes | casualty后继续/all-3 ↑ | casualty后target captures / waves ↑ | W2/W3 capture with<12 ↑ | 曾到达exact 11/10/9 active |
|---|---:|---|---|---|---|
| PERSIST-A-TIMEOUT-CONTINUE | 4 | 4/4；4/4 | 21 / 8 | 3 / 3 | 3 / 2 / 0 |
| PERSIST-B | 5 | 5/5；5/5 | 27 / 10 | 4 / 4 | 3 / 2 / 0 |
| PERSIST-C | 5 | 5/5；5/5 | 25 / 10 | 4 / 4 | 4 / 1 / 1 |

14个casualty episode全部继续并完成三波capture；其中11局在旧strict停止点还未完成三波。11、10 active后继续capture有直接观察；9 active仅C的一局，不能宣称degraded-team latency不变或服务可靠。三组zero-active、Wave4、global horizon stop均0。

relaxed最终20个recovery failure：A′4 casualty+3满编、B5+2、C5+1，总计14+6。14个casualty后的12-active coverage资格不可满足，不能独立解释为policy不会recovery。六个满编miss均完成三波capture，随后700步recovery超时；静态重算显示每局都曾同时满足RMS/max阈值，但连续geometry hold最大仅19–26步，小于30。下表hold忽略其它资格门槛，作为实际有效hold上界。

| 满编case | seed | 同snapshot最佳CE RMS / max ↓ | 几何达标步数 ↑ | 最长hold / required ↑ |
|---|---:|---|---:|---|
| PERSIST-A-TIMEOUT-CONTINUE e01 | 2026100702 | 0.045000 / 0.088343 | 92/700 | 26 / 30 |
| PERSIST-A-TIMEOUT-CONTINUE e02 | 2026100703 | 0.045444 / 0.087912 | 57/700 | 25 / 30 |
| PERSIST-A-TIMEOUT-CONTINUE e06 | 2026100707 | 0.043303 / 0.077941 | 152/700 | 25 / 30 |
| PERSIST-B e01 | 2026100702 | 0.042353 / 0.084964 | 96/700 | 19 / 30 |
| PERSIST-B e07 | 2026100708 | 0.042015 / 0.085915 | 116/700 | 25 / 30 |
| PERSIST-C e08 | 2026100709 | 0.047007 / 0.082338 | 81/700 | 26 / 30 |

| arm | obstacle ↓ | boundary ↓ | agent-agent ↓ | evader_contact ↓ | capture-region ↓ | early-refresh ↓ | post-capture recovery ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| strict PERSIST-A | 1 | 1 | 1 | 1 | 2 | 0 | 2 |
| strict PERSIST-B | 1 | 1 | 2 | 1 | 2 | 2 | 1 |
| strict PERSIST-C | 1 | 2 | 1 | 1 | 3 | 0 | 2 |
| relaxed PERSIST-A-TIMEOUT-CONTINUE | 1 | 2 | 1 | 1 | 2 | 0 | 3 |
| relaxed PERSIST-B | 1 | 1 | 2 | 1 | 2 | 2 | 1 |
| relaxed PERSIST-C | 1 | 2 | 1 | 2 | 4 | 0 | 2 |

以上是events，不是互斥episode分类；capture-region为涉及pursuer在前一trajectory snapshot距active target≤8，early-refresh为W2/W3刷新后首20步（10秒）。阶段标签可重叠；pre/post capture在核验JSON中互斥记录。它们为新增离线诊断口径，未修改formal标准。

## 9. 代表GIF核验

6个e00 representative均存在、非零、逐帧解码通过；event seed与trajectory对应，采样帧覆盖完整三波capture过程。均seed2026100701；不是把20局另跑一遍。稀疏GIF末帧可稍早于trajectory endpoint，完整step和hash见核验JSON。

| arm | GIF完整路径 | bytes / frames |
|---|---|---|
| strict PERSIST-A | `/home/yjq/rl/CoCap1/z05-repeated-arrival-formal-20261007-heldout20x3/gifs/persist-a_e00_representative.gif` | 2603938 / 75 |
| strict PERSIST-B | `/home/yjq/rl/CoCap1/z05-repeated-arrival-formal-20261007-heldout20x3/gifs/persist-b_e00_representative.gif` | 1003681 / 29 |
| strict PERSIST-C | `/home/yjq/rl/CoCap1/z05-repeated-arrival-formal-20261007-heldout20x3/gifs/persist-c_e00_representative.gif` | 1447434 / 42 |
| relaxed PERSIST-A-TIMEOUT-CONTINUE | `/home/yjq/rl/CoCap1/z05-repeated-arrival-casualty-robustness-20261006/a_timeout_continue/gifs/persist-a-timeout-continue_e00_representative.gif` | 2603938 / 75 |
| relaxed PERSIST-B | `/home/yjq/rl/CoCap1/z05-repeated-arrival-casualty-robustness-20261006/b_survival_relaxed/gifs/persist-b_e00_representative.gif` | 1003681 / 29 |
| relaxed PERSIST-C | `/home/yjq/rl/CoCap1/z05-repeated-arrival-casualty-robustness-20261006/c_survival_relaxed/gifs/persist-c_e00_representative.gif` | 1447434 / 42 |

## 10. 联合科学判断与claim boundary

| 能力/问题 | 当前结论 | 边界 |
|---|---|---|
| coverage primitive | 原生Z05与历史positive controls支持可学、可完成 | 不等于任意规模/布局可靠；30-step hold存在少量不稳定 |
| capture primitive | 原生独立20/20、relaxed repeated三组均20/20 | 全局可靠性、多训练seed稳定性未证明 |
| repeated capture | 强证据支持零额外训练复用三波capture，包括casualty后继续 | relaxed continuation不是原collision-free full-team service |
| evidence-conditioned coordination | Z更新、reactivation、release与跨wave行为切换诊断通过 | Local-vs-Z正式机制因果收益尚无完整比较 |
| behavior reuse | event-derived capture/support participant跨wave复用、support→capture转换存在 | 不称emergent roles，也不是角色因果证明 |
| full-team persistent service | 观察到45%–70%strict service能力 | 尚未证明稳定/可靠 |
| safety/collision | relaxed失败中的14/20与casualty合同资格关联 | 配对cohort诊断，不能推断普遍因果占比；需独立门禁 |
| recovery stability | 6个满编miss达到CE数值区域却未hold30步 | 支持候选研究方向，不自动解锁训练 |
| casualty后coverage | 现合同要求12active，因此剩余11/10/9后不可能成功 | 未建立degraded-team coverage service合同 |
| scalability / friend truncation | native最新独立20局已出现28次friend截断/194482agent-observations，涉及4/60局；friend cap=8 | scalable capture不等于safe service；不改变token容量合同 |

**已核验证明**：本cohort120局合同诊断通过、参数更新0、三波slot/clock/物理状态连续；relaxed60/60三波capture；casualty continuation与12-active coverage资格可分离。**强证据支持**：repeated capture和behavior reuse可复用；安全与hold限制full-team service。**暂时趋势**：Evidence候选筛选差异。**尚不能下结论**：Z稳定优于Local、Global经验上界排序、传播的论文主机制因果收益、通用fault-tolerant service可靠性。

## 11. 下一步decision gate

**N0 / NO_EXTRA_TRAINING_YET**，保持Persistent T0。N1 recovery-focused与N2 safety-focused均有候选证据，但Evidence正式比较未完成，不据不完整representation结果选择新训练方向；N3 representation主要瓶颈尚未证明；不选择额外N4实验。下一步仅完成既定Local最后筛选、selection、fresh corrected三臂50局held-out、actual seed/fingerprint/hash/诊断核验与报告。完成后再次联合科学判定；多training-seed验证仍是论文机制稳定性需求，本轮不启动。

## 12. Provenance、运行状态与同步

- Evidence：`evidence/local-global-20261006` @ `6b7aab28546e1a30c60f84d12de3bde70930f670`，代码、completion snapshot、tests与audit文档已提交。
- Persistent：`evaluation/z05-repeated-arrival-demo-20261006` @ `751ee2fdf8e0134504e9b4f033904cc610dd6891`；其父`906e2bc`为原来未同步的casualty实现，formal结果commit`80cc077`。
- Z05独立reference：`evaluation/iqn-z05-independent-20260923` @ `1a9da054ecb5652726005d00d08ab05a2188dd8e`。
- Central：`ops/ac-master-dag-20260921`；快照基点`3940d87e3a12f8c3d51e8c68aa720e2370f7c1c0`，最终交付HEAD由git与handoff给出，避免自引用hash。
- 所有checkpoint路径、训练配置SHA、selected SHA、summary/event/trajectory/GIF SHA、remote refs及42条worktree登记见 [integration verification](../artifacts/2026-10-08_experiment_integration/verification.json)。
- 根盘可用140.47GiB。Evidence训练进程已结束；当前supervisor14306与Local700k CPU筛选1001961/1001974/1001975仍在运行。snapshot progress=12/60，只作为进度，不能据ETA宣称完成。
- 另有TERL-MAPPO训练PID992932 / supervisor992900及其独立25k evaluator仍运行；两卡还被其它用户任务占用。没有停止/修改这些任务，Evidence lease已结束但GPU并非空闲。
- Validation：Evidence31 passed、Persistent27 passed；120局artifact核验PASS，6份final checkpoint payload/hash/finite审计PASS，compile与diff-check PASS。
- 本文和中央DAG/state经commit/push后再次fetch与ls-remote验证；同步回执见integration目录的remote verification，实际最终HEAD另由handoff给出。live coordinator_state.json仍是未提交动态runtime文件，既有Z05 raw artifact目录也未提交；本轮只同步审核后的代码/文档/状态/snapshot，不把未完成动态runtime包装为完成证据。

主要源文档：
- [Evidence audit](https://github.com/Jayyeah/cocap-voradj/blob/6b7aab28546e1a30c60f84d12de3bde70930f670/docs/evaluations/IQN_EVIDENCE_COMPLETION_AUDIT_20261008_ZH.md)
- [Persistent strict report](https://github.com/Jayyeah/cocap-voradj/blob/751ee2fdf8e0134504e9b4f033904cc610dd6891/docs/evaluations/Z05_REPEATED_ARRIVAL_DEMO_20261006_ZH.md)
- [Casualty / hold / collision report](https://github.com/Jayyeah/cocap-voradj/blob/751ee2fdf8e0134504e9b4f033904cc610dd6891/docs/evaluations/Z05_CASUALTY_TOLERANCE_ROBUSTNESS_20261006_ZH.md)
- [Z05最新独立结果](https://github.com/Jayyeah/cocap-voradj/blob/1a9da054ecb5652726005d00d08ab05a2188dd8e/docs/evaluations/report.md)
