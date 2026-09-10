# Forward Final corrected PPO：D smoke / 更新健康 Gate

起点 `f9532d2dcc49aad36d729bfc8e0eb393e4ed59c8`，本次pull确认远端无更新。C3三种冻结策略formal100已全部完成并PASS；详见 [bridge台账](FORWARD_FINAL_MAPPO_BRIDGE_LEDGER_20260908_ZH.md)。本页只记录新增D，不重新审计旧Pure-Capture。

**本阶段最新入口：[PPO root-cause台账](FORWARD_FINAL_PPO_ROOT_CAUSE_20260909_ZH.md)。P0/P1先行；已证实critic context aliasing，25k继续HOLD。旧第12/13节下一实验建议被本阶段Gate覆盖。**

**历史至22046dd（2026-09-09）：C3 bridge PASS保留；D扩预算HOLD。固定BC完整回报critic诊断已完成，phase校准门槛仍未全过；未启动25k PPO。BC与既有PPO-512各40rollout/20GIF已全部完成，详见文末及GIF台账。此前运行状态/NEXT WAKE-UP为历史快照。**

## 1. D共同parent、Final训练语义与预算

- 同一个C3 BC `actor_epoch_030.pt`，SHA `7916a970226a0452a8f71a22235524f6ba4511aa4c9bda907a9c8368a60bf2cd`。
- Actor保持local LegacyVorAdj输入，直接输出integer AW9；新增central action-free V，hidden256/8heads/4layers，4 agent、8 enemy、5 obstacle容量。全局state仅给训练critic。
- 恢复Final原生 `CoCapTrainer._reset_task`、`_evader_actions`、`_set_coverage_ce_control_weights`。不调用旧AC `configure_environment/make_components`；只复用通用rollout容器与P0 MAPPOTrainer。
- 与Final相同按回合交替mixed/coverage，不等于两任务训练步数各半。capture snapshot pool容量1000、capture reset概率.75、剩余部分map-random概率.5；pool从空开始，按自身在线完整mixed episode的capture snapshot填入。复制机制，未伪造历史2m时的pool内容。
- 奖励时钟=`teacher age2m + online interaction steps`，实际CE speed weight **.0005**。两分支相同。冻结eval仍沿用C3 reset reward默认0；reward不反馈进冻结actor，success/观测/物理不变。D不是沿用C1保存的0系数reward做off-policy更新。
- native post-capture mission deadline保留true terminal；总horizon的`too long episode`是truncation，bootstrap使用reset前物理世界；GAE在任一种episode结束断开。
- PPO两边共同：lr actor3e-5、critic1e-4，rollout256，epochs3，minibatches2，gamma.99/lambda.95，clip.2，entropy.01，targetKL.02，ValueNorm开启。PPO阶段允许整个Actor（含backbone）更新；warm-up只调用critic optimizer，整个Actor逐bit检查不变。

计划的正式比较是**同25k总交互预算**：D1全部PPO，D2前4096步critic-only，之后20904步PPO。warm-up计入预算，不能称等Actor更新次数或“只改变初始critic却不改变训练过程”。固定warm-up是明确处理变量，不由EV阈值动态改变预算。first25k完成后的同evaluator Gate通过，才考虑50k。

## 2. 工程核验与512步 smoke

新增 `tools/train_forward_final_ppo_20260909.py`，首先检查C3 PASS、BC SHA、runtime；每个rollout更新前验证old/new log-prob。保存完整trainer/ValueNorm/optimizers、环境与recovery pool状态、rollout及Python/NumPy/Torch/CUDA RNG，同时导出可冻结加载的Actor。每25k保存，不覆盖已有文件。当前CLI首Gate只允许≤25k；没有自动resume到50k。

21项相关CPU测试通过（D3项 + bridge8项 + P010项）：真实capture snapshot reset消费、末期CE权重、timeout next-value来自reset前世界、warm-up Actor不变/V变化、完整checkpoint还原、零更新概率、phase/target等合同。GPU smoke各512步；D2前256步warm-up（仅缩短的工程smoke，不能评价正式4096步warm-up收益）。

`smoke_parity.json`核验两个launch只有branch标签、GPU标签、warmup_steps不同；Actor tensor SHA、critic初值SHA、初态fingerprint、所有runtime完全相同。两个GPU任务实际都经历mixed和capture-snapshot coverage reset。

| 512步最后一次update | Direct | 256warm-up→PPO |
|---|---:|---:|
| PPO交互步数 | 512 | 256 |
| zero-update max log-prob error | 3.10e-5 | 2.50e-5 |
| minibatch KL max | .0551 | .1394 |
| clip fraction | .1768 | .2197 |
| critic EV | .183 | .108 |
| KL early-stop | 触发 | 触发 |

**不能把“零更新合同通过”写成“优化已健康”。** 原MAPPOTrainer的targetKL是更新循环的early-stop信号，不是硬trust-region约束；它记录minibatch的更新前likelihood差异并在optimizer动作后触发停止，最终整批policy KL未由这些数字界定。上述值足以要求先看闭环任务保持，但本身不证明erosion。

warm-up第256步Actor逐bit不变，critic EV约.127；这仅证明critic-only阶段执行有效，**不证明cold-start已排除**。本轮没有修改PPO算法来事后消除KL，没有teacher-KL或新网络，也没有重新蒸馏。

## 3. 下一 Gate：512步 checkpoint 冻结task eval20

`tools/evaluate_forward_final_ppo_20260909.py`直接复用C3同一`run_episode`，全模式均sample：BC parent、Direct512、Warm-up512。新seed base `2026098101`，每个20mixed+20pure coverage；三者同BC parent、同collision/phase/sensing/evaluator，比较初态fingerprint。训练seed2026097101与此分离。

GPU0先BC parent再Direct512；GPU1 Warm-up512。只有冻结eval，无额外梯度更新。任务完成后用：

```
PYTHONPATH=src:. python3 tools/summarize_forward_final_d_screen_20260909.py --root artifacts/2026-09-09_forward_final_d/smoke_task_eval20
```

预声明更新健康Gate：每场景≥20，safe/CE/capture≥.9，较parent下降≤10百分点，collision增加≤10百分点，成功条件mission P90≤parent1.25倍。whole-episode paired统计同步报告。任一不满足为`HOLD_PPO_SCALE`，先诊断更新漂移；全部满足才`ELIGIBLE_FOR_25K_NOT_AUTOSTARTED`。这是小样本工程screen，不是统计非劣效证明或Direct/Warm-up胜负判定。

所有结果依赖均需结束后读取，不根据局部trajectory作胜负判断。不存在自动25k/50k supervisor。下次醒来只读三个report及汇总Gate，再决定是否启动既定25k对照；不因GPU空闲另开任务。

## 4. 前次会话冻结结果与历史交接（其中Direct现已完成）

截至本次会话结束，BC parent与Warm-up512均完成20mixed+20pure；Direct512尚在运行。BC mixed safe/capture19/20、collision1/20；Warm-up512均20/20、0collision。两者pure CE均20/20、0collision，但pure完成时间mean由79.9升至102.6秒，P90由121.25升至196.05秒（+61.7%）。这触发既定1.25倍P90效率门槛，**Warm-up512为HOLD_PPO_SCALE**，不启动25k。该小样本结果不能宣布正式4096步warm-up失败，更不能比较尚未完成的Direct；它证明只看binary success会漏掉更新后settling变慢。

快照：2026-09-09T01:58:59.401134+08:00。GPU0 Direct512 worker PID264405（flock PID263398），32/40回合，当前step131；ETA 2026-09-09T01:59:49.496155+08:00。GPU1评估完成并空闲。checkpoint均为已有512步smoke Actor，没有25k checkpoint。完整PID/throughput输入记录见 `artifacts/2026-09-09_forward_final_d/SESSION_HANDOFF_20260909_0200.json`。

**NEXT WAKE-UP：2026-09-09 02:05 CST，或Direct512报告完成时。** 先汇总全部配对结果，再诊断一次/两次PPO update为何改变coverage settling。当前唯一待完成GPU任务是GPU0 Direct512；GPU1不再补实验。下一正式GPU1任务需由完整健康Gate收敛，不预开25k/teacher-KL/continuous。

## 5. Direct512完成：旧Gate PASS不等于效率保持

起点再次pull为`64e11b140288bd7a71f5b24eb3f98ce376d7da1b`。`smoke_task_eval20/comparison.json`现已完整汇总。Direct mixed/pure均20/20 safe、0collision；mixed mission mean/P50/P90=147.35/137.25/192.95秒，capture均值47.725秒，capture→CE均值99.625秒。BC分别为118.5/106/159秒、capture47.842秒、capture→CE70.658秒（BC mixed19/20 safe）。

在共同safe的19对上，Direct mixed mean慢31.579秒，探索性paired bootstrap95%区间[7.37,59.24]秒；13对更慢、6对更快。主要差距在capture后coverage恢复，不能归因为capture追不上。pure mean差+6.825秒，区间[−10.28,24.15]。Warm-up的pure mean差+22.7秒，但该小样本区间[−1.65,48.03]跨零；此前P90+61.7%的工程HOLD保留，不能升级成总体显著退化或正式4096warm-up无效。

**不改写原Gate：** Direct的mixed P90比值约1.213，因此按此前1.25阈值仍为`SMOKE_HEALTH_PASS`。它暴露了只设P90门槛会漏掉mean/P50变化。总体仍HOLD（Warm-up不通过）；研究判断也不支持因Direct粗PASS就扩25k。新的mean/P50阈值只前瞻用于下述新诊断，不回填历史判定。

## 6. 唯一新增路线：Direct Actor LR 3e-5→3e-6

两条均从同一C3 BC parent、同初始central V、同seed2026097101，512总交互，完整Actor可训练、无warm-up。唯一设置差异是actor_lr；critic lr、Final runtime/reset/schedule、PPO early-stop规则不改。新CLI值检查实际optimizer lr，不能只写YAML。

为补原artifact缺失的整批KL，GPU0仅重放一次原LR512（约17秒）；GPU1跑低LR512（约18秒）。不是新增算法/训练树。整批诊断在eval/no_grad下计算所有active row、全部9动作的`KL(old||new)`；不使用sampled-action近似、不推进RNG。测试验证探针不改参数/RNG。原LR重放出的Actor文件SHA **fa46fdedc1a1116f62cfa947bd5409fe549a3307ea3acb69d7d5e0210ef9bfba**与历史Direct512完全一致，确认诊断没有改变训练轨迹/结果。

| 每轮完整更新之后 | 原LR3e-5 | 低LR3e-6 |
|---|---:|---:|
| 第一轮整批mean KL | .226007 | .006693 |
| 第一轮state KL P90 / max | .66258 / 8.33245 | .01331 / .31409 |
| 第一轮argmax翻转率 | 19.34% | 3.91% |
| 第二轮整批mean KL | .019437 | .003262 |
| 每轮实际minibatch更新数 | 2 | 6 |

**CONFIRMED：** 原LR第一次完整update造成远超targetKL=.02的平均分布变化；旧日志max minibatch KL=.069甚至低估其最终整批.226。targetKL的early-stop不是最终更新幅度保证。降低LR在共同第一rollout上显著降低这种变化。第二rollout已经因策略不同而状态不同，不能称完全相同数据。

**新盲点：Actor early-stop同时截断critic训练。** 原设置每轮只做2个critic minibatch，低LR因不触发early-stop做6个。形式上只改LR，但作用路径同时包含Actor幅度及critic拟合量，不能把后续任务效果全部归因于Actor步长，也不能将不同rollout上的EV直接作critic质量对照。这是现有更新循环的真实耦合；本轮不再改第二个变量去拆分它。

**UNPROVEN：** 较小KL是否保住或改善完整任务、能否稳定累计到25k、是否存在reward/advantage与CE settling目标错位。没有据此启动25k，更没有使用teacher-KL或解冻/冻结新方案。

## 7. 低LR512冻结Gate（历史启动记录，现已完成）

唯一GPU任务：GPU1运行`lr_probe/low_lr_eval20`，与已有BC parent相同diagnostic20 seeds、同C3 evaluator，40回合。GPU0空闲。新前瞻Gate在eval启动前写入`lr_probe/task_gate.json`：safe/CE/capture≥.9且较BC下降≤5百分点，collision增加≤5百分点；各scene完成时间mean≤BC1.10倍、P50≤1.15倍、P90≤1.25倍。继续报告共同safe paired时间，不将成功条件统计等同全体任务指标。此为固定诊断种子的screen，不是正式泛化证据。

完成后运行：

```
PYTHONPATH=src:. python3 tools/summarize_forward_final_lr_probe_20260909.py --root artifacts/2026-09-09_forward_final_d/lr_probe
```

通过只得到短预算候选资格，不自动训练。若仍不通过，停止LR扫参，先用现有轨迹及rollout检查reward/advantage/phase与settling变化；不把问题包装成需要新算法或网络。

结束快照：2026-09-09T02:57:30.357241+08:00。GPU1 worker PID279456 / lock PID279455，32/40回合，当前step100，约9.81回合/分钟；ETA 2026-09-09T02:58:19.272298+08:00。GPU0空闲。完整快照为`lr_probe/SESSION_HANDOFF_20260909_0300.json`。

**NEXT WAKE-UP：2026-09-09 03:00 CST，或低LR完整report出现后。** 仅汇总并裁决完整任务健康度，不自动进入25k。本轮4项D测试（新增KL探针）+10项P0测试，共14 passed。


## 8. 低LR冻结结果：较小KL没有自动解决完整任务效率

本次再次`git pull --ff-only`，远端仍为`64e11b140288bd7a71f5b24eb3f98ce376d7da1b`。不重跑C0–C3或旧Pure-Capture A/B。完整结果见`lr_probe/comparison.json`，预声明Gate为 **HOLD_PPO_SCALE**。

| frozen sample，20回合/场景 | BC parent | Direct512 LR3e-6 |
|---|---:|---:|
| mixed safe / capture / collision | 19 / 19 / 1 | 19 / 20 / 1 |
| mixed成功条件mission mean / P50 / P90（秒） | 118.5 / 106 / 159 | 133.447 / 138.5 / 169.9 |
| capture→CE完成样本均值（秒） | 70.658 | 86.579 |
| pure safe | 20 | 20 |
| pure mission mean / P50 / P90（秒） | 79.9 / 74.25 / 121.25 | 82.925 / 61 / 136.25 |

低LR的mixed mean、P50超过前瞻阈值；其余门槛通过。mixed共同safe为18对，均值差+10.639秒；两边19个成功回合并非同19个，不能把成功条件均值差当paired效应。此次同一diagnostic种子族不是新的泛化formal。

**裁决：** 3e-6已显著减小第一rollout的整批KL（.226→.00669），但尚未保持mixed任务效率；停止继续扫LR，25k保持HOLD。不能因此断言所有PPO均必然侵蚀BC，更不能归因于categorical Actor表达不足（C3仍PASS）。

## 9. CPU phase / reward / GAE审计：冷baseline问题是候选原因

`tools/audit_forward_final_advantages_20260909.py`生成`advantage_audit_cpu/report.json`。这是冻结BC的CPU shadow512，不是之前CUDA采样的逐bit重放。两个256窗口均使用同一个未更新central V/ValueNorm，Actor及critic哈希不变；实际Final训练reset/reward consumer，CE speed=.0005。

第一窗口pre-capture 240行，raw GAE均值+53.16；post-capture 612行，−1.164，正值比例0；pure coverage 172行，−3.391，正值比例0。第二窗口post/pure仍几乎全部非正。将V完全置零计算GAE仍类似，因此不只是全局advantage normalization把正值翻成负值。实际coverage阶段以负CE cost为主，capture含大正terminal reward；第一窗口post总reward−56.075，其CE center−58.824、control−.783、PBRS+3.532。component字段有重叠，不可全部相加。

**LIKELY / 待因果验证：** 从近零V开始时，对负cost的coverage价值未校准，有限rollout内的phase credit可能不利于保持既有动作。**负GAE本身不是reward/GAE实现错误的证据**：与动作无关的baseline在期望下不会改变正确policy gradient；单次轨迹的正负比例也不是baseline质量判据。没有修改Final reward、phase权重或网络。

## 10. 4096步critic-only与独立完整回报校准

GPU1执行既定D2的前4096步，只训练central V/ValueNorm；零Actor/PPO更新，整个Actor逐bit等于C3 BC。16次update、96个critic minibatch，约132秒；完成12mixed+12pure，capture pool12。checkpoint `critic_calibration_4096/step_004096.pt` SHA `b2b850b131b9ba32633b5e374ae03d9513c206741ad5a9556cc47c6b5dceeb2c`。最后训练EV=.80896、ValueNorm mean=11.8517/std=33.4585。

该EV是对当前rollout的bootstrap GAE return target计算，不是对真实完整回报的留出拟合。随后以固定BC、fresh seed2026099101、6mixed+6pure完整回合评估cold/warm两V；两个critic看完全相同轨迹，gamma=.99的完整Monte Carlo return不含value bootstrap。使用Final训练reset池及reward(.0005)，不是C3的独立场景formal采样。无truncation；这是校准diagnostic，不替代C3可靠性统计。

| phase | cold→warm MC RMSE | warm within-phase EV | warm mean bias |
|---|---:|---:|---:|
| pre-capture | 131.791→153.234 | .0121 | +81.040 |
| post-capture | 102.079→100.969 | −.0343 | +27.441 |
| pure coverage | 10.681→10.503 | .0590 | −5.275 |

预声明校准screen要求post/pure各≥3回合、RMSE降到cold的≤.75、within-phase EV>0；两项未通过，**HOLD_CRITIC_CALIBRATION**。warm后post/pure raw GAE正值比例升至58.5%/73.5%，但这不能代替校准——baseline过负同样会制造正advantage。

**证据边界：** MC target包含策略采样及失败的回报噪声，不等于每个state的精确期望V；8984行高度回合内相关，不能当8984个独立样本。仅6mixed中有一次capture后碰撞，可能支配mixed RMSE。因此在作机制归因前，同种子重放一次并保留逐行return/prediction，补安全完成/其他结果分层；原all-outcome Gate保留，不能删掉失败样本后宣称PASS。此处不声称4096warm-up已经消除cold-start，也不声称增加warm-up时长必然无效。


## 11. 碰撞尾部分层完成：不能只归因尾部，也不能夸大critic结论

`heldout_mc_stratified/replay_parity.json`核验12回合outcome、长度及所有原aggregate RMSE/bias/EV与首次eval完全一致（最大差0）；新增统计不改轨迹。保存`calibration_rows.npz`及SHA，可直接CPU重算，无需再次GPU重放。

一次capture后碰撞占warm post-capture平方误差的 **91.74%**。安全完成的5个mixed回合中：post RMSE32.086→30.575（仅降4.7%），bias+10.936→+2.044，within-phase EV仍−.0231；安全pre-capture RMSE106.058→98.085（有改善）。pure全部6回合安全，但RMSE仅降1.7%。所以warm-up有一定平均baseline校正，**不能写成完全没有学习**；同时远未证明充分的状态价值区分或消除cold-start。

当前可靠结论分级：

- **CONFIRMED：** C3完整policy迁移仍成立；低LR512没有通过预声明mixed效率Gate；4096critic-only确实不改Actor；训练EV与留出MC校准结论不同；碰撞尾部确实主导此次post平方误差。
- **LIKELY：** cold/不充分校准的phase价值、Actor更新幅度及其与critic更新数耦合值得先处理，优先于新算法/网络。
- **UNPROVEN：** critic误差是否导致观察到的PPO效率下降；MC目标替换是否改善；critic信息不足是否为根因；4096warm后接PPO是否优于Direct（本轮尚未做该Actor更新）。
- **不得采用：** “training EV高→critic健康”“GAE变正→warm-up有效”“较小KL→完整任务保持”“负coverage reward→reward写错”。

## 12. 下一条GPU1唯一建议与交接

**下一条建议：固定BC完整回报的critic拟合诊断，Actor保持逐bit冻结。** 用Forward Final实际训练reward/reset收集一个小型固定trajectory bank，保存global state、完整MC return、phase/role及reset lineage；训练/留出使用不同seed stream和独立capture pool，防止mixed及其派生coverage reset跨split。只训练现有central V，训练集估计并固定归一化统计，完整MC监督目标；不换网络、不改reward、不更新Actor。预先限定小预算（例如40个完整回合、≤100个critic minibatch update；不因未通过自动延长）。

问题只问：现有critic是否能够在训练集拟合、并在按episode/来源隔离的留出集预测固定BC真实回报？同时报告训练/留出、pre/post/pure的RMSE/bias/EV，全部结果与安全/失败分层。MC含采样噪声，比较训练与留出的差距，不把逐行return当state真实期望V。与本次bootstrap warm-up相比，数据及拟合流程也变了，**不能宣称是target-only因果A/B**。若训练也不拟合，先查目标尺度/优化；若训练好而留出差，优先查信息可辨识性/分布覆盖；只有证据改善后才安排既定短PPO Gate。本建议尚未启动。

不再开LR/warm-up时长网格，不进入25k/50k、teacher-KL、continuous或VXY。GPU0空闲；GPU1的4096critic-only及两次冻结校准均完成，当前本项目无待运行GPU任务。ETA=0。**NEXT WAKE-UP：下次会话落实上述唯一critic诊断；没有需要定时轮询的任务。**

本轮最终回归：D4项 + P010项 + critic校准3项，共 **17 passed**（仅既有protobuf弃用警告）。校准新增测试覆盖真实terminal完整回报、EV不能遮蔽常量bias、失败回合分层保留平方误差。两次`git diff --check`通过。


## 13. 固定BC完整MC回报拟合完成：存在学习，phase校准仍不足

使用`tools/fit_forward_final_mc_critic_20260909.py`，GPU1从同一初始central V开始；Actor始终冻结C3 BC；Forward Final实际训练reset/reward（CE speed=.0005）。train seed2026099201与heldout seed2027099201，两个独立stream/capture pool。30train+10heldout完整回合，19360+7384 active-agent行；分别15+15、5+5 mixed/pure。无truncation，**40回合都安全完成，没有失败样本**，所以本次不能检验collision-tail的V校准。

保存每一步global state、完整gamma.99 MC return、active/phase/episode及各回合reset source。不是拿C1 teacher IQN数据冒充BC数据，不是在训练/留出之间共享capture snapshot。ValueNorm只从train目标更新一次后冻结，mean17.0289/std47.1809；heldout不参与统计、梯度或checkpoint选择。固定100个critic minibatch update（每批64个完整joint-state），约188秒，无Actor optimizer动作。模型和预测已保存，bank/模型文件本地保留，JSON/SHA可复核。

| phase | train cold→fit RMSE | train within-phase EV | heldout cold→fit RMSE | heldout within-phase EV |
|---|---:|---:|---:|---:|
| pre-capture | 102.530→53.812 | .271 | 93.937→51.971 | .136 |
| post-capture | 7.028→5.826 | .004 | 7.352→5.731 | .097 |
| pure coverage | 5.968→3.804 | .374 | 7.105→5.601 | .108 |

预声明post/pure的RMSE≤cold.75倍、EV>0门槛仅train/pure通过：**HOLD_MC_FIT_CALIBRATION**。不能事后把heldout约.780/.788比值改阈值称PASS；同时不能把这一启发式门槛未过等同算法完全失败。均值bias明显改善（heldout post+4.21→+.34、pure+3.94→−.49）；训练post EV≈.004提示目前主要学到阶段平均水平，尚未充分区分该phase内部的回报。

CPU读取同一bank/prediction的`phase_loss_audit.json`发现：pre-capture只占train行数23.3%，却占fit后平方误差 **97.2%**；post占行数42.5%但误差仅2.1%；pure占行数34.2%但误差仅.7%。**CONFIRMED**是MSE贡献严重不均；**LIKELY**是单一总体归一化下的损失尺度值得优先检查；**UNPROVEN**是梯度被怎样主导、是否因此造成PPO效率下降、调整权重是否有效。平方误差份额不等于梯度范数。

相比前次bootstrap warm-up，这次target/data/update流程均有变化，不是target-only A/B；MC仍是单次随机回报，不能把残差都视为可消除的V误差。也不据此设计新Transformer或改变Final reward。

**下一研究诊断建议（尚未启动）：** 复用这份固定bank、同critic初值/100更新/相同batch索引，仅对critic MSE按train phase回报尺度加权，检查post/pure拟合能否改善且pre-capture不过度损失；仍然冻结Actor。权重只来自训练集，不修改环境reward，不使用heldout调参。它用于区分loss尺度与表示/优化问题，不自动转为正式PPO改法。当前先完成下述可视化，不启动新GPU学习任务。

## 14. BC / PPO rollout GIF与本轮验证

用户要求增加可视化：每个模型mixed20/10GIF、coverage20/10GIF，默认sample；BC与已有低LR512均用同seed/evaluator。新入口复用C3 transition engine与历史renderer，只加入可选快照hook、共用Actor loader、隔离快照及输出编排；renderer仅为防CE标题裁切增加换行。没有复刻旧AC环境，也没有再跑PPO训练。

标准配置、两卡后台调用命令、输出与易错点见[GIF台账](FORWARD_FINAL_ROLLOUT_GIF_20260909_ZH.md)。实际可视化检查确认770×726、100ms、CE指标完整；短smoke保留首尾、绘出友军Voronoi/感知圈/CE centroid。共享代码相关回归 **28 passed**，包括C0/C1/C2合同、P0/D、critic统计、快照RNG/动作/事件不变及事件列表无序比较（改变事件内容仍拒绝）。本批实际每个rollout再核验现成eval20的fingerprint、任务指标、事件内容和action histogram。

生产目录为`artifacts/2026-09-09_forward_final_visual/{bc_sample,ppo_low_lr512_sample}/`；startup两次记录单独保留，不混入统计。两卡后台任务正常后继续文档/CPU分析；本轮结束时进度/ETA见该目录的最新SESSION_HANDOFF，不在线等待GIF完成。

结束状态（2026-09-09 13:58 CST）：BC与PPO各40/40回合、20/20张GIF完成，逐回合reference parity PASS，双方初态配对一致；GPU0/GPU1任务均结束。总计80 rollout/40 GIF，统计均复现既有eval20，不增加独立样本量。预览根入口`artifacts/2026-09-09_forward_final_visual/index.html`，对应run内有全部20张图。ETA=0，无定时NEXT WAKE-UP；不再启动任何训练。


## 15. 新阶段：P0 reward / P1 state / fixed-BC calibration

从远端`22046dd4fea91497b04f7dffb15e216c74a6cc31`重新接手，先读最新代码/artifacts，不重证C3。文献检索、reward公式与runtime context审计、旧fixed-MC bank统计补全见[新台账](FORWARD_FINAL_PPO_ROOT_CAUSE_20260909_ZH.md)。

- **RUNTIME FACT：** 相同critic输入和动作，post计数10/499导致下一步继续/终止；相同CE几何、hold0/29导致继续/成功终止；release counter1/9在同raw label下产生不同next role。`STATE ALIASING / NON-MARKOV CRITIC INPUT`已证实，不等于已证明PPO退化因果。
- **EXPERIMENT RESULT：** 原40episode fixed-MC bank零新训练；train-only phase/elapsed-time baseline在heldout post RMSE5.218优于原V5.731，pure5.379优于5.601；pre/closure原V较好。新增MAE、EV、calibration slope/intercept、rank、closure与outcome桶；原bank无失败，collision校准仍不可判。
- **LITERATURE-BACKED INTERPRETATION：** 当前优先D类缺context，不能按train EV宣布critic健康，也不先实施旧phase-weighting建议。
- GPU0补采已有BC/PPO及同seed IQN冻结reward轨迹；GPU1只做greedy物理中心continuous head蒸馏+frozen mean/sample eval。两条均无PPO。完整结果、Gate与结束运行状态以新台账末节为准。


### 第15节完成结果补充

**EXPERIMENT RESULT：** P0冻结120条全部完成。共同safe mixed18对PPO慢10.639秒，但discounted return也降1.137；recovery reward–time关联弱，不能把退化写成普遍“拖延刷reward”。P0允许critic-only进一步定位，不是严格时间目标对齐PASS。continuous mean/sample80条全部safe且0collision，但mean pure P90=170.45秒未过效率Gate，`HOLD_CONTINUOUS_REPRESENTATION`，支线STOP。

**CODE FACT / 运行协议：** 50维critic-only phase/time/history schema已实现，Actor输入不变。GPU0最小context对照重放原40条MC bank并逐bit校验geometry/target，再以原100更新/同batch训练；GPU1结束。详见[root-cause第8–10节及末节](FORWARD_FINAL_PPO_ROOT_CAUSE_20260909_ZH.md)。25k仍HOLD，P3尚不具备前提。


## 16. 最新状态：context对照工程中断，已修复来源追踪

**RUNTIME FACT：** 原context worker PID586169已退出，29/40回合、critic更新0。中断来自“capture snapshot来源必须等于reset后坐标”的错误断言；原生reset会按新障碍/间距约束修复坐标。不是context V训练负结果，也没有context calibration结论。

**CODE FACT / 验证：** 已改为reset前记录实际选中snapshot对象SHA，保留reset后是否修复作为诊断；原bank逐bitparity和split隔离断言继续保留。故意无效snapshot的source/RNG/初态对照及其余相关测试共5 passed。未重启实验，旧运行快照与NEXT WAKE-UP失效；无本项目GPU任务运行。

下一步仅重跑同一40episode/100更新context诊断，然后按phase/heldout/baseline决定后续；25k PPO HOLD，continuous支线STOP。具体结论、修复及命令见[root-cause第12节](FORWARD_FINAL_PPO_ROOT_CAUSE_20260909_ZH.md)。


## 17. 修复后P2重跑已启动

来源追踪修复及5项测试通过，GPU上初始V函数误差=0；新增已有输出保护及失败状态记录。按用户授权在空闲GPU1启动`context_critic_lineage_fixed`，仍是同40回合/100次critic更新、Actor冻结。任务未完成时不得宣称context校准通过；最新PID/进度/ETA/NEXT WAKE-UP见[root-cause第13节](FORWARD_FINAL_PPO_ROOT_CAUSE_20260909_ZH.md)与`SESSION_HANDOFF.json`。原失败目录保留，25k继续HOLD。


## 18. 修复后P2完成：HOLD_CONTEXT_VALUE_CALIBRATION

**EXPERIMENT RESULT：** 40回合/100critic更新完整完成，Actor逐bit不变，原bank数组逐bit一致，train/heldout初态与source池隔离通过。heldout RMSE geometry→context：pre51.971→50.610，post5.731→5.875，pure5.601→5.708；post/pure仍未胜简单baseline5.218/5.379。MAE有改善，不能称完全无学习。train post EV仍约.0042，预声明6checks仅train/pure通过。

**结论：** 已识别context缺失成立，但补context+同100更新尚未解决校准，未证明它单独解释PPO变慢。下一方向是同bank的phase残差/bias及实际梯度贡献审计，不先扫参数或追加PPO。failure桶仍无样本，P3与25k继续HOLD。无本项目运行任务，NEXT WAKE-UP取消；完整结果及证据边界见[root-cause第14节](FORWARD_FINAL_PPO_ROOT_CAUSE_20260909_ZH.md)。


## 19. Fixed-bank梯度审计完成：共享phase干扰优先验证，继续HOLD

**EXPERIMENT RESULT：** 无新增rollout/Actor/critic更新。train pre占96.86% critic loss，但完整bank的gradient norm-mass仅trunk48.00%、head24.53%；pre↔post/pure cosine在trunk为−.494/−.482、head为−.742/−.739，post↔pure分别+.975/+.995。固定终点16个原minibatch中，pre trunk norm-mass中位数78.69%，pre↔recovery约半数batch冲突；不能把全库平均或loss占比等同历史Adam更新。

**EXPERIMENT RESULT：** heldout恢复前50步post target均值−8.006、critic−2.572、baseline−6.525；pure为−6.343/−2.666/−5.118。去掉train估计的phase bias后仍未胜heldout baseline；存在可预测时间轮廓欠拟合，不能判target不可预测。train pure的RMSE实际优于baseline，避免全盘失败叙述。

**HYPOTHESIS / 唯一下一建议：** 优先验证pre与recovery共享value head干扰：仅提出一个pre/recovery两head、同bank/初始化/normalizer/batches/100更新的critic-only对照，post与pure合组；未实施、未启动。尺度不均共存、共享trunk冲突和普通优化不足仍是边界。P2/P3/25k/continuous继续HOLD。全部phase target/residual/loss/实际gradient表、复现与证据边界见[root-cause第15节](FORWARD_FINAL_PPO_ROOT_CAUSE_20260909_ZH.md)。


## 20. Two-head同条件对照完成：CASE 3，停止拆head

**EXPERIMENT RESULT：** same bank/init/100批索引/Adam/ValueNorm/global loss/clip/100 updates均通过断言，Actor逐bit冻结，初始全bank V差0。heldout RMSE single→two：pre50.610→51.185，post5.875→6.974，pure5.707→7.026；train recovery也变差。early50局部RMSE改善，但recovery整体V下移约5.3–5.4，后期target≈−1却预测≈−7，整体校准失败。

**EXPERIMENT RESULT：** cross-head cosine按路由归零，train trunk pre↔post/pure cosine由−.494/−.482转为+.512/+.500，heldout也转正。冲突减小并不等于critic健康。归类CASE 3；不支持继续PCGrad或拆trunk，不启动very-short PPO。

**HYPOTHESIS / 唯一下一建议：** 停止head结构方向，改做无训练的early-recovery transition state/full-return construction审计，核对phase边界、context/reward/MC target对齐及真实terminal规则。未实施该下一审计。P2/P3/25k/continuous继续HOLD；4项相关测试及独立checkpoint重载通过，全部结果和边界见[root-cause第16节](FORWARD_FINAL_PPO_ROOT_CAUSE_20260909_ZH.md)。
