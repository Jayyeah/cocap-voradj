# Forward Final corrected PPO：D smoke / 更新健康 Gate

起点 `f9532d2dcc49aad36d729bfc8e0eb393e4ed59c8`，本次pull确认远端无更新。C3三种冻结策略formal100已全部完成并PASS；详见 [bridge台账](FORWARD_FINAL_MAPPO_BRIDGE_LEDGER_20260908_ZH.md)。本页只记录新增D，不重新审计旧Pure-Capture。

**当前（2026-09-09）：C3 bridge PASS保留；D扩预算HOLD。低LR512完整eval20仍未保持mixed效率；4096步critic-only warm-up虽训练EV≈.81，独立完整回报校准未通过。Actor在整个校准阶段逐bit不变，未启动25k PPO。第4/7节的运行状态及NEXT WAKE-UP均为历史快照；以文末最新交接为准。**

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
