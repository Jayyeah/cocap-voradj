# Forward Final corrected PPO：D smoke / 更新健康 Gate

起点 `f9532d2dcc49aad36d729bfc8e0eb393e4ed59c8`，本次pull确认远端无更新。C3三种冻结策略formal100已全部完成并PASS；详见 [bridge台账](FORWARD_FINAL_MAPPO_BRIDGE_LEDGER_20260908_ZH.md)。本页只记录新增D，不重新审计旧Pure-Capture。

**当前：D训练桥接已实现；CPU合同测试与双GPU512步smoke通过数值/信息合同；更新KL偏大，正在同种子冻结eval20检查。25k暂缓，不自动扩50k。**

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

## 4. 已完成的冻结结果与停止扩预算决定

截至本次会话结束，BC parent与Warm-up512均完成20mixed+20pure；Direct512尚在运行。BC mixed safe/capture19/20、collision1/20；Warm-up512均20/20、0collision。两者pure CE均20/20、0collision，但pure完成时间mean由79.9升至102.6秒，P90由121.25升至196.05秒（+61.7%）。这触发既定1.25倍P90效率门槛，**Warm-up512为HOLD_PPO_SCALE**，不启动25k。该小样本结果不能宣布正式4096步warm-up失败，更不能比较尚未完成的Direct；它证明只看binary success会漏掉更新后settling变慢。

快照：2026-09-09T01:58:59.401134+08:00。GPU0 Direct512 worker PID264405（flock PID263398），32/40回合，当前step131；ETA 2026-09-09T01:59:49.496155+08:00。GPU1评估完成并空闲。checkpoint均为已有512步smoke Actor，没有25k checkpoint。完整PID/throughput输入记录见 `artifacts/2026-09-09_forward_final_d/SESSION_HANDOFF_20260909_0200.json`。

**NEXT WAKE-UP：2026-09-09 02:05 CST，或Direct512报告完成时。** 先汇总全部配对结果，再诊断一次/两次PPO update为何改变coverage settling。当前唯一待完成GPU任务是GPU0 Direct512；GPU1不再补实验。下一正式GPU1任务需由完整健康Gate收敛，不预开25k/teacher-KL/continuous。
