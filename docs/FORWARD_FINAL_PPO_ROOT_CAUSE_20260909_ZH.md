# Forward Final 强 BC → PPO 效率退化：根因阶段

事实起点：2026-09-09 同步 `origin/experiment/small-step-ac-migration-20260828`，`git fetch --prune` + `git merge --ff-only @{u}` 后 HEAD=`22046dd4fea91497b04f7dffb15e216c74a6cc31`。该提交在 `515d54d` 之后，已经包含 fixed-MC critic 100-update 拟合与完整 GIF 结果。`f890b33` / `f9532d2` / `515d54d` 均为祖先。用户本阶段指令覆盖旧台账下一实验建议及 AGENTS 中旧预算顺序。C3 PASS 保留，不重证 Actor/BC 迁移。

**当前裁决：25k PPO HOLD。P1 已证实 critic state aliasing；现有 critic 的 heldout post/pure RMSE 未稳定胜出简单 phase/time baseline。没有新 PPO、没有 critic loss weighting、没有 LR/warm-up/width sweep。** P0与continuous均已完成（第8/9节）；首次context对照在拟合前中断；来源追踪已修复，现按同预算在GPU1重跑（第13节）。

所有新增结论采用 CODE FACT / RUNTIME FACT / EXPERIMENT RESULT / LITERATURE-BACKED INTERPRETATION / HYPOTHESIS。历史台账中的 CONFIRMED 等标签不回写。

## 1. 文献 / implementation review（2026-09-09 检索）

以下来自原论文、作者项目或实现文档，不把相邻研究的效果当 CoCap 因果证据。

| 工作及一手来源 | 成熟原则及适用边界 | 当前先解决哪层 |
|---|---|---|
| [DAPG, Rajeswaran et al., RSS 2018](https://roboti.us/lab/papers/RajeswaranRSS18.pdf) | BC 初始化 + demonstration augmented policy gradient；示范项随学习减弱。原算法不是 PPO，不能照搬其步长和加权量。 | 强初始化需保护，但不能修复错误 reward 或缺失状态；保留后续单一 regularized 方向。 |
| [Kickstarting, Schmitt et al. 2018](https://arxiv.org/abs/1803.03835) | 在学生访问状态上加 teacher cross-entropy，允许 teacher 影响逐渐减弱并超越 teacher；原实现不是当前 MAPPO。固定 teacher 下 CE 与 KL(teacher\|actor) 对 actor 梯度相同。 | 保护更新而不永久锁死；不是 P0/P1/P2 的替代品。 |
| [Adaptive BC Regularization, Zhao et al.](https://arxiv.org/abs/2210.13846), [作者实现](https://github.com/zhaoyi11/adaptive_bc) | offline→online 分布突变会导致性能坍塌；自适应 BC 强度在稳定性与改进间折中。原方法含 Q ensemble/off-policy 机制，不能把整套算法移入本阶段。 | 借鉴限制早期 drift 的原则；不据此开新的算法树。 |
| [Demonstration-Regularized RL, ICLR 2024](https://arxiv.org/abs/2310.17303) | 研究先 BC 再向 BC policy 作 KL 正则的 RL 及理论样本复杂度。KL 方向、覆盖假设与具体定理条件不能省略；理论保证不是当前神经 MAPPO 的保证。 | 如果 objective/value/advantage 都通过后 Direct 仍无增量，只允许一个退火 regularized 版本。 |
| [PPG, Cobbe et al.](https://arxiv.org/abs/2009.04416), [CleanRL PPG 实现文档](https://docs.cleanrl.dev/rl-algorithms/ppg/) | policy 与 value 分阶段优化，辅助 value 学习时约束 policy 变化、允许 value 数据复用。 | 当前 Actor/V 已分网，不存在同一 backbone 的 value-gradient 干扰；但 PPO KL early-stop 截断 critic minibatch 的调度耦合确实存在。应区分这两种耦合。 |
| [PPG Reloaded, ICML 2023](https://proceedings.mlr.press/v202/wang23aw.html) | Procgen 实证强调 policy regularization 与 data diversity，而非单纯高 value reuse / 低频 distillation。 | 不把更多 warm-up/value epochs 当通用答案；分 split/phase 看泛化。 |
| [Time Limits in RL, Pardo et al.](https://arxiv.org/abs/1712.00378) | 内生有限任务时限需要 time-aware state；外生采样截断应 bootstrap。 | 直接对应 post-window / CE-hold 等历史依赖；同时保留 episode-horizon truncation 与 mission-terminal 的不同解释。 |
| [GAE, Schulman et al.](https://arxiv.org/abs/1506.02438) | value-based 多步估计存在 bias/variance 权衡。准确 state V 可减小方差；bootstrap 使用错误/混叠 V 时可产生偏差。 | 负 advantage 或 aggregate EV 都不能单独定罪；健康 critic 后才做 teacher-Q 按 phase/role 对照。 |
| [PIRLNav, CVPR 2023](https://arxiv.org/abs/2301.07302), [作者实现](https://github.com/Ram81/pirlnav) | 强 BC 到在线 RL 先 critic-only，之后渐进打开 actor 学习；目标是避免随机 critic 破坏已有导航行为。 | 支持 frozen-policy return calibration，但不证明固定4096步足够。 |
| [ADEPT, 2026-08-19 v1](https://arxiv.org/html/2608.19182v1), [作者项目](https://adept-dexterity.github.io/) | BC distillation → frozen-actor critic warm-up → conservative PPO。论文区分 observation mismatch、value misalignment 和 excessive drift；其消融中降低 actor LR 防止坍塌，warm-up/BC进一步改善；clip 收紧作用较小。 | 当前 BC representation 问题已通过；小 LR 后效率仍退化，下一层应看 reward 与 conditional value，不能机械复制其百万步 warm-up 或认定小 KL 必定有效。 |

**LITERATURE-BACKED INTERPRETATION：** 当前属于强先验 policy 的 online adaptation 稳定性问题，且已经发现 contextual value 不充分。成熟原则是先定义正确任务目标与 value 条件，再校准 frozen-policy critic，最后限制 policy drift。没有一篇文献支持“train EV .81 就排除 critic”，也没有证据要求改 Actor backbone。ADEPT 这里指 dexterity 的 `2608.19182`，不是同名 diffusion-environment 工作 `2506.01759`。

## 2. P0：reward 与 full-mission efficiency

**CODE FACT：** `envs/coverage_ce.py::ce_transition_reward` 与 `VorAdjEnv.step::coverage_task_reward` 实际给 coverage

`r = 10[-c_next - u + gamma*Phi_next - Phi_now]`，`Phi=-c`，`gamma=.99`。

本合同 `coverage_ce_acceleration_weight=0`、`coverage_ce_angular_velocity_weight=0`；D训练时 `u=.0005*(speed/3)^2`。C1/C3冻结采集 reset 默认 speed weight=0。不能误用函数默认 `.002/.001` 为实际 acceleration/turn cost。CE terminal 成功额外 bonus 为0；native phase/terminal PBRS correction 已存在，不能因为调用中出现 `terminal=False` 就误判漏清零。`coverage_ce_pbrs_reset_mode=phase_and_all_terminal`。post/pure 没有独立常量每步时间代价。

在整段 post/pure、固定初态且正确 terminal correction 下，discounted PBRS 总和只剩初始势项；优化的基础量是 discounted CE error + speed cost，不是 mission time。本段内正 shaping 不能机械解释为“刷奖励”。pre-capture 混合角色切换、support 的 .5 权重和 capture phase boundary 需要逐项审计，不直接套整回合全局单一势函数定理。

**CODE FACT：** capture terminal reward 为 `goal_reward * factor`，普通围捕 factor 与参与数/角间隙离散程度有关；大正 capture 奖励支配 full-return 尺度。full mission reward 有追求 capture 速度的动力，但“更快 CE 连续达标30步”不等同“更低累计 centroid error”。30步 hold、CE阈值与控制能耗是不同目标。

**HYPOTHESIS：** 策略可能接受更长的低误差尾段换取较小 discounted cost；接近质心时 time penalty 缺失与折扣会弱化速度信号。跨 episode 难度会混淆 reward–time 相关，必须结合共同safe配对和分量变化，而非仅凭负/正相关判冲突。

P0输入及限制：

- `cpu/iqn_existing_alignment.json`：直接读取 C1 的200条 episode/shards，逐文件 SHA 校验；safe mixed99/pure99；原始 reward0，独立采集种子。
- `cpu/bc_mc_alignment.json`：复用现有40条 fixed-BC MC bank，按 `r_t=G_t-.99G_(t+1)` 恢复 reward（float32舍入限制）；原训练 reward .0005、native独立recovery pools。未记录CE最终误差，quality标 unavailable。
- `p0_replay/`：因已有 BC/PPO eval/GIF报告缺逐步reward，有限冻结重放 IQN/BC/PPO 各20mixed+20pure，same diagnostic20 seeds；实际 reward consumer .0005。BC/PPO 每回合校验旧report的初态、动作诊断、length/capture/recovery/mission/safe/collision完全一致。没有重新训练，也不增加原BC/PPO的独立可靠性样本量。IQN同seed新增reward比较不属于重证C0/C3。
- return采用每步4名pursuer reward均值，再沿时间求和；同时报告 discounted/undiscounted。phase discount 在该phase起点重启，不能把不同起点的phase returns直接相加成总discounted return。
- `p0_alignment.json` 的每组含 Pearson/Spearman、fast/medium/slow时间三分位内 return P10/P50/P90/mean；safe outcome条件统计，失败保留在records。不能按agent rows算相关样本量。

**EXPERIMENT RESULT（现有IQN数据）：** mixed full mission Spearman：discounted −.452 / undiscounted −.248；capture：−.844 / −.054；post恢复：−.265 / −.369；pure：−.228 / −.317。这显示 capture 折扣目标方向较清楚、coverage效率关系弱，尚不是PPO退化的因果解释。

## 3. P1：实际 central-V 输入审计

**CODE FACT：** `training/continuous/central_schema.py::build_central_global_obs` 生成 `self[4,9]`、全局pursuer/evader/obstacle几何速度、mask；`models/small_step_ac.py::CentralValueNetwork.forward` 只消费这些。pursuer token的role槽恒0；focal self含本机当前pursuing flag，但不含release倒计时。模块头注释声称body-frame，全局token实现实际world-frame，以实现为准。完整global几何不等于完整环境state。

| context / consumer | 实际是否可用 | 判定 |
|---|---|---|
| remaining target geometry / velocity / count | live enemy tokens + mask可用 | 当前活动目标充分；捕获/失效/初始不存在在mask上合并。 |
| pre / post / pure phase | 有live enemy可推断pre；无enemy不能区分post/pure | 缺失；两种无目标状态的deadline语义不同。 |
| capture / target flags | deactivated enemy不输出，captured flags未输入 | 固定4v1下部分可从历史先验猜测，跨mixed/pure不等价。需显式场景与captured状态。 |
| post-capture remaining window / started | `post_capture_started/post_capture_step`只在env | **STATE ALIASING**：native window500真实terminal。 |
| CE连续达标累计 | `distribution_hold_steps`影响30步成功，input无历史计数 | **STATE ALIASING**；当前几何不能推导此前连续达标长度。 |
| pursuing release memory | flag部分可见，`_pursuing_release_counters[4]`与其他agent记忆不全 | **STATE ALIASING**；K10 memory影响下一role/Actor输入/奖励。 |
| stationary capture累计 | `_stationary_capture_counters`未输入 | CODE FACT 缺失：stationary hold10启用；此次具体rollouts触发频率未证明。 |
| remaining episode horizon | `episode_step`无输入 | MC有限完整episode需要；当前总horizon被定义为外生truncation并bootstrap，不能与post-window一概而论。MC bank无truncation并不消除隐藏time。 |
| CE speed schedule age | 未输入；D锚定2m+online，当前后段恒.0005 | 本阶段是已验证常量，不单独构成随时间变化的aliasing。跨早期schedule数据混用才需要系数/clock。 |
| reset source / pool lineage | 未输入；目前bank只存source标签 | per-episode terminal G不依赖下一episode pool，故缺source标签本身不证明非Markov。pool必须split隔离；几何相同且当前context相同不应仅因source不同给不同V。 |
| curriculum/stage | 无输入；本轮固定4v1/1obs、相同参数 | 当前常量；未来混stage需显式context/足够物理参数，不先扩大网络。 |
| legacy settle/grace / zone scheduler | 成员存在，但CE下settle关闭、zone关闭、成功立即结束 | 不能仅凭字段存在就认定当前缺状态；以实际启用分支为准。 |
| PBRS role/phase reset | 当前几何可算势，角色历史/phase边界部分隐含 | terminal correction实现存在；不足来自输入context，不是漏写terminal公式。 |

**RUNTIME FACT：** `p1/context_probe.json`：完全相同critic输入与动作，post计数10 vs499 → 下一步done False vsTrue。`p1/context_extended.json`：相同收敛几何与动作、hold0 vs29 → 下一步done False vsTrue；同raw coverage labels、release counter1 vs9 → 下一effective flag False vsTrue。均调用实际runtime consumer，不是仅阅读字段名。

**证据边界：** 这些是context干预反例，证明输入非Markov；没有证明该反例在低LR PPO访问了多少次，更没有证明deadline aliasing支配了效率退化。尤其成功轨迹通常远离500步deadline，不能把“存在deadline aliasing”自动等同主要原因。CE hold和release memory更靠近日常settling/role变化，仍需context-conditioned fixed-BC诊断检验。

## 4. 复用现有P2结果：补全统计与廉价baseline（本轮零神经网络训练）

**EXPERIMENT RESULT：** `cpu/existing_mc_enrichment.json` 对既有30train/10heldout、100更新模型新增：每phase、closure附近（最后10个pre步）、safe/failure的RMSE/MAE/EV、`target=slope*prediction+intercept`、Spearman。全部40回合safe，failure各桶明确0 rows，不伪造失败校准。现有失败tail bank来自别的run，不能混入当本次heldout失败结果。

baseline固定为训练集 `E[G | phase, floor(elapsed_phase_steps/50)]`，未见bin退回训练phase均值。elapsed来自已过步数，**绝不使用真实未来completion剩余时间**。无heldout参数拟合/调参/模型选择。三种预测同一bank、同一active-agent rows比较；baseline自身不含完整context，也不是oracle。

| phase | train 原V fit RMSE | train baseline | heldout 原V fit RMSE | heldout baseline |
|---|---:|---:|---:|---:|
| pre | 53.812 | 61.956 | 51.971 | 55.762 |
| post | 5.826 | **5.162** | 5.731 | **5.218** |
| pure | **3.804** | 4.218 | 5.601 | **5.379** |
| closure末10步 | **49.202** | 76.061 | **54.630** | 65.487 |

**EXPERIMENT RESULT：** train/heldout完整初始global-state SHA集合无交集。**CODE FACT：** 原collector每split新建FinalMissionStream及capture pool，seed独立。初态hash不重合本身不证明source池独立，二者证据分别记录。原bank没存pool snapshot逐条fingerprint；新增采集须保存完整lineage与source-snapshot hash。

**LITERATURE-BACKED INTERPRETATION：** 现有复杂V在pre/closure学习有效，post接近均值预测；heldout post/pure未稳定胜出简单phase/time predictor，不支持C类“critic健康”。当前优先归类 **D（确定缺context）**，伴随A类phase内拟合不足；不能仅凭固定100更新判网络容量不够，也不能把目前结果只判B类generalization。区分不可约policy sampling噪声、context缺失、优化不足仍需最小补context fitting。

## 5. 后续Gate与最小修复候选

1. P0若发现安全轨迹的时间与return存在实际tradeoff，先保持PPO HOLD，报告冲突。候选是明确的full-mission running-time cost/成功终止目标，或在现有CE cost上加入最小时间项；都改变目标，必须单独审核，不在本轮直接改reward。调整gamma不保证修复速度目标，不能用gamma sweep替代定义任务。
2. P1已FAIL：critic-only context至少包括phase、post窗口剩余、CE hold、pursuing release、stationary capture history、必要目标flags及明确schedule系数；保留actor local合同。先做版本化schema、逐步runtime断言及配对输入反例测试，不能将缺失context从旧bank伪造为零。
3. 新fixed-BC bank需保存真实pre-action context、独立train/heldout seed/pool、source snapshot SHA。只比较原geometry V与context V，固定小预算、同batch索引/同targets/训练归一化，不同时phase weighting或改reward；明确这是context-only对照。按现有phase/outcome/closure统计及廉价baseline评估；失败样本不足就标不可判，不隐藏。
4. P0/P1/P2全通过后才P3；teacher Q[9]是teacher价值，不是BC/PPO的精确A，Q-gap冲突是诊断。分ordinary/support/closure/post/pure，检查GAE、bootstrap、normalization、采样及clipping。当前不能回答“critic健康时advantage是否正确”。
5. 25k保持条件启动：same BC、Final、actor LR3e-6、无附加变量，5k/10k/25k保存完整resume与eval；可靠性严重破坏/数值失败停。效率小幅早降可续25k；25k无恢复则停止，仅再考虑一个有退火teacher/BC正则版本。当前旧runner只验证C3，不会自行检查本阶段三Gate，**不得直接按旧默认命令启动**；在Gate可用前不开放25k。

## 6. GPU1 continuous AW representation bridge

**CODE FACT / RUNTIME FACT：** 复用同BC encoder与policy隐藏层初始化，encoder冻结；2维tanh head对teacher greedy AW9物理中心作归一化MSE。没有soft概率中心标签。30epoch/batch2048/lr3e-4，验证MSE选checkpoint；不是continuous RL。使用已有continuous AW积分器、共享C3 transition engine；同scene/seed/runtime，九个grid centers实际step及短teacher轨迹均验证和离散完全一致。continuous adapter默认yaw=0会改变初态与RNG，本入口显式`yaw.init=random_uniform`恢复Final语义，记录此冲突。

frozen mean + clipped small-noise sample：归一化std=.05，分别20mixed+20pure，同C3前20 seeds；记录boundary clipping质量，不能把它称无界Gaussian。nearest-center agreement是proxy，不是精确物理action一致率；连续entropy不可用，旧字段零placeholder不作结论。支持/closure见共用MissionEventTracker，capture/coverage/safe/time/collision按同一终止合同。

Gate预先写launch：两mode场景safe≥.9、mixed capture≥.9、collision≤.1；mean mission P90≤同seed BC argmax的1.25倍。若不能通过，STOP continuous；即使通过只标representation bridge，不允许continuous PPO。初态fingerprints逐条配对，Actor eval参数hash验证不变。

## 7. 可复核入口

- P0/P1：`tools/audit_forward_final_root_cause_20260909.py`、`tools/audit_forward_final_state_context_20260909.py`。
- CPU统计：`tools/summarize_forward_final_root_cause_20260909.py --out artifacts/2026-09-09_root_cause/cpu --replay artifacts/2026-09-09_root_cause/p0_replay`。
- Continuous：`tools/bridge_forward_final_continuous_aw_20260909.py`。
- raw NPZ/PT本地保留，Git保存JSON/SHA/config/scripts；不覆盖历史artifact。作业launch含PID/GPU/source SHA，progress含当前step/throughput/ETA。
- 共享evaluator/bridge/visual/PPO/critic回归18 passed；新增phase折扣、calibration方向、runtime反例、continuous grid-center短trajectory测试3 passed；仅既有protobuf弃用警告。实际完整BC/PPO逐条parity是额外证据。


## 8. P0完整结果与Gate（120/120重放完成）

**EXPERIMENT RESULT：** `cpu/p0_alignment.json` complete=true；BC/PPO各40episode全部通过原诊断report逐项parity。IQN同seed40episode完成。共同safe的18对mixed：PPO−BC mission **+10.639秒**，总discounted return **−1.137**，undiscounted **−6.957**；post单段discounted **−1.160**。按episode配对bootstrap95%时间区间[−6.528,+26.167]秒、discounted-return区间[−9.057,+6.097]；点估计效率退化不能升级成总体显著因果效应。

| safe episode组 | full mission vs return Pearson / Spearman | capture vs return Pearson / Spearman | recovery vs return Pearson / Spearman |
|---|---|---|---|
| IQN mixed20 | −.332 / −.392 | −.757 / −.764 | −.281 / −.392 |
| BC mixed19 | −.607 / −.725 | −.825 / −.823 | +.111 / +.129 |
| PPO mixed19 | −.230 / −.353 | −.850 / −.880 | −.089 / −.114 |

表中均为discounted。undiscounted、时间三分位return分布、pure CE质量相关及每episode原始数值见JSON。pure total Spearman（discounted / undiscounted）：IQN −.620/−.661、BC −.464/−.561、PPO −.492/−.534；pure terminal CE RMS与return关联弱，BC约−.051、PPO−.126（discounted Spearman）。

共同safe mixed中8/18对出现“full mission更慢但总return更高”；12/18对recovery更慢，其中4对post单段return更高。因此非严格排序等价确实存在，但**不能解释成本次PPO普遍通过变慢赚取更高return**，其配对均值实际降低。post分量PPO−BC：CE center −2.389、speed control −.00763、PBRS −.02199、安全项 +1.258，合为−1.160；实际不是降低speed penalty造成大幅return收益。安全项包含near-boundary代价，safe completion不代表安全reward始终0。

pure20对：时间+3.025秒，discounted return+.08095、undiscounted−.15846，显示折扣下存在局部效率tradeoff；对应区间均跨0，不能凭这点宣布全面reward mismatch。

**LITERATURE-BACKED INTERPRETATION / Gate：** P0为 `MIXED_DIRECTIONALLY_ALIGNED_RECOVERY_WEAK_NOT_STRICT_TIME_OBJECTIVE`：允许继续critic-only因果诊断，不是严格时间目标对齐PASS，更不放行PPO。主退化不能单独归因于reward harvesting；reward对recovery时间的弱信号保留为HYPOTHESIS。没有直接改reward；常量time-cost候选只在目标冲突后续确认时讨论。分析图为`cpu/reward_efficiency.png`和独立PDF。

## 9. Continuous完整结果：HOLD，STOP

**EXPERIMENT RESULT：** 30epoch选epoch29；train/validation归一化MSE .14485/.14111；validation物理MAE为a=.09893、ω=.11709，nearest-center agreement69.25%。backbone逐bit等于BC。mean/sample各20mixed+20pure完成，全部safe且0collision；但效率Gate未全过：

| frozen mode | mixed mean/P50/P90秒 | pure mean/P50/P90秒 |
|---|---|---|
| continuous mean | 110.10 / 113.50 / 137.15 | 100.65 / 73.00 / **170.45** |
| continuous sample | 117.40 / 104.75 / 149.25 | 156.80 / 77.75 / **210.55** |

mean pure P90超过同seed BC argmax的1.25倍预声明门槛，`HOLD_CONTINUOUS_REPRESENTATION`。sample pure长尾更重；不按高safe率覆盖效率失败。support/closure/capture/CE事件与角色agreement保存在`continuous/report.json`；没有continuous PPO。

**CODE FACT：** 最终汇总曾因NumPy bool不能JSON序列化中断，80条records和全部评估已经保存。仅修复显式bool转换并使用`--summarize-only`重建report；没有重跑rollout或改变Gate。GPU1任务已结束。

## 10. D类最小context对照（启动协议，不预判结果）

新增`training/forward_final_value_context.py` schema `forward-final-value-context-v1`；50维训练期context追加到central self，含phase、episode/post remaining、CE hold/success、release initialization/counters/flags、stationary counters、target存在/捕获/失败、CE speed系数。Actor仍只有原local输入。原hidden256/8heads/4layers不变；新增输入列零初始化，初始V函数与旧几何V误差必须<1e−6。不是加宽critic或新Transformer。

`tools/fit_forward_final_context_critic_20260909.py`在GPU0重放原30train/10heldout BC种子及独立pool，只补真实pre-action context与完整source lineage。每个旧global-state、MC target、phase、episode、active数组必须逐bit相同，任一不同立即失败，不把新数据冒充同bank。记录实际初始fingerprint、全部pool snapshot SHA、capture reset源匹配，断言train/heldout无初态及pool snapshot交集。

同原train-only归一化、critic LR1e−4、100 minibatch updates、batch64、seed2026099301/相同batch索引；不更新Actor，无phase weighting/teacher KL。比较context V、历史geometry V、同train-only phase/time baseline；全部phase/outcome/closure指标保存。旧bank40回合全safe，这一对照仍不能证明failure-tail校准健康。预声明post/pure两split RMSE≤cold.75、EV>0，另heldout context RMSE须不差于phase/time baseline；通过也只标`CONTEXT_VALUE_SCREEN_PASS_FAILURE_UNTESTED`，不自动跑PPO。

新增context维度/初始function不变测试通过；本阶段新增4 tests通过，加共享回归18 tests。P1已识别反例被输入区分，不等于所有hidden state已穷尽。结果和会话结束进度见下一节。


## 11. 会话交接快照

快照时间：2026-09-09T19:08:08.594633+08:00。GPU0 tmux=`cocap_context_critic_20260909`，PID=586169，alive=True；实际阶段`collecting_context`，critic update=0，采集完成=25/40。throughput=12.931678193716923 episodes/min；ETA=2026-09-09T19:09:33.191168+08:00。**NEXT WAKE-UP：2026-09-09T19:10:33.191168+08:00，或report出现时。** 这是运行快照，不是Gate完成；episode内部step在本collector未上报，不能假称已完成100更新。GPU1已结束，continuous HOLD后STOP；P0已完成。完整状态见`artifacts/2026-09-09_root_cause/SESSION_HANDOFF.json`。

作业启动源码另在`source_snapshots/`归档并逐SHA匹配launch；P0运行时加载的是22046dd原evaluator，continuous加载的是新增物理policy接口版本。22项相关测试通过，`git diff --check`通过。所有后台任务仅本项目，未占用或停止其它用户进程。


## 12. 快速状态更新：context对照中断，非算法负结果

**最新状态覆盖第11节快照：无本项目后台任务运行；旧NEXT WAKE-UP失效。** GPU0另有PID590300占约17GB，非本context作业，本轮未占用或停止它；GPU1 continuous评估早已结束。

**RUNTIME FACT：** PID586169在train episode29（第30回合）开始时触发`capture source must match actual reset positions`，只完成29/40回合。**Actor更新0、critic更新0**；无新context critic checkpoint/report。因此不得写作“context fitting失败/无效”，本轮只有工程中断，context修复的价值尚未检验。原progress残留running现已更正，`failure_audit.json`保存旧快照及日志SHA。

**CODE FACT：** `envs/base.py::reset`接收capture snapshot位置后，会按新障碍和pursuer最小间距重新验证，冲突时重采样位置。于是`reset_source=capture_snapshot`表示初始化来源，不保证最终坐标原样保留。此前诊断工具用最终坐标反查source并硬断言相等，是错误的provenance假设。原Final环境语义未改，此发现不推翻P0/C3结果或已有state-aliasing反例。

**修复及验证：** `RecordedSourceStream`在调用原生env.reset前，从`initial_pursuer_positions`的实际来源对象记录选中snapshot SHA；之后另存`capture_positions_preserved`和坐标匹配列表，允许原生修复。继续要求source属于本split pool，保留原MC bank的geometry/target逐bit匹配及train/heldout source disjoint断言，没有绕过真正的数据隔离Gate。新增故意无效snapshot测试，验证修复前source身份准确、reset后初态fingerprint与Python RNG逐项等于未加记录的原生stream。相关**5 tests passed**；无新训练。

**LITERATURE-BACKED INTERPRETATION：** 目前最强证据仍是“缺context确实存在”以及“复杂geometry V的heldout recovery不胜简单phase/time baseline”；它们支持先修state再校准，但仍不能证明PPO变慢由critic aliasing引起。P0未显示mixed策略普遍用效率换更高return；continuous的纯coverage长尾门槛未过。二者均不支持跳入新RL算法。

**唯一下一步：** 在fresh目录重跑修复后的同一40episode/100更新context对照，先过原bank逐bitparity，再看train/heldout pre/post/pure/closure的RMSE、MAE、EV、calibration与baseline。此次快速更新未启动重跑，不新增超参或预算。若context后post训练仍仅学均值，先对固定bank做phase梯度贡献/信号诊断；若train好heldout差，先数据多样性；若两边改善，仍须补failure/outcome校准后再讨论P3。25k继续HOLD。

重新执行入口（先核对空闲GPU；下例仅指定当时空闲的GPU1，不自动启动）：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 tools/fit_forward_final_context_critic_20260909.py --out artifacts/2026-09-09_root_cause/context_critic_lineage_fixed --device cuda:0
```


## 13. P2已修复并重跑（用户授权）

**CODE FACT / 验证：** 来源追踪保持reset前实际source身份、允许原生坐标修复；5项相关测试通过。新增GPU preflight验证50维context零初始化后，64个真实bank joint-state上的初始V最大误差=0；无policy更新。检查normalizer状态可保存。另修复异常退出遗留running进度：本进程拥有的run将保存failure_audit并标记failed；重复使用已有输出目录被拒绝时不修改其中任何文件，独立保护测试PASS。

**RUNTIME FACT：** 新目录`context_critic_lineage_fixed`，GPU1/process-local cuda:0，tmux=`cocap_p2_retry_20260909`。同原30train+10heldout种子、独立pool、原geometry/target逐bitparity、同100更新/64 batch/LR/归一化/Actor冻结全部保留；无新超参，不覆盖首次失败目录。启动源码按launch SHA归档。

运行快照 2026-09-09T20:46:58.625414+08:00：PID=641168；阶段=collecting_context，采集21/40，critic update=0，throughput=12.458263111867833 episode/min；ETA=2026-09-09T20:48:45.130946+08:00。**NEXT WAKE-UP：2026-09-09T20:49:45.130946+08:00，或完整report生成时。** 不按进度提前判P2 PASS。CPU检查结束，按约定不在线等待；后续先读取新目录report/failure_audit，25k仍HOLD。
