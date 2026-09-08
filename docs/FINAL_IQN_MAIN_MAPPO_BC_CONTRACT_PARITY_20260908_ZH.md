# main Final IQN-AW ↔ 当前 MAPPO-BC Contract Parity Audit

> **路线更新（用户确认，2026-09-08/09）：** 本文保留历史审计事实；第7节原 A/B 必经 Gate 和 GPU1-only 调度已失效。A/B 现为 `OPTIONAL_DIAGNOSTIC`。当前主线是 Forward Final C0→C1→C2→C3，允许双 GPU，进度见 [Forward Final bridge 台账](FORWARD_FINAL_MAPPO_BRIDGE_LEDGER_20260908_ZH.md)。不得按本页旧 NEXT WAKE-UP 重启 Pure-Capture formal100。

审计日期：2026-09-08。已 pull/fetch；canonical 是远端 **main@a5814f49fa29d869cdc3fb8d8e0df4722aa11f00**（2026-08-04 Final integrated release）；实验分支起点为 **4bf3c6935917d2515c50132ad088c991125226a8**。不能用实验分支名字或 teacher 文件名替代这两个身份。本页更新 P0 中“下一步直接做 Final-side 冻结比较”的建议：先完成下文 A，再安排明确定义的 B。

## 1. 核心裁决与历史意图

**当前 BC 不是 Final IQN 的 only-architecture-changed 版本。** 精确名称是：**Final Stage2@300k IQN 权重，在 B0/K10 corrected Pure-Capture Legacy-VorAdj 环境上生成数据，再冻结86个 backbone state-dict keys，蒸馏 categorical head 的策略。** BC 本身不是已经完成 PPO 优化的 MAPPO。

两条 lineage 必须分开：

- **权重 parent**：main Final CR-MS/VCT-LS/CE 的 Stage2 8v2@300k，teacher SHA `ef58ae9bdd018633afee0d16f2242d2f157a6cc18611e41474f115aed254b87e`。它的 IQN 是在 Final mixed/coverage 课程里训练出来的。Stage3 实际继承当时的 Stage2@500k，并非后来 release 选中的300k，不能重写该历史。
- **目标环境 parent**：`positive_feedback_ladder_20260807/stage5b_mixed_aw.yaml` → `parallel_ce_legacy_voradj_20260809/legacy_voradj_oldmix_4p1e1obs_200k_aw_allagent.yaml`（显式回退 legacy capture 与 all-entity VorAdj）→ `...allagent_noclip_scratch.yaml` → **CF0 capture-first/local** → **CF3 support-full** → **B pure_capture_local_allcapture** → **B0 K10** → `small_step_ac_migration_20260828/common.yaml` → `mappo9_v2_20260830/common.yaml/seed1.yaml`。

历史意图有直接文本证据：8月28日小步台账把 Golden 定为 `3ee4d949...` / IQN@125k 的 corrected Pure-Capture；8月30日 common 注释明确继承该环境不变；9月3日台账明确先在 MAPPO exact target 环境 qualify 三个 Final teacher，再选择 Stage2 采集600回合。因此 **capture-only、旧奖励、规模4v1及 corrected collision 是有意沿用诊断任务，不是无意漏载 Final YAML**。但这些选择只支持相对旧 Golden 的小步对照。

三类问题不可混淆：

1. **有意简化**：capture-only / 4v1 / 1000步 / terminal-on-capture / all-capture奖励与team terminal，服务于当时 AC 可学性诊断。
2. **legacy AC 继承**：Legacy-VorAdj、旧-MS/front、capture/coverage两套障碍构图、无效的CE/半径/support权重残留字段。来源明确，不能仅凭“继承”一词判定每项都是 bug。
3. **真正 migration drift**：宣称20m敌方局部感知却实际未执行；把 Final teacher 权重资格写成 Final task/信息资格；把 CE helper flag 或残留1/1 support权重当成实际奖励；混用不同 runtime/seed 的 paired 证据。应修正后续迁移合同与实验标签，不篡改旧环境或旧结果。

## 2. main 的 canonical Final 实际身份

`main` 的 release manifest、三阶段 YAML、IQN checkpoint SHA 与实验分支一致；但 env/base、VorAdj、Robot 源文件有后续变化。已在隔离 checkout 中真正导入 **main 的模块**，分别 reset 4v1/8v2/12v3，调用实际 sensing/support/capture/CE getters 和构图函数，并运行隐藏敌方、支援、terminal探针。没有把新分支 runtime 当作 main runtime。

相同探针在4bf3c69下重复：三阶段 resolved+actual profiles、信息探针、奖励探针、20步固定 AW9 动作 trace 的位置/奖励/done/obs hash 全部相等。代码静态审查中 main 的 Final 分支已经使用 effective K10；不能将实验分支中间版本的K10错误倒写成 main 历史错误。这些有限探针**不证明所有 collision / inactive roster / 长历史完全等价**；严格 canonical 冻结基准优先使用 main 运行时，后续修正分支另起明确合同。

证据：`artifacts/2026-09-08_contract_parity/canonical_main_runtime.json`、`branch_final_runtime.json`、`main_branch_comparison.json`。main 导入路径随 artifact 保存。

## 3. 逐项 effective contract 表

以下 Final 指 **main release 三阶段完整训练任务**，不是某个另行截短的 capture evaluator；规模相关以4v1为比较起点，同时列8v2/12v3。`INTENTIONAL_DIFFERENCE` 表示历史已有设计依据；`MIGRATION_DRIFT` 表示声明与有效行为或新目标不符，不假定作者曾有意授权泄漏。SAME 只针对本行，不代表整体公平。

| 合同项 | Canonical main Final IQN-AW | 当前 BC dataset / target PPO / BC eval | 分类；实际 consumer |
| --- | --- | --- | --- |
| 地图/捕获spawn | 120×120；边界0..120；双方map_random；edge margin12；同类及初始敌我min sep15；静止初始化 | 相同4v1捕获spawn，含相同随机yaw；已核对同seed初态 | SAME；`CoCapEnv.reset`、spawn采样 |
| 规模/场景分布 | 4v1/1obs→8v2/2obs→12v3/3obs；每阶段含无敌方coverage | 只4v1/1obs、capture | INTENTIONAL_DIFFERENCE；task merge、scene_cycle |
| coverage spawn/recovery | pure coverage cluster/capture snapshot/map-random混合；捕获态池1000，capture ratio .75，余下map ratio .5 | 数据没有这些rollout；继承的pure_ce任务配置未被使用 | NOT_APPLICABLE（BC）；`CoCapTrainer._reset_task` / collector只选capture |
| horizon | 3000 action steps；无独立pre-capture截断（0）；post窗口500/600/700 | episode与pre-capture均1000步（500秒） | INTENTIONAL_DIFFERENCE；`VorAdjEnv.step`读取真实实例/配置 |
| AW9物理动作 | a={-.4,0,.4}，w={-π/6,0,π/6}，按a外层w内层的9个index | 相同9个物理中心 | SAME；`Robot.action_list`、`CategoricalGridActor.action_grid` |
| AW执行入口 | integer index→legacy Robot AW积分 | categorical index→显式(a,w)向量→AW adapter→Robot AW积分 | INTENTIONAL_DIFFERENCE（接口）；未因此变为连续策略 |
| 时间/动力学 | physics dt=.05，N=10，decision=.5秒；pursuer vmax3，drag=.4/3；yaw随w变化 | 相同；不是VXY fixed-yaw servo | SAME；实际Robot属性及固定动作探针 |
| APF evader | `v2_fixed`、15离散AW动作、vmax3.5、drag=.4/3.5、感知20 | 相同 | SAME；evader/APF消费者；不同于 pursuer 感知漏洞 |
| enemy sensing半径 | **20m表面clearance**；全向；当前圆模型p/e半径各√1.25，等价该组合中心距约22.236m | YAML/getter也是20，**实际敌方token无有限米制半径截断**，由Voronoi邻接筛选；不是所有敌人必见 | MIGRATION_DRIFT；`_pack_agent_obs`的VCT-LS/legacy分支 |
| friend sensing/shared-state | 全图有效友军位置用于构图；graph邻居友军位置/速度/role，按role/shared-count/距离排序，截断8个token；无20m友军距离阈值 | 同样需要全体友军状态，邻接可能另受敌方站点影响 | SAME（共享范围）；`_voronoi_map/_pack_agent_obs` |
| obstacle显式感知 | 表面clearance≤20m；min_obs也只在可见障碍里选 | 中心距≤20+障碍半径，或coverage cell交集等条件；min_obs取全图最近障碍中心距；不是同一20m表面传感器 | MIGRATION_DRIFT（“same local sensing”声明）；`_pack_agent_obs` |
| 静态map/geofence信息 | 全图静态障碍与边界进入自由空间掩码、投影centroid；并非未知地图探索合同 | 同样使用全图障碍/边界，另有上述非局部显式量 | SAME（已知地图前提）；coverage map + boundary features |
| VorAdj构造 | capture/coverage均pursuer-only，`free_mask_projected`，grid60 | capture为all-entity + `legacy_assign`；coverage为all-entity + `free_mask_projected`，grid60 | INTENTIONAL_DIFFERENCE（legacy消融继承）；`_capture/_coverage_voronoi_map` |
| evader参与派生feature | evader不作为Voronoi site，不直接改变centroid/几何友军邻接 | evader作为site，影响centroid/邻接/cell相关obstacle token | INTENTIONAL_DIFFERENCE（构图选择）；把其解释成半径局部是下一行drift |
| 超半径敌方影响 | 全队都未感知的敌方瞬时移动不影响已审查actor输入/Q；邻居自己看见目标时，允许通过role位影响本机 | 无人满足20m感知仍可经enemy token及派生feature影响actor | MIGRATION_DRIFT；state→preprocessing→token→actor，见§5反事实 |
| global enemy token | 无无条件全局广播；VCT-LS只输出本机直接测量 | `global_evader_visibility=false`，但不能推出米制local | SAME（开关关闭），不是信息范围SAME；`_pack_agent_obs` |
| capture dense具体版本 | `ring_importance_ms_v0`（CR-MS）weight2，旧approach/MS/front=0，direct timestep0 | **B0/K10 Pure-Capture allcapture legacy branch**：direct approach1+旧MS2+front.5；informed approach1；uninformed0 | INTENTIONAL_DIFFERENCE；`capture_task_reward` vs `pure_capture_dense_reward` |
| CR-MS实际启用 | true；preferred8、outer10.5、σ2、progress clip3 | false；虽残留omega_ring_ms=2及ring参数，branch不可达 | INTENTIONAL_DIFFERENCE；`_ring_importance_ms_enabled`实测 |
| support奖励 | enabled；.5×neighbor-visible approach-only+.5×CE，不是.5×CR-MS | 普通support blend disabled；残留1/1权重无效；B的一跳informed专用attraction，无CE | INTENTIONAL_DIFFERENCE；blend getter与pure_capture优先分支 |
| role/K10 | direct触发pursuing，失去检测保留10个动作transition；非pursuing图邻居若有pursuing则support；否则coverage | effective K10仍在，但由legacy邻接决定；另有direct_capture/one_hop_informed/uninformed奖励分区，不是三任务分配 | INTENTIONAL_DIFFERENCE；`_task_labels_from_map`、B observable partition |
| CE/coverage奖励实际执行 | `centroid_energy_v0` PBRS scale10、κ1、γ.99，phase/all-terminal reset；coverage和support可达 | CE helper flag=true，**dense CE/PBRS路径被allcapture绕过**，本次转移实测0 | INTENTIONAL_DIFFERENCE；有helper flag不代表有任务奖励 |
| CE速度成本 | 训练时0→global_step200k后.0005；a/w成本0；新建冻结eval env只读配置为0，须与训练调度区分 | 残留速度cost0，且CE奖励不可达 | NOT_APPLICABLE（BC优化目标）；trainer scheduler vs env消费者 |
| CE success | normalized RMS≤.05、max≤.10连续30步；min active4/8/12；CV<.15只是诊断 | 继承相同CE阈值/min active2、CV<.20，但不是capture-only成功目标 | NOT_APPLICABLE（BC主任务）；CE success helper与终止路径分开 |
| formal moving capture K | K=3、中心距≤8m、最大角隙≤π、最大/最小gap比≤3 | 相同 | SAME；`_loose_capture_events`，不是outer-ring人数 |
| stationary capture | enabled；速度≤.2、hold10、至少2机 | 相同 | SAME；实际比较条件是≤，不能只报total capture |
| 捕获终奖 | 每个capture event：120×几何factor，给participants；stationary factor1 | 相同金额公式，但给**全部active pursuers**，每事件一次 | INTENTIONAL_DIFFERENCE；terminal_recipients分支，CPU实测3人vs4人 |
| 安全成本 | collision/boundary -160、近距/边界proximity -10；hard boundary死亡 | 相同主要金额与边界规则 | SAME（金额）；不代表collision检测等价 |
| collision semantics | **main legacy_end_step**；不能把swept称为main原合同 | **synchronized_swept_v1** | INTENTIONAL_DIFFERENCE（corrected Golden）；后续双方统一修正需另命名 |
| done / terminal | capture后继续；CE成功或post窗口/总horizon结束；失活/evader lost等；min active=全队4/8/12 | all-target capture立即done；1000步timeout；min active2（可继续尝试stationary） | INTENTIONAL_DIFFERENCE；`VorAdjEnv.step`，CPU捕获转移主分支false、BC true |
| post-capture/mixed | 500/600/700步窗口；无settling terminal/reward；捕获与pure CE episode 1:1交替 | 捕获即terminal；残留window500、mixed_crms/pure_ce配置没有进入collector/PPO scene cycle | NOT_APPLICABLE（BC采集）；不能把残留配置称为full-task覆盖 |
| observation形状/编码 | self9、friends8×7、enemy8×7、obstacle5×5，总22token；robot-frame/map-diagonal距离、sqrt(N)centroid缩放 | 相同tensor schema与坐标规范，但信息源/role含义不同 | SAME（schema）；不等于feature语义等价 |
| backbone | IQN entity Transformer256/8/4，target/max/mean/summary/pursuing分支 | 复用Legacy decision backbone，**86 keys与真实teacher逐bit相同** | SAME（已加载feature权重）；dropout=.1但冻结eval禁用 |
| head / 学习与数据 | distributional IQN Q，课程含capture/support/coverage/mixed | categorical logits，softmax(Q/T)监督，T=1，只有Pure-Capture状态；BC更新只动head | INTENTIONAL_DIFFERENCE；distiller，非PPO优化成功证据 |
| central critic | 无执行期central V；IQN训练奖励可用全局几何 | MAPPO central V仅后续PPO训练用；BC/frozen actor不读它 | NOT_APPLICABLE（零PPO BC）；不能以CTDE解释preprocessing泄漏 |

## 4. 两个实际奖励合同

### main Final：CR-MS + support approach/CE + CE persistent mission

令 `C_i=(sqrt(N) ||p_i-centroid_i|| / map_diagonal)^2`，`u_i`为归一化速度/加速度/角速度平方成本。Final ordinary coverage transition：

`r_CE = 10[-C_i' - u_i + γΦ_i' - Φ_i]`，`Φ_i=-C_i`、γ=.99；phase切换及terminal另做PBRS reset correction。速度权重训练200k前0、之后.0005，其他两项0。普通冻结env reset的即时getter为0，**这不否定历史训练scheduler之后的.0005**。

Direct capture：`r_direct = 2 clip(||p_i-m_i|| - ||p_i'-m_i||, -3, 3)`，`m_i`为CR-MS在占据/障碍/边界过滤及速度/角度加权环带上构造的目标；preferred8m、outer10.5m、σ2、cell1.5、occupancy radius4；不是旧mean-shift再叠front。旧三个omega均为0；direct timestep=0。

Support：`r_support = .5 clip(d_before-d_after,-3,3) + .5 r_CE`，approach内部weight1；目标仅选 pursuing图邻居直接看到的敌方。**support capture部分仍是approach-only，不是CR-MS项**。该邻居目标坐标用于训练奖励计算，不放进本机enemy token。global几何奖励属于训练特权，部署需保留角色和观测合同而非在线计算reward。

各角色另加安全项；捕获终奖 `120 (2π/n) exp(-std(angle_gaps))` 给capture参与者，stationary为120；CE额外hold/repeat/settling奖金清零。CPU构造的非捕获support机实测 capture=.171043、coverage=-.065308，证明两个分量实际可达。

### 当前 BC / PPO target：B0-K10 corrected Pure-Capture allcapture

不能只叫“legacy”：是8月9日oldmix显式legacy奖励（approach1、旧MS2、front.5）经CF0→CF3→B allcapture→B0 K10保留下来的**B专用分区奖励**。

- `direct_capture`：`clip(Δd,-3,3) + 2 M_old + .5 F_old`。旧MS依靠35m候选搜索、1.5m cell、占据radius4等构造引导点；front按敌方速度方向与半径8的站位打分。它不是CR-MS。
- `one_hop_informed`：只给 `clip(Δd,-3,3)`；`uninformed`：0 dense task reward。所谓可见在当前legacy里指邻接，不是20m。
- B分支**没有通用timestep -1**，即使env.timestep_penalty=-1仍在配置；也没有coverage dense/PBRS。另加安全与all-active共享capture终奖。
- 捕获后直接terminal，未采集任何post-capture action。600回合数据的support标签表示capture过程中旧role/知情分区，不是Final完整support优化的证明；“direct rows=0”的VCT-LS元数据字段在legacy下本来就不工作，不能解读为没有捕获个体。

Dataset manifest config hash本轮重新从collector的解析路径计算，**完全等于** `259bdd77a72353292c50ed4994d3de6fb196ce8206a3859da9020a9f5b809db5`；dataset、当前PPO target、BC eval的核心runtime getters一致。实际600回合/171052 agent rows；权重与环境来源均有证据，没有把新合同追认给旧数据。

还须区分**环境reward**与**teacher Q来源**：冻结teacher不会因为在B环境rollout就重新学成B的Q。采集器储存的是Final训练权重在legacy观测上的固定τ Q及greedy action，reward仅是记录；BC loss是T=1的`KL(softmax(Q/T)||π)`，不使用环境reward重新计算return。这进一步限制“value/task迁移已成立”的说法。

## 5. 敌方信息路径及部署合同

### 当前代码事实

main Final：`state → pursuer-only Voronoi(全友军+已知map) → own local enemy/obstacle radius过滤 + neighbor role → token → IQN`。所有机都看不到的目标(42,42)在四角友军场景中移动到(48,42)，obs与固定τ Q均不变。敌方不进入centroid/几何邻接的站点集合。

但是“本机半径外绝无任何敌情”不符合main：将敌方放到(10,55)，本机(10,10)仍没有enemy token，邻居(10,70)能看见，传来的role位由0变1。这是已有support机制所需的**显式允许通信**；排序也可能随role改变。移动后的friend轨迹或K10记忆同样可承载历史敌情，不能把它们错称即时oracle坐标泄漏。

BC：`全队与敌方state → 含evader的Voronoi → centroid/邻接/obstacle cell候选 → token → actor`。四机与敌方表面距离均37–43m时，四机仍得到enemy token。进一步固定显式enemy rows/mask为移动前的值，只让真实派生feature发生变化，冻结BC logit最大仍变 **.450499**；这是分量中介诊断，不是假称该clamped输入自然来自环境。未clamp时P0已测得约.5082。说明**仅在最终打包时mask远处enemy token不足以修复合同**，还必须修构图/centroid/role来源。

### 建议的最终可部署合同（不冒充现成无线实现）

- 共同坐标系定位、自身运动状态；已知静态全图障碍/geofence。
- 全队有效友军ID/位置/速度/active/timestamp共享，或另行证明等价的分布式邻居发现；每机独立构建pursuer-only Voronoi。当前实现不是只用“20m内友军”或“最多8个输入token”就能复原拓扑。
- 自机enemy测量20m表面clearance；图邻居只交换自己的role/pursuing位，不无条件广播enemy坐标。允许的role信息通道写进方法定义。
- actor不读取global critic、全局敌方位置或训练reward；公平baseline使用相同友军/map/感知预算。无线范围、延迟、丢包、定位误差和未知地图是待验证的新条件。

所以用户对main Final的理解**大体成立**；需补充“support不是完整CR-MS复用”“已知地图与全友军构图”“允许邻居role敌情”“legacy collision仍是历史canonical”四个限定。

## 6. 真正只替换 Final IQN 决策算法的 MAPPO 应是什么

现在的 `configure_environment(...mappo9_v2)` 与runner仍服务于capture-only合同，**不能仅换teacher路径，或把scene名改为mixed_crms就叫Final MAPPO**；旧B开关仍会绕开CE。

明确实现目标是：从main release三阶段配置与runtime出发，保留Final观测、pursuer-only map、CR-MS/support/CE、K10、APF/物理AW9、终止/阶段/恢复spawn及CE成本调度。categorical actor使用现有经过验证的Legacy decision backbone + 9-logit head，直接输出整数AW9索引给同一环境；增加training-only centralized V、PPO/GAE/ValueNorm及P0 eval-forward/log-prob合同。一个actor仍对每机局部token独立执行。不是continuous接口迁移。

允许因算法而变化的是distributional Q/replay/epsilon→categorical PPO/on-policy rollout/entropy/central V。保留scene/reset分布，不硬搬IQN离策略replay四buffer比例作为PPO重复旧样本；相关采样策略若改变要单独记录。full-task BC初始化需另采Final分布，旧B数据不能默认为包含coverage。

若决定采用synchronized-swept，应命名为**Final任务＋共同collision修正**，IQN与MAPPO双方重评；不能称为main逐项无差异。当前审计不实现此迁移、不启动训练。

## 7. 冻结评估裁决与100%结果边界

**当前100%只支持：Final teacher能在B环境完成任务；复用teacher backbone并训练head的BC在该4v1 capture分布上argmax/sample均有很强任务保持。** 不能支持Final full-task parity、MAPPO scratch探索成功、PPO优化增益、严格半径local部署或coverage泛化。Sample100/100、0碰撞、123.78步，与历史argmax100/100、3碰撞、71.16步是本合同内描述；后者缺新状态fingerprint，不在本轮补造配对显著性。

**GPU1 下一条：A — same-contract imitation formal100。** 使用dataset hash对应的B0/K10环境（含corrected collision），固定同一Stage2 teacher和BC SHA，IQN greedy / BC argmax / BC sample在GPU1串行；共同100个未用于teacher选择和dataset采集的seed，记录resolved/runtime、初态fingerprint、逐回合safe capture/time、新closure/support事件。候选seed `2026091801..2026091900`，启动前查已有资产重用情况。对argmax讨论greedy imitation，对sample讨论软标签策略的性能代价，不要求sample动作等同greedy。

A与B是不同问题。A后B应固定**canonical main运行时**与Final4v1的三种场景（capture诊断/pure coverage/full mixed，场景任何截断需明确标记）。输入形状兼容已确认，但语义存在已知OOD。不能沿用旧AC scene_config。优先用同一AW integer执行wrapper，使两actor真正进入同一Final状态转移；先main原collision，若改collision必须双方同改并另标合同。**本轮不直接启动B**，更不同时变信息、碰撞、horizon后把下降归于MAPPO表达能力。

B若失败，分别标记信息/观测迁移、legacy→Final任务与奖励语义、dataset只有capture、coverage/support/mixed状态OOD；冻结执行时reward不直接反馈进actor，但训练历史与soft-Q标签语义仍重要。B不能单独识别所有因素，必要时只在失败后增加有针对性的中间合同，不预先展开实验树。

状态：CPU审计完成；未启动GPU评估或训练；ETA=0，无后台等待。NEXT WAKE-UP：执行唯一A同合同冻结评估；完成后再确定B的最小wrapper与证据边界。只用GPU1串行，GPU0保持空闲。
