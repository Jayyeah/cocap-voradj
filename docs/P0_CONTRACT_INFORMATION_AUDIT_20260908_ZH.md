# P0 合同修复与信息来源审计（2026-09-08）

> **2026-09-08 canonical main / BC parity补充：** [完整逐项合同表与奖励/信息审计](FINAL_IQN_MAIN_MAPPO_BC_CONTRACT_PARITY_20260908_ZH.md)。main@a5814f4才是Final parent；BC环境是B0/K10 corrected Pure-Capture，未迁移Final完整任务。下一条改为A同合同三策略冻结评估，B Final-transfer随后单独定义；不是直接把旧BC放进混合修改后的“Final-side”环境。

本页覆盖旧台账中有关 PPO 健康、BC/PPO 配对、local execution 和 closure 的过强结论。起点是远端最新 `7725f88ba2cacb5fa2014dad95a005b5b362d9d0`；本轮先 pull，9 月 8 日再次 fetch 确认无更新。未改奖励、物理能力、网络结构或历史 checkpoint；没有启动训练。所有 GPU 工作仅物理 GPU1，GPU0 保持空闲。

## 1. 证据等级与裁决

| 判断 | 最新等级 | 证据与适用范围 |
| --- | --- | --- |
| 原 categorical PPO likelihood 合同正确 | WRONG ASSUMPTION | backbone 实际缺省 dropout=.1，BC encoder eval，而 PPO 曾重新启用 train forward；未更新参数时重采样 dropout 已产生 approx KL≈.089、clip fraction≈.455。不能把这些数归因为优化步长。 |
| 本轮 categorical forward / stored likelihood 合同可验证 | CONFIRMED（代码、CPU 回归及 GPU1 smoke） | act/update 固定 eval forward，仍保留 autograd；每次 update 前重算所有 active rollout 的 action log-prob，误差>1e-4 或非有限即拒绝。GPU1 初始 probe 的 log-prob 与 ratio 误差均为0。尚不代表长训优化健康。 |
| BC Actor 可以表达很强的 argmax 策略 | CONFIRMED，限旧 AC 4v1 capture 合同 | 旧100回合 argmax capture=100%；不是 Final CR-MS/mixed 任务资格证，也不是 stochastic rollout 已达100%的证据。 |
| Direct PPO 有显著 erosion | UNPROVEN，撤回旧“配对显著”结论 | BC 实际 seed=2026090301..0400，native PPO 实际 seed=2026100301..0400（入口隐含+10000）。旧 BC↔Direct/Warm paired bootstrap 无效；本次审查此前给出的 exact McNemar p=.125/.0078 同样撤回，这是审查自身的配对错误。100/96/92%只能作为各自样本描述。 |
| critic warm-up 已排除 cold start | UNPROVEN | 原 forward 合同有混杂，单分支结果不能排除该机制；Warm↔Direct 实际同 seeds，但比较仍受上述合同影响。 |
| Final 与旧 AC 的 local observation 信息等价 | WRONG ASSUMPTION | Final 的目标 token 用局部表面距离过滤；旧 AC 的 legacy_voradj 按含敌方站点的 Voronoi 邻接，无半径过滤，且派生 centroid/邻接也可依赖未感知目标。 |
| support11 是有效的奖励因果实验 | WRONG ASSUMPTION | 错 namespace 实际 .5/.5 未变。本轮错误 namespace 与 runtime NOOP 都能被拒绝；未重训。 |
| VXY 慢主要证明策略差、或 AW/VXY 已物理公平 | UNPROVEN / WRONG ASSUMPTION | AW 纵向 .4 与 VXY 全平面 servo .4 不是相同可达加速度集合；历史比较只能按实际控制接口解释。没有在 P0 改物理预算。 |

IQN 保留为当前性能锚及成果主线；最终算法不预先指定。MAPPO 只有在共同信息/碰撞/任务合同下提供安全完整任务时间、困难场景或部署上的明确增益，才升级论文方法组成。

## 2. 已实施的合同修复

- `training/small_step_ac.py::MAPPOTrainer`：rollout 与 update 使用同一个 deterministic forward；随机探索来自 categorical sampling。`eval()` 不冻结参数，测试确认 encoder/actor 可反传更新。所有 active 行在首次 optimizer step 前验证 old/new log-prob，inactive padding 不参与。continuous MAPPO 也固定 eval forward，但其独立 likelihood 迁移不在本次 categorical Gate 内。
- `run_small_step_ac_migration.py`：full-resume checkpoint 写入 `mappo-eval-forward-v1`。缺该标记的旧 PPO resume 不得恢复训练；eval-only 可读旧 checkpoint。历史权重可以显式作为新 rollout 的初始化，不得携带旧 likelihood 继续更新。
- 新 `training/runtime_semantics.py`：实际 reset 后读取 topology、实际目标 token 规则、support getters/blend、capture R/K、碰撞语义、dt/physics_dt、实际机器人 a/w/drag、adapter 参数及活动 dropout 模块。支持 `runtime_semantic_assertions` 的逐字段期望；错误 namespace、未知断言字段、数值不符立即失败。categorical preflight 还核对真实 AW action centers 与全动作 zero-update ratio。
- `assert_only_changed(before, after, allowed)` 必须验证**恰好**所声明的 live 字段变化，声称修改而实际不变也失败。CPU 真实环境中 .5/.5→1/1 恰好改变两个有效 getter。此项是后续候选实验 preflight 的必用接口；不是已经为所有历史脚本实现完整配置 schema。
- AC runner、BC evaluator 写 runtime preflight；IQN reset 验证并保存每任务首个 runtime snapshot。resolved config 与实际实例结果必须同时保存；跨进程环境 reset 前仍需 `set_global_config`，全局 ConfigManager 的隔离尚未重构。
- native MAPPO eval 显式 exact seed，记录实际 seed 与初始状态 SHA256；其它 `_screen` 调用保留旧默认 offset 以免静默重定义历史 suite，同时明确标记 `legacy_plus10000`。`assert_paired_records` 对 seed/初始状态缺失或不等均拒绝；历史无 fingerprint 的输出不得直接当作已验证配对。

## 3. closure/support 的新定义

共享实现：`evaluation/mission_events.py`；已接入 BC evaluator、native AC screen、AW/VXY cross-retention evaluator。schema=`same-target-mission-events-v1`，原字段保留供历史读取并标明 legacy，不可再称为新 closure 指标。

- 实际 capture 区域是目标中心距 `r<=capture_distance`（本配置8m），独立记录目标 ID；不再用外侧环带 `8<=r<10.5`，也不取多个目标的最大人数拼接事件。
- 每个目标首次/再次从少于2人进入2+为机会起点，分别跟踪达到3+及满足实际 K、最大角隙和角隙比的时间。进入2+时已3+可形成0时长事件。跌回2人以下记 `coalition_lost`，下次重新触发新机会。
- support 机会：本机未直接观测某目标、但其已观测的友军 token 对应个体直接观测该目标。分别跟踪同一 agent-target 到 direct detection、到真实 capture 区域。这里的 target association **只在 evaluator 内用于统计**，不是声称策略接到了带目标坐标的支援请求，也不是任务分配成功率。
- 每个机会保留 completed / support_lost / coalition_lost / target_removed / agent_inactive / episode_end。未完成机会不能从分母删去；episode_end 右删失，reset 时已存在的机会标记左删失。捕获后环境先删除目标的情况，从权威 capture event 的目标/参与者快照恢复末步完成判据；stationary 不能冒充几何闭合。
- 输出完成率、各终止原因、完成条件下的均值，以及排除左删失的 completed_known_start_mean_seconds/count；原 completed_mean_seconds 包含左删失下界，必须带标签读取。不要把条件完成均值当全体任务性能，也不要把 target_removed 自动当作协同失败。
- 时间分辨率为 decision dt=.5秒。事件可重复且同回合相关，未来置信区间应按 episode/seed 聚类；不把每个事件当独立试验。不把中断作为无信息删失直接套普通 Kaplan–Meier。

论文主指标仍应是固定任务 horizon 下安全完成率与完整任务时间/预算内安全完成曲线，配套 collision/boundary、capture 后 CE 恢复；closure/support 是机制诊断。本次没有重新跑 Final mixed，也没有以这次 capture-only eval 宣称完整任务改进。

## 4. state → preprocessing → token → action 的实际信息路径

主要源锚点：`envs/voronoi_adjacency.py::_capture_voronoi_map/_coverage_voronoi_map/_pack_agent_obs/_has_enemy_neighbor`，`models/iqn.py`、`models/continuous/local_entity_token_encoder.py`、`models/small_step_ac.py`，以及 `envs/base.py::_loose_capture_events`。

| 输入/处理 | Final VCT-LS | 旧 AC legacy_voradj | 部署含义 |
| --- | --- | --- | --- |
| Voronoi 站点 | 全体有效友军位置，不含 evader | 友军与 evader 全状态 | AC 的信息依赖从 token 之前已开始，不能只检查 global_enemy_flag |
| coverage centroid | 上述友军、全图边界及静态障碍地图，active count 做 sqrt(N)尺度 | 可额外依赖敌方站点 | Final 的局部 obstacle token 不意味着 centroid 只用了局部地图 |
| self | 自身速度/yaw、边界向量、centroid、direct/role状态；按配置打包 | 类似维度但含义及 min obstacle 信息范围不同 | 同维度、同 backbone 不是信息合同等价 |
| friend tokens | Voronoi 邻居位置/速度/role，按 role、shared-count、距离排序，最多8个 | 邻接本身可能受敌方影响 | token 只留8个不等于构图只需8个；排序也需要候选邻居 role |
| enemy tokens | 局部表面 clearance<=20m；没有无限广播目标坐标 | Voronoi 邻接即可，不受20m硬限制 | legacy 历史结果不得声称符合 Final 半径局部感知 |
| obstacle tokens/min distance | 局部表面距离过滤；另有已知地图进入 centroid | 全局 nearest obstacle / cell intersection 可进入 | unknown-map 部署是另一个问题 |
| actor forward | 自己打包的 tokens→entity encoder/Transformer→IQN Q 或 actor categorical head | 同形输入可携带不同源信息 | 网络内部没有运行时读取 centralized critic |
| action | 离散AW索引→9个(a,w)→Robot子步动力学 | categorical索引→同AW centers→continuous adapter→Robot | 物理动作值对齐不自动对齐碰撞算法；Final原配置 legacy_end_step，AC synchronized_swept_v1 |
| critic/reward | IQN训练奖励可用全局几何；无执行时 centralized V | central V全状态仅训练，actor仍token输入 | 合法训练特权必须与执行输入区分，不能用它掩盖 preprocessing 的信息依赖 |

CPU 受控反事实已冻结真实 Final IQN / BC 权重：四机(10,10),(10,70),(70,10),(70,70)，目标(42,42)，所有表面距离37–43m。移动这个所有机均看不到的目标：Final 全部 obs 与固定 tau 的 IQN Q 均不变；AC 四机都有 enemy token，BC focal logit 最大变化 .5082。另移动局部传感范围外的已知障碍，Final obstacle token 仍无，但 centroid feature 改变；可见目标正控制 token=1。代码依赖审查支持一般结论，有限场景反事实本身不是完整非干扰证明。

### 最小、明确且可公平执行的当前信息假设

1. 已知静态 geofence/障碍地图；所有机有共同坐标系的定位与自身速度估计。动态未知地图、遮挡感知未获验证。
2. 每个决策周期可获得友军 roster：ID、位置、速度、active、时间戳；active count 可推导。每机独立计算 pursuer-only Voronoi 与 centroid。当前代码以理想同步、无丢包状态供给模拟这项通信；不宣称已实现无线协议。
3. 各机自己的目标位置/速度估计只来自局部20m表面感知。Voronoi 图邻居交换自己的 pursuing/role 位，允许排序与支援判断；不广播敌方坐标。构图前候选角色不可仅限截断后的8个 token。
4. 不需要一个中心节点输出动作，但**需要友军全队运动状态共享**或另行实现且证明等价的分布式邻居发现。现有证据不能用“有限通信半径+仅最近8机”替换这一前提。团队规模、通信量及误差必须在部署验证中报告。
5. 所有算法得到相同地图、友军状态、目标测量和延迟合同；相同 capture/collision/终止/动力学预算；critic/训练 reward 的全局信息另列。

这是当前 Final 代码的简单充分信息合同，尚不是通信比特数下界或部署验证。它满足分散执行与局部敌方感知意图；旧 AC 历史环境不满足。P0 不静默改 legacy 环境后继续沿用原 checkpoint 的“100%”标签。

## 5. 验证与运行状态

- 37项相关 CPU 测试通过，覆盖真正梯度更新、stale likelihood更新前失败、不同目标隔离、重复/未完成support、捕获目标删除、真实8m与角度判据、信息依赖反事实、wrong namespace/NOOP、旧resume拒绝。仅有既存 protobuf 弃用警告。
- GPU1 两回合 stochastic smoke：2/2 capture，2/2 无碰撞捕获；不足以形成性能结论。
- 100回合 GPU1 stochastic eval：2026-09-08 14:11 CST启动，PPO=0；状态以 `artifacts/2026-09-08_p0/bc_sample_formal100/progress.json` 和 `FORMAL_DONE` 为准。启动 manifest 固定 GPU UUID、checkpoint hash 与本轮源文件 hash。最终结果在本页后续状态块更新。
- 原始 large dataset/checkpoint 不入 Git；本轮轻量 runtime/测试/评估证据随提交保存。未运行长训或冻结政策之后的 PPO 分支。

## 6. 下一步只推荐一条 GPU1 实验

**冻结 IQN teacher 与 BC Actor 的 Final 信息合同资格评估**：同一组4v1初始状态，Final pursuer-only Voronoi / 局部敌方感知，统一 corrected collision，AW9物理中心，比较 IQN greedy、BC argmax、BC categorical sample，三策略在 GPU1 串行、零梯度。用每回合 seed+初始状态 fingerprint 验证配对，报告安全 capture 概率/时间与新事件指标，独立标明它是 Final-sensing capture 资格证，不是 mixed 全任务证据。禁止顺手 PPO 或重蒸馏。

理由：先回答旧强 BC 在真正目标信息合同下是否仍强；否则“如何让 PPO 保持好策略”是在维护另一个问题的好策略。若资格通过，再讨论有限预算的修正合同 PPO；不同时推荐第二条训练路线。

P0 当前工程与审计完成，stochastic正式评估按下方状态认定。P1 是上述唯一新评估；P2 是用户确认后的完整任务与物理/通信鲁棒性验证。teacher-KL/TD3/SAC/support重训/MAPPO-VXY继续暂停。


## 7. 最终状态（2026-09-08 14:16:38 CST评估完成）

**P0 所列工程修复、信息审计及 BC stochastic formal100 均完成。** 无训练、无待运行 GPU 任务。

| BC零更新策略 | 捕获 | 碰撞 | 平均回合步数 | dt=.5秒下平均时间 |
| --- | ---: | ---: | ---: | ---: |
| 本轮 sample / eval-forward / 旧AC合同 | 100/100 | 0/100 | 123.78 | 61.89秒 |
| 历史 argmax / 旧AC合同（仅作描述参照） | 100/100 | 3/100 | 71.16 | 35.58秒 |

新发现是：BC 的 categorical sampling 本身也能完成这个 capture benchmark，但任务时间明显不等于 argmax 的时间；不能把“BC stochastic本来就失败”当作PPO失败的既定解释。0/100碰撞不是零风险证明；本轮不对它与历史argmax的安全改善作显著性结论，不将纯capture时间当完整任务时间。

新事件统计：真实区域2+→3+共有205机会、102完成、103 coalition_lost；2+→几何完成100/205，其余105中断。即使回合捕获100%，闭合过程仍有大量反复，说明 success 饱和没有使机制指标失去信息。support→direct为176/284，另18 support_lost、90 target_removed；这些目标移除很多是队友已完成任务，不能一律解读为支援能力差。support→capture-region为82/196。完整事件与左删失标记保存在 gate_report。

- 评估 Actor SHA256：`bb8f971201f6e0a55af3ccb62bcae117f96afa1b07fad2c95c3da1dc552fa35e`。
- PPO updates=0；实际 seeds `2026090301..2026090400`，100条初始状态 fingerprint。
- 运行报告 decision=`STOCHASTIC_EVAL_COMPLETE_REVIEW_REQUIRED`，不是自动PPO批准。报告原有 teacher retention/action-agreement checks 是旧 argmax Gate 的诊断字段，不能作为 stochastic策略通过/失败的标准。
- `visited_2plus/3plus` 是 legacy outer-ring 诊断；解释 closure 一律用 `mission_event_summary`。
- CPU native 入口补充 smoke：两回合各两步，实际初始 seed/fingerprint 与本轮 BC formal100前两回合完全一致；这是对入口修复的集成验证，不是性能样本。formal100 resolved_config另按相同解析路径保存并标注为评估后重解析。
- 当前 ETA：无后台任务，0；**NEXT WAKE-UP：用户确认后，仅执行第6节 Final 信息合同资格评估。** 不自动启动任何训练。
