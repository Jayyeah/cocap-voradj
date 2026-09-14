# Forward-Final Full-Task Scratch MAPPO — PRE-FLIGHT

**FORMAL TRAINING: HOLD**

本路线只回答：不加载 IQN/BC、不使用 BC dataset、teacher-Q、teacher-KL 或 imitation loss，random-init MAPPO 能否达到当前 BC 的可接受邻域，或出现可持续的 Full-Task 学习趋势。本轮只做配置、审计、随机策略 occupancy、unit/CPU/CUDA smoke。没有正式训练、没有 curriculum、没有 backbone 替换。

## 1. 最新事实与适用范围

2026-09-14 启动时 `git fetch origin experiment/small-step-ac-migration-20260828`；本地与远端均为 `75e336109a88b9f4183f2bb340f1404a0fe579b9`，无须合并。工作区已有 root-cause 台账修改及 P2/P3 新诊断，未覆盖。事实顺序：最新本地 gate → 当前源码 → 最新台账 → 历史结果。

当前未找到独立命名为 MASTER 的总闸文件；权威入口是 [root-cause §17–18](FORWARD_FINAL_PPO_ROOT_CAUSE_20260909_ZH.md) 及其链接的诊断，不能从旧 C3 PASS 推导 MASTER P1/P2/P3 全通过：

- `P1 TRANSITION/RETURN SEMANTICS: PASS`，仅此子项；critic state aliasing 仍是未解除的问题。
- [最新 P2/P3 诊断](FORWARD_FINAL_P2_P3_CRITIC_ADVANTAGE_DIAGNOSTIC_20260914_ZH.md)：`P2 VALUE CALIBRATION: FAIL`、`P3 ADVANTAGE SEMANTICS: INCONCLUSIVE`。该文是接手时已有的本地未跟踪文件，不是本轮 scratch 产物。
- [corrected PPO ledger](FORWARD_FINAL_CORRECTED_PPO_LEDGER_20260909_ZH.md) 与 [transition/return audit](FORWARD_FINAL_TRANSITION_RETURN_AUDIT_20260914_ZH.md) 继续 HOLD，不能据此启动 5k/25k 正式 PPO。
- 本轮用户额外明确只允许 PRE-FLIGHT。因此即使以后外部 gate 更新，本入口也没有正式训练模式。

研究范围首先是 Stage1：120×120、4 pursuers、mixed 1 evader/1 obstacle 与 pure coverage 0 evader/1 obstacle，完整 capture→recovery/coverage。**Full-Task 不等于已覆盖 Stage2 8v2 / Stage3 12v3。**

已读取 [Historical Final ↔ 旧 BC parity](FINAL_IQN_MAIN_MAPPO_BC_CONTRACT_PARITY_20260908_ZH.md)、[Forward bridge C0–C3](FORWARD_FINAL_MAPPO_BRIDGE_LEDGER_20260908_ZH.md)、Historical/Forward YAML、[历史 MAPPO-9-v2 audit §10–11](AC_CTDE_GAP_AUDIT_20260830_ZH.md)、[scratch ledger §18.2](SMALL_STEP_AC_MIGRATION_LEDGER_20260828_ZH.md)，以及当前 MAPPOTrainer/ValueNorm/GAE/CentralValueNetwork/central schema。

历史 MAPPO-9-v2 为 corrected **Pure-Capture**，非本任务：3×400k，best deterministic capture 10%/50%/5%，mean best 21.67%，mean best collision 78.33%；final capture 0%/5%/5%。它证明旧任务有最低限度的 scratch 学习先验，不能证明 Final coverage/support/post-capture 学习已成立，也不能用 best-checkpoint 非零捕获替代持续趋势。

## 2. Canonical Scratch Contract 与 parity

配置：[canonical.yaml](../configs/experiments/forward_final_scratch_mappo_20260914/canonical.yaml)。这是专用 preflight 的严格 schema，**不是 `train.py`/IQN trainer 的输入**；不继承历史 `total_timesteps=2m` 启动逻辑。环境通过配置中的 Forward YAML 及同一 `scene_config` 解析，架构只读取已验证的形状标量。

比较基准明确为当前 `tools/train_forward_final_ppo_20260909.py` 的 **Direct production recipe**：actor LR3e-5、cold geometry V、无 warm-up。3e-6 是后来单独的 LR diagnostic，不把该 probe 宣称成新的 canonical recipe；context/two-head critic 也未通过 gate，不能静默晋升。若 MASTER 将来提升新 recipe，须先更新两路线共同基准并重做 parity。

`SAME` 为有效设置相同；`SCRATCH_REQUIRED` 为取消教师训练/权重依赖的必需变化；`IMPLEMENTATION_ONLY` 为不改变 MDP/PPO 的执行、记录或初始化来源修正；`UNEXPLAINED` 为未解释差异。以下 table 的分类计入机器可读 audit，目标 **UNEXPLAINED=0**；这不等于现有基准所有科学问题已解决。

| 项目 | Forward-Final BC→PPO Direct | Canonical scratch | 分类 |
|---|---|---|---|
| 环境入口 | Forward Stage1 YAML + `scene_config` | 同一入口，禁止旧 AC configure_environment | SAME |
| 地图、规模、spawn、APF | 120×120、4v1/4v0、1obs，Final APF v2_fixed | 不变 | SAME |
| observation | self9、friend8×7、enemy8×7、obstacle5×5；robot-frame、sqrt(N) centroid | 不变 | SAME |
| sensing / topology | own enemy/obstacle20m surface；pursuer-only Voronoi/free-mask；允许邻居role、已知map/全友军构图 | 不变；critic信息不进入actor | SAME |
| 动作/物理 | integer AW9，a±.4、w±π/6，vmax3、drag .4/3、dt.05×10 | 不变 | SAME |
| collision | synchronized_swept_v1 | 不变 | SAME |
| transition revision | terminal-priority-truncation-bootstrap-weighted-ce-v2 | 锁定同一新版本 | SAME |
| direct reward | CR-MS/ring_importance_ms_v0，weight2；旧项0 | 不变 | SAME |
| support reward | .5 approach-only neighbor-visible + .5 CE | 不变 | SAME |
| coverage reward | centroid energy/PBRS scale10，gamma.99；正确的加权reset correction | 不变 | SAME |
| safety/capture bonus | 原安全成本、capture参与者终奖 | 不变 | SAME |
| reward clock | 2m+online，CE speed=.0005 | 仍为2m+online、.0005；只是数值时间锚，不读取teacher年龄资产 | SAME |
| capture / role | R8/K3/角度条件、stationary min2/hold10；K10 release | 不变 | SAME |
| Full-Task结束 | capture后继续；horizon3000、post window500、min active4、CE hold30 | 不变 | SAME |
| episode task cycle | mixed / coverage 逐回合交替，不保证步数各半 | 不变 | SAME |
| native recovery reset | pool1000、capture比例.75、剩余map-random .5；池初始为空 | 同机制；只接受本策略在线capture snapshot | SAME |
| 实际访问/池填充速度 | BC自身轨迹 | random policy自身轨迹；不是人为重采样/预填池 | SCRATCH_REQUIRED |
| actor backbone/head结构 | local Legacy decision Transformer256/8/4 + categorical9 head | 原结构、dropout配置.1；PPO eval模式关闭dropout | SAME |
| actor权重 | C3 BC parent（含复制的IQN backbone） | 全部参数随机初始化；允许常规bias0/LayerNorm1，不加载已训练tensor | SCRATCH_REQUIRED |
| actor可训练性 | Direct中全部参数可训练 | 不变；不冻结backbone | SAME |
| central V | action-free geometry V_i(s)，256/8/4，4/8/5 padding，自身9维 | 同类同schema，fresh；同seed逐位相同initial V | SAME |
| PPO/GAE | rollout256；γ.99/λ.95；clip.2；epochs3/minibatches2 | 不变 | SAME |
| optimizer/entropy | Adam eps1e-5；actor3e-5/critic1e-4；entropy.01/value1/grad.5/targetKL.02 | 不变 | SAME |
| ValueNorm | beta.99999/eps1e-5，var floor.01；raw GAE、normalized V clipping | 不变；全零统计、初始std1 | SAME |
| optimizer初始状态 | Direct cold Adam state={}、update_count0 | 不变；不是加载resume启动新seed | SAME |
| warm-up/teacher loss | Direct无warm-up，PPO无teacher损失，但parent来自BC | 无warm-up；无任何imitation/KL/Q监督 | SAME |
| 资产加载/启动gate | BC factory加载IQN再BC；启动检查C3 parent hash | 直接建网；拒绝IQN执行/旧artifact读取；正式gate独立保持HOLD | SCRATCH_REQUIRED |
| 冻结eval | actor路径仍无条件算teacher Q来记agreement | 同Final环境、AW9与APF，argmax/sample仅算actor；不生成teacher agreement | IMPLEMENTATION_ONLY |
| checkpoint谱系 | 导出包含teacher/BC SHA；完整trainer/RNG/stream | 专用scratch schema/config/seed/完整状态；不伪造teacher parent | IMPLEMENTATION_ONLY |
| preflight预算 | 历史512步等独立smoke | 40个unique training steps、16步rollout，只验证工程；未来仍256 | IMPLEMENTATION_ONLY |
| 未来预算/gate | BC保持性的25k起步gate | 25k评估cadence、100k首learnability审查、最多分段到400k | IMPLEMENTATION_ONLY |

**保留的边界：** Historical Final 的原collision仍是 legacy_end_step；Forward共同修正后，不能声称与历史逐转移相等。9月14日terminal/PBRS修复也改变部分边界reward；旧C3/PPO artifact不得重标为新transition版本。

当前 geometry central schema 缺phase/window/hold/history等context，已有state aliasing证据。Scratch不擅改critic；`UNEXPLAINED=0` 表示可解释地复用该基准，**不是 critic learnability/calibration PASS**。空pool机制相同，但没有capture时 `.75` 不等于75%真实recovery exposure；实现会fallback，须记录实际reset source。

## 3. 实现与工程审计

入口：[preflight_forward_final_scratch_20260914.py](../tools/preflight_forward_final_scratch_20260914.py)。只接受 `occupancy` / `smoke`，无formal、无长训steps、无自动续训。配置 `formal_training` 改成PASS也会被拒绝；解锁正式训练需要后续独立实现和最新gate证据。

- `FinalMissionStream` 复用原生Final reset/scheduler/APF；`collect_transition` 复用当前 inactive masking、reset前 next V 和termination splitting。导入旧模块是代码复用；不调用BC factory/save/eval。
- 直接实例化 `LegacyVorAdjFeatureBackbone` / `CategoricalGridActor`，初始化head gain=.01，整个actor可训练。不能调用 `load_legacy_iqn_state_dict`。critic在单独重设同seed后构造，保留Direct相同初值。
- 执行guard禁止 `CoCapIQN` 构造/load/forward及legacy权重导入，同时禁止访问当前run目录以外的已有artifact。Scratch路径在没有teacher checkpoint、BC actor、dataset、Q bank时成立；resume只读取本run自己写的scratch checkpoint。
- GAE先按旧ValueNorm反归一化values/next_values；true terminal不bootstrap，timeout用reset前状态bootstrap，二者都断开GAE递推；inactive rows不参与Actor、优势统计或V loss。capture本身不误作Full-Task terminal。
- ValueNorm每rollout按active return更新一次，V loss保留旧normalized prediction做clipping。当前实现不是PopArt；不在此改算法。
- actor保持eval模式但autograd开启，概率来自Categorical。零更新log-prob合同由trainer校验；记录整批exact KL，不能把minibatch early-stop当硬trust region。Actor early-stop同时截断critic更新这一已知耦合仍保留。
- checkpoint保留actor/V、两个Adam state、ValueNorm、update_count、env/stream/recovery pool、partial rollout、Python/NumPy/Torch/CUDA RNG；验证后续转移及下一次update逐位一致。

## 4. Random-policy phase exposure：完成

预声明测量：CPU、seed2026091401、**frozen random-init categorical sample**，40个原生episode（20 mixed+20 coverage），无梯度/optimizer，无强制capture或shortened horizon。用原生交替训练stream、空pool及成熟CE系数。该批是单初始化的有限暴露诊断，不是正式实验或学习效果。

标签口径：pre_capture/post_capture/pure_coverage为互斥的action前phase；capture发生的transition仍属pre_capture，其后动作才属post_capture。support是来自实际reward metadata的重叠agent-row角色，不是第4个互斥phase。ring2+/3+为同一目标中心距离≤8m的后继state人数，capture时由event participants补回已移除目标；不是CR-MS outer10.5m环带。3+ hold记录连续decision steps，每步.5秒。当前4v1保证不跨目标拼人数。

每个episode保留capture/normal/stationary、ring step/episode occupancy、3+最长hold、phase steps/active rows、support reward rows、reward正值/非零密度及capture/CE/safety/terminal分量、collision/boundary、CE RMS/area CV、episode length、完整失败/删失、entropy/action histogram、reset source/pool。分母为active-agent rows或env steps；不将agent rows当独立episodes。未访问phase的reward density为null，不写成“奖励恒0”。

完整统计见本节结果与 [occupancy report](../artifacts/2026-09-14_scratch_mappo_preflight/occupancy_seed1/report.json)。纯coverage有学习信号不等于capture后的recovery有暴露；发生大量dense安全惩罚也不等于有任务进展信号。

### 4.1 完整结果与统计边界

CPU耗时336.54秒，40/40原生episode全部结束；**13,973 env steps、55,892 active-agent rows，actor hash始终不变，optimizer updates=0**。这是随机策略观测步数，不是13,973步训练。

| 指标 | Mixed（20局） | Pure coverage（20局） |
|---|---:|---:|
| env steps / active rows | 5,913 / 23,652 | 8,060 / 32,240 |
| capture / normal / stationary | 0 / 0 / 0 | N/A |
| 2+ / 3+ visited episodes | 0 / 0 | N/A |
| 2+ / 3+ step occupancy | 0 / 0 | N/A |
| 3+ max hold | 全部0步 | N/A |
| post-capture action steps | **0** | N/A（不能以pure代替） |
| support active rows | 6,058（25.613%；19/20局出现） | 0 |
| CE / safe completion | 0 / 0 | 0 / 0 |
| collision / boundary episodes | 20 / 19 | 20 / 15 |
| episode length mean / P50 / P90，步 | 295.65 / 194.5 / 590.3 | 403 / 272 / 827.8 |
| terminal CE RMS mean / P50 | .19690 / .17388 | .19540 / .19064 |
| terminal area CV mean / P50 | .29087 / .30566 | .24939 / .23540 |
| categorical entropy，nats | 2.19719959 | 2.19720029 |

全体action前phase占比：pre_capture42.317%、pure_coverage57.683%、post_capture0%。support为重叠角色，占全部agent rows10.839%。40局全部失败，无成功completion time；episode早结束不能解释成任务快。CE/CV是terminal几何诊断，可能已包含失活队形，不等于全队安全coverage质量。

| 实际奖励桶 | rows | 非零密度 | 正值密度 | mean reward |
|---|---:|---:|---:|---:|
| pre_capture | 23,652 | 96.516% | 9.923% | −.83378 |
| support（pre内重叠） | 6,058 | 100% | 20.238% | −.63521 |
| pure_coverage | 32,240 | 100% | .1675% | −1.09453 |
| post_capture | 0 | N/A | N/A | N/A |

pre的capture/coverage/safety分量非零密度分别29.503%/88.293%/3.877%，mean分别−.00188/−.31101/−.52089；pure coverage分量密度99.926%、mean−.52691，safety mean−.56762。capture terminal奖励全批0。奖励并非全面稀疏，但没有真实recovery训练样本，也没有terminal capture正反馈。

AW9索引0..8动作次数为 `[6315,6139,6165,6079,6181,6167,6376,6239,6231]`，各动作10.876%–11.408%；entropy接近ln9=2.197225，未观察到初始化动作塌缩。

实际完成的reset为mixed `environment_default`20、coverage `ordinary_map_random`13/`synthetic_cluster`7；**capture_snapshot=0，pool=0**。底层stream结束最后一局时会预建下一局，因此原始reset_counts有21个environment_default；上表按40个已观测episode计，不把额外reset算样本。

裁决：**PHASE VISITATION BOTTLENECK**。本批随机策略在到达2+/3+ coalition前就已频繁碰撞/触边，完全没有post-capture action；纯coverage fallback和support奖励无法证明真实recovery已可学。不加入curriculum，不预填teacher capture pool，不据此改reward。

统计限制：单一random初始化、20个mixed原生episode不能证明“几乎永远无法捕获”。即使近似独立同分布Bernoulli，0/20的单侧95%上界仍约13.91%；在线reset/RNG与pool依赖意味着该界只是尺度提示。13,973 steps也不是独立Bernoulli试验数。因此“本批严重暴露缺口”有直接证据，“MAPPO经学习后永远到不了recovery”没有证据。

## 5. 未来正式单seed gate（仅设计，未授权启动）

所有预算以**joint environment decision steps**计，不按4个agent乘数；rollout256，checkpoint/eval每25k。不能为迁就25k checkpoint而缩短PPO rollout：可在25000保存168步partial rollout，暂停后原样恢复；训练窗口仍256。训练stream/RNG应在独立eval前后保持完全一致。该生产cadence尚无正式launcher，本轮仅验证partial resume基础。

解锁前置：最新MASTER明确P1/P2/P3 PASS，适用transition/schema与本配置一致，runtime断言及完整resume通过，GPU明确空闲/分配；以后若critic/PPO recipe变动，先更新共同baseline、重做parity和smoke。**当前不满足。**

建议最低可判读预算：先承诺单seed100k训练（seed2026091401），预留有条件的100→200k；25k/50k零capture不能单独判死。只有明确正向趋势才批准200→400k；不预承诺3×400k。

| 时点 | 必看证据 | 决策 |
|---|---|---|
| 全程每update/每checkpoint | finite loss/grad/value；log-prob一致；runtime/teacher guard；entropy、exact KL、clip fraction、EV、ValueNorm及各phase occupancy | 任何非finite、合同漂移或resume污染立即STOP_IMPLEMENTATION；保留checkpoint，诊断后才谈恢复 |
| 25k / 50k / 75k | normal/total/stationary capture、2+/3+、3+ hold、support、collision/boundary、pure CE/CV、时间/失败及phase | 学习曲线观察；零capture不是停止理由，不追加reward/curriculum |
| ~100k首次Gate | 对step0和25/50/75/100k检查下述持续趋势；尤其post exposure与pool来源 | BC邻域成立：候选成功；有partial/full趋势：CONTINUE_TO_200K；几乎无recovery且无前驱进展：PHASE VISITATION BOTTLENECK / HOLD诊断，不宣称MAPPO不可能 |
| 100–200k | 至少3个相邻25k checkpoint的同协议趋势、更新健康、无持续安全恶化 | 达到下述Full-Task趋势才允许200→400k；只有单点capture、stationary投机或only-coverage改进不足以扩400k |
| ~200k | 足够暴露仍无可持续进步；或post始终零暴露、前驱也不再进展 | STOP_THIS_RECIPE / EXPOSURE_LIMITED；不自动加curriculum、不改backbone、不无限延长 |
| ≤400k | 持续趋势/BC邻域的独立种子场景确认 | 单seed通过后才考虑另外2个training seeds；仍失败则封存该recipe结果 |

更新健康前瞻阈值：exact整批KL>.05或clip>.3记WARN；exact KL>.1连续3次update、clip>.5连续3次且KL>.05，或entropy<.1 nats连续3个checkpoint且任务无进展，HOLD做更新诊断。phase EV在target variance不足时报告不可判，不能以总体EV掩盖post calibration；EV<−1连续3个checkpoint且相应phase episode样本≥10，进入critic诊断HOLD。负EV初期/没有post样本本身不宣判scratch失败。

**趋势须前瞻定义，避免挑best：** step0使用同配置random-init，不使用本preflight短smoke作为性能基准。固定eval种子，每checkpoint每scene每mode20个完整episode（共80）；argmax/sample分开，sample为训练策略分布主指标。记录whole-episode paired delta/95% bootstrap及二项区间，paired时间只在共同成功子集报告，另报全体失败率/安全完成CDF。未完成不能记作短completion time。训练occupancy与固定eval occupancy分表，避免reset分布混淆。

以step0作为固定参照，连续3个相邻checkpoint均满足以下方向才叫**持续Full-Task趋势**：

1. mixed链条：normal capture或safe completion较step0至少+10百分点；或2+ visitation +15百分点且3+ visitation +10百分点（20局中的3局/2局增量），同时3+ hold中位数不退化、support→direct/ring参与率不退化。只看总capture而stationary增加不合格。
2. coverage：pure CE成功至少+10百分点；若尚稀疏，则在所有episode上CE RMS和area CV中位数均至少下降15%，同时collision不升；有成功时间后必须同时报告mean/P50/P90及删失。
3. 真正recovery：最近3个eval checkpoint累计≥10个mixed回合含真实post action，报告post CE成功/CE RMS/完成时间方向，且最后一个不低于第一个；mixed与pure collision均不高于step0+5百分点。只有前两项而缺post暴露标PARTIAL_TREND，允许100k之后有限观察到200k，不足以扩400k。
4. 三个checkpoint不能只由一个罕见seed反复贡献：保留episode IDs、配对区间，跨独立确认集方向必须相同。粗20局阈值是成本受限screen，不能宣称统计显著或总体非劣效。

**第一阶段可接受BC邻域：** 在新代码同环境、新paired initial fingerprints下重评冻结当前BC（仅独立baseline evaluator；严禁其数据回流scratch更新）。argmax/sample分别比较：mixed normal/safe、pure CE/safe均≥.90且较对应BC下降≤.10；collision≤.10且较BC增加≤.05；共同safe的mission/recovery mean、P50≤BC1.25倍、P90≤1.50倍，且失败/删失不恶化到上述门槛以外。该邻域比后续精确非劣效目标宽，服务于“scratch是否建立”的第一阶段；**没有超过IQN的要求**。若时间共同成功对<10，则时间结论INCONCLUSIVE，补确认而非算PASS。

历史C3仅作目标锚：BC argmax/sample mixed normal99%、safe98%，pure CE100%；mixed mean117.02/112.98秒、P90163.55/157.9秒。它们是修复前版本的冻结结果；不能直接与新runtime作严格paired gate，须保存新baseline版本/seed/hash。独立baseline中读取BC用于比较，不属于scratch训练依赖。

**何时扩到3 seeds：** 单seed连续3 checkpoints满足BC邻域或上面的Full-Task趋势，且用预留确认seed base2026102401、每scene每mode≥50局复核方向、安全与post暴露；全部健康/语义gate仍PASS。随后增加2个random-init training seeds，recipe/预算/停止规则预先固定，原seed也计入3个。报告每seed final/曲线及失败，不能只平均各自best。未达上述条件不因GPU空闲扩种子。

预算必须另列eval：100k阶段4次×80局=320局，加step0 80局；BC基准另80局，确认另200局，仅在候选成立后追加。eval env steps、wall time/GPU时长单独计账，不能把“100k训练”写成总成本100k。未做正式吞吐benchmark，不按40步smoke速度预测100–400k时长。

## 6. Smoke与交接

执行示例（均为PRE-FLIGHT；CUDA只在确认分配物理卡空闲后使用）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. python tools/preflight_forward_final_scratch_20260914.py --mode occupancy --episodes 40 --output artifacts/<fresh-dir>
CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. python tools/preflight_forward_final_scratch_20260914.py --mode smoke --device cuda:0 --output artifacts/<fresh-dir>
```

39项定向测试（scratch13、P1 transition9、MAPPO-v2 7、P0 10）通过；覆盖random constructor的teacher/load禁用、same/different seed、fresh V/optimizer/ValueNorm、配置漂移拒绝、无teacher eval可复现、GAE/timeout/inactive及既有概率/ValueNorm合同。传统回归测试可能构造其原来的BC测试fixture；scratch命令自身在teacher/artifact fail-closed guard内运行。

首轮CPU smoke在PPO/resume通过后因独立eval的APF host缺`eval`键失败；只修复这个新入口并清除冗余未执行分支，在新目录重跑，原checkpoint与源码快照保留。没有修改共享trainer/环境/现有台账，也没有把失败尝试删除冒充首次成功。

最终版在恢复后的第3次tiny update额外校验finite和两个分支metrics逐位相同；增加2项CLI拒绝formal/25k参数测试后，39/39通过。该断言变更后在新目录再跑CPU/CUDA smoke，均PASS；没有重跑或筛选occupancy样本。

| 最终smoke | CPU | 物理GPU1 / cuda:0 |
|---|---:|---:|
| unique训练交互 / 恢复重放 | 40 / 4 | 40 / 4 |
| 实际逻辑PPO updates | 3（第3次另在clone重放） | 3（第3次另在clone重放） |
| 第1/2次update exact full-batch KL | .00014309 / .00004869 | .00006792 / .00002685 |
| 第1/2次update clip fraction | 0 / 0 | 0 / 0 |
| 第1/2次update EV | .1209 / .3924 | .1283 / .5509 |
| fresh optimizer / ValueNorm | PASS | PASS |
| resume后transition/RNG/optimizer逐位一致 | PASS | PASS |
| argmax/sample × mixed/coverage（各4步） | PASS | PASS |

这些数字只证明小批执行finite；**不能据此判训练稳定、value校准通过或Full-Task可学**。CPU/CUDA采样RNG不同，未要求跨设备轨迹逐位一致。每次smoke为2×16步rollout及恢复后的8步rollout，当前正式recipe仍256。

复跑与失败尝试的成本全部保留：1次CPU失败尝试、2次CPU成功、2次CUDA成功；每次40个unique训练步（另4步恢复验证），总200个unique smoke训练步、20个验证重放步，成功4次eval各16步。失败发生于首个eval动作前。随机occupancy完全独立、无optimizer。所有CUDA都是确认GPU1无compute进程后运行的数秒smoke，没有抢占或终止任何现有正式实验。

最终证据：[parity/provenance audit](../artifacts/2026-09-14_scratch_mappo_preflight/contract_audit.json)、[CPU](../artifacts/2026-09-14_scratch_mappo_preflight/cpu_smoke_final/report.json)、[CUDA](../artifacts/2026-09-14_scratch_mappo_preflight/cuda_smoke_final/report.json)、[tests](../artifacts/2026-09-14_scratch_mappo_preflight/tests.txt)。各run保留launch/source snapshot/report；checkpoint按仓库规则仅本地保留。原occupancy的源码快照早于独立eval host修复；occupancy执行函数没有变化。final report为完成依据，原运行progress曾停留running，交接时仅补完成状态，不改统计。

## 7. 最终七问

1. **工程上具备吗？** 已具备无需teacher的random-init建网、同Final环境rollout、现有MAPPO更新、fresh ValueNorm/Adam、完整resume及双模式eval的基础，CPU/CUDA smoke通过。但尚未具备“科学gate全通过且可正式放行”的条件；geometry V的state aliasing/P2校准和P3仍阻塞，正式长训launcher也有意未启用。
2. **统计上有严重phase bottleneck吗？** 本批有：20 mixed中capture/2+/3+/post均0，真实capture pool为空，只有support和pure coverage的密集信号。标PHASE VISITATION BOTTLENECK；有限样本不能证明学习后的不可达。
3. **最小正式预算？** 解锁后1 seed先100k训练，25k cadence；按趋势条件预留到200k，正向Full-Task趋势成立才到400k。评估另计，不用smoke吞吐估算长训成本。
4. **单seed stop/continue？** 25k/50k零capture不判死；100k看持续前驱/coverage/recovery信号，100–200k确认；工程异常立即HOLD，200k仍无持续学习或始终无post且前驱停滞则停止该recipe，禁止无限延长。
5. **何时3 seeds？** 单seed连续3 checkpoint达BC邻域或持续Full-Task趋势，经独立完整eval确认且全部gate保持PASS，再加2 seed；固定recipe，报告全部seed而非各自best。
6. **成功后能否摆脱IQN？** 在训练依赖意义上可以：actor/V/optimizer都从随机初值开始，只用环境回报；无需IQN权重、BC数据或teacher推理。复用网络结构不构成teacher训练依赖。仍须准确注明结构/任务设计来源，且4v1成功不能外推更大规模。
7. **现在能否正式启动？** **不能。FORMAL TRAINING: HOLD。** P1仅transition子项PASS，P2 FAIL、P3 INCONCLUSIVE，本轮仅PRE-FLIGHT。

<!-- OVERNIGHT_20260914_BEGIN -->

## Overnight bounded diagnostic（最新自动快照）

快照：2026-09-14T23:12:57.211766+08:00。两条均为 `EXPLORATORY_NON_GATE`，用户显式授权的非正式预算例外；不覆盖既有科学裁决。

P1 transition/return：四类修复后 PASS；central-V state aliasing：UNRESOLVED；P2 FAIL/HOLD；P3 INCONCLUSIVE/HOLD；formal PPO / formal Scratch：HOLD。

- scratch: `step0_evaluation`，decision=`PENDING`，step=0；PID 622682，tmux `cocap_overnight_scratch_20260914`；[run](../artifacts/2026-09-14_overnight/scratch_seed1)。
- bc_ppo: `training`，decision=`PENDING`，step=3400；PID 622686，tmux `cocap_overnight_bc_ppo_20260914`；[run](../artifacts/2026-09-14_overnight/bc_ppo_seed1)。

[逐 checkpoint 指标与 matched delta](../artifacts/2026-09-14_overnight/MASTER_SUMMARY.json)；[启动协议及 morning handoff](FORWARD_FINAL_OVERNIGHT_DIAGNOSTIC_20260914_ZH.md)。BC 5k 仅在全部 operational checks 成立时续至 10k；Scratch 100k 硬停，25k/50k 零 capture 不早停。禁止自动 25k PPO、>100k Scratch、200k 或新 seeds。任何正结果只提出下一 Gate，负结果不证明算法不可行。

<!-- OVERNIGHT_20260914_END -->
