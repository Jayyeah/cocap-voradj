# RA-L论文实验准备状态（2026-10-15）

## 状态枚举

- **READY**：报告、checkpoint、seed/配置和评估版本关联足以按现有口径引用。
- **MISSING_ARTIFACT**：主体实验可能已发生，但必要来源、配置、报告或原始可视化资产缺失。
- **NEED_EVALUATION**：需要按已修正评估器/冻结指标完成新的评估；本轮不启动。
- **NEED_DECISION**：运行前仍需作者决定方法、对照的信息条件或论文主张。

## 逐项清单

| 实验资产 | 状态 | 当前有据事实 | 缺口 / 论文使用边界 |
|---|---|---|---|
| Z05 Stage1/2/3 selected checkpoint | **READY** | 3个selected二进制均已读入校验：checkpoint SHA与selection report一致；PyTorch weights_only安全读取；按匹配源码实例化CoCapIQN后strict state_dict加载通过。stage、step、配置和字节数见checkpoint CSV。 | 仅使用Stage3 step600000作为本文最终Z05 policy。Stage1/2用于curriculum阶段/基线表，不替代Stage3。 |
| Z05 12/3/3原生评估 | **READY** | Stage1、Stage2、Stage3各20×Coverage/Capture/Mixed；每批60 episode；report、episodes.jsonl、mission events、selected seed GIF及逐帧source JSON存在。 | Coverage CE是几何指标，不可写成“地图搜索完成”。时间和比例请保留原分母、删失和collision字段。 |
| Z05 Stage3固定策略规模泛化：4/1/1、8/2/2、16/4/4、20/5/5、24/6/5 | **READY** | 4与8为downward测试；16/20/24各20×3场景；checkpoint均为Stage3 8ee5…d095；配置/resolved hash和evaluator SHA写在报告。 | 没有为这些规模重训；严格称zero-shot/cross-scale evaluation。24/6/5不要与另一个24/3/3扩展合并。 |
| Z05额外paired-seed与跨规模：12/3/3、18/3/3、24/3/3 | **READY** | 10月6日source commit 2eeec7e6638640295a8469583500e5d3ad265682；12/3/3 20×3有9 GIF；18/3/3 20×3无GIF；24/3/3由4个5×3 shard组成20×3，无GIF。 | 这是不同目标密度与seed家族，单独标表/附录；不可用作24/6/5结果替代。 |
| Z07 Stage3 native对照 | **READY** | 20×3 native report；source SHA 2eeec7e6638640295a8469583500e5d3ad265682；selected cp 0180…3fd3；三场景各一份固定episode-index-0 GIF。 | 用作Z07版本对照。输入Z/Z-policy配置不同，不能把它写成只改变单一变量的因果消融。 |
| Z07 Stage1/2 native对照 | **MISSING_ARTIFACT** | selected checkpoint本体存在。 | 本次指定输出树中未找到对应Stage1/2 native逐局报告；不能仅由selection report推断独立评估分数。 |
| Strict repeated arrival PERSIST-A/B/C | **READY** | 20×3=60局，checkpoint SHA匹配Z05 Stage3；60/60 contract diagnostics pass；60 trajectories、60 event JSON、15 GIF；verification.json核了selected GIF decode、seed与轨迹hash。Persistent service分别9/20、13/20、14/20。 | 结论应写“观察到完成能力，但未证明稳定/可靠”；A/B/C是不同到达策略，不能据当前单一policy/seed线做因果机制结论。 |
| Relaxed A′/B/C casualty tolerance | **READY** | 20×3=60局；60 trajectories、60 event JSON、15 GIF；严格12机CE门槛保持12，eval survival stop改动明确记录。 | 补充degraded-team诊断，不替换strict结果。A′还更改timeout-continue语义。 |
| 3×3 repeated-arrival demo | **READY（演示）** | 9局、9 trajectories、9 event logs、3 regime GIF。 | 每组仅3局；GIF未与具体episode seed在summary中绑定。不可作为formal rate或已知seed案例。 |
| Evidence旧held-out 50×3三臂输出 | **NEED_EVALUATION** | 实际发现Z05/Local Binary/Global Oracle三个report，均status=complete，每臂150条记录，合计450。三臂seed集合与每个scene内initial-state fingerprint完全配对；checkpoint SHA与selected checkpoint匹配；optimizer/replay update=0、checkpoint mutation=false。 | 它们在旧目录final_heldout_50，而不是corrected目录final_heldout_50_corrected；Coverage time_to_ce_seconds对成功局仍全空，三臂run source SHA未写入report，独立episodes.jsonl/事件文件/trajectory/GIF也缺失。可作为有来源限制的provisional端点，不作为corrected final close-out。 |
| Evidence selected训练来源 | **MISSING_ARTIFACT** | Local Binary、Global Oracle Stage1–3 checkpoint都存在且直接安全加载；当前审计checkout是6b7a…f670。 | checkpoint/status/selection/report没有记录真正训练source SHA；6b7是审计checkout，不是运行commit证明。不能用它冒充训练来源。 |
| Final is_pursuing历史release | **MISSING_ARTIFACT（评估）** | 三个发布checkpoint均存在，manifest SHA一致，weights_only读取与严格加载通过；结构含pursuing embedding。 | release manifest标明不含rollout/GIF，也未给原始config/source commit；报告和trajectory缺失。仅历史checkpoint对照，非本文Z05主策略。 |
| Local/Global公平对照判定 | **NEED_DECISION** | Local Binary禁用连续Z；Global Oracle使用privileged/global信息条件。 | 决定论文中以matched-local消融还是privileged upper-bound呈现。不得默认全对照同信息。 |
| 传统分层围捕baseline算法 | **NEED_DECISION** | 可复用本报告冻结的地图、动力学、capture、碰撞、seed与horizon合同。 | 此轮不选算法。选择后还需登记其信息、通信、任务分配、动作与调参预算。 |

## Evidence corrected held-out 50×3缺口与资源

服务器上确实有450条已完成记录，因此无需重复“旧schema”这一批。原始报告位置：

- /home/yjq/rl/CoCap1/iqn-evidence-comparison-20261006-runtime/final_heldout_50/z05/report.json
- /home/yjq/rl/CoCap1/iqn-evidence-comparison-20261006-runtime/final_heldout_50/local_binary/report.json
- /home/yjq/rl/CoCap1/iqn-evidence-comparison-20261006-runtime/final_heldout_50/global_oracle/report.json

Old report的Coverage CE time为空；旧episode总时长包含strict CE后post-capture grace，不能作为first strict-CE完成时间的替代。corrected evaluator会记录实际strict CE success transition，所以若需要该指标，需执行独立的corrected 50×3 final-only评估。当前预期输出路径final_heldout_50_corrected不存在。精确参考命令见同目录PAPER_VISUALIZATION_TOOLKIT_ZH.md的2.3节；本轮未执行。

旧报告记录每个arm采用2个CPU worker、GPU训练仍活动。已观测每臂elapsed约：Z05 3174 s（53 min）、Local Binary 4986 s（83 min）、Global Oracle 1387 s（23 min）。若串行执行类似任务，约159分钟加启动/审计开销；资源充足且并行时墙钟时间不能按该和数保证。建议沿用报告的CPU-only inference、2 workers/arm和线程限制。是否使用训练仍活动的服务器资源，应运行前重新看resource gate。

## 论文图表落点建议

| 论文单元 | 可直接使用的现有资产 | 目前结论 |
|---|---|---|
| Table I：selected模型/checkpoint/配置 | PAPER_CHECKPOINT_REGISTRY.csv | 可直接审稿；证据变体source provenance缺口需脚注 |
| Table II：native coverage/capture/mixed | Z05 Stage1/2/3各report.json；Z07 Stage3 report作历史对照 | Z05 Stage3作为主要行；Z07只作为不同策略版本结果 |
| Table III/Fig.：跨规模成功率与延迟 | Z05 16/20/24标准扩展；附录列18/3/3与24/3/3 | 标准三组与追加两组分表，不拼成一条规模曲线 |
| Fig.：典型coverage/capture/mixed轨迹 | rollout CSV中按seed规则选择的Z05 Stage3 native GIF；另外可用一个G3 mixed碰撞失败案例 | 只能标注具体seed和checkpoint；不表示平均行为 |
| Table/Fig.：三波持续任务 | Strict A/B/C 20×3 summary与逐局trajectory/event | 可写为completed evaluation；成功率45–70%，目前不能称可靠能力 |
| Supplement：degraded-team | Relaxed A′/B/C report、15 GIF与逐局event | 明确它更改了eval survival/timeout语义 |
| Evidence comparison表 | 现有450条old-schema report可作provisional表 | Corrected timing、run source SHA与trajectory/GIF仍缺；不进入最终主结论表，直到修复闭合 |

当前输出目录不含大型checkpoint/trajectory/GIF副本；CSV只记录服务器绝对路径、大小和hash/seed关联。

## 后续最值得补的三项实验

1. **优先：corrected paired held-out 50×3 close-out。** 复用已确认的三个Stage3 checkpoint和同一seed manifest；只补齐corrected evaluator source SHA、CE transition time、paired report和新output root。预计450个evaluation episode；不训练。
2. **冻结信息条件后的传统分层围捕baseline。** 先由作者决定算法，再使用同地图/动力学/种子/normal capture/swept collision/horizon合同；同时登记该方法的local/global sensing、通信、assignment和调参预算。此处不预选算法。
3. **Z05与Local Binary的独立训练seed复现及机制指标。** 现有paired Evidence为单checkpoint/representation；无法确认Z优势跨训练初始化是否稳定。新训练前先定预算，并增加first-target detection time、目标等待接近时间、service rate和CI，避免把CE几何成功直接解释成搜索增益。

本文件是状态登记，不启动以上任一任务。
