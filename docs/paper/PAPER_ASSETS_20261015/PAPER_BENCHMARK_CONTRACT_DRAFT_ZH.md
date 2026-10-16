# CoCap 经典实验合同草案（2026-10-15）

**状态：候选合同，尚待论文作者冻结。** 本文把真实 Z05 Stage3 配置和已运行评估协议整理成可复现基线；不选择最终传统围捕算法，不改环境语义，也不把 Global Oracle 当作完全公平的对照。

## 1. 事实源与版本绑定

| 内容 | 事实源 | SHA / 版本 |
|---|---|---|
| CoCap Z05/Z07训练代码 | experiment/iqn-z-unified-decay-dual-curriculum-20260919 | f0c0dfca8a508233f30ddbb5158699400d92a00f |
| Z05独立评估器 | evaluation/iqn-z05-independent-20260923 | 各run report.json.head；正式批次为 5c146f74752073414af416b0db6c686e1c588a07；10月配对批次为 2eeec7e6638640295a8469583500e5d3ad265682 |
| Z05 Stage3配置 | configs/experiments/iqn_z_unified_decay_curriculum_20260919/z05_stage3_12p3e3obs_700k.yaml，继承Stage2、Stage1、common与CR-MS基础配置 | 每份报告另存 resolved_config_hash；具体值见rollout CSV |
| Z05 Stage3 selected | .../z05/stages/stage3/training/checkpoints/step_600000.pt | 8ee5c162c32883984f72aa4be4b86e82338ae1e8d8c912011d181987a476d095 |
| 重复到达严格合同 | evaluation/z05-repeated-arrival-demo-20261006 | formal evaluator commit b308050f4ad99d49180538af4fe0159340f430c2 |
| relaxed casualty诊断 | 同上 | evaluator实现提交 906e2bcfd205d6d0f21072e2eed00a7409b2f532；只作诊断 |
| 10月 Evidence 当前审计checkout | evidence/local-global-20261006 | 6b7aab28546e1a30c60f84d12de3bde70930f670；旧50×3 report未嵌入实际运行source SHA，不能把该checkout SHA倒填成运行来源 |
| 历史 main | /home/yjq/rl/cocap-voradj | a5814f49fa29d869cdc3fb8d8e0df4722aa11f00；仅历史比较 |

每个可发表统计应同时登记：模型checkpoint SHA、训练source commit（若旧产物未记录必须标缺）、原始及resolved配置hash、评估器source commit、seed规则、环境/指标合同版本、报告hash。注册表把未记录项标作 UNRECORDED，不从当前checkout推断。

## 2. 标准 Z05 Stage3 场景

### 2.1 地图、规模和初始化

- 地图为120 m × 120 m，边界[0,120] × [0,120]，启用硬边界；出界是终止/碰撞事件。
- 原生任务为 **12架 pursuer / 3个 evader / 3个障碍物**。Stage1、Stage2原生训练任务分别是4/1/1、8/2/2；它们是curriculum阶段验证，不与Stage3原生数字合并。
- pursuer、evader采用 map_random 初始化；初速度为0；配置的初始最小间距为15 m。障碍半径从1.0–1.1 m区间生成，地图边缘留12 m。
- pure coverage不生成目标，pursuer在中心半径13 m的随机簇内初始化，队内最小间距7 m。Capture与Mixed使用地图随机初始化。
- 一般化评估固定同一Stage3 checkpoint，不重训：16/4/4、20/5/5、24/6/5（pursuer/evader/obstacle）；另外存在独立的18/3/3、24/3/3配对seed扩展批次，须与24/6/5区分。

### 2.2 动力学、动作和感知

- 决策周期 Δt=0.5 s，物理积分 dt=0.05 s，每个决策周期执行10个物理子步。
- pursuer采用离散unicycle AW9动作：纵向加速度 a∈{-0.4,0,+0.4} m/s²，角速度 ω∈{-π/6,0,+π/6} rad/s；最大速度3.0 m/s；drag系数0.133333…；初速度0。动作索引为加速度与角速度的3×3组合，固定midpoint-32 greedy IQN评估。
- evader最大速度3.5 m/s，按 v2_fixed APF动力学更新。Stage3基础配置将goal attraction与goal noise设为0；障碍/边界作用、排斥和速度项仍按配置实现，不能简写成静止或随机游走目标。
- NormSense V2使用free-mask有效面积和活动pursuer数量解析共同表面感知半径：
  R = max(20 m, 0.8715 × sqrt(A_eff / N))。
  目标和障碍按表面距离过滤；配置floor为20 m。已登记native/大规模场景报告没有触发floor；复现实验仍应记录每局有效半径与floor触发状态。global_enemy_visibility=false，目标坐标不会全队广播。
- 友机通信拓扑为free-mask投影后的friendly Voronoi相邻关系。该环境用全队状态计算Voronoi几何，再将有限token输入策略；论文描述局部观测时应说明这一几何依赖。

### 2.3 碰撞、捕获、CE与episode终止

- 碰撞合同为 synchronized_swept_v1：同步推进时按连续扫掠轨迹检查碰撞，不只比较决策周期端点。记录类型至少区分 obstacle、boundary、agent-agent、evader-contact；collision和boundary分别报告，并保留原始事件数。
- **normal capture**：至少3架active pursuer的中心距目标不超过8 m；按环形方位角排序，最大角间隙≤π，且最大角间隙/最小角间隙≤3。条件成立的仿真transition立即移除该目标，不另加normal-capture hold。
- **stationary fallback**：目标速度≤0.2 m/s，至少2架pursuer在捕获距离内连续10个决策周期。它单独报告，不计入normal capture成功；Z05 Stage3原生纯capture 20局为0次fallback。
- **Coverage CE**：对每个active pursuer i，取位置p_i和自身Voronoi cell质心c_i，地图对角线L=sqrt(120²+120²)：
  d_i = sqrt(N_active) × ||p_i-c_i||₂ / L；
  CE_RMS = sqrt((1/N_active) Σ_i d_i²)，CE_max=max_i d_i。strict CE要求活动人数达到合同阈值（Stage3为12）、CE_RMS≤0.05且CE_max≤0.10连续保持30个决策周期。CE衡量Voronoi质心几何均衡，不等于已探测完整地图或搜索成功。
- 标准episode horizon为3000决策步（1500 s）；Stage3全部目标捕获后继续同一策略进行coverage恢复，最多700步（350 s），其间strict CE成功则终止；窗口到期仍未达成则超时。普通episode不自动刷新目标。
- 目标碰撞失效、pursuer碰撞/失活、边界事件、终止状态和horizon均保留独立原因。标准capture、mixed任务不得把stationary fallback、目标碰撞损失或普通捕获混成单一“围捕成功”。

## 3. 目标重新出现：分开报告的三波评估

Repeated-arrival是评估器注入新目标的协议，不是训练时学会的在线任务调度。严格版本使用Stage3 selected、3波×每波3目标、120×120、3障碍、3000总决策步、每次捕获后的700步recovery窗口。各regime使用配对初始world seed和target-refresh RNG；spawn合法性拒绝可能改变最终坐标，**不宣称各regime Wave2/3坐标逐点相同**。

- **Strict PERSIST-A**：等待当前波的strict CE恢复后再放入下一波；恢复窗口到期未恢复则停止，不继续刷新。
- **Strict PERSIST-B**：全队Z归零后触发下一波；Wave1/2 recovery可被新波中断。Wave3 arrival trigger只记事件，不生成Wave4，继续同一episode直到Wave3恢复或终止。
- **Strict PERSIST-C**：全队Z归零后，再按独立delay RNG抽取额外决策步后触发下一波；Wave3之后同样不生成Wave4。
- persistent_service_complete = all_3_captured AND final_recovery_success AND no_disqualifying_failure。原合同中累计碰撞、terminal failure或active-pursuer loss均属于disqualifying failure。all_3_safe_complete对B/C只作诊断，因为前两波recovery会被下一波打断。
- Relaxed A′另改中间recovery timeout策略；Relaxed B/C把episode survival stop改成active_count==0，但strict 12机coverage条件保持12。它们是degraded-team诊断，绝不替代Strict A/B/C主结果。
- 报告每个regime 20局总分母、seed、配对差异、每波结果、全部碰撞和队伍人数；不把10月6日3×3 demo数字并入formal 20×3。

## 4. 指标定义、单位和缺失/删失规则

设E为episode集合，n=|E|，Δt=0.5 s。成功率分母始终是所有开始的episode，失败局不从分母删除。

| 指标 | 统计定义 | 单位 / 优劣方向 | 失败、删失处理 |
|---|---|---|---|
| Normal capture success | Σ_e 1(all targets normal-captured)/n；stationary fallback单列 | 比率，↑ | 目标碰撞、episode终止、horizon或未捕获均算未成功；同时给Wilson 95% CI |
| Capture completion steps/time | 最后一个目标normal capture step减目标出现step；report每局及跨局median/p90 | 决策步、秒，↓ | 未完成局没有观测完成时间；不能填0或从均值中无说明删除。报告成功数/删失数，辅以竞争风险累计发生率与到合同τ的RMST |
| 最大目标等待接近时间 | 每个目标从spawn到min_active ||p_i-p_target||≤8 m的首次时间；episode取该波目标最大值 | 秒，↓ | 一直未进入8 m者在碰撞终止、episode结束、horizon或下一波arrival右删失；按目标逐项保留删失，不能用成功目标最大值冒充完整局最大值 |
| Repeated三波累计围捕成功率 | Σ1(三波全部normal capture)/n | 比率，↑ | 未生成的后续波按regime协议分类为未完成/不可观测；明确stop reason，不把未生成波当成功 |
| Persistent service completion | Σ1(persistent_service_complete)/n | 比率，↑ | 严格按上节布尔公式；所有失败仍在分母。报告完成时间成功局数与删失原因 |
| 三波完成时间 | episode开始至Wave3最后一个目标normal capture；另列最终recovery结束时间 | 秒，↓ | 不完成Wave3 capture的局右删失在terminal/horizon；Wave3 recovery失败保留为persistent service失败 |
| 存活个体数量 | capture结束、每波结束及episode结束的active pursuer数；同时报min/median/p90 | 个体数，↑ | 失活不可逆；不得只在成功局统计 |
| Collision events及类型 | 按swept事件去重计数，分obstacle/boundary/agent-agent/evader-contact；报告每局事件数、至少一次碰撞episode率和每100 team-hour率 | 次、比率，↓ | 以完整episode曝光时长为分母计算暴露率；被碰撞终止仍保留该事件和终止原因 |
| Search dispersion | D_t=sqrt((1/N_active)Σ_i||p_i-p̄||²)，episode统计有效步上的时间均值和p90 | 米；↑表示队伍位置更分散，但无全局单调“越大越好” | 同时报CE、边界/碰撞和active count；失活后只按active队员计算并报告分母 |
| Coverage CE | 每步CE_RMS、CE_max；strict success按N_active阈值与30步hold定义 | 无量纲，CE↓、strict success率↑ | 未完成者在实际episode终止/horizon右删失；post-capture time_to_strict_CE不得从episode总时长代替 |
| 跨规模泛化 | 固定Z05 Stage3 checkpoint，在16/20/24规模及另列18/3/3、24/3/3运行同场景；比较成功率差、时间差与碰撞差 | 比率/秒/次；成功率↑、时间与碰撞↓ | 不重训、不重新挑checkpoint；每个规模独立给分母/CI。24/6/5与24/3/3不是同一测试条件 |

对时间主结果建议给Kaplan–Meier/竞争事件累计发生率及预先固定τ的RMST，至少列出成功n、未成功n和stop reason；PERSIST配对比较按共同seed bootstrap差值及95%区间。旧summary只有complete-case均值时，可作为历史描述，不应冒充删失校正估计。

## 5. 信息公平性与传统baseline冻结要求

现有对照的信息条件并不相同：

- Z05：NormSense V2局部enemy/obstacle token、friendly Voronoi邻接、连续Z状态；不全局广播敌人坐标。
- Local Binary：观测/网络输入关闭连续Z；应以训练配置和输入token实际实现登记后再作为消融结果引用。
- Global Oracle：全局可见性/目标相关信息条件高于局部观测，适合标作信息上界或privileged reference，不默认是完全公平baseline。
- Final is_pursuing：旧布尔role位与pursuing embedding结构，checkpoint维度与Z05不同，属于历史方法/结构对照。
- 传统分层baseline尚未选定。本草案只规定它们必须共享地图、动力学、初始seed、捕获/碰撞判据、horizon和重出现协议；单独登记目标坐标/检测半径、通信拓扑与频率、队伍全局信息、任务分配器、内存、动作限制及调参预算。若信息不匹配，结果标签中明确写matched-local或privileged-global，不做完全公平的优劣因果结论。

## 6. 每次运行的最小登记模板

    experiment_id: <unique-id>
    method_id: <Z05|Z07|LocalBinary|GlobalOracle|baseline-name>
    training_source_sha: <full-commit-or-UNRECORDED>
    checkpoint_path: <server-absolute-path>
    checkpoint_sha256: <64-hex>
    stage_step: <stage, step>
    model_config_sha256: <hash>
    environment_config_sha256: <hash>
    resolved_config_sha256: <hash>
    evaluator_source_sha: <full-commit>
    evaluator_script_sha256: <hash>
    contract_version: cocap-classic-z05-draft-v1
    scenario: <coverage|capture|mixed|persistent>
    pursuers_evaders_obstacles: [12, 3, 3]
    seed_manifest: <path-and-sha256>
    report_json: <path-and-sha256>
    episodes_jsonl: <path-and-sha256>
    event_log: <path-or-embedded-field>
    trajectory_and_gif: <path, seed, render-source>

缺任一来源字段时标为MISSING_ARTIFACT/UNRECORDED，不用当前HEAD补写成历史实验事实。
