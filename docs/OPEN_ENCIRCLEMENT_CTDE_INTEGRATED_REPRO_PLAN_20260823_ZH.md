# 开源连续 CTDE 围捕整合复现计划（2026-08-23）

> 状态：R-1、R0、R1、R2、CPU/CUDA smoke 与六条 L0（3 seeds × 2 algorithms × 150k）均已完成；工程 Gate 通过，但稳定围捕性能 Gate 未通过（2026-08-27 回填）。
>
> 主工程底座：`tinyzqh/light_mappo`。
>
> 原始可运行锚点：`reinshift/MADDPG_Multi_UAV_Roundup` 的环境、MADDPG 与随仓库模型。
>
> 延后扩展：`reinshift/KF_AA_MARL` 的多目标任务、`isRounded` 与 MATD3；进入前必须完成 critic/replay/checkpoint 硬审计。

## 1. 决策与声明边界

本路线不是继续修补先前的 MAADPG 独立重实现，也不把三个开源软件仓库冒充为论文官方复现。三个候选均没有一篇与代码逐项对应、可据此声明“原样复现”的正式围捕论文。

本路线的第一目标是建立一个能够稳定围捕、合同透明、连续动作、CTDE、可多算法复用的轻量基准。第二目标才是在完全相同的环境合同下比较 MADDPG、MAPPO 与当前 CoCap MASAC。第三目标是逐个加入 CoCap 的困难因素，定位正式 Legacy-VorAdj 围捕失败的真实来源。

声明分三层：

1. `Upstream-Raw`：上游原代码和原权重的恢复结果，包含已知缺陷，只回答能否重现仓库演示。
2. `Audit-Corrected`：修正明确实现错误后的独立基准，不再声称与上游原结果严格同合同。
3. `CoCap-Bridge`：从简化基准逐项加入 CoCap 动力学、局部感知、K3、碰撞与多任务机制的研究实验。

未经对应 Gate，不得跨层使用 checkpoint、replay、结果或成功率。

## 2. 三个上游项目的固定角色

### 2.1 `MADDPG_Multi_UAV_Roundup`

仅承担：

- 原始 3-hunter/1-target 三角形围捕环境来源；
- 标准 centralized-Q MADDPG 对照；
- 随仓库模型的 `Upstream-Raw` 可运行性锚点。

已知必须审计的问题：

- 围捕和terminal奖励使用 `rewards[0:2]`，遗漏第三个hunter；
- score同样遗漏第三个hunter；
- 三角形包含使用浮点精确相等 `Sum_S == S4`；
- 速度限幅对完整 `multi_current_vel` 矩阵求norm，而不是逐agent求norm；
- `action_space` 声明无界，但执行前又把动作模长裁至0.04；
- `main.py` 默认 `evaluate=True`；
- target也是学习agent，造成共同训练的对手非平稳性；
- checkpoint只有网络权重，没有optimizer/replay/RNG/runtime。

这些问题在 R0 原样评估中只记录、不修；进入 R1 后才修复。

### 2.2 `light_mappo`

作为主要工程底座，复用：

- MAPPO runner、centralized value critic、shared actor；
- 并行 rollout、GAE、ValueNorm、checkpoint与视频输出；
- 简洁的环境adapter接口；
- 普通CPU/GPU均可运行的轻量依赖。

但其 UAV 环境与 Roundup 高度同源，也继承上述 reward slice、浮点判定和速度norm问题。README中target“idles”的表述与实际“逃离最近hunter+墙壁排斥”脚本矛盾，正式合同以代码审计后冻结的脚本为准。

### 2.3 `KF_AA_MARL`

第一批不纳入主基线。后续只考虑复用：

- 6-hunter/2-target任务；
- target assignment；
- `isRounded`距离+最大逃逸角判据；
- MATD3算法候选。

进入前强制确认：

- twin critics是否真正读取central joint state和joint action；
- actor更新的teammate action是否停止梯度；
- delayed actor update、target smoothing和target network是否正确；
- target done后其他agent的transition/done/bootstrap语义；
- replay采样单位；
- checkpoint是否包含optimizer/replay/RNG/runtime。

若任一项无法确认，它只能作为环境/几何donor，不能作为MATD3复现证据。

## 3. 冻结的复现对象

### 3.1 `roundup_raw_v0`

完全保持上游原代码：

- 3 hunters + 1 learned target；
- 2×2场地；
- 3个静态圆障碍；
- world-frame连续 `[a_x,a_y]`；
- 100-step episode；
- 原始obs、reward、碰撞和三角形判定；
- 原始提供权重。

该合同只用于 R0，不允许生成后续研究结论。

### 3.2 `corrected_roundup_v1`

第一版可信基准固定为：

- 3 learned hunters + 1 scripted target；
- target脚本固定为 `light_mappo` 的“逃离最近hunter+墙壁排斥”，不学习；
- 2×2场地、3个静态圆障碍；
- decision `dt=0.5`；
- hunter `v_max=0.1`、`|a|<=0.04`；
- target `v_max=0.12`、`|a|<=0.05`；
- hunter连续动作 `[a_x,a_y]`；
- 每个hunter 26维观测：self位置/速度、两名队友位置、16线LiDAR、目标距离/方位；
- Actor始终知道目标，因此标记为 `full-target-information`，不得称为Legacy局部感知；
- episode上限100 steps；
- success：目标位于三hunter凸包/三角形内，且三个hunter到目标距离均不超过0.3；
- 第一版success为单step instantaneous，不加入CoCap hold；
- 修正reward覆盖全部3个hunter；
- 第一版保留原任务级碰撞响应，不偷偷换成CoCap swept collision；后续碰撞语义必须另开版本。

几何判定必须使用数值稳定的barycentric/凸包判断或带尺度容差的面积判断。禁止继续使用浮点精确相等。

## 4. 预期阶段与Gate

### R-1：上游获取、许可与不可变快照

任务：

- 记录三个仓库URL、license、branch、commit SHA、抓取日期；
- 将上游原代码放入独立vendor/snapshot目录，不直接在快照内改代码；
- 保存原始文件SHA256；
- 建立 `upstream_manifest.json`；
- 确认原始模型文件、依赖版本、训练入口和评估入口；
- 检查是否存在Git LFS、缺失资产、错误相对路径或隐藏下载。

Gate：快照可校验，上游license保留，任何本地修复只发生在adapter/corrected副本。

### R0：原始Roundup权重恢复

任务：

- 使用上游声明依赖加载原始模型；
- 不修任何环境bug；
- 固定seed分别做20 deterministic和20带原探索噪声rollout；
- 保存成功率、episode length、三角形包含、三机距离、collision和动作范围；
- 保存最多5个代表性GIF；
- 检查原始模型是否同时包含3个hunter和learned target。

Gate：模型确实加载、动作/观测shape一致、rollout可结束、结果可在同seed重跑。R0成功率只作为仓库恢复记录。

### R1：`corrected_roundup_v1`实现

只允许以下明确修复：

1. `rewards[0:2]`与score改为全部三个hunter；
2. 每个agent独立计算速度norm与限幅；
3. action space和真实执行边界统一；
4. 三角形/凸包判断改为数值稳定实现；
5. 固定target脚本、seed和reset随机源；
6. 明确terminated/truncated；
7. 增加正式metrics和artifact manifest；
8. 增加模型-only milestone和唯一rolling full bundle。

强制单测：

- 目标在三角形内/边上/外部；
- 退化/共线三角形；
- 三个hunter均得到stage和terminal reward；
- 逐agent速度限幅互不干扰；
- 声明动作与实际动作逐元素一致；
- scripted target方向和边界排斥；
- fixed seed reset/rollout可复跑；
- timeout只truncated，capture按合同terminated；
- oracle/几何控制器在简化无障碍场景能稳定成功。

Gate：全部测试、`py_compile`、`git diff --check`通过；oracle成功；random policy不得伪造高成功；32-step CPU smoke finite。

### R2：同环境双算法基线准备

#### R2-A：MAPPO（主基线）

冻结配置起点：

- shared actor；
- centralized `V(concat(o_1,o_2,o_3))`；
- hidden=64、layer=1；
- actor/critic LR=`5e-4`；
- PPO epoch=15；
- clip=`0.2`；
- entropy coef=`0.01`；
- gamma=`0.99`；
- GAE lambda=`0.95`；
- ValueNorm；
- Huber value loss；
- 明确记录实际max grad norm，不沿用help文字中的旧值；
- 8 rollout envs；
- 第一阶段150k env steps。

#### R2-B：MADDPG（算法对照）

冻结配置起点：

- 只训练三个hunter，target使用与MAPPO相同脚本；
- actor读取各自26维obs；
- 每个hunter一个central critic，读取joint state和三个hunter joint action；
- actor/critic两层128；
- actor LR=`1e-4`；
- critic LR=`3e-3`作为upstream起点，但必须显式记录scheduler；
- gamma=`0.99`；
- tau=`0.01`；
- replay=`1M`；
- batch=`256`；
- 第一阶段与MAPPO matched 150k env steps；
- 后续是否扩至500k由150k Gate决定。

共同检查：

- actor只读取本地26维obs；
- MAPPO critic读取联合obs；
- MADDPG critic读取联合obs和联合action；
- reward、target脚本、spawn、episode与success完全相同；
- deterministic eval关闭探索；
- 32-step CPU和约2k-step短CUDA smoke finite；
- checkpoint resume后step、optimizer、normalizer/replay和RNG按各算法能力连续；若上游框架缺项，先补齐再长训。

Gate：短smoke无NaN/OOM、环境吞吐稳定、checkpoint可加载、固定seed eval可复跑。完成此Gate后才允许进入长训。

### L0：MAPPO/MADDPG正式长训

这是新agent本轮必须实际启动、但不应持续等待其完成的阶段。

正式计划：

- 两算法各3个seed；
- 先到150k env steps；
- 每25k保存model-only checkpoint并做20 deterministic rollout；
- 50/100/150k保存重点曲线；
- rolling full bundle只保留最新一个；
- 150k保存唯一final full bundle；
- 每个最终checkpoint补20 stochastic/reduced-noise rollout和5个代表性GIF。

至少记录：

- distinct capture episodes；
- deterministic/stochastic capture rate；
- triangle/hull containment fraction；
- 三个hunter同时进入capture radius的fraction和hold；
- largest angular gap、pairwise angle separation；
- collision及类型；
- minimum/mean target distance和progress；
- action saturation、速度、episode length；
- MAPPO value health；
- MADDPG Q/TD/actor/critic gradient health。

150k Gate：

- PASS：三seed中至少2个deterministic capture rate不低于40%，且三seed中位数不低于50%，collision rate不高于20%；
- STRONG PASS：中位deterministic capture rate不低于80%；
- PARTIAL：重复出现真实三机闭合/hull containment，但成功率未过PASS；
- FAIL：仍主要是单向追逐、collision或没有重复三机几何。

分支：

- MAPPO PASS、MADDPG FAIL：MAPPO成为轻量主基线，先审计MADDPG优化而非调环境；
- 两者均PASS：保留两条，优先使用更稳定/更高成功者作为后续bridge anchor；
- MADDPG PASS、MAPPO FAIL：核查MAPPO centralized state、reward/advantage scale与on-policy预算；
- 两者均FAIL：停止扩步，回到环境/reward/geometry/collision audit，不移植CoCap复杂合同。

### R3/L1：CoCap MASAC同合同算法对照

在 `corrected_roundup_v1` 不变的前提下接入当前MASAC：

- 保持3H1E、脚本target、full-target-information、world `[a_x,a_y]`、100-step、相同reward和success；
- 只把算法换为shared stochastic Actor + centralized twin-Q MASAC；
- 禁止引入Legacy-VorAdj、K10、support/coverage、APF、`(a,w)`和post-capture。

目的：区分“MASAC实现/优化问题”和“CoCap任务合同过难”。训练预算、seed和eval与L0 matched。

### R4：逐项桥接回CoCap

只有至少一个L0基线稳定PASS后启动。每次只改变一个主变量：

1. `3H/all-three triangle → 4P/K3 ring`，仍保持full target和`[a_x,a_y]`；
2. scripted target → 当前moving APF evader；
3. world `[a_x,a_y]` → body-frame `(a,w)`；
4. full target information → Legacy-VorAdj local sensing；
5. 简化碰撞 → synchronized swept/substep collision；
6. 加入direct/informed/uninformed信用分级；
7. capture稳定后才恢复post-capture coverage。

每一步使用上一层已通过合同的matched baseline，并重新做scratch线；warm-start只能作为另列诊断，不能替代scratch因果对照。

### R5：正式论文方法复现

当 `corrected_roundup_v1` 至少有一个高成功CTDE anchor后，优先选择 PMLR/ACML 的 *Faster Target Encirclement with Utilization of Obstacles via Multi-Agent Reinforcement Learning*：

- 先实现contributing-angle reward；
- 再实现lion-inspired two-stage encirclement；
- matched MADDPG为论文算法对照；
- 每个新增机制独立ablation；
- 不同时加入CoCap local sensing或`(a,w)`。

这一阶段才可以写“对公开论文方法的独立复现尝试”；仍需声明没有作者代码和权重。

### R6：KF_AA多目标扩展

仅在完成MATD3硬审计后启动。先在 `corrected_roundup_v1` 做3H1E MATD3算法校验，再扩展到6H2E；禁止把算法切换和多目标任务切换合并成一个实验。

## 5. 与现有CoCap主线的隔离

- 新建独立worktree与branch，不覆盖 `ablation/all-agent-oldmix-20260810`；
- 独立config、artifact、tmux与run name；
- 不读取当前CF3/B0/B1/PC0 replay；
- 不warm-start当前CoCap Actor；
- 不改当前CoCap reward、动作空间、K10或碰撞默认合同；
- R3以前不得把Legacy-VorAdj代码路径混入上游基准。

## 6. Artifact与存储合同

每个run保留：

- resolved config与upstream/local commit；
- manifest、seed、依赖版本和命令；
- metrics/episodes JSONL；
- 25k model-only milestones；
- 唯一rolling full bundle；
- final model与唯一final full bundle；
- final 20-rollout统计和最多5个代表性GIF。

不保留：

- 每个checkpoint的完整replay副本；
- 已由final full bundle取代的rolling full副本；
- 重复、随机且无诊断价值的大量GIF；
- 已被正式线取代的smoke/pilot replay；
- vendor仓库的重复clone和构建缓存。

删除任何训练产物前，必须确认正式进程退出、final full可加载，并记录精确路径与释放空间。

## 7. 上一次MAADPG尝试的归档边界

上一轮独立worktree：`/home/yjq/rl/CoCap1/cocap-voradj-maadpg`。

保留：

- 全部代码和Git历史；
- `docs/maadpg_reproduction_20260823/`合同、歧义、假设与审计文档；
- 六条正式线的`final-model.pt`、`final-full.pt`、50/100/150k model-only milestones；
- manifest、episodes、diagnostics、stdout和run ledger；
- pilot/preflight的文档与轻量metrics。

2026-08-23已清理：

- 六条已完成正式线的`rolling-full.pt`；
- 两条25k pilot的`rolling-full.pt`；
- 两条被取代formal-preflight的`rolling-full.pt`；
- 一条signal-preflight的`rolling-full.pt`。

清理前约1.8 GiB，清理后约902 MiB，释放约0.9 GiB。删除内容是不可恢复的未跟踪artifact，但所有正式线仍保留完整`final-full.pt`和推理模型，因此不影响最终分析或从最终状态恢复。上一轮结果只能作为“论文信息不完整下的独立MAADPG负结果”，不得warm-start或污染本路线。

## 8. 新agent本轮停止条件

新agent应完成R-1、R0、R1、R2代码/配置/测试与短smoke，随后实际启动L0正式长训。只有在确认长训进程、恢复资产、指标与资源均无异常，后续唯一事项只是等待step增长时，才允许结束会话；不得为了等待25k或150k持续占用会话。

当下面全部满足时结束：

1. 上游来源、commit、license与模型清单已冻结；
2. 原始模型能够加载，或缺失/不兼容证据已完整记录；
3. `corrected_roundup_v1`实现与强制测试通过；
4. MAPPO/MADDPG共享同一环境合同；
5. 32-step CPU smoke和短CUDA smoke finite；
6. checkpoint/resume/eval合同通过；
7. L0三seed配置、supervisor和启动命令准备完成；
8. 至少启动当前资源允许并与既定GPU调度一致的L0正式run，未启动的seed必须已挂入可靠的自动调度；
9. 启动后观察足够的真实env steps，确认step/update持续增长、metrics finite、GPU/VRAM/RAM正常、replay/normalizer按算法合同工作、rolling full可完整写入并可加载；
10. 文档、测试、`git diff --check`通过并commit；
11. 报告实际PID/tmux/GPU、当前step、实测吞吐、首个25k与150k ETA。此后若只剩训练增长即可结束，不tail、不轮询、不等待checkpoint。

## 9. 给执行agent的启动提示词

下面提示词与本文件共同构成新agent合同；用户可直接复制使用。

```text
你现在接手 CoCap-VorAdj 的开源连续 CTDE 围捕整合复现任务。

工作仓库：/home/yjq/rl/CoCap1/cocap-voradj
当前参考worktree：/home/yjq/rl/CoCap1/cocap-voradj-allagent-oldmix
第一必读文档：docs/OPEN_ENCIRCLEMENT_CTDE_INTEGRATED_REPRO_PLAN_20260823_ZH.md
历史失败边界：docs/MAADPG_REPRODUCTION_PRINCIPLES_AND_START_PROMPT_20260823_ZH.md，以及独立worktree /home/yjq/rl/CoCap1/cocap-voradj-maadpg 中的 docs/maadpg_reproduction_20260823/。

目标：建立一个可信、轻量、可稳定围捕的连续动作CTDE基准。使用 tinyzqh/light_mappo 作为主工程底座；使用 reinshift/MADDPG_Multi_UAV_Roundup 的原始环境、MADDPG和随仓库模型作为raw锚点；KF_AA_MARL只作为后续多目标/MATD3 donor，本轮不直接长训。

必须先建立事实快照：Git/worktree/branch、GPU/进程/RAM/磁盘、三个上游仓库的URL/commit/license/模型/入口/依赖。新建独立worktree、branch、artifact，禁止污染正在使用的CoCap分支和训练资产。不要停止其他项目进程。

如果需要使用子agent，只能使用 `gpt-5.6-luna` 且 `reasoning_effort=max`。仅委派边界清晰、可并行验证的子任务；主agent必须亲自读取合同、负责关键语义判断、集成、测试与最终验收，不能把整项复现或核心审计整体外包。

执行顺序：

R-1：冻结三个上游仓库快照与SHA，保留license，原始snapshot只读，本地修复放adapter/corrected目录。

R0：原样加载MADDPG Roundup提供模型，不修上游bug；固定seed完成20 deterministic与20原噪声rollout（若很快），记录capture、三角形、三机距离、collision、episode length、动作边界，最多5 GIF。R0只作为软件恢复记录。

R1：实现corrected_roundup_v1。固定3 learned hunters+1 scripted target、2×2、3静态障碍、world [ax,ay]、dt=.5、100-step、26维hunter obs、full target information、三机凸包内且三机距目标<=.3即成功。只修：reward/score覆盖3个hunter、逐agent速度norm、action声明与执行一致、稳定凸包判定、固定target脚本和seed、terminated/truncated、metrics/manifest/checkpoint。第一版不加入CoCap K10、Legacy-VorAdj、APF、(a,w)、support/coverage、swept collision或hold。

强制测试：三角形内/边/外/退化；三hunter reward；逐agent限速；实际动作一致；target脚本；seed复现；terminated/truncated；oracle在简化场景稳定成功。运行py_compile、聚焦pytest、git diff --check。

R2：在同一个corrected_roundup_v1上准备两条算法。
- MAPPO：shared actor、central V、hidden64/layer1、lr=critic_lr=5e-4、PPO15、clip.2、entropy.01、gamma.99、GAE.95、ValueNorm、Huber、8 env、150k计划。
- MADDPG：只训练3 hunters、相同scripted target、local actor+joint state/action critic、128x2、actor lr1e-4、critic lr3e-3及明确scheduler、gamma.99、tau.01、replay1M、batch256、matched 150k计划。

两条都必须完成32-step CPU smoke和约2k短CUDA smoke，验证finite、CTDE输入、executed action/replay一致、deterministic eval、checkpoint/resume连续。准备3-seed L0 configs、每25k model-only checkpoint、唯一rolling full、150k final full、20-rollout+5GIF evaluator与supervisor。

完成全部前置Gate后必须实际启动L0正式长训：按可用GPU安全启动当前可并行的run，并把其余seed接入可靠的自动调度；不得与其他项目争抢GPU或造成OOM。启动后观察足够的真实env steps，确认PID/tmux、step/update增长、metrics finite、GPU/VRAM/RAM、replay/normalizer以及rolling full写入和加载均正常。只有当训练已经稳定运行、未启动seed已有自动调度且后续唯一事项只是等待step增长时，才结束会话。不要等待25k/150k，不持续tail或轮询。报告：新worktree/branch、upstream commit、R0结果、修复清单、测试、smoke、存储预算、实际PID/tmux/GPU、当前step、实测steps/hour、25k/150k绝对时间ETA和自动调度状态。

关键边界：前三个开源项目没有对应正式围捕论文，不得声称论文原样复现；previous MAADPG final/replay不得warm-start；任何环境/奖励/目标策略变化必须新spec/version；不要根据演示GIF代替多seed正式rollout。

完成后更新本复现文档、commit代码和文档；若网络/依赖/GPU阻塞，进行有限次安全排查并保留证据，无法解决时停止，不要无限重试、tail或等待。
```

## 10. 本轮执行进度（2026-08-23）

权威执行记录位于 `docs/open_encirclement_ctde/EXECUTION_LOG_20260823_ZH.md`，逐文件上游来源与 SHA256 位于 `docs/open_encirclement_ctde/upstream_manifest.json`。本节只记录 Gate 状态，不改变前文冻结合同。

- R-1：PASS。三个 MIT 上游已按固定 commit 导出到 `vendor/upstream/<repo>/<sha>/`，clone 未发现 Git LFS 文件；Roundup 随仓库 16 个模型文件均已冻结。
- R0：PASS。原权重包含 3 hunter + 1 learned target，20 deterministic capture rate 为 60%，20 原探索噪声 capture rate 为 15%；raw bug 全部保留，仅作为软件恢复记录。
- R1：PASS。`corrected_roundup_v1`、稳定三角形判定、三 hunter reward、逐 agent 限速、有界动作、scripted target、seed、terminated/truncated、metrics/state 与 moving-target oracle 已实现；oracle 在无障碍 10 seeds 为 10/10。
- R2：PASS。MAPPO actor 为 local 26、central V 为 joint 78；MADDPG 三个 actor 为 local 26、三个 critic 为 joint 78+6。完整 bundle 覆盖 optimizer、ValueNorm/replay、RNG、runtime 与环境状态。
- Smoke：PASS。聚焦 pytest 18 项通过；MAPPO/MADDPG 32-step CPU smoke 各完成 1 次真实更新；CUDA 2k 分别完成 10/98 updates，metrics finite。MAPPO 已从 600-step rolling full 实际恢复到 2k，MADDPG 的含 replay full bundle 也已实际加载。
- L0：COMPLETE / PERFORMANCE FAIL。六条 150k 于 2026-08-23 12:26:19 +08:00 全部完成，supervisor 6/6 completed、0 failed。MAPPO 三 seed 的 deterministic/stochastic capture 均为 2/60（3.33%），平均碰撞约 9%；MADDPG 分别为 1/60（1.67%）和 0/60，平均碰撞约 63%。训练/评估/checkpoint/recovery 工程合同通过，但没有形成稳定围捕；逐线和里程碑证据见 `docs/open_encirclement_ctde/L0_FINAL_RESULTS_20260823_ZH.md`。
