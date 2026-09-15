# Forward-Final Corrected MAPPO 单任务因果分解（2026-09-15）

## 授权、来源与边界

用户在本轮明确授权两条独立 random-init 单 seed：GPU0 Pure-Capture 500k；GPU1 Pure-Coverage 200k，按条件自动执行 attribution probes 与最多一次 R2 200k。该授权覆盖旧 preflight / overnight / AGENTS 的历史预算限制；不启动 Full-Task PPO，不重开旧算法线，不启动 R3。两GPU训练不是代理委派。

启动先 `git fetch origin`、`git merge --ff-only origin/experiment/small-step-ac-migration-20260828`：Already up to date。共同基线 HEAD=`28cc8ba`。最近提交为 overnight结果/历史比较 `28cc8ba,488c69f,cbc0591`、overnight runner `08b2506`、scratch preflight `8f6f4f3`、四项修复 `75e3361`。完整读取 overnight results、Scratch preflight、PPO root-cause、transition-return audit、canonical scratch YAML 与历史 MAPPO-9-v2 config/result；代码/runtime/artifact优先。原未跟踪文件未纳入本实验。

历史 Full-Task Scratch100k无capture/CE成功，仅作分解动机。历史 MAPPO-9-v2 使用旧 Pure-Capture奖励、不同任务终止/horizon/schema，绝不作为新capture配置父项。旧critic state aliasing/P2/P3疑点保留；这次以单任务学习实验回答问题，不声称已修复它们。

## 共同 Single-Task Scratch Contract

唯一配置：[common.yaml](../configs/experiments/forward_final_single_task_20260915/common.yaml)，父配置：[canonical scratch](../configs/experiments/forward_final_scratch_mappo_20260914/canonical.yaml)。两个CLI task使用同一新runner，仅复用最新scratch建网、Final collector与当前MAPPO update函数，不调用历史capture runner或overnight预算/stop gate。

| 共同项 | Capture = Coverage |
|---|---|
| 环境 | 4 pursuers，120×120，1 obstacle，半径1–1.1；Final APF/physics |
| 出生 | Final mixed共同root：map_random、min separation15、margin12；无recovery pool |
| 感知/拓扑 | local surface sensing20m、friendly VorAdj/free_mask_projected、sqrt(N) centroid；不输入global enemy |
| 动作 | AW9，a±.4、w±π/6、vmax3、drag.4/3、dt.05×10 |
| Actor | 全随机 LegacyVorAdjFeatureBackbone，256 hidden / 8 heads / 4 layers；self9、8/8/5 padding、pursuing embed8、dropout配置.1，PPO eval()禁dropout；正交head gain.01 |
| central V | 全随机 action-free geometry V，256/8/4，max_agents4、evaders8、obstacles5 |
| PPO | rollout256、γ.99、λ.95、clip.2、epochs3、minibatches2、actor LR3e−5、critic LR1e−4、entropy.01、value1、gradclip.5、targetKL.02、Adam eps1e−5 |
| ValueNorm | fresh，β.99999、epsilon1e−5，原variance floor.01；独立实例 |
| 语义 | `terminal-priority-truncation-bootstrap-weighted-ce-v2`、synchronized_swept_v1、active-only Actor、reset前next V、truncation bootstrap及GAE断递归、weighted CE correction |
| 时间 | horizon3000、min_active4；成熟CE speed coefficient.0005、reward clock2m+online |
| 初始/隔离 | seed2026091501，两个进程各自RNG/optimizer/ValueNorm/output/checkpoint；相同初始Actor/V hash |
| 教师 | baseline `teacher_dependency=0`；constructor禁止IQN/旧artifact读取；probe另进程不向RL回流 |
| 评估 | 每0/25k checkpoint，argmax20+sample20完整episode、固定seed2026191501起、保留训练RNG，逐episode记录 |

Full-Task 原coverage overlay的 cluster/min_sep7及mixed/coverage recovery reset混合不继承：两条单任务统一使用上述共同root，避免出生与recovery引入额外变量。Coverage reward函数与系数保留原Final；此处改变的是共同单任务reset合同，明确记录，不冒充原Full-Task训练分布。

### Capture vs Coverage parity

| 唯一差异叶 | Capture | Coverage | 分类 |
|---|---|---|---|
| env.num_evaders | 1 | 0 | TASK_REQUIRED |
| voradj.single_task_objective | capture | coverage | TASK_REQUIRED |
| capture_episode_ends_on_capture | true | false（CE true terminal） | TASK_REQUIRED |
| support_reward_blend_enabled | true，capture权重1、coverage权重0 | false，无support | TASK_REQUIRED |
| scenes | [capture] | [coverage] | TASK_REQUIRED |
| 授权预算 | 500k | 200k | TASK_REQUIRED |

机器逐叶比较：[parity.json](../artifacts/2026-09-15_single_task/parity.json)，`UNEXPLAINED=0`。相同叶及parent完整嵌入artifact；runtime消费者另行断言。

Capture direct保持Final CR-MS。Support保留Final approach_only/neighbor_visible，预声明归一化 `.5 / sum(surviving .5)=1`，coverage=0。uninformed角色无capture dense reward；不把无信息者改为oracle追踪。新的single-task开关仅在coverage reward入口返回0，保留原token几何；不启用历史legacy Pure-Capture reward开关。capture当步终止，无post action/recovery，CE/PBRS/control分量逐transition必须0。原safety和capture terminal reward保留。

Coverage无enemy/capture/support。原centroid-energy、PBRS scale10、speed.0005、accel/turn0、safety和strict RMS.05/max.1/hold30均保持；无completion bonus。共享min_active4意味着一次失活可真正终止，不能沿用历史min_active2。

## 预算、监控与可观测性

预算按joint env decision steps，不乘4。Capture 0→500k；250k仅review，任何zero capture不早停。Coverage0→200k；100k仅review。除工程/非finite错误，无指标早停。25k不能整除256：保存partial rollout，后续继续到256再update，不改变rollout长度；末尾partial随terminal checkpoint保留。

每checkpoint：checkpoint含Actor/V/Adam/ValueNorm/RNG/env/partial rollout、SHA及源码manifest；eval保存所有episode分布和固定seed。Capture记录capture/normal/stationary、2+/3+ visitation、最长3+hold、collision/boundary、capture time/episode length、所有reward components。Coverage记录strict success、terminal及trajectory CE RMS/max、area CV、最长strict hold、time-to-CE、碰撞/边界、PBRS/control/safety/return。每PPO更新另记录entropy、KL/clip、EV/value loss、raw/normalized advantage mean/std/quantiles、Actor/value grad；不把没有完成的episode长度当完成时间。

后台supervisor每300秒读进度，只管理自己启动的PID；评估/完成节点更新独立ledger与MASTER并commit/push白名单小文件。大checkpoint、teacher数据和fixed rollout留本地，保存SHA。作业不自动重启；工程错误保留产物，其他独立作业继续。PID/step/checkpoint/throughput/ETA/NEXT WAKE-UP写SESSION_HANDOFF。

## 前瞻判据

### Capture500k

`STRONG_STABLE_LEARNABILITY`：至少3个相邻checkpoint，argmax和sample capture均≥50%、collision均≤20%，终点也满足；高capture无安全性不能算strong。曾出现capture但不满足上述持续/终点要求：`WEAK_OR_UNSTABLE_LEARNABILITY`；全无capture：`NO_MEANINGFUL_LEARNABILITY`。完整曲线、best、terminal、所有连续3点窗口及最后窗口均报告，不能只用best。

历史三seed分别报告best/terminal和全部3点窗口，旧best=10%/50%/5%，terminal=0%/5%/5%；历史配置/评估SHA保存在historical_reference.json。它与新线不是matched MDP，因此仅历史参考，不做单变量改进归因。

### Coverage200k

训练策略sample为主要判据，argmax独立报告。连续3点success≥10%，或连续3点RMS/CV中位数相对step0均改善≥15%、strict hold均增加≥3步且窗口首末RMS/CV不反向、hold不下降：`PURE_COVERAGE_LEARNABLE`。任何如此持续学习都阻止自动reward修改；若后期回落明确标记later_regression。否则进入两个attribution probe。粗阈值服务single-seed operational screen，不宣称总体显著。

### Probe A

仅Coverage无持续学习后：独立pure-coverage新采集，当前Final BC只作为教师。200 train + 50 heldout完整独立episode；whole-episode split，teacher成功率须≥80%，否则工程/证据不足，不能误判representation。学生使用整个随机Actor（backbone/head都随机），所有层可训练，AW9 argmax硬标签；30 epochs、batch512、Adam3e−4、gradclip.5、不改变模型。记录初始/最终backbone hash、每epoch train/heldout agreement；只评估最终epoch，eval seeds不参与拟合/选择。

最终argmax success≥50%，或相对random step0 RMS/CV改善≥25%且hold+3，且collision≤20%：`REPRESENTATION_SUFFICIENT_FOR_SUPERVISED_LEARNING`；否则`REPRESENTATION_LEARNABILITY_SUSPECT`，禁止R2。supervised失败只是suspect，不是token信息不足的证明。

### Probe B

同当前reward实际重新运行strong BC与scratch200k、同20个seed×双模式，逐条assert scratch重放与原200k结果/初态/return完全一致。报告CE PBRS、safety/control、discounted return分布、success/RMS/CV/time；配对return delta分布、正delta比例、5000次episode配对bootstrap区间，reward与RMS improvement/success/mission time的Spearman（pool及policy内分别报告）。时间仅共同成功/有成功的记录可用，失败是删失。

High-quality return稳定更高需要teacher成功≥80%、paired mean区间下限>0、≥80% matched pair delta正、中位delta正；不以一个均值差裁决。R2候选仅在teacher合格且CE分量相对safety/control绝对份额中位数<.5，或matched return未稳定分离时成立；这只是幅值对照的理由，不是幅值已成因果瓶颈。

### R2 gate

必须baseline无学习、Probe A充分、Probe B支持候选；否则不启动。固定scratch200k Actor/V，从独立seed采8×256无更新rollout，保持state/action/logp/V/terminal/active不变，只将现有`CE center + PBRS（含terminal correction）`乘2。原control/safety/collision/success bonus不变，PPO/Actor/V不变。重算raw return/GAE、真实ValueNorm更新后的统计、normalized advantage、sign flips/cosine和CE/total绝对比。

若8窗口全部normalized advantage RMS变化<.05、sign flips<1%、cosine>.995，或CE/total比及raw return未实际变化：`REWARD_SCALE_INTERVENTION_EFFECTIVELY_NULL`，不启动200k。有效才random-init同seed R2 0→200k，每25k同评估。gate绑定baseline checkpoint/common config/source SHA，禁止无gate CLI启动。

R2因果改善需最后3个对应checkpoint均比baseline success+20pp，或RMS/CV各改善≥20%且hold+3，collision不超过baseline+5pp。只回答当前shaping幅值是否不足；不能推断success reward需求。失败只提出`Coverage-R3: explicit strict CE completion / hold achievement reward`，本轮不运行。

## 2×N证据索引

具体数值与完成状态见下方自动结果和 [MASTER.json](../artifacts/2026-09-15_single_task/MASTER.json)。条件未触发的probe/R2记为未运行，不能填作负结果。

| 任务 | corrected baseline | historical reference | supervised representation | reward separability | R2 |
|---|---|---|---|---|---|
| Capture | [500k逐checkpoint双模式评估](../artifacts/2026-09-15_single_task/capture/)；best/terminal/所有3点窗口分别统计 | [MAPPO-9-v2三seed best/terminal/窗口](../artifacts/2026-09-15_single_task/historical_reference.json) | 不适用 | 不适用 | 不适用 |
| Coverage | [200k baseline逐checkpoint评估](../artifacts/2026-09-15_single_task/coverage/) | 当前Final reward合同 | [Probe A：train/heldout agreement和rollout](../artifacts/2026-09-15_single_task/attribution/representation.json) | [Probe B：matched分布、bootstrap及rank](../artifacts/2026-09-15_single_task/attribution/reward_separability.json) | [仅gate通过才运行](../artifacts/2026-09-15_single_task/coverage_r2/)，状态见MASTER |

## 最终MASTER规则

完成两个baseline及必要probes/R2后，只输出用户规定五种结论之一。Capture只有strong才算可靠learnable；弱信号保留原分类。按representation疑点、R2明确改善、双方可靠可学、capture可学+coverage监督可学但RL失败、其余不可靠的顺序落账。最终2×N表和原始模式/窗口不被结论替代。全流程运行中`causal_conclusion=null`，禁止把启动/排队状态称为已完成实验。

最终结论对应的下一重点（仅建议，不自动启动Full-Task或R3）：

| 最终因果结论 | 下一重点 |
|---|---|
| CAPTURE_AND_COVERAGE_BOTH_INDIVIDUALLY_LEARNABLE | Full-Task capture/coverage gradient interference |
| CAPTURE_LEARNABLE_COVERAGE_RL_NOT_LEARNABLE_BUT_SUPERVISED_LEARNABLE | reward / critic / advantage |
| COVERAGE_REPRESENTATION_LEARNABILITY_SUSPECT | token / backbone的可学习性；不自动R2 |
| COVERAGE_REWARD_SCALE_CAUSALLY_LIMITING | 已有CE/PBRS shaping幅值；不能据此回答success reward需求 |
| SINGLE_TASKS_STILL_NOT_RELIABLY_LEARNABLE | 停止Full-Task扩展，重新审查基础AC合同 |

## 工程验证

初版6项定向测试通过：共同随机初始化、Final CR-MS/Support×2配对、Capture真正terminal且CE=0、Coverage原reward逐位一致、timeout bootstrap/success priority、R2仅改CE/PBRS、无recovery stream。两GPU完整rollout256 + partial checkpoint/resume + tiny resumed update +双模式短eval smoke通过，optimizer/RNG/state逐位恢复。最终组合回归33 passed；新增gate/监督梯度后定向9 passed（6项重叠，共36个不同测试）。最终源码两GPU smoke再次通过。固定rollout幅值分析另在260-step工程checkpoint完成8×256无更新检查，结果仅证明分析代码可执行，不授权R2。见tests.txt、gate_tests.txt、smoke_*_final/report.json、smoke_scale/report.json及launch/source manifest；smoke不是学习结果。

<!-- AUTO_SINGLE_TASK_RESULTS -->

## 最新自动结果

状态：`IN_PROGRESS`。最终因果结论：`PENDING — 尚无最终结论`。

| 任务 | 当前 corrected baseline | 历史/归因对照 | R2 |
|---|---|---|---|
| Capture | RUNNING / None | MAPPO-9-v2：best 10%/50%/5%；terminal 0%/5%/5%；完整持续窗口见 historical_reference.json | 不适用 |
| Coverage | PENDING / None | Probes pending or not required | NOT_STARTED |

### Capture best / terminal / sustained window

| 模式 | 新线 best capture（step） | 新线 terminal capture | 最后3点 capture mean / min | 最后3点 collision mean |
|---|---|---|---|---|
| argmax | 0.0% (25000) | 0.0% (25000) | 尚不足3点 | — |
| sample | 0.0% (25000) | 0.0% (25000) | 尚不足3点 | — |

### 运行交接

```json
{
  "capture": {
    "pid": 827847,
    "alive": true,
    "returncode": null,
    "step": 28200,
    "checkpoint": "/home/yjq/rl/CoCap1/cocap-voradj-small-step-ac/artifacts/2026-09-15_single_task/capture/step_025000.pt",
    "throughput": 9.358152008872482,
    "eta_seconds": 50415.936773914924,
    "status": "training"
  },
  "coverage": {
    "pid": 827848,
    "alive": true,
    "returncode": null,
    "step": 42200,
    "checkpoint": "/home/yjq/rl/CoCap1/cocap-voradj-small-step-ac/artifacts/2026-09-15_single_task/coverage/step_025000.pt",
    "throughput": 14.002990415897823,
    "eta_seconds": 11269.021495640465,
    "status": "training"
  },
  "supervisor": {
    "pid": 827802,
    "next_wake_up": "2026-09-15T13:07:05.673700+08:00"
  },
  "causal_conclusion": null
}
```

机器可读完整结果：[MASTER](../artifacts/2026-09-15_single_task/MASTER.json)。
历史完整曲线与窗口：[historical_reference](../artifacts/2026-09-15_single_task/historical_reference.json)。
未经完整预算和所有必要 gate，不形成五类最终结论。
