# EXP-TERL-MAPPO-01：迁移合同（编码前冻结）

日期：2026-10-08，Asia/Shanghai。研究问题：原生 TERL Stage1 中 random-init MAPPO 的 cooperative capture 可学习性。用户本轮授权优先于 AGENTS.md 的历史 CoCap 路线与训练限制；不修改其他实验或中央状态。

事实源：TERL 远程 HEAD/采用 SHA `143359b2722d49c29b4fecc0ad1fd8d46326e45a`（本轮 git ls-remote 核验未更新）；CoCap learner/reports parent `80debaef0f5983d8128b5059c90d65acc9216c10`，corrected learner lineage `25dd0f8`。远程 scratch 和 Capture Stage1 分支分别同本地 `80debae` / `640a26dc4d4c8df3ae224162a2707cf61720947f`；中央 DAG remote `3940d87e3a12f8c3d51e8c68aa720e2370f7c1c0`；默认 main `a5814f49fa29d869cdc3fb8d8e0df4722aa11f00`。独立分支从 scratch learner/report parent 创建；主分支不包含较新 learner，不能按名字选择 main。

原 TERL 工作树 robots/evader.py 与 thirdparty/APF.py dirty，故使用 git archive 固定 SHA 的 byte-identical vendor，不读取该工作树的修改。vendor 保留 MIT LICENSE。

| 项目 | TERL-IQN 实际源码 | 既有 CoCap-MAPPO scratch | 目标 TERL-MAPPO | 分类 |
|---|---|---|---|---|
| 环境 | MarineEnv，120×120，Stage1 3P1E0obs4cores，初始 P-E>10 | VorAdjEnv，4P1E1obs | 原生 MarineEnv + trainer生命周期适配 | SAME（TERL）；EXPERIMENTAL_DIFFERENCE（与CoCap） |
| 课程 | 0/2M/4M/5M/6M，五阶段共7M | 独立capture primitive | 仅冻结首阶段，不启动完整课程 | EXPERIMENTAL_DIFFERENCE |
| 观测 | self4、friends5×7、enemy8×7、obs5×5，共19token | self9、friends8、enemy8、obs5，共22token | 保留原观测，不加Z/CR-MS/VorAdj | SAME |
| 信息 | 友军/障碍有限距离；敌方未执行距离过滤；pursuing由最近敌方≤18m计算 | NormSense V2 enemy surface R≈52.22，role/neighbor合同 | 保留原生信息；actor无central输入 | SAME；与CoCap为EXPERIMENTAL_DIFFERENCE |
| backbone | TERLConfig默认256/4/3，但入口Agent默认覆盖为256/8/4；dropout .1、PreLN、FF4H | 256/8/4 LegacyVorAdj | **256/8/4有效训练入口**；保留entity/type/Transformer/self+max/target fusion | SAME |
| pooling | max包含padding query输出（key被mask不代表query置零） | Legacy decision pooling | 兼容原始unmasked max；masked选项仅诊断 | SAME；意图差异UNRESOLVED |
| no-target | attention全-inf导致NaN | corrected fully-masked fallback | empty目标权重0，fusion保留self；全mask临时self开放 | CORRECTNESS_REQUIRED |
| 动作 | a-major笛卡尔积AW9，a±.4，w±π/6；P vmax3、E3.5 | AW9，额外CoCap合同 | 原生索引/动力学、dt .05×N10=.5s | SAME |
| APF/current/spawn | 原APF含filtered acceleration索引疑点，4涡核、随机reset | 历史APF/NormSense/task地图 | 全部原生；不静默修APF | SAME；APF设计意图UNRESOLVED |
| Dense reward | active -1；最近距离<4/friend/obs<4加-5；每敌方≤8给5，远端硬编码5exp[-.05(min_d-8)] | CR-MS/support capture、reward clock offset2M | 原生reward不缩放，无新shaping | SAME |
| cooperation | ≥3先分支，3人+5/4人+3.5/5人+2；≥5的elif不可达；另有any≥5时全员-10；count不排除inactive | CoCap CR-MS/role reward | 原生保留；Stage1只3人分支 | SAME |
| capture | ≥3、半径8、最大angle≤π且max≤3min；helper返回**其他agent数量** | CR-MS/K3 + stationary等 | 原生正常几何，无stationary fallback | SAME |
| goal credit | 每个checker参与者记录捕获并获120×(2π/other_count)exp(-std)；3人等角约376.99每人 | 参与/共享终奖合同不同 | 保留（不改成120或251.33） | SAME |
| collision/boundary | decision末检测，collision -80；trainer设deactivated；越界只罚-5，无boundary terminal | synchronized_swept_v1 / minactive4 | 原生末检测、越界惩罚；不混swept | SAME |
| done/reset | capture的step done可能false；trainer检查all captured / active<3；horizon检查在递增前，实际3001 decisions | terminal-priority/truncation、同步reset | 原step untouched；按原trainer episode reset，capture/collision joint true terminal，timeout truncation；terminal优先 | SAME（episode）；CORRECTNESS_REQUIRED（PPO masks） |
| actor head/init | IQN quantile/cosine→Q；随机初始化 | categorical、random init、无teacher | 删除quantile/cosine、categorical9、head orthogonal gain.01 | ALGORITHM_REQUIRED |
| dropout forward | IQN act eval，训练更新train | corrected actor始终eval（autograd有效） | actor/critic eval，sampling探索，避免PPO似然漂移 | ALGORITHM_REQUIRED / CORRECTNESS_REQUIRED |
| critic | 无V | centralized action-free Transformer V/ValueNorm | 复用corrected V，原生全状态适配，包括洋流核与时间；actor不读 | ALGORITHM_REQUIRED |
| learner | replay/epsilon/target IQN | PPO/GAE/ValueNorm，256 rollout，LR3e-5/1e-4 | 复用corrected learner，active loss、raw GAE、normalized value clipping | ALGORITHM_REQUIRED |
| evaluation协议 | 内置eval半horizon1501、P-E初距>20 | Stage1 primitive自己的评估合同 | 本次screen使用training Stage1初距>10、horizon3001；不同于原内置eval | EXPERIMENTAL_DIFFERENCE |
| budget/eval | 7M，默认保存200k | 100k scratch；0/25k/50k/75k/100k argmax20+sample20 | 首轮100k bounded；异步CPU evaluator，0/25/50/75/100k双模式；不因早期0capture改reward | EXPERIMENTAL_DIFFERENCE |

关键风险：原step done不能直接作MAPPO terminal；critic必须使用pre-reset state bootstrap；ValueNorm旧预测必须按rollout旧stats反归一化；critic无位置embedding以保持agent排列等变；原reward有正的驻留奖励和大terminal奖励；旧CoCap结果不能解释为同任务算法优劣。必要修复逐项测试，不修公开源码的行为疑点。

已有证据：M-COV200k argmax/sample strict CE80%/100%；M-CAP500k终点capture4/40、collision36/40；20261006 Capture scratch100k仅50k sample1/20 capture，终点两模式0/20、collision20/20，但ring2/3改善，分类partial。中央状态仅MASTER写；本分支提供handoff，不写中央DAG/state。
