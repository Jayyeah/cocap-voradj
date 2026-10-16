# CoCap MAPPO 迁移路线最终历史（2026-10-15）

## 范围与当前状态

本次恢复 ops/ac-master-dag-20260921、Batch01 TERL-MAPPO矩阵、Batch02 E1/N2/R2/N3/R3/V3、500k Stability Recovery、M-CAP/M-COV和DAG A0–A3，以及独立 IQN→Actor BC→PPO/Critic 线。以运行manifest、progress、checkpoint与最新screen为准；较新的本地结果不被旧状态文档覆盖。

- Delivery branch：delivery/batch02-cocap-mappo-20261011，写入前HEAD 45171da3cfe004c1b35aa9480284ed38c4ce5a6d；master DAG HEAD 2155fc576608f33e32d774d126668d7995a4b64b。
- Oct12 17:05云端状态报告是历史快照。R2/R3 stability continuation随后完成到1M并写出paired screen；V3 progress更新至Oct12 19:03，998,808/1M，last checkpoint 975k，随后进程退出。因此R2/R3续训是已完成但未同步；V3续训未闭合，不能记成完成。
- Oct12 17:05前的云快照有N2/R2/R3 baseline摘要，但manifest标明未包含checkpoint权重。之后R2/R3 continuation和V3 stability运行结果仍在本地/服务器目录。
- Oct15 21:43检查没有CoCap/MAPPO训练PID。yjq_mappo和yjq_z tmux pane为Codex shell。两张RTX A6000由Drone6D训练使用，约8.2/8.0 GiB；GPU0/1当时利用率0%/98%。没有触碰这些进程。
- delivery的AGENTS.md规定报告同步走bounded supervisor；本任务没有找到对这五份新归档件的GitHub发布授权。文件仅在本地新增，不commit/push、不覆盖旧记录。

capture可能代表capture event或capture-completion；full-task success专指成功CoCap mixed合同中完成capture、post-capture recovery/CE并安全结束。每mode 10/20 episode的screen是单seed描述值。best、selected、last分别为screen候选、选择规则结果、训练末点。

## 两条不同初始化的路线

~~~mermaid
flowchart LR
  subgraph S[Scratch MAPPO迁移线]
    A[TERL native MAPPO] --> B[PPO stability]
    B --> C[CoCap E1 environment bridge]
    C --> D[observation / reward migration]
    D --> E[CoCap backbone N3 / ledger R3]
    E --> F[V3组合]
    F --> G[500k stability continuation]
  end
  subgraph T[成功策略蒸馏与Critic线]
    I[IQN teacher] --> J[Actor BC]
    J --> K[PPO continuation]
    J --> L[Critic / advantage / Q-ranking audit]
  end
~~~

Batch02 N2/R2/N3/R3/V3均由随机actor、随机中央V、fresh optimizer/ValueNorm/env/RNG起步；未加载IQN或BC actor。IQN→BC→PPO从已成功策略起步，critic/Adam/ValueNorm另行fresh或训练。两类run不可混作同一训练起点。

## TERL native与Batch01矩阵

| run | parent/source与初始化 | 实际合同与learner/critic | steps与结果 | delta和结论 |
|---|---|---|---|---|
| TERL Stage1 native | source 143359b2722d49c29b4fecc0ad1fd8d46326e45a；random scratch，100k后full-state resume，只延长预算 | 3P1E0O MarineEnv、AW9、TERL local obs/reward；MAPPO/PPO、256/8/4 actor与action-free central V | 1M；best/selected 775k SHA 590d486876d4fca1cd75311c2dcc1255f847a1f756d91eed202e737193b58db4，正常capture45/50、collision5/50；last screen capture30%、collision50–60% | 学到TERL Stage1围捕，不是CoCap能力。末期KL-stop、entropy、EV与行为共同恶化，因果未确定。 |
| Batch01 P1 split | 从TERL selected 775k完整状态分叉；control runtime 0e5b03b3c39fe4615b91060d7225b3034d820952，treatment 5b53c3f4f56265dbb8039c9b9ba8bcf53ed75892 | 唯一处理target_kl .02→.01；其余PPO/TERL固定 | 1M；control last capture/collision 40/45；treatment last 95/70，best 850k argmax、950k sample均100/0 | 单seed正向late-retention信号，不是一般稳定性结论。 |
| R1 | random scratch | distance dense reward改为原reward减distance、加2×clipped nearest-enemy distance progress；TERL actor/V/PPO | 1M；last capture10%，argmax 2/20 capture、18/20 collision | reward上升不等于任务能力；单线不能定因果。 |
| N1 | random scratch | actor使用LegacyVorAdjFeatureBackbone但保留TERL tokens/native info；central critic不变 | 1M；中期best约325k/725k为100/0，last capture/collision 75/75 | 中期screen未保留；未证明CoCap原生feature增益。 |
| T1 | TERL selected775k actor；Stage2 fresh critic/optimizer/ValueNorm/env | TERL Stage2 native scale，PPO | 100k；last capture85%、collision55% | 保留起始能力，未证明继续训练有增益；无Stage3。 |
| C0 | random scratch | 真正tanh-Gaussian AW连续动作，不做AW9 quantization；TERL任务 | 1M；best mean750k argmax100/0、sample70/15；last40/40 | TERL Stage1动作空间实验，不是CoCap。 |

Batch01全矩阵为单seed PROVISIONAL_LONG，base QA仍pending、未正式晋级。P1是强Stage1 checkpoint续训，其余scratch arm不能混作相同初始化。

Batch01常用PPO设定为gamma .99、GAE .95、clip .2、epochs3/minibatches2、actor LR 3e-5、critic LR 1e-4、entropy .01、ValueNorm、target_kl .02；P1 treatment唯一改target_kl至.01。不同任务arm的reward/actor输入/动作变化已在表中分别列出。

## CoCap environment bridge与Batch02

成功IQN Z05 Stage1/Forward-Final V2是任务合同参照：source 1a9da054ecb5652726005d00d08ab05a2188dd8e，resolved hash 6955ecf28383f7e146a0ee485ef3e33cb1d1782b197a51ceadd2ef93968d3b5a；Stage1 selected 1.3M IQN SHA d35b7984d8ca03fa7b536811794578ff402e9b8ad48589e3318f2cb1077d702b。它不是MAPPO初始化权重。

E1 source 8827c11ef3534e5e9894bfb6b223405f24544be3。它保留CoCap 4P1E1O mixed、AW9、self9/22-slot observation、VorAdj/NormSense/Z05、CR-MS/support/CE/PBRS/safety/event reward、同步swept collision及capture后recovery/CE；通过64 scripted transitions+11 edge fixtures、256 rollout decisions/真实PPO updates和fresh-process checkpoint恢复。保留TERLActor、action-free CentralValueNetwork、256/8/4，只绑定输入尺寸（critic self25→9）。结论是工程桥，不是学习能力。

Batch02 A runs manifest共享parent 53ed5b2eabe3b48c280741e29629038b450e7d7b、冻结E1 8827c11ef3534e5e9894bfb6b223405f24544be3、experiment SHA 8ce559335d656b7338a23886eb9cc958fa1a96cc。N2/R2均fresh scratch、1M joint decisions、成功CoCap provider完整mixed任务/reward、PPO和action-free central V。

| run | 科学delta与critic/合同 | best / selected / last | capture、collision、full task | 状态 |
|---|---|---|---|---|
| N2 | TERL encoder/head消费CoCap self9/entity slots；CoCap full reward；central V沿用 | best900k；selected=last1M SHA e33d38b1663742726977baaf2ff095f941844e0b054b16d77a1d0ba7ebe25aef | capture completion约17%；full success0 | 预算和评估完成。 |
| R2 | TERL observation/backbone消费CoCap state；CoCap full reward；critic不变 | best screen925k；selected550k SHA 112e2a5babfd2250d66dea40473975a7c41dc7a371c4531c7395f8274a4b806a；last1M SHA 78e74ac65d409c30a407a95f41e41e39024a6eeb8f3ed3e8320190d5c2085583 | selected capture62/62%、collision38/58%；last capture32/40%、collision70/74%；success0 | baseline final完成；capture伴高碰撞且末点退化。 |
| R3 | R2加read-only六类reward ledger；PPO直接消费provider原reward数组，账本不改reward | selected550k SHA d449d1aca86d43720d95b73b82d901699f0603abe43d21ee49c138aa4dc4d467；last1M SHA e5256b3c499aa9c547951b8065b89d4e7eda8067e48f48198ea68b686eb893cb | R2相同screen数值；success0 | 相对R2只差telemetry；非独立重复或新reward处理。 |
| N3 | actor encoder换成功IQN VorAdj feature/single-action feature；TERL AW9 categorical head；没有IQN Q/quantile权重；central V不变 | selection/last1M SHA d9bf9819029e03badb3a60544217b4f64f9ace0e0fcae2d2822458881850e270 | selection capture5/0% argmax/sample、collision60/80%、success0 | 预算到点、selection存在；formal final closure缺失。 |
| V3 | N3 actor与R3 ledger代码组合；critic未改 | selection/last1M SHA e811bbd1e0c60f94429e0bb05c1f33c8de009a3a1b5a43e1f26fbeb35349a83d | selection capture5/0%、collision60/80%、success0 | final未闭合；不是critic实验。 |

运行provenance逐线为：N2/R2 source experiment SHA 8ce559335d656b7338a23886eb9cc958fa1a96cc、代码parent SHA 53ed5b2eabe3b48c280741e29629038b450e7d7b；N3 source 3575b742548f8ce5ceba68f3db776533fb5d83ba、parent 8ce559335d656b7338a23886eb9cc958fa1a96cc；R3 source 3575b742548f8ce5ceba68f3db776533fb5d83ba、parent N3 SHA 28a33379b85b8fd48c22fb271c3e41f681b0bcdf；V3 source同为3575b742548f8ce5ceba68f3db776533fb5d83ba、parent R3 SHA 7cb8bc07b6ed95c29cacdc6c308e2f64ef59c380。各线另有冻结E1 provider SHA 8827c11ef3534e5e9894bfb6b223405f24544be3。所有Batch02 actor/V仍random scratch。

N2/R2不是observation-only或reward-only因果分解：二者都接入完整CoCap provider，但actor处理路径不同。

## 500k stability continuation

R2/R3/V3从各自500k last完整训练状态恢复到1M；唯一改动target_kl .02→.01。续训实现experiment SHA均为b479a869027262ad60bf30b0604263f5b99b5341。actor LR 3e-5、critic LR 1e-4，无scheduler；entropy .01、clip .2、PPO epochs3/minibatches2、gamma .99、GAE .95、ValueNorm等不变。full-state parent checkpoint：R2 204a4a2e53091cd25ce6790e0a755f21b5068810ed02a46fcdeb0449d0ee74539；R3 9e831f92693cc468d798e19907b3caa789571e53f14adc18041d19ec9b438a2b；V3 7f2daa809b8f703549c9a62a891b57a8bb831bd167bc566e781142c885f5652c。

| run | last checkpoint | 末点评估 argmax/sample（capture completion；collision；full success） | 状态 |
|---|---|---|---|
| R2 | treatment best screen argmax725k 90%、sample750k 90%；无正式promotion/selected；last1M SHA 50f1d7e7aaa5fcd1854058074c55f967de5cfc5e35c43c9b717e2953f6b0047b | 1M capture50/20%；collision40/90%；success0 | PAIRED_SCREENS_COMPLETE，provisional。 |
| R3 | treatment best screen argmax725k 90%、sample750k 90%；无正式promotion/selected；last1M SHA de3ecece8c00b00e276ab6c91a1c080248b3521a11c9f1a366232715cdf98a67 | 与R2逐screen相同 | PAIRED_SCREENS_COMPLETE；不是独立复现。 |
| V3 | treatment best screen argmax850k 40%、sample900k 20%；无正式promotion/selected；998,808 decisions，last975k SHA ef1ebad53abc1a5ed862abbee0156ddbe5b0901b400ddc600da137c5627f72d2 | 975k capture30/0%；collision70/100%；success0 | 进程退出，无1M checkpoint或paired final report。 |

R2/R3 treatment把KL early-stop从约1.4%升到18%，说明target_kl确实影响early-stop；但800–950k七个screen的treatment capture/collision均值33.6/85.0%，baseline45.0/63.6%，没有late-window综合收益。Batch01 TERL P1的正向单seed终点与此不一致，不能宣称低target KL普遍提升CoCap稳定性。

## DAG A0–A3、M-COV/M-CAP

DAG task A0–A3与critic identifiability audit的A0/A1是不同命名空间。

| run | 起点/合同 | 结果 |
|---|---|---|
| DAG A0 / M-COV repro | corrected scratch MAPPO，Pure Coverage 200k，source 668d5f710c68 | CE reproduction PASS；terminal step_200000.pt，旧记录无权重SHA。非capture。 |
| DAG A1 | no-entropy local-Q coverage scratch100k，last SHA 12ce59dfed9fe07e3a053a33253f8c919411acc90f95f6e6cd194bf5afa39f39 | final evaluator不兼容CoCapIQN state dict，未完成rollout/update；entropy归因incomplete。 |
| DAG A2 | discrete SAC coverage scratch100k | transient learning；entropy temperature target过强，未继续；非capture MAPPO。 |
| DAG A3 | capture-init I0 ring-anchor scratch100k，SHA b30946f2b693ee205ba2d472c0c80cfd2f71ec9e02812d52d8dfa1927a203e8f | 缺final evaluator/seed语义；training recent capture0%、collision97%、success2%；I1–I3未跑。 |
| M-COV | pure coverage scratch200k，last SHA 3adfdbe90970cfc1333a4a5cba82ca496393d170c56b789a842f0c4ba06fe68f | 中间100k/150k CE100%；last argmax CE80%/collision5%，sample CE100%/collision0%。 |
| M-CAP | NormSense V2 pure capture scratch500k，last SHA 01eeb67c689dd323c2e8e680549ed8e5bd871b7b0a75b8b989e19aa02e953197 | terminal capture4/40、collision36/40；ring3 visitation15/30%、strict hold0；collision agent-agent21、evader9、obstacle6。诊断为geometry/ring3限制，collision放大失败。 |

M-CAP 100k scratch reproduction末点两种模式均20/20碰撞、capture0；50k曾有一次sample capture。coverage/capture子任务均不等于mixed full-task。

## IQN teacher→BC→PPO与Critic线

| 阶段 | 初始化/合同 | 真实结果 |
|---|---|---|
| 旧IQN→BC | teacher SHA ef58ae9bdd018633afee0d16f2242d2f157a6cc18611e41474f115aed254b87e；600 episodes/171,052 active rows；BC best epoch20 SHA bb8f971201f6e0a55af3ccb62bcae117f96afa1b07fad2c95c3da1dc552fa35e | 旧B0/K10 pure-capture合同：pre-PPO capture100%、collision3%、agreement约.966；不能冒充current full task。 |
| Canonical Forward-Final BC | IQN teacher 2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89；BC actor 7916a970226a0452a8f71a22235524f6ba4511aa4c9bda907a9c8368a60bf2cd | mixed teacher capture100%/collision1%/CE99%/safe completion99%；BC argmax capture99%/collision1%/CE98%/safe98%，sample99/2/98/98%；coverage CE100%。 |
| 旧Direct/warm-up PPO | 旧BC bb8f…；fresh critic，MAPPO-9-v2 | 100k描述值Direct capture.96/collision.04；warm-up .92/.08。旧matched causal claim已撤回。 |
| Forward-Final PPO探查 | canonical BC actor state hash d626fe2cd19b1cc40fadabf85fdbff8d8e151969c707d9a777f081d5b893cfa0；fresh critic/Adam/ValueNorm；PPO期间无teacher | 5k后capture100%、collision0，但full safe CE completion从100%降至0；argmax CE20/20→0/20，sample20/20→3/20。标为exploratory non-gate。 |
| Critic/advantage audits | frozen strong BC actor、teacher与held-out MC/Q data，不更新actor | early-recovery value bias、local/central fit、AW9 Q-ranking、counterfactual和双head研究。详见另外两份审阅。 |

## 收束

TERL Stage1 capture已确认；CoCap pure coverage有局部信号，CoCap capture能力弱且安全受限；CoCap mixed full-task MAPPO没有成功checkpoint。IQN→BC保留full-task能力；PPO小步更新的CE退化是重要预警，但Critic因果机制未识别。见 [失败与正信号审阅](MAPPO_FAILURE_AND_POSITIVE_SIGNAL_REVIEW_ZH.md)、[Critic选项](MAPPO_CRITIC_RESEARCH_OPTIONS_ZH.md)、[最终决定](MAPPO_CONTINUE_OR_ARCHIVE_DECISION_ZH.md) 与 [checkpoint注册表](MAPPO_RUN_CHECKPOINT_REGISTRY.csv)。
