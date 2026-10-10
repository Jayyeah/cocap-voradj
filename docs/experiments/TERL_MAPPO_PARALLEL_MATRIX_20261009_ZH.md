# TERL-MAPPO Batch01 实验矩阵与运行入口

当前运行快照、best/last、collision及曲线统一见[2026-10-10状态文档](BATCH01_STATUS_20261010_ZH.md)。实时状态使用`python tools/batch01_status.py --current`，中央事实源为[batch_state](../../artifacts/2026-10-09_terl_mappo_batch01/batch_state.json)与[run registry](../../artifacts/2026-10-09_terl_mappo_batch01/run_registry.json)。旧启动PID、25k上限及V1/V2状态已移至[完整历史记录](TERL_MAPPO_PARALLEL_MATRIX_HISTORY_20261010_ZH.md)，不覆盖当前状态。

## BASE与执行分类

V3r2 candidate：`9cc2d47c532c76239a61a0f6a19c8603358c92bf`；canonical lock：`384b9ba587b905b879a08f01fa1073ca470bdc0a836459067d193d97b03a84ab`；交付HEAD：`d6e201052f0eac319342c35cd4efd9171df3fd54`。`CORE_V3_SELFTEST_PASS / QA_PENDING / BASE_FREEZE_BLOCKED`，`BATCH01_BASE_SHA=null`，正式共同parent未冻结。CPU87/CUDA3/A3脚本12探针通过属于自测；最新独立QA仍是V2的c20b632阻断报告，未伪造V3签字。

原25k pilot已按预算停止。用户后续明确授权长线PROVISIONAL：P1两分支分别到1M，R1/N1/C0到1M，T1 Stage2到100k。授权原件及hash固定在[长线预算合同](../../artifacts/2026-10-09_terl_mappo_batch01/commander_v3r2/provisional_long_authorization_20261010.json)，不自动扩展、不使用final、不追认为正式证据。旧V3r1六个pilot隔离保留；当前长线仅复用合法V3r2状态。

## 已实现的科学差异与初始化

| 线 | 相对T0的科学差异 | 实际初始化 | 授权PROVISIONAL终点 |
|---|---|---|---|
| P1-control | target_kl=0.02；同状态配对control | selected775k完整actor/critic/两个Adam/ValueNorm/env/APF/RNG/counters分叉；strict恢复 | 1M |
| P1-treatment | 唯一PPO参数差异target_kl=0.01 | 同一775k完整checkpoint；live consumer核验0.01 | 1M |
| R1 | 仅native distance dense项替换为CR-MS类几何shaping；native其它raw分量、事件和物理保留 | actor/critic/Adam/ValueNorm scratch | 1M |
| N1 | 仅CoCap-style 256/8/4 actor；native self4、friend5×7、enemy8×7、obstacle5×5及mask/type/ordering；critic固定T0 | actor及ValueNorm scratch；无VorAdj/Z/额外目标信息 | 1M |
| T1 | Native Stage2 4P1E1O6C，Stage3工程支持7P2E2O8C；固定容量central critic包含完整8-current槽位和mask | selected775k actor warm transfer；新critic/两个Adam/ValueNorm/env/RNG/counters；跨规模不是bit-exact resume | Stage2 100k首个review；Stage3未启动 |
| C0 | AW9变为真实物理continuous a∈[-0.4,0.4]、omega∈[-π/6,π/6]；tanh Gaussian、joint logprob/Jacobian/pre-tanh latent | 连续actor head及ValueNorm scratch；native reward/critic/PPO保留 | 1M |

R1不能称为Full CoCap Reward。N1不引入新信息。C0不量化回AW9，deterministic模式称mean；其differential entropy可为负，不能与categorical Shannon entropy直接混用。各线CPU/CUDA rollout/update、evaluator、完整checkpoint恢复门禁已通过；详见各worktree gate_summary与中央registry。新模块来源均经声明delta绑定，训练源码不热修改。

## 评估与升级合同

新ARM协议由V3 candidate绑定：screen2056100900、selection2066100900、final2076100900；sample动作offset100000；scene offsets0/1000/2000。same-scene arms配对物理seeds，两个动作模式共享初态但采样RNG隔离。历史T0 screen/selection/final维持原值；新final与训练、旧观察final及selection不重叠。regular screen每模式10，milestone20；selection每模式20，正式final50；全部完整horizon，由独立CPU evaluator执行。PROVISIONAL禁止final。

P1报告775k之后每25k保持曲线及best-observed、last和collision；新control不可用历史1M替代。当前best-observed是screen观察记录，未创建formal selection receipt。T1升级使用selection域的Stage2及Stage1 retention，各模式20局且normal≥80%、collision≤20%；还须独立QA、冻结BASE和合法阶段迁移receipt。Stage2 100k是review而非失败界线；200k后续窗口是预登记设计，不等于已授权追加预算。当前Stage3仍被门禁阻止。

## 来源与资源管理

每线有独立worktree/branch/config/delta/run_id/output/tmux/manifest；唯一管理Agent负责全部进程。独立QA签字不得由同一Agent的自测替代。正式节点继续WAITING_BASE，工程执行状态单独登记PROVISIONAL_LONG。

启动时优先完全空闲GPU；两GPU均有外部任务时，每卡最多两条自有计算线，并检查实测容量。资源监督器由我们启动，用户2026-10-10要求后，运行期RAM/GPU/磁盘/租约仅记录告警，不能作为监督器杀停理由；源码/checkpoint/预算门禁仍执行。已锁定runner自身硬资源检查未热改、当前未触发。[资源策略原件](../../artifacts/2026-10-09_terl_mappo_batch01/commander_v3r2/runtime_resource_policy_20261010.json)。外部任务不操作，Evidence/Persistent/历史TERL及恢复锚点不删除。

每25k checkpoint/screen，保留latest、必要milestone与pending evaluation；本次有限预算下保存量仍在已测容量内。旧资源中断及严格恢复回执保留，未保存尾段按记录回滚；预算不延长。持久服务为`batch01_commander_v3r2_long_supervisor`及`batch01_commander_resource_recovery`，恢复状态可在CLI断连后从registry与progress读取。

## 2. 四个参照合同：实际值与目标值分开

“历史 CoCap”在本批特指上述 corrected scratch Capture；Coverage R20正控另列背景，不能混为同一个信息合同。“下一步 CoCap”是迁移目标，尚无冻结训练合同。

| 轴 | TERL-IQN 原始实际 | T0 TERL-MAPPO 实际 | 历史 CoCap-MAPPO Capture 实际 | 下一步 CoCap-MAPPO 目标（未冻结） |
|---|---|---|---|---|
| Environment | MarineEnv，120×120，原APF/current，decision末碰撞；boundary只罚 | 同原生；wrapper复刻trainer生命周期 | VorAdjEnv/Forward-Final V2，synchronized_swept_v1，boundary/safety合同 | 先 E1 bridge 审核，再采用 V2；不直接替换 T0 |
| Network | entity/type Transformer 256/8/4实际入口，self+unmasked-max、Target Selection/fusion、IQN后端 | 同TERL特征；categorical9；action-free central V 256/8/4；empty-mask修复 | LegacyVorAdjFeatureBackbone，self9、friend8、enemy8、obs5；categorical9/central V | CoCap actor +经桥接验证的V；N1只验证结构 |
| Reward | time/distance/global/emergency/collision/goal；驻留dense、多目标循环 | 原值原顺序重构assert；不缩放 | Final ring_importance_ms_v0/CR-MS，capture support=1、CE=0、clock offset2M | R2完整Capture合同；Coverage/Mixed需独立冻结 |
| Information | self4/friend5×7/enemy8×7/obs5×5=19token；敌方不距离过滤；友军/障碍有限距离 | 同原生19token；actor不读central；V含current/time | self9/22token，NormSense V2 enemy surface R≈52.217m，neighbor_visible/approach_only，historical global-friendly VorAdj；无Z | N2全CoCap观测；M2才审Z/evidence；global-friendly保留，local-friendly V3延后 |
| Task | 原生≥3几何capture，8m/related18m；active<3、capture terminal；timeout实际3001 | 同任务；明确terminated/truncated与pre-reset V | 4P1E1obs，Pure Capture成功terminal，无recovery，min_active4，horizon3000 | E1/R2后完整Capture；V1 Coverage、M1 Mixed另设任务 |
| Action | a-major AW9，a∈{-0.4,0,0.4}、w∈{-π/6,0,π/6}；dt .05×10=.5s；P vmax3/E3.5 | 同原生索引与物理 | AW9同类接口但环境物理必须逐项断言，不能仅凭AW9名认为相同 | C0原生连续a/w为独立诊断；不自动改CoCap目标的动作合同 |
| Algorithm | off-policy IQN/replay/epsilon/target Q | corrected on-policy PPO/GAE/ValueNorm/active loss；actor与V独立Adam；eval-mode防dropout似然漂移 | 同corrected PPO核心；其它环境consumer不同 | 共用经QA的PPO；任何稳定性改动单独登记 |
| Curriculum | 0/2M/4M/5M/6M切换，共7M | 仅Stage1累计1M；selected775k | scratch Capture 500k及独立100k复现；Coverage200k | T1先短程迁移；不得自动走7M或第二批 |
| Initialization | random IQN，无BC | seed9/109 random-origin；100k→1M精确全状态resume | fresh actor/V/Adam/ValueNorm，无teacher | 每条线单独冻结迁移方式；不得伪称跨结构bit-exact |

## 历史、QA与后续工作入口

- [完整历史矩阵及V1/V2协调证据](TERL_MAPPO_PARALLEL_MATRIX_HISTORY_20261010_ZH.md)
- [既有V3r2独立QA交接原件](BATCH01_V3R2_QA_HANDOFF_20261009_ZH.md)，保持原文；其中25k是历史初始授权，新预算以本页及授权JSON为准。
- [Scientific Changelog](BATCH01_SCIENTIFIC_CHANGELOG.md)
- [当前状态、原始screen、曲线与云端SHA](BATCH01_STATUS_20261010_ZH.md)

Batch02/T2、Evidence/Persistent及其它旧实验继续保持原门禁。此文档整理不批准新科学delta、额外预算或正式训练。
