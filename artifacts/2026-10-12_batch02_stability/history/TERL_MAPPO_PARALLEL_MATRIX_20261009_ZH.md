# TERL-MAPPO Batch01 实验矩阵与运行入口

当前运行快照、best/last、collision及曲线统一见[2026-10-11状态文档](BATCH01_STATUS_20261011_ZH.md)。实时状态使用`python tools/batch01_status.py --current`，中央事实源为[batch_state](../../artifacts/2026-10-09_terl_mappo_batch01/batch_state.json)与[run registry](../../artifacts/2026-10-09_terl_mappo_batch01/run_registry.json)。旧启动PID、25k上限及V1/V2状态已移至[完整历史记录](TERL_MAPPO_PARALLEL_MATRIX_HISTORY_20261010_ZH.md)，不覆盖当前状态。

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

## 2. 三个主要参照合同：成功IQN优先

本轮完整差异、配置消费者和transition原件见[成功IQN合同](TERL_COCAP_ENVIRONMENT_CONTRACT_20261011_ZH.md)。B0 Final roleful/R20与V2 Z05分别有成功证据，不把二者混成已验证的新组合。以下以V2 Z05 Stage1为E1首个桥接参考；N2采用Z05还是先roleful需用户明确。

| 轴 | TERL-IQN 原始实际 | 当前TERL-MAPPO实际 | 成功IQN-CoCap V2参考 / Batch02含义 |
|---|---|---|---|
| Environment | MarineEnv，120×120，原APF/current，decision末碰撞；boundary只罚 | T0同原生；T1改变规模，C0改动作接口 | VorAdjEnv、NormSense V2、零current、v2_fixed APF、synchronized_swept_v1、硬边界/死亡；E1串行bridge |
| Network | entity/type Transformer 256/8/4，Target Selection/fusion、IQN后端 | T0 TERL actor；N1仅CoCap-style backbone；action-free central V | 成功IQN voradj_single_head；MAPPO categorical actor/central V为算法适配，须与task observation分开 |
| Reward | time/distance/global/emergency/collision/goal | 同原；R1仅替换distance项 | 成功Final CR-MS、cartesian_disk、support=.5/.5、CE/PBRS/safety及真实clock；R2恢复原语义 |
| Information | self4/friend5×7/enemy8×7/obs5×5=19；敌方不距离过滤 | N1保持native信息；V含current/time | self9/22slot、NormSense直接探测、global-friendly VorAdj；Z05=.5衰减/.10 floor、physical_only、无role late fusion；N2全合同 |
| Task | ≥3几何capture，8m/related18m；active<3、全capture terminal；timeout3001 | wrapper明确terminated/truncated/pre-reset V | Stage1 4P1E1O，min_active4/horizon3000；mixed捕获后继续recovery/CE hold30/window500；M1/M2目标为完整生命周期 |
| Action | a-major AW9，a={-.4,0,.4}、w={-π/6,0,π/6}；.05×10 | C0 continuous physical AW为独立诊断 | 成功AW9保留；continuous若成为目标须另标MAPPO adaptation |
| Algorithm | off-policy IQN/replay/epsilon/target Q | on-policy PPO/GAE/ValueNorm；actor/V独立Adam | 不移植off-policy replay为PPO task概率；稳定性/初始化调整独立登记 |
| Curriculum | 0/2M/4M/5M/6M，共7M | Batch01仅既有授权；T1 Stage2 100k完成 | 成功IQN 4/1/1→8/2/2→12/3/3，各阶段表现不同；不自动继承预算或宣称全规模成功 |
| Initialization/eval | IQN random-origin | P1完整775k配对分叉；其余scratch或T1仅actor warm | 恢复selected1300k成功Stage1作为只读参考；新MAPPO初始化/clock待确认，best screen不替代正式评估 |

历史失败CoCap-MAPPO corrected scratch Capture（support=1、CE=0、clock offset2M、无recovery、无Z）和Coverage失败仅保留在[历史矩阵](TERL_MAPPO_PARALLEL_MATRIX_HISTORY_20261010_ZH.md)，用于失败模式/debug对照。这些差异不得直接继承为R2/N2/M1/M2的新任务合同。

## 历史、QA与后续工作入口

- [完整历史矩阵及V1/V2协调证据](TERL_MAPPO_PARALLEL_MATRIX_HISTORY_20261010_ZH.md)
- [既有V3r2独立QA交接原件](BATCH01_V3R2_QA_HANDOFF_20261009_ZH.md)，保持原文；其中25k是历史初始授权，新预算以本页及授权JSON为准。
- [Scientific Changelog](BATCH01_SCIENTIFIC_CHANGELOG.md)
- [当前状态、原始screen、曲线与云端SHA](BATCH01_STATUS_20261011_ZH.md)

Batch02准备依赖以[2026-10-11更新DAG](../ops/COCAP_MAPPO_BATCH02_DAG_20261011_ZH.md)为准；[E1实施报告](BATCH02_E1_BRIDGE_REPORT_20261011_ZH.md)记录最新已授权的成功CoCap闭环与必要输入绑定，工程通过、独立验证未完成。T2、Evidence/Persistent及其它旧实验状态不变，无额外长期预算或正式训练。
