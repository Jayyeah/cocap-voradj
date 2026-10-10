# Batch01 Scientific Changelog

Master-only、append-only。科学源码版本与中央文档提交分别登记；文档提交不是BASE。

| Entry | Date | Class | Scope / decision | BASE |
|---|---|---|---|---|---|
| B01-000 | 2026-10-09 | FACT_RECOVERY | 核验T0 seed9 Stage1累计1M、selected775k、两模式45/50 normal及5/50 collision、末点30%退化；T0交付bb794ca、best SHA256590d4868…b58db4 | 历史T0；不填Batch BASE |
| B01-001 | 2026-10-09 | PLANNING_ONLY | 登记T1/N1/R1/P1/C0与P1-control、四参照九轴矩阵；预算/阈值/初始化为提案；全新run WAITING_BASE | UNFROZEN / SHA=null |
| B01-002 | 2026-10-09 | CONTRACT_COORDINATION | Core candidate→独立QA同SHA→Master冻结；共同env/PPO/eval hash、strict consumer assertions、Storage-before-lease、有限并发benchmark、固定进程source与v1→v2迁移 | UNFROZEN |
| B01-003 | 2026-10-09 | PREREGISTRATION_ONLY | BATCH02 E1/R2/N2/V1/M1/M2；T2 deferred；没有实现/训练/lease | UNFROZEN |
| B01-004 | 2026-10-09 | FACT_RECOVERY | Evidence旧supervisor complete报告已存在但corrected目录缺失/time为空；保留corrected-final pending及Persistent N0。其它worktree原件不修改 | 无科学代码变化 |

## 后续记录字段

`entry_id/date/author/incident_id/base_version/base_sha/arm/run_id/class/changed_axes/old_effective_values/new_effective_values/companion_required_changes/core_fix_sha/qa_report_sha/evidence_paths/comparability/retrain_decision/master_decision`。

分类限于 `FACT_RECOVERY`、`PLANNING_ONLY`、`CORRECTNESS_REQUIRED`、`ALGORITHM_REQUIRED`、`EXPERIMENTAL`、`BASE_FREEZE`、`RUN_REGISTERED`、`VERSION_MIGRATION`、`RESULT_CLASSIFICATION`。真实科学变更必须写旧/新值，不能以“bugfix”“配置优化”遮蔽reward/PPO/obs/terminal差异。

## A1-MASTER-02（2026-10-09T15:18:55.622062+08:00）

| Entry | Class | Scope / decision | BASE |
|---|---|---|---|
| B01-005 | FACT_RECOVERY | A0外部原件/TSV/保护清单核验；home261.60/shared136.77GiB，quota UNKNOWN；74/98GiB均含30 reserve；20.66 reclaim未授权，无删除/归档 | UNFROZEN |
| B01-006 | FACT_RECOVERY | A2科学candidate `40bd91b56cd199e527f7a2bb2317b2c1c5b61e4a`；delivery `bdbd502ff2f071b0cac07a0d0d956a2f35015ea1`；canonical `91dc6ea76c0f7b76ffa43441286db40b9e4ea1e9e19885a328df16ab6d024bf3`；78源码hash/T0未变核验；Core61不等于QA_PASS | SHA=null；QA_SAME_SHA=PENDING |
| B01-007 | CONTRACT_COORDINATION | 发放 `B01-QA-CUDA-40bd91b-20261009T151855`，仅GPU0有界QA correctness；300s测试/600slease、128decisions/8PPO、2GiB torch/4GiBdriver、1thread/1GiB新写入；Master未执行，正式lease/launch仍关闭 | UNFROZEN |
| B01-008 | CONTRACT_COORDINATION | Core旧T0评估域与中央Batch新域不一致，登记冻结前待决；QA已有未提交CPU/延迟导入锁逃逸记录，仅观察未验收；不改candidate | QA_SAME_SHA=PENDING |

科学axes/arm预算和初始化提案本轮未改变；以上为交付登记与QA门禁，无BASE发布、版本迁移或正式结果。下一步仅接收独立QA并评审，不自动冻结。

分类补充：继续沿用B01-002的`CONTRACT_COORDINATION`记录共享合同和有界QA授权；只涉及管理门禁，不代表已发布科学BASE。历史条目保留。

## A1-MASTER-03（2026-10-09T15:38:22.471208+08:00）

| Entry | Class | Scope / decision | BASE |
|---|---|---|---|
| B01-009 | FACT_RECOVERY | 接收A3正式report `974cdd171898a261b7c862622463537cdc31a487`，被测candidate `40bd91b56cd199e527f7a2bb2317b2c1c5b61e4a`、原lock91dc6ea7…24bf3；24份证据hash与JUnit核验；正式BASE_QA_BLOCK，T0 CPU parity/full resume/775k strict load通过，CUDA未执行 | UNFROZEN / SHA=null |
| B01-010 | CONTRACT_COORDINATION | CORE_V1_QA_BLOCKED / WAITING_CORE_V2 / BASE_FREEZE_BLOCKED；登记B1–B4/Q1，等待用户已安排的Core定向修复交新SHA/lock，再A3同新SHA独立复验；旧未使用QA lease关闭，无新lease/训练 | candidate修订，不是BASE发布/迁移 |
| B01-011 | RESULT_CLASSIFICATION | 保留T0 COMPLETE成功与late degradation；A0审计和四参照九轴/预算/初始化提案继续有效，无重新规划；五线及P1-control仍WAITING_BASE | 无新正式实验结果 |

Core/QA实际科学修复由对应owner在各自branch提交；Master本轮只接收结论、登记复验要求与关闭旧权限，不修改共同PPO/env/reward/observation/terminal代码。

## B01-012 — 2026-10-09T17:38:31.228981+08:00 Core V2交付登记（尚未QA验收）

科学candidate `863a0cf55aca0ace8a0aaab36d9166aa4c97268f`，canonical lock `885ec7b8617cf88e2537b08ebf041e1d8bfc41f120b7ec4aef1708c4a513a8a4`；Core证据HEAD `d49cabbbb514b65c2bd28a1b14cf834eea614a20`另列。Master96文件/lineage/diff/证据hash核验通过；T0科学语义保持，V1 B1/B2/B3/B4/Q1修复仅记Core报告已修、待A3。Core77/3skip不是独立QA。状态CORE_V2_DELIVERED / WAITING_QA_V2 / BASE_FREEZE_BLOCKED，QA_SAME_SHA=PENDING、BASE=null；V1阻断历史保留，五线与P1-control WAITING_BASE。无BASE release/migration或训练。

## B01-013 — 2026-10-09T17:38:31.228981+08:00 V2限定QA CUDA lease与A0动态复核

A0所有原件hash复核一致，容量审计/保护清单继续有效；共享可用136.73GiB、RAM available103.58GiB、两卡采样无compute/util0，quota UNKNOWN，无清理/归档授权。新lease `B01-QA-CUDA-V2-863a0cf-20261009T173831`仅V2 GPU0/cuda:0，正式256/8/4，1进程/1线程，300s CUDA/600s active/1024 joint decisions/8 PPO，2GiB Torch/4GiB自身driver/8GiB RSS/2GiB新输出，至`2026-10-09T19:38:31.228981+08:00`且fresh preflight。原V1未使用lease仍关闭，不复用。真实V2 pin checkpoint/source/RNG补充为必验；Master未执行CUDA/benchmark，正式lease与训练门禁不变。

## B01-014 — 2026-10-09T17:38:31.228981+08:00 评估seed差异重核与A3 handoff

Core V2 T0历史2026100800/2036100800/2046101800保持；未来ARM提案2056100900/2066100900/2076100900仍不同、未冻结。必须显式实际集合冻结与QA disjoint/paired-physical/action-RNG核查，禁止把观察过T0 final当新held-out。未更改九轴矩阵/预算/初始化。A3接口：[V2 handoff](BATCH01_V2_INDEPENDENT_QA_HANDOFF_20261009_ZH.md)，CPU/static立即可行；用户交同SHA独立结论及lease收尾后唤醒Master，后续评审才能决定冻结。


## Commander V3 2026-10-09

- V2→V3：实际code/source编译比对、startup execution witness、fail-closed缓存负测、新ARM protocol绑定；不修改T0环境/actor/critic/PPO/reward/动力学。candidate0b2686a，lock9ca6c24e…cdf19。
- CORE_V3_SELFTEST_PASS不等于A3独立验收。BASE未冻结；用户只允许合格且最多25k的PROVISIONAL pilots，不能追认。P1配对control和treatment保留完整775k状态，只有target_kl .02/.01差异。

## Commander V3 六路工程与pilot交付

- P1-control .02/treatment .01：同selected775k完整严格恢复，仅live target KL不同；各25k已停止于800k。新screen配对曲线保存在commander_v3/p1_pilot_screen_curve.csv；best-observed和last只用于pilot报告，不生成final选择。
- R1：仅native distance dense替换CR-MS类annular mean-shift shaping；其余raw分量、MarineEnv、actor/critic/PPO不变；scratch25k已停止。N1：仅CoCap-style256/8/4 actor，native self4/friend5x7/enemy8x7/obstacle5x5，无新增信息；critic固定T0原架构；scratch25k已停止。
- T1：selected775k actor严格warm，新fixed7P2E2O8C critic/Adam/ValueNorm及Stage2场景；包含全部currents、partial agent、多目标native reward和terminal；Stage2/3同规模完整恢复自测通过。跨规模不是exact resume。原升级阈值率.8/.2用于selection20/模式的Stage2与Stage1 retention，final保持盲态；Stage3仅接受绑定hash的正式promotion bundle，禁止pilot升级。
- C0：双维tanh Gaussian含scale/tanh joint Jacobian、pre-tanh latent；连续物理AW直接传入未改原生积分、不AW9量化；九点parity/zero-update概率/full-resume通过，独立mean/sample evaluator。
- 全部CPU/CUDA/source/delta/consumer/checkpoint/new-seed与GPU1并发门禁SELFTEST_PASS；T1/C0各25k pilot真实update后启动。无独立V3签字，所有pilot不自动扩预算、不消费final、不追认为正式。BASE SHA仍null。单Agent执行，无嵌套Agent。

## Commander 状态同步 2026-10-10

- FACT_RECOVERY：V3r2 candidate9cc2d47c、lock384b9ba5及Core交付d6e2010；QA_PENDING、BASE未冻结。Core/QA/六个长线ARM远程HEAD与本地逐一一致，原独立QA证据及交接文档不修改。
- RUN_REGISTERED：用户明确授权的长线PROVISIONAL中，P1-control/treatment、N1/C0完成1M，T1 Stage2完成100k，R1继续到1M；T1/C0由资源恢复队列在槽位释放后真实严格恢复、PPO update验证后运行并已完成。预算未扩大，Stage3未启动，正式RUNNING=0。
- RESULT_CLASSIFICATION：同协议原始screen按字节归档，整理全部曲线及P1 775k→1M paired normal/collision保持图；best-observed与last同时报告，仅描述性PROVISIONAL结果，不生成selection或追认。
- CONTRACT_COORDINATION：用户要求运行中资源约束不再由监督器强停；监督器RAM/GPU/磁盘/租约改为告警，源码/checkpoint/预算/QA门禁保留。锁定训练源码未热修改。
- FACT_RECOVERY：旧矩阵/DAG全文保留为历史，当前入口统一至BATCH01_STATUS_20261010_ZH.md；六任务的工程状态从实际registry刷新，正式WAITING_BASE状态保留。证据与文档同步GitHub，checkpoint权重仍保留服务器、同步路径/hash。
