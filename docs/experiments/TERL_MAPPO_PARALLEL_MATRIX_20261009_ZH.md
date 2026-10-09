> 2026-10-09 更新：V3r1 自身 SourceGuard 可命中同 size/mtime 旧V2缓存，启动来源门禁不完整。全部六条旧pilot已隔离，不作正式证据；V3r2源码启动修复候选5b6224f正在回归。QA_PENDING、BASE未冻结。监督器registry锁等待已修复，重启前保持停止。两GPU目前均有其他用户任务，之后每卡最多两条自己的线。

## Commander 六路pilot交付状态（覆盖下方历史启动快照）

P1-control/treatment均已完成775k→800k，R1/N1均已完成scratch25k，并按硬上限停止；T1 Stage2 PID1621721与C0 PID1623197已真实PPO update后登记PROVISIONAL_RUNNING，均限定25k。所有训练使用GPU1，GPU0外部任务受保护。T1三规模mask/current/partial生命周期与Stage2/Stage3完整恢复、C0九点原生物理parity/连续概率合同/完整恢复、CPU/CUDA256/8/4与新seed evaluator均SELFTEST_PASS。正式RUNNING=0，QA_PENDING，BASE未冻结；Stage3正式另需Stage2 selection与Stage1 retention条件。P1已生成两条pilot screen保持曲线与best-observed/last/collision，不能作为正式比较结论。

持久监督：`python tools/batch01_status.py --json`；`tools/batch01_supervise.py`在独立tmux运行，不自动重启或扩预算。checkpoint hash、PID身份、磁盘reserve/容量、GPU外部占用与lease均纳入监测；每run完整registry可在CLI断连后恢复。

# 2026-10-09 Commander V3 当前状态（覆盖以下历史快照）

**CORE_V3_SELFTEST_PASS / QA_PENDING / BASE_FREEZE_BLOCKED**。科学candidate `0b2686a06e900a090a34fd3be4e0143ba3f794f0`，canonical lock `9ca6c24e881e4e13de58513bde32e2008805f3e1fd8fe55645c66a9c413cdf19`；交付HEAD `5ecb8774612f2002576b66f9a8a6cba46dad09a6`。最终CPU83 passed+CUDA3 passed；A3原脚本12探针由同一Commander执行，不能冒充独立签字。V2独立报告c20b632的BASE_QA_BLOCK完整保留。BASE共同正式parent仍为空。

用户Commander V2授权一个Agent实施所有ARM，并在对应科学/资源门禁通过后运行最多25k additional joint decisions的明确PROVISIONAL pilot。正式运行仍要求独立V3 QA及冻结；pilot不可自动扩预算、使用final或追认为正式证据。中央run_registry的`pilot_runs`与正式`runs`区分。新ARM协议绑定2056100900/2066100900/2076100900，sample offset100000、scene offsets0/1000/2000，与历史实际种子不重叠；T0历史final不变。最新进程/门禁读取run registry及各run progress.json。

---

# TERL-MAPPO Batch01 并行迁移矩阵与中央管理合同

日期：2026-10-09，Asia/Shanghai。中央唯一写入者Master。当前CORE_V2_DELIVERED / WAITING_QA_V2 / BASE_FREEZE_BLOCKED；T0成功、A0审计和原科学矩阵继续有效。仅授权V2专用conditional CUDA correctness，不冻结BASE或启动实验。

## A1-MASTER-04：Core V2接收，等待A3同SHA复验（覆盖下方历史快照）

登记`2026-10-09T17:38:31.228981+08:00`。科学candidate **`863a0cf55aca0ace8a0aaab36d9166aa4c97268f`**，canonical lock **`885ec7b8617cf88e2537b08ebf041e1d8bfc41f120b7ec4aef1708c4a513a8a4`**，Core交付HEAD **`d49cabbbb514b65c2bd28a1b14cf834eea614a20`**另列。Master核验96共同文件candidate/delivery/worktree一致、T0原科学源/配置不变、candidate→delivery只有14个报告/证据文件增量。Core77 passed/3 CUDA skipped/0 failed只记自测，五项V1问题为CORE_V2_REPORTED_FIXED_PENDING_A3。V1 QA_BLOCK历史及旧lease closeout保留，released BASE/migrations仍空。

`BATCH01_BASE_SHA=null`、`QA_SAME_SHA=PENDING`；T1/N1/R1/P1/C0/P1-control继续WAITING_BASE。当前只发V2 QA lease **`B01-QA-CUDA-V2-863a0cf-20261009T173831`**：物理GPU0/进程内cuda:0，单进程/CPU线程1，正式256/8/4、两个CUDA节点加真实pin source/RNG/full-state补充，总CUDA≤300s/active≤600s/1024 joint decisions/8真实PPO，Torch≤2GiB、自身driver≤4GiB、RSS≤8GiB、新输出≤2GiB。有效至`2026-10-09T19:38:31.228981+08:00`，激活须fresh preflight且剩余≥10分钟；Master未执行CUDA。正式GPU_LEASE仍NOT_REQUESTED。

A0报告/TSV/保护清单重新hash匹配，本轮资源样本`2026-10-09T17:38:21.419399+08:00`–`2026-10-09T17:38:31.226733+08:00`，共享可用136.73GiB、RAM available103.58GiB、两卡util0且未见compute PID；其它用户CPU任务受保护。quota UNKNOWN，正式启动前动态容量/IO/retention gate未解除，无清理授权。

**evaluation seed proposal仍与Core当前T0不同**：T0历史screen/selection/final为2026100800/2036100800/2046101800；未来ARM提案2056100900/2066100900/2076100900未冻结。保持T0历史，未来ARM实际seed集合须显式冻结并经QA核查，禁止复用已观察T0 final。本轮只更新交付与门禁，不改九轴/预算/初始化提案。

直接转交[A3 V2独立QA handoff](BATCH01_V2_INDEPENDENT_QA_HANDOFF_20261009_ZH.md)，对应[机器lease](../../artifacts/2026-10-09_terl_mappo_batch01/master04/qa_smoke_lease.json)与[Core核验回执](../../artifacts/2026-10-09_terl_mappo_batch01/master04/core_v2_verification.json)。现有formal test占位manifest必须补充真实V2 pin checkpoint guard；CPU/static可先行。NEXT WAKE-UP为用户转交同SHA/lock的A3正式报告、CUDA lease收尾/WAIT与seed reconciliation；后续Master才可能评审冻结，不自动推进。

## A1-MASTER-03：A3正式阻断，等待Core V2（覆盖下方历史快照）

登记时间`2026-10-09T15:38:22.471208+08:00`。已fetch核实A3远程branch `audit/terl-mappo-batch01-base-qa-20261009`、正式HEAD **`974cdd171898a261b7c862622463537cdc31a487`**；[固定QA报告](https://github.com/Jayyeah/cocap-voradj/blob/974cdd171898a261b7c862622463537cdc31a487/docs/TERL_MAPPO_BATCH01_BASE_QA_20261009_ZH.md)结论 **BASE_QA_BLOCK**，被测科学candidate **`40bd91b56cd199e527f7a2bb2317b2c1c5b61e4a`**，不是Core文档HEAD。Master核验报告和24份证据hash、JUnit、candidate/lock；QA提交仅新增报告/证据、原78共同源码不变。详见[正式接收回执](../../artifacts/2026-10-09_terl_mappo_batch01/master03/qa_block_receipt.json)。

中央明确登记 **CORE_V1_QA_BLOCKED / WAITING_CORE_V2 / BASE_FREEZE_BLOCKED**；`QA_SAME_SHA=BASE_QA_BLOCK`、`BATCH01_BASE_SHA=null`，五条新线及P1-control仍WAITING_BASE。旧candidate永不因Core61通过或局部CPU通过自动冻结。本轮等待已由用户安排的Core定向修复，不再派Agent。

| Issue | A3正式结论 / Core V2复验门禁 |
|---|---|
| B1 | 高：未声明lazy/dynamic依赖改变后rollout/PPO/full checkpoint保存恢复仍通过；须在受影响rollout被接受或learner状态改变前拒绝，save/load也必须fail closed |
| B2 | 连续动作adapter的callable identity启动检查可绕过；须验证实际构建adapter的动作能力 |
| B3 | 连续density guard额外采样entropy消耗RNG；须使用density-only或恢复检查RNG |
| B4 | critic EV元数据应说明post-update denormalized V对pre-update GAE returns；保持历史learner数值不变 |
| Q1 | independent QA branch测试不可移植；读取committed lock的fixture与Core-only生成权限测试分开，不放宽生产门禁 |

**T0成功保留**：Stage1 seed9累计1M、selected775k，两模式normal45/50与collision5/50、1M末点normal6/20退化不变。A3独立正式CPU256/8/4、完整256 decisions/768active transitions/6paired minibatches、默认T0精确parity/完整resume及真实775k strict load通过。原suite55pass/4fail/2CUDA skip，committed-lock fixture59pass/2CUDA skip；不将局部通过、finite或CUDA skip升级为BASE PASS。

**Storage审计及现有矩阵继续有效，不重新规划**：A0原件/保护清单/容量登记保持；quota UNKNOWN、共享余量不是独占quota，正式启动前动态复核；无清理授权。四参照九轴profile、每线科学delta/预算/初始化提案未改，QA列出的arm剩余证据另记门禁。

旧V1 smoke lease已未使用并关闭：[closeout](../../artifacts/2026-10-09_terl_mappo_batch01/master03/qa_lease_closeout.json)，原grant保留为历史。A3本轮实际CPU-only、CUDA未执行；当前无Batch01活跃QA或正式GPU lease，旧窗口不可用于V2。Master不运行/停止任何实验，不发正式lease。

**NEXT WAKE-UP**：接收Core V2新的科学candidate SHA、canonical lock与B1–B4/Q1修复/diff证据，再安排A3对完全相同的新SHA和lock独立复验。V1通过项不可自动转移；CPU/static可先行，V2 CUDA需fresh资源检查和新的专用有界lease。参见[Core V2复验合同](BATCH01_CORE_V2_REQA_CONTRACT_20261009_ZH.md)。共同QA通过后仍须后续Master评审才可能冻结；各arm正式启动门禁独立保留。Core V1/V2是candidate轮次，BASE-v1尚未发布。

## A1-MASTER-02 历史交付登记（当前由上方Master03覆盖）

登记时间：`2026-10-09T15:18:55.622062+08:00`。恢复中央local=origin=GitHub `5c0641635af6cbbf6a8dcd38baeb619024e9ebdc`。Core交付HEAD `bdbd502ff2f071b0cac07a0d0d956a2f35015ea1` 已push且从其可达科学candidate **`40bd91b56cd199e527f7a2bb2317b2c1c5b61e4a`**。二者不能互换；`BATCH01_BASE_SHA=null`、`QA_SAME_SHA=PENDING`。

Master对candidate Git对象、交付对象、工作树的78个lock源码hash逐一一致核验；canonical digest **`91dc6ea76c0f7b76ffa43441286db40b9e4ea1e9e19885a328df16ab6d024bf3`**。candidate至交付仅4个证据/文档文件变化；T0原源码/配置保持一致。Core JUnit确有61 passed/0 failure/error/skip，属于Core自测，不能自动赋值QA_PASS。详见[Core核验回执](../../artifacts/2026-10-09_terl_mappo_batch01/master02/core_candidate_verification.json)和[收到的candidate lock](../../artifacts/2026-10-09_terl_mappo_batch01/master02/received_core_candidate_lock.json)。后者只是交付快照，中央冻结lock的digest仍为null。

A0实际原件在`/home/yjq/storage-audit-A0-20261009`，并未假定已进入中央Git。已核验报告、TSV行数/hash、保护清单及零删除/归档记录；中央仅保存[容量审计回执](../../artifacts/2026-10-09_terl_mappo_batch01/master02/storage_a0_receipt.json)。closing `/home/yjq`261.60GiB、共享可用136.77GiB；个人quota UNKNOWN，共享余量不是独占quota。5线74GiB、8并发98GiB均含30GiB reserve；20.66GiB可清理候选仍未授权。状态为**CAPACITY_AUDIT_COMPLETE_DYNAMIC_RECHECK_REQUIRED**；正式启动前重审实际output mount、quota、保留峰值与其它任务增长。

资源窗口 `2026-10-09T15:09:02.192187+08:00`–`2026-10-09T15:09:12.049591+08:00`：两卡util mean/peak均0、未观察到compute PID；GPU0 min free48521MiB、GPU1 48522MiB；RAM available106.14GiB、128CPU load1=3.31、共享disk available136.76GiB。只支持一次有界QA正确性smoke，不构成并发吞吐或容量结论。[QA lease](../../artifacts/2026-10-09_terl_mappo_batch01/master02/qa_smoke_lease.json)：物理GPU0/进程内cuda:0，单进程/单线程，两个允许节点串行，总CUDA测试≤300秒、active lease≤600秒、joint decisions≤128、PPO calls≤8、torch allocation≤2048MiB/自身driver VRAM≤4096MiB、RSS≤8GiB、新写入≤1GiB、evaluator workers=0；有效至`2026-10-09T17:18:55.622062+08:00`，启动前必须重新采样并留足10分钟。Master未执行smoke，正式GPU_LEASE仍NOT_REQUESTED。

QA工作树观察HEAD `40bd91b56cd199e527f7a2bb2317b2c1c5b61e4a`，branch `audit/terl-mappo-batch01-base-qa-20261009`，已有未提交CPU suite/问题复现；[interim回执](../../artifacts/2026-10-09_terl_mappo_batch01/master02/qa_interim_observation.json)仅记进展，不代替验收。其延迟依赖源码锁逃逸记录为REPORTED_PENDING_FINAL_QA_TRIAGE，等待QA正式结论。

冻结前待解决：Core的screen/selection/final沿用T0 `2026100800/2036100800/2046101800`，中央Batch01新域提案是`2056100900/2066100900/2076100900`。先区分历史T0复现与新arm held-out合同，检查实际seed集合与采样RNG；QA返回reconciliation，Master审定后由Core另交合法版本/配置接口。不得在当前candidate上改lock或把已见过T0 final冒充新held-out。

直接转交[BASE独立QA handoff](BATCH01_BASE_INDEPENDENT_QA_HANDOFF_20261009_ZH.md)。本轮结束等待QA验收，不自动冻结BASE，不启动T1/N1/R1/P1/C0/P1-control。下方初次恢复段和旧PID仅为历史快照。

## 1. 权威状态与恢复事实

唯一中央状态：[AC_MASTER_DAG](../ops/AC_MASTER_DAG_20260921_ZH.md) 与 [state.json](../../artifacts/2026-09-21_ac_master_dag/state.json)。本目录文档和 Batch01 JSON 是中央登记的附件；发生不一致时 fail closed，不能自行挑选较宽松的状态。

初次规划恢复快照：local=origin=GitHub HEAD为`381b61dba6c94478ed23b712d39018fc70e2341f`，当时中央clean。TERL交付为`bb794ca8435f06b9fa5693c0fed98f567320decf`；Core/QA当时均在该提交，Core clean、QA detached clean。**该提交是已验证T0交付来源，不是BATCH01_BASE_SHA。** 当前Core/QA状态见顶部A1-MASTER-04登记；不合并或修改其它worktree科学代码。

固定事实来源（按提交读取，旧启动段落不能覆盖最终结论）：

- [Stage1 Final](https://github.com/Jayyeah/cocap-voradj/blob/bb794ca8435f06b9fa5693c0fed98f567320decf/docs/TERL_MAPPO_STAGE1_FINAL_20261009_ZH.md)、[Migration Contract](https://github.com/Jayyeah/cocap-voradj/blob/bb794ca8435f06b9fa5693c0fed98f567320decf/docs/TERL_MAPPO_MIGRATION_CONTRACT_20261008_ZH.md)、[Implementation](https://github.com/Jayyeah/cocap-voradj/blob/bb794ca8435f06b9fa5693c0fed98f567320decf/docs/TERL_MAPPO_IMPLEMENTATION_20261008_ZH.md)、[1M Learning](https://github.com/Jayyeah/cocap-voradj/blob/bb794ca8435f06b9fa5693c0fed98f567320decf/docs/TERL_MAPPO_1M_LEARNING_20261008_ZH.md)、[Central Handoff](https://github.com/Jayyeah/cocap-voradj/blob/bb794ca8435f06b9fa5693c0fed98f567320decf/docs/TERL_MAPPO_CENTRAL_HANDOFF_20261008_ZH.md)。
- TERL-IQN 原始 vendor 固定 `143359b2722d49c29b4fecc0ad1fd8d46326e45a`；历史 CoCap MAPPO/report parent `80debaef0f5983d8128b5059c90d65acc9216c10`，corrected learner lineage `25dd0f8`。历史报告包括 `MAPPO_SCRATCH_PRIMITIVES_CONTRACT_AUDIT_20260923_ZH.md` 和 `MAPPO_CAPTURE_SCRATCH_REPRO_20261006_ZH.md`，不能用分支名替代 SHA。
- Evidence remote `6b7aab28546e1a30c60f84d12de3bde70930f670`；Persistent remote `751ee2fdf8e0134504e9b4f033904cc610dd6891`。Persistent strict service A/B/C=9/20、13/20、14/20，relaxed all-three capture 均20/20、service=13/20、13/20、14/20；保留既有 COMPLETE 分类和 N0 决策。
- Evidence 本地未提交旧 supervisor 输出在 10-09 01:25 声称 complete，Local S3 selected=600k；`final_heldout_50` 三臂报告存在，Coverage time n=0，`final_heldout_50_corrected` 目录缺失。登记 `SELECTION_REPORTED_PENDING_AUDIT / LEGACY_FINAL_PRESENT / CORRECTED_FINAL_PENDING`，不宣称 corrected 正式比较完成，不提交其动态原文件，不解锁第二批训练。

### T0 已核验锚点

Stage1=3P1E0obs4cores，seed9/actor109，random-origin，无 teacher/BC。100k full-state 精确续训至累计1M；仅预算改变。实际1,000,000 joint decisions、3,000,000 active transitions、3,920 PPO calls、22,994 paired optimizer minibatches。41点 screen 共980局、selection120局、final100局；两个模式共享50个物理初态，不能视为100个独立 seed。

selected=775k；实际 `step_000775000.pt` 与 `best.pt` SHA256 均重新核对为 `590d486876d4fca1cd75311c2dcc1255f847a1f756d91eed202e737193b58db4`。final seed base=2046101800；argmax/sample normal capture 均45/50，collision均5/50。1M末点 screen normal 均6/20，collision=10/20与12/20；late degradation 保留为事实，不能写成 reward、critic 或 entropy 的已证因果。历史约24.31 decisions/s只是已结束单 run 的累计指标，不是当前并发容量。

服务器锚点：`/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/runs/terl_mappo_stage1_seed9_1m_continuation/checkpoints/step_000775000.pt`。初次恢复审计见[recovery_snapshot.json](../../artifacts/2026-10-09_terl_mappo_batch01/recovery_snapshot.json)：旧trainer/controller/evaluator已退出，当时外部PID1335255占用两卡。本轮新采样未观察到compute PID；任何其它任务仍受保护，不能认领、停止或修改。

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

## 3. 首批登记与相对 T0 的变量合同

`BATCH01_BASE_SHA = null`，`BASE-v1 = UNFROZEN`。五条新线与P1配对control均 `WAITING_BASE`。T0 `COMPLETE` 是已完成历史锚点，**不属于等待新BASE才能成立的训练**；其Batch01可运行绑定仍待BASE。T2 `DEFERRED`，T1失败后只重新审查需求，不自动启动多seed。

| Arm / owner接口 | 问题与唯一主变量 | 初始化提案 | 有界预算提案 | 状态 |
|---|---|---|---|---|
| T0 / Master | 原生Stage1可学锚点；保留末点退化 | 已完成seed9 random-origin | 已完成累计1M，selected775k | COMPLETE（历史） |
| T1 / Curriculum | accelerated native curriculum：先4P1E，再7P2E | selected775k actor/V迁移；shape/state兼容性待Core/QA | 每阶段100k additional，25k/50k review，累计新增最多200k | WAITING_BASE |
| N1 / Backbone | 仅actor结构迁至CoCap style；原生信息不变 | fresh seed9/109，V/Adam/ValueNorm fresh；不移植不兼容775k actor | scratch1M，25k cadence | WAITING_BASE |
| R1 / Reward | 仅native distance dense term替换为CR-MS类capture dense shaping | fresh seed9/109，全状态fresh，避免旧V/ValueNorm与新reward混淆 | scratch1M，25k cadence | WAITING_BASE |
| P1 / Stability | 唯一配置变量 target_kl .02→.01 | 775k完整actor/V/Adam/ValueNorm/env/RNG/counters分叉 | additional225k，到累计1M | WAITING_BASE |
| C0 / Continuous | 原生AW9 categorical→bounded continuous a/w | fresh seed9/109，全状态fresh；不把离散head当连续pretrain | scratch1M，25k cadence | WAITING_BASE |

预算仅为计划上限，不是本轮launch授权。N1/R1/C0与T0做同任务单seed描述性对照；新的Batch01 final seed域独立于已经看过的T0 final。若需要同BASE fresh T0对照，由Master另登记，不让各线自行增加baseline run。

### 逐轴有效差异（其余继承 T0）

| 轴 | T1 | N1 | R1 | P1 | C0 |
|---|---|---|---|---|---|
| Environment | 原native Stage2=4/1/1obs/6cores/>13m；Stage3=7/2/2obs/8cores/>15m，保留native map/APF/末碰撞 | T0 | T0 | T0 | T0物理；新增连续action入口校验 |
| Network | 同actor；critic须支持变长agents/current，参数兼容或显式迁移 | CoCap-style actor；hidden256/8/4与head固定，critic T0；适配器与pooling属于结构delta | T0 | T0 | TERL encoder/V不变，仅连续mean/logstd head |
| Reward | 同native函数；人数global分支、多敌方dense重复是实际伴随变化 | T0 | 只替换distance项；time/global/emergency/collision/goal、capture判定不变；CR-MS系数/clock/target分配待精确提案 | T0 | T0 |
| Information | 原生同过滤规则；更多agents/targets及critic current容量是课程伴随变化 | 原生19token信息、可见性、ordering、mask；self4→CoCap输入只可确定性映射/常量padding，不新增检测、VorAdj、role或Z | T0，reward不可泄漏给actor | T0 | T0 |
| Task | 4P1E→7P2E完整原生任务；不是仅人数单变量 | T0 | T0成功/terminal规则 | T0 | T0 |
| Action | T0 AW9/物理 | T0 | T0 | T0 | tanh-squashed Gaussian：a=.4 tanh(u_a)、w=(π/6)tanh(u_w)；禁止snap回AW9 |
| Algorithm | common PPO；critic维度/optimizer迁移须登记 | common PPO、central V不变；adapter必要变化须QA | common PPO，归一化统计按fresh初始化 | common PPO其余参数完全不变；KL guard只减少update不提前停止总预算 | common PPO ratio/GAE/V；必要连续logprob含tanh和affine Jacobian、entropy估计/确定性mean模式；不是第二个自由超参搜索 |
| Curriculum | 两阶段各100k；当前阶段完成且gate PASS后才进入下一阶段 | T0 Stage1，scratch1M | T0 Stage1，scratch1M | T0 Stage1，775k→1M | T0 Stage1，scratch1M |
| Initialization | actor/V兼容权重warm start；新env/current/agent形状不能直接exact resume；Adam/ValueNorm reset/迁移选择待冻结 | 全fresh | 全fresh | 完整状态strict restore，仅target_kl与run输出/budget元数据登记变更；需要resume显式allowlist | 全fresh |

四参照逐线比较的完整机器表：[cross_arm_delta_matrix.json](../../artifacts/2026-10-09_terl_mappo_batch01/cross_arm_delta_matrix.json)，每个arm×参照×九轴记录参照值、有效提案值、分类和未解决项。

| Arm | 相对TERL-IQN原版 | 相对T0 | 相对历史CoCap-MAPPO | 距下一步CoCap目标的剩余差异 |
|---|---|---|---|---|
| T1 | IQN→MAPPO必要变化；保留native阶段但压缩预算、775k迁移 | 课程与对应env/task/信息容量变化；不改奖励公式 | native环境/信息/奖励/课程仍不同，actor非CoCap | E1/R2/N2尚未桥接；规模迁移不代表CoCap full-task |
| N1 | 替换actor结构和IQN算法；原生任务信息保留 | actor结构+无新增信息的adapter；fresh初始化是明确对照设计 | actor family接近，19token原生信息及env/reward/V状态仍不同 | 仍缺E1环境、R2完整奖励、N2完整观测；无Z |
| R1 | MAPPO必要变化+一个dense reward项迁移 | distance shaping替换；其余原生reward分量保持 | 仅CR-MS类dense项接近；非完整Final reward/safety/clock | 仍需R2全部capture合同、E1/N2；不能称Full CoCap Capture |
| P1 | MAPPO必要变化；原生Stage1保留 | 只target_kl .02→.01，完整775k状态 | PPO family接近但环境/结构/信息/任务全不同 | 是稳定性诊断，不推进环境/奖励/观测迁移 |
| C0 | IQN→连续MAPPO；原物理边界与任务保留 | action distribution/head与相应logprob/entropy/eval必要变化 | categorical AW9不同，native环境/信息/奖励仍不同 | 连续a/w结论单独成立；不自动选为CoCap统一动作 |

### 各线评审门禁提案

统一报告normal capture、collision类型、ring2/ring3/strict、时间mean/median/p90与success/failure/time-limit n、boundary计数、raw reward组件、KL/ratio/clip/entropy、梯度、EV与return variance、ValueNorm、updates/active transitions。finite是工程门禁，不代替科学学习。

- T1：先做zero-shot及Stage1 retention；每阶段完整100k后独立50物理seed/每模式final。建议迁移PASS为两模式normal≥40/50、collision≤10/50，并在Stage1 retention同样满足；只作本cohort资格，不宣称总体成功率下界。阶段失败标记结果并交Master审T2，不扩预算。不得靠stage2失败自动绕到stage3。
- N1/R1/C0：保留固定预算末点和预声明selected两种结果、完整曲线，比较normal/collision/几何与时间；没有多seed不宣称actor/reward/action普遍优越。R1直接验证逐项reward重构、驻留反事实与native事件不变。
- P1：必须有 `P1-control`（同775k完整状态，target_kl=.02）配对分叉计划。历史775k→1M是已有退化参照，但新BASE full-state control的兼容性需QA；未获运行授权不能执行。主要指标是末点normal/collision及775k后固定25k点的保持曲线/AUC，不能再以best selection掩盖末点；两模式分别报告。建议维持资格为末点normal≥40/50且collision≤10/50；优于配对control只称此状态分叉的描述性证据。

## 4. BASE、锁与run登记

附件：[batch_contract.lock.json](../../artifacts/2026-10-09_terl_mappo_batch01/batch_contract.lock.json)、[run_registry.json](../../artifacts/2026-10-09_terl_mappo_batch01/run_registry.json)、[batch_state.json](../../artifacts/2026-10-09_terl_mappo_batch01/batch_state.json)、[Scientific Changelog](BATCH01_SCIENTIFIC_CHANGELOG.md)、[统一handoff](BATCH01_HANDOFF_TEMPLATE_20261009_ZH.md)。只有Master更新这些中央附件和DAG/state。

冻结顺序：Core提交并push真实candidate SHA/完整源码hash清单 → 独立QA对**同一个candidate SHA**验收并push报告 → Master核对交付、填写真实 `BATCH01_BASE_SHA` 与 lock digest、冻结BASE-v1 → 各线从该SHA建立delta branch → 审核arm差异、Storage、benchmark和lease → 后续授权训练。不能用T0 SHA、placeholder、分支名或QA自己的报告HEAD填BASE。冻结前五条线始终WAITING_BASE。

冻结时归档不可变版本到 `artifacts/2026-10-09_terl_mappo_batch01/bases/BASE-v1/`（lock、source/dependency manifest、Core/QA引用）；后续BASE-v2建立新目录，当前lock只作版本入口，旧run引用原不可变lock/hash。Storage容量审计已经接收；QA CUDA以本轮A0审计+当前smoke资源门禁+专用短lease为前置，启动时再复核。正式训练Storage/GPU/benchmark门禁独立保留，不形成QA等待冻结的循环。Master本轮未执行GPU工作。

每run必须登记 `base_version/base_sha/base_lock_sha256/arm_head/delta_hash/resolved_config_sha256/scientific_sources_sha256/runtime_contract_hash/evaluator_sha256/dependency_lock_sha256/parent_checkpoint_sha256/init_mode/seed_manifest/counter_units/output_root/lease_id/command`。训练source树必须clean并固定到科学源码hash，依赖、原vendor和所有import consumer纳入manifest；启动前hash验证失败即退出。各arm的common env/PPO/evaluator/dependency hashes必须一致；只有批准delta文件和配置允许不同，R1/C0/N1通过common入口选择各自插件。禁止各线拷贝或私改共同PPO、native环境和评估器。

统一基础PPO精确值来自T0 manifest：rollout256、gamma.99、lambda.95、clip.2、3epochs/2minibatches、actor LR3e-5、V LR1e-4、entropy.01、value coef1、gradclip.5、targetKL.02、Adam eps1e-5、ValueNorm beta.99999/eps1e-5。P1只允许targetKL=.01；C0连续概率计算是必要实现差异，连续entropy定义/初始std仍须明确写进arm合同，不能默默调entropy coefficient。

统一评估：新screen/selection/final域提案分别2056100900/2066100900/2076100900起；scene_index偏移100000、episode_index递增，same-scene arms配对物理seeds，不同动作sampling RNG隔离。与历史T0三个域及旧100k final逐项做集合disjoint检验，QA通过后冻结。regular screen10/模式、milestone20/模式、top3 selection20/模式、final50/模式；同场景完整3001 horizon/spawn，用parameter_updates=0独立CPU evaluator。C0的deterministic mean/sample明确标名，禁止将mean说成离散argmax。fingerprint/hash/计数来自实际episode，不以expected manifest代替验证。不得使用final调参或选checkpoint。

## 5. Core / QA / Storage 交付登记与待验收

| 接口 | 必需交付 | 当前状态 |
|---|---|---|
| Core | push candidate SHA；共有config/arm插件/scene接口/连续动作入口/strict resume allowlist；T0不变路径parity；变人数/洋流critic及权重迁移规则；runtime consumer assertions；源码与依赖hash；统一eval/seed/retention/benchmark instrumentation；commit影响分类 | CORE_V1_QA_BLOCKED；WAITING_CORE_V2；新SHA/lock与B1–B4/Q1修复交付待收 |
| QA | 独立审同candidate；vendor字节、T0actor有效路径/physics/reward/info/terminal/pre-reset bootstrap/GAE/ValueNorm/active loss/dropout；CPU full-state restore与CUDA后续经lease的有界路径；N1信息不扩展、R1组件、C0Jacobian/action bounds、T1 multi-target/current、P1唯一参数；final seed隔离与manifest fail-closed；报告SHA/限制/FAIL项 | BASE_QA_BLOCK；正式报告974cdd17已接收；CPU parity通过、CUDA未执行，等待新candidate同SHA复验 |
| Storage | 按GPU计划审root及实际output mount、df字节/inode、各run目录du、checkpoint/resume/metrics/GIF/eval保留峰值与atomic临时文件；并发总峰值+其它任务增长+≥15GiB margin；write/fsync速度、ETA、清理建议/保护清单；有时效的PASS或WAIT | A0容量审计已接收；formal动态复核仍需完成，quota UNKNOWN，禁止清理 |

无需等待GPU做独立CPU规划；没有可评审新交付时以handoff结束，不长期轮询。用户转交Core/QA/Storage handoff后Master继续冻结；不给现有工作树发消息、不启动它们的任务。

## 6. 资源门禁与有限并发benchmark（设计，未执行）

目标每GPU3–4条，允许按容量降为1–2条；不是最低并发承诺。正式训练/benchmark须先Storage动态容量PASS，再申请独立lease；QA correctness smoke单独依A0已完成审计、fresh资源门禁与本轮专用短lease执行，不解锁正式门禁。`CUDA_VISIBLE_DEVICES=physical_gpu`，程序内`cuda:0`；必须核对GPU UUID，不能依logical编号claim设备。

建议benchmark分级：1/GPU基线 → 2/GPU → 3/GPU → 4/GPU；前一级PASS且总吞吐收益明显才升级。每级至少3个重复窗口，warm-up256 decisions，measurement2048 decisions/run或5分钟上限；未完成2048时如实报告实际steps与置信不足。使用冻结BASE、全宽256/8/4、Stage1及T1最高计划场景/C0代表混合，真实PPO update、25k样式checkpoint写入和完整CPU evaluator负载；benchmark专用fresh输出/seed，不进入formal结果、不复用其权重。

每秒记录GPU util/temp/VRAM及每run peak、CPU per-core/%/load/线程数、RSS/可用RAM/swap-in/out、iowait、写字节/s与checkpoint save/fsync p50/p95、所有已有受保护jobs heartbeat。主目标 `sum(valid completed joint decisions)/measurement wall seconds`，另报active transitions/s、updates/s、各场景吞吐；不可将7P与3P的agent transitions直接当同难度比较。必须同时量eval production/consumption、pending checkpoint数、oldest age、queue增长率和drain ETA，避免训练加速把评估/磁盘拖垮。

建议接受条件（由Storage/benchmark实测复核后冻结）：无OOM/NaN/Inf、无受保护job stall/明显降速；新增一级总有效throughput≥前一级1.10倍、单run throughput≥其独占基线0.5倍；按实际available CPU留≥25%余量，iowait p95<10%、无持续swap增长；RAM余量≥max(16GiB,20%)；VRAM余量≥max(8GiB,20%物理容量)；checkpoint p95不超过单run2倍且无持续写入队列；evaluator backlog可稳定或在≤一个25k checkpoint周期内drain；磁盘可用仍覆盖Storage并发峰值与15GiB margin。单一GPU utilization低不构成PASS。

evaluator初始每run最多2 CPU workers，**全机Batch初始总额4 workers**，由CPU实测增减；不能给8条run各自默认2 workers而遗漏总量。隔离output/worktree，latest+milestones+screen top3/selected保留策略须先测full-state文件实际大小；hardlink不得重复计费，原775k锚点与P1父状态永久保护，不清理无关任务。

推荐顺序：CPU提案并行准备五条线 → Core/QA冻结 → Storage审计 → 单run/混合有限benchmark → T1在GPU0、P1+配对control按GPU1实测容量安排 → 容量允许加入N1、R1 → C0正确性通过后最后加入。若只允许1/GPU，先T1和P1，control/N1/R1/C0依次排队；若2/GPU，T1+N1、P1/control同卡先完成配对窗口再排R1/C0；不得在未登记lease时启动任何一条。第二批继续锁定。

## 7. 动态修复与版本迁移协议

1. BASE-v1冻结后各线delta从同一commit派生；run的源码、resolved合同与eval hash固定。
2. 发现common-code bug：arm提交BUG handoff，包含触发输入、首次影响step/版本、source hash、受影响消费者和最小repro；Master登记incident，相关**新launch**关闭。存活process不hot patch。会产生无效样本的run可经Master决定暂停/终止，只操作被授权本批进程；无关jobs不动。
3. Core在修复分支最小修正，区分CORRECTNESS_REQUIRED、ALGORITHM_REQUIRED与EXPERIMENTAL；不得顺手改reward/PPO/obs/terminal。QA独立复现并审parity，出受影响矩阵。
4. Master发布BASE-v2真实SHA、锁digest、common hashes与migration ledger；旧BASE-v1不可覆盖，旧工件保留。尚未开始的arm重base并重新审delta/runtime；在跑arm仅记录旧版本，不能自动改import路径。
5. Master逐run裁定：记录层修复且科学轨迹不变可继续训练并用隔离新eval补报告；policy/loss/reward/obs/terminal/physics改变则旧结果不与v2拼接，需要登记RETRAIN_REQUIRED或受控新run。若同因果对照两臂受影响，成对重训；只影响单arm也须评审对照可比性。所谓不受影响须QA证据，不能默认。
6. full-state resume跨版本必须保存parent SHA/hash、optimizer/VNorm/RNG/env兼容证明、迁移allowlist与exact/functional分类；语义改变不得称bit-exact。新run ID、新output、source hash落盘后才可重新申请lease与运行授权。

迁移账本记录 `incident_id/from_base/to_base/core_fix_sha/qa_report_sha/affected_runs/semantic_axes/last_valid_step/comparability/retrain_decision/migration_method/master_decision/evidence_paths`；changelog append-only，禁止已运行各线出现未登记reward/PPO/observation/terminal差异。

## 8. 第二批预登记（没有实现或训练授权）

| ID | 依赖与前置gate | 预期合同差异 | 可并行准备 |
|---|---|---|---|
| E1 Environment Bridge | T1规模与T0/P1状态兼容证据；Core/QA runtime断言 | Marine→VorAdj/V2、APF/current/spawn/collision/boundary/terminal；逐项桥接不一次归因 | CPU源代码差异表与paired transition fixture设计 |
| R2 Full CoCap Capture | R1诊断+E1 PASS；representation/reward/non-null门禁；Master审 | 完整Final CR-MS/support/goal/safety/clock，成功判定/credit差异明确；不是首批R1 | 奖励公式与native/CoCap事件账本 |
| N2 Full CoCap Observation | N1结构/信息隔离PASS+E1 PASS | self9/22token、NormSense V2、detector、ordering、VorAdj/role全部观测合同；无自动Z/local-friendly V3 | 信息可得性图、字段/slot/mask映射 |
| V1 Pure Coverage | E1 PASS、完整Coverage reward/info/non-null；历史M-COV仅背景 | 无敌方、CE/PBRS/hold/safety、coverage success/terminal/time；新任务 | 正控来源审计与eval定义 |
| M1 Mixed | E1/R2/N2/V1 PASS，capture→recovery与retention gate | 混合phase、reward时钟、post-capture recovery/CE、terminal与service | phase状态机与held-out manifest设计；Full-Task PPO历史门禁仍须显式裁定 |
| M2 Evidence Representation | N2/M1合同ready；Evidence corrected final合同审计完成；Persistent N0联合再审 | Z/local/global信息差异与传播、slot/TTL/update；global oracle不能称公平distributed baseline | CPU重读handoff、传播诊断与不泄漏检查设计 |

上述接口名称采用 `BATCH02-*`，不覆盖历史B/R/N/C任务。真正Batch02 BASE、预算、seed、gate和训练授权必须依据Batch01结果再次冻结。T2多seed延后；T1失败只是审查触发，不能绕过当前无训练授权。

## 9. 交付与复核

中央登记校验工具：[validate_terl_mappo_batch01.py](../../tools/validate_terl_mappo_batch01.py)，仅验证元数据一致性、不启动环境或模型。正常commit/push中央branch，push后fetch并直接ls-remote核验；最终交付HEAD由Git/final handoff给出，避免把包含自身的commit SHA写进本commit。当前附件中的HEAD字段明确是恢复或核验快照，不能冒充未来交付SHA。

交付核验：中央metadata consistency PASS，4项内存注入负检查均拒绝（文档HEAD冒充candidate、冻结前正式lease、QA lease扩大为正式工作、共享空间冒充独占quota）；readiness按预期FAIL（QA/BASE/arm/Storage动态复核/seed/benchmark/正式lease/授权未完成）。此前规划commit `a5d5072c16e0cb0e1f38c2dfc4f1eb3ad95921ff` 的[旧同步回执](../../artifacts/2026-10-09_terl_mappo_batch01/remote_verification.json)保留为历史。A1-MASTER-02交付另行正常push/fetch/ls-remote核验；当前交付HEAD见Git与本轮handoff，不自引用。

A1-MASTER-02登记交付 `a2e8de0fe5964bdfa325206ef1e6e13e52ab7434` 已正常push，完成fetch且local=origin=GitHub ls-remote。见[本轮同步回执](../../artifacts/2026-10-09_terl_mappo_batch01/master02/remote_verification.json)。回执commit另行推送核验，最终HEAD以Git和handoff为准；BASE仍null，QA_SAME_SHA仍PENDING。

A1-MASTER-03正式QA_BLOCK登记交付 `dd83367e56e068e3c1deb4d5d8cdec58e26c9d96` 已正常push、完成fetch，local=origin=GitHub一致。[Master03同步回执](../../artifacts/2026-10-09_terl_mappo_batch01/master03/remote_verification.json)记录此核验点；回执commit另经push/fetch核验。当前WAITING_CORE_V2/BASE_FREEZE_BLOCKED、BASE=null，无活跃lease，不冻结或训练。
