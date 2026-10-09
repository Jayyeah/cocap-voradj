# EXP-TERL-MAPPO-01：完整交付审查

最终更新（2026-10-09）：按用户授权忽略100k performance gate，已从100k精确续训至累计1m，全部独立筛选、selection和隔离final完成。最终分类 **`TERL_MAPPO_STAGE1_LEARNABLE`**，selected775k的argmax/sample各45/50正常捕获、5/50碰撞。以下清单已按实际完整证据更新；后半部早期0/25k记录仅保留为历史。见 `TERL_MAPPO_STAGE1_FINAL_20261009_ZH.md`、`TERL_MAPPO_1M_LEARNING_20261008_ZH.md`。

本审查保留附件全文的原始目标：TERL原生合同上的random-init MAPPO迁移、正确性测试、有限Stage1学习验证及报告。**本次完成判定依据实际1m预算和隔离final，而非启动成功。** 训练/evaluator/controller已退出，冻结科学source无变化；独立收尾审计全部通过。最终文档提交、push及remote HEAD核验完成后，goal可标记complete；未要求本轮解决的科学设计意图继续保留UNRESOLVED。

## 要求与证据

| 附件要求 | 权威证据 | 当前判定 |
|---|---|---|
| §1 TERL远程HEAD、采用SHA、dirty本地隔离 | 固定143359b2；remote ls-remote一致；vendor字节与原Git对象逐文件测试；未使用dirty Evader/APF | 完成 |
| §1 CoCap local/remote/worktrees/DAG与lineage恢复 | 80debae scratch parent；corrected25dd0f8 lineage；中央初始3940d87、最终只读复核381b61d；ac-capability640a26d；本轮独立worktree和只读DAG | 完成；历史AC-CAP补审见下文 |
| §1 三方合同及五类差异、编码前风险 | MIGRATION_CONTRACT首提交前已落盘，包含有效8/4 vs默认4/3、奖励/信息/算法/终止等逐项 | 完成 |
| §2 仅一个新模型、原TERL前端、categorical AW9、random init/no teacher | model.py，manifest/config，actor原特征parity/删除cosine测试，输出9 logits | 完成 |
| §2 CTDE/action-free central V、不泄漏、agent排列 | CentralValueNetwork复用；native central_state focal25+P/E/current/time；critic等变/actor隔离测试 | 完成 |
| §3 原生MarineEnv、Stage1与动作/物理/current/APF/reset/capture/collision/reward | 未修改vendor；runtime_contract与启动断言；random-seed转移+event parity；3P1E0obs4cores120×120、AW9、dt.05×10 | 完成 |
| §4 十项奖励/NaN/mask/terminal审计 | MIGRATION_CONTRACT、IMPLEMENTATION、原始/兼容padding反事实、3/4/5/6合作reward、capture信用与done、每步reward重构assert | 完成；公开APF/current/padding设计意图仍UNRESOLVED且原行为保留 |
| §5 PPO/GAE/ValueNorm/active/clip/KL/optimizer及计数 | corrected MAPPOTrainer、配置及metrics.jsonl；env decisions/active transitions/rollout calls/paired minibatches明确区分 | 完成 |
| §6 环境合同测试 | upstream Git字节、3seed序列、capture/collision/timeout/boundary、AW9、per-agent done/joint reset | 完成 |
| §6 actor测试 | shape、正常特征parity、padding反事实、target/no-target/全mask、argmax/sample、likelihood/entropy、共享参数/梯度 | 完成 |
| §6 critic/PPO测试 | state shape/排列、GAE真实terminal与truncation、active loss、独立clipping计算、ValueNorm、CPU exact resume、CUDA RNG replay及finite | 完成；42-test suite+1独立pre-reset V测试 |
| §6 实际网络CPU/CUDA smoke与精确扩预算 | 256/8/4 CPU/CUDA512、1024 CUDA preflight finite；32步完整CUDA rollout/PPO/state/RNG逐位一致；真实CLI与非budget guard/retention预检；实际100k anchor恢复 | 完成 |
| §7 Stage1实质学习验证、强/部分/无可信信号分类 | 累计1m、31个screen点重复normal positive；selected775k隔离final两模式90% normal、10% collision；初始化0%；最后1m screen30%如实保留 | **完成：TERL_MAPPO_STAGE1_LEARNABLE** |
| §8 独立并行screen、selection-heldout、final隔离 | evaluate.py两CPU workers；41点screen共980局；3候选selection共120局；final100局；1m final2046101800与旧100k final2046100800及screen/selection均隔离 | 完成；1200局逐局审计通过 |
| §8 每checkpoint完整指标 | eval JSON完整几何/时间mean median p90/碰撞类型/删失/奖励/entropy；同step PPO/EV/loss/KL/梯度/ratio关联；所有3920行finite；0-step optimizer不适用 | 完成；全41点表与completion_audit JSON |
| §9 独立资源与retention，不干扰Evidence | 共享GPU0按用户授权训练，峰值allocated≈616MiB；2CPU evaluator；仅本线30个已完成screen/hash点按ledger清理；latest/best/milestones/top3保留，未删除parent或其它实验 | 完成；retention/hash/模型finite再核验通过 |
| §10 后续三组扩展与跨任务差异 | 原IQN源码保留；本轮TERL actor/critic/runner分离；未实现CoCap-backbone组；三方表明确CR-MS/Z/NormSense/sensing差异 | 完成 |
| §11 四类报告、central handoff、commit/push与remote HEAD | Migration/Implementation/全1m Learning/Final Follow-up及handoff；1m自动结果5eeb3c8已push核验；最终只读审计与收尾报告本轮提交；中央不写 | 科学交付完成；Git以最终提交后核验为准 |
| §12 12项最终问题 | Final report完整覆盖合同/区别/迁移/必要修改/tests/训练/各checkpoint/positive/best hash/未解决项/同步/后续建议 | 完成；末尾退化、单训练seed与未测boundary计数均明示 |

## 最终独立验证

`tools/audit_terl_mappo_1m_completed_20261009.py` 实际执行通过，输出 `artifacts/2026-10-08_terl_mappo_1m/completion_audit_20261009.json`（`all_checks_passed=true`）。逐局重算1200局count/rate、normal与collision排斥、成功时间及删失、raw reward分量与clock、paired初态、各seed域隔离；重算screen前三与heldout best规则，best/final SHA256一致。

3920条训练记录逐条核对25k保存切点与256 rollout、3000000 active transitions、22994 paired minibatches、有限指标；parent392行精确相同，首条续训100256/393/2358。仅budget变化，冻结source/runtime一致，100k anchor重新hash；保留checkpoint全部参数finite，删除点由完整screen、metadata与retention ledger互证。

基础43项correctness tests、CPU/CUDA smoke、budget-only完整CUDA逐位恢复和CLI预检均已通过，科学source此后未变。收尾仅增加只读audit和文档/图，不再启动训练。所有实际训练与评估已完成，无failure/partial工件。best775k hash为 `590d486876d4fca1cd75311c2dcc1255f847a1f756d91eed202e737193b58db4`；完整后续决定见Final report。

## 历史gate证据（0/25k，以下时态对应当时）

## 补恢复的历史AC-CAP事实

`experiment/ac-capability-cap-stage1-20260920`（640a26d）对应 **PS-Local-Discrete-AC**：共享local categorical actor＋独立local single-Q，off-policy replay、target actor/critic、Polyak tau=.005、epsilon-mixture、lr1e-4。它不是MAPPO，不能复用replay/target机制作为本实验算法。

该分支P0及finite recovery文档：`docs/audits/AC_CAPABILITY_P0_UNIFIED_PREPARATION_20260920_ZH.md`、`AC_CAPABILITY_CAP_MIX_FINITE_RECOVERY_20260920_ZH.md`。根因是terminated next_obs全mask，使target attention NaN，乘0仍NaN；最小self-token fallback后CAP短gate6000步/1313updates有限。后续还有CUDA RNG restore修复640a26d。

中央DAG对旧AC-CAP登记为 `NONFINITE_AFTER_300K_FORMAL`；不宣称旧resume bit-exact（replay reset后的functional continuation）。本地eval存在300k和400k孤立JSON，两者deterministic20均0 capture/0 collision、1000-step horizon；400k文件不构成500k完整预算完成（无formal_report.json）。本轮保留中央closeout和这项本地不完整证据，不修改中央结论，不拼接到MAPPO学习曲线。

## 首个有效独立evaluation

0-step random-init checkpoint，hash `ee4e5956210d99a0e0f6b8873fa43f683a836f348e75904b0471fd187aaf40ed`，argmax/sample各20局，40局完整落盘，screen wall time约745s。

| mode | normal/capture | ring2/ring3/strict | collision | censored | entropy | episode return |
|---|---|---|---|---|---|---|
| argmax | 0/0（n20） | 0/0/0 | 1/20（P-E接触） | 20/20 | 2.19718 | -16841.61 |
| sample | 0/0（n20） | 0/0/0 | 2/20（P-E接触） | 20/20 | 2.19715 | -8516.23 |

无成功，capture mean/median/p90为null。初始global reward均值约-14037.92/-6945.17（per-episode per-agent sum）；Stage1合作项只可能非负，故该负值主要来自原生反复boundary penalty。**boundary越界不属于本环境collision terminal**；未将边界强行算作碰撞，也不以低collision宣称初始策略安全/有效。该0-step结果仅是随机初始化基线，不是学习失败证据。

当时未修改训练超参、reward、actor pooling、难度或科学source，要求后续检查capture/ring/entropy/boundary/global/critic/KL/ratio后作bounded判断。100k无信号不能直接写“MAPPO不能围捕”；后续用户明确授权无视gate精确续训1m，现已完成。

## 每个checkpoint的证据关联

`tools/audit_terl_mappo_results_20261008.py` 是独立只读审计入口，不属于已冻结的训练源码，也不修改checkpoint、evaluation或selection。它验证不可变checkpoint的SHA256、两种动作模式的配对初态、episode计数及screen/selection/final seed隔离，并将完整evaluation与**相同environment decision step**的PPO telemetry关联；初始化0-step的optimizer指标明确为不适用。输出保存在 `artifacts/2026-10-08_terl_mappo/checkpoint_evidence_audit.json`。

为避免误读删失，它同时报告collision failure与纯time-limit censor数量，保留原evaluator“所有未capture均censored”的汇总口径。训练metrics要求步数严格递增；若未来resume导致重复/乱序，工具会要求明确核对run lineage，不能静默拼接。该工具已对完整0-step 40局、真实metrics与checkpoint hash运行通过；25k及以后结果尚在执行时不纳入完成结果。

复核命令：`python tools/audit_terl_mappo_results_20261008.py --run runs/terl_mappo_stage1_seed9_100k --output artifacts/2026-10-08_terl_mappo/checkpoint_evidence_audit.json`。`scope_complete=false` 表示仍缺后续screen或隔离final测试，不能据启动成功作科学结论。

## 25k完整筛选与自动收尾

25k checkpoint SHA256：`8afb878a99e3ef99e7ad1b4f6a8dcad24c37287101eef0770213dbfe02e7cbe3`。argmax/sample各20局均0 capture、0 ring2/3/strict；collision分别1/20、3/20，sample类型包括1局P-E与2局P-P。此早期点尚无积极学习证据，不能据此否定MAPPO。公开Stage1 global reward的非负合作项与-5越界项给出越界agent时间占比下界：argmax约98.38%、sample约57.12%；低collision不能掩盖该boundary行为。

`tools/finalize_terl_mappo_results_20261008.py` 在冻结源码之外生成训练窗口数值统计、图和独立Follow-up report。真实0-step/25k与当前训练数据的snapshot审计已通过；未完成run被final closeout guard明确拒绝，不能写完成结论。自动模式等待**既有且已验证身份的supervisor PID退出**，再要求100k budget完成、全部screen、selection、隔离final双模式各50局、best alias/hash与科学source一致，最后只提交指定报告文件并push/核验remote HEAD。不启动或恢复训练，不修改原selection/classification。

详见 `TERL_MAPPO_FOLLOWUP_DECISION_20261008_ZH.md` 与artifact `training_diagnostics.{json,png}`。训练episode中的ring字段是terminal-step snapshot，不能当作累计visitation；PPO actor loss含entropy项，梯度norm是裁剪前值的rollout均值。正式学习判断仍使用独立evaluation。
