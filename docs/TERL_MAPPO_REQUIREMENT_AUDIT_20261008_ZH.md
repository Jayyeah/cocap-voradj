# EXP-TERL-MAPPO-01：完整交付审查

本审查保留附件全文的原始目标：TERL原生合同上的random-init MAPPO迁移、正确性测试、有限Stage1学习验证及报告。**启动成功不是完成学习验证；当前goal不标记complete。** 用户明确允许训练健康、只剩等待时结束交互，训练与独立评估继续在tmux中执行。

## 要求与证据

| 附件要求 | 权威证据 | 当前判定 |
|---|---|---|
| §1 TERL远程HEAD、采用SHA、dirty本地隔离 | 固定143359b2；remote ls-remote一致；vendor字节与原Git对象逐文件测试；未使用dirty Evader/APF | 完成 |
| §1 CoCap local/remote/worktrees/DAG与lineage恢复 | 80debae scratch parent；corrected25dd0f8 lineage；中央3940d87；ac-capability640a26d；本轮独立worktree和只读DAG | 完成；历史AC-CAP补审见下文 |
| §1 三方合同及五类差异、编码前风险 | MIGRATION_CONTRACT首提交前已落盘，包含有效8/4 vs默认4/3、奖励/信息/算法/终止等逐项 | 完成 |
| §2 仅一个新模型、原TERL前端、categorical AW9、random init/no teacher | model.py，manifest/config，actor原特征parity/删除cosine测试，输出9 logits | 完成 |
| §2 CTDE/action-free central V、不泄漏、agent排列 | CentralValueNetwork复用；native central_state focal25+P/E/current/time；critic等变/actor隔离测试 | 完成 |
| §3 原生MarineEnv、Stage1与动作/物理/current/APF/reset/capture/collision/reward | 未修改vendor；runtime_contract与启动断言；random-seed转移+event parity；3P1E0obs4cores120×120、AW9、dt.05×10 | 完成 |
| §4 十项奖励/NaN/mask/terminal审计 | MIGRATION_CONTRACT、IMPLEMENTATION、原始/兼容padding反事实、3/4/5/6合作reward、capture信用与done、每步reward重构assert | 完成；公开APF/current/padding设计意图仍UNRESOLVED且原行为保留 |
| §5 PPO/GAE/ValueNorm/active/clip/KL/optimizer及计数 | corrected MAPPOTrainer、配置及metrics.jsonl；env decisions/active transitions/rollout calls/paired minibatches明确区分 | 完成 |
| §6 环境合同测试 | upstream Git字节、3seed序列、capture/collision/timeout/boundary、AW9、per-agent done/joint reset | 完成 |
| §6 actor测试 | shape、正常特征parity、padding反事实、target/no-target/全mask、argmax/sample、likelihood/entropy、共享参数/梯度 | 完成 |
| §6 critic/PPO测试 | state shape/排列、GAE真实terminal与truncation、active loss、独立clipping计算、ValueNorm、CPU exact resume、CUDA RNG replay及finite | 完成；42-test suite+1独立pre-reset V测试 |
| §6 实际网络CPU/CUDA smoke | 256/8/4 CPU/CUDA512，各2 PPO updates；1024 CUDA4 updates、24 paired minibatches、finite、allocated≈590MiB | 完成 |
| §7 Stage1实质学习验证、强/部分/无可信信号分类 | 正式100k PID992932运行；完成初始0-step argmax20/sample20；后续25/50/75/100k尚待生成 | **未完成，需已启动run/checkpoint评估结果** |
| §8 独立并行screen、selection-heldout、final隔离 | evaluate.py两workers；screen2026100800、selection2036100800、final2046100800；预声明排序及四类规则 | 实现完成，后续evaluation执行待完成 |
| §8 每checkpoint完整指标 | eval JSON逐episode capture/normal/time/ring2/3/strict/collision/type/censored/reward/entropy；相同步数PPO指标在metrics.jsonl；0-step optimizer指标不适用 | 初始评价完整；后续pending |
| §9 独立资源与retention，不干扰Evidence | 共享GPU0，user授权忽略util门槛；1训练+2CPU评估线程；少量milestone/latest/best hardlink；不写旧run，磁盘>140GiB free | 完成；运行期持续遵循 |
| §10 后续三组扩展与跨任务差异 | 原IQN源码保留；本轮TERL actor/critic/runner分离；未实现CoCap-backbone组；三方表明确CR-MS/Z/NormSense/sensing差异 | 完成 |
| §11 四类报告、central handoff、commit/push与remote HEAD | 本分支Migration/Implementation/Stage1/Handoff；启动c6ce039与健康快照62a3be7均push验证；中央不写 | 当前阶段完成；最终学习报告/分类待结果 |
| §12 12项最终问题 | 原合同/迁移/理由/tests/source/sync有报告；checkpoint表已具备0-step结果；positive/best/后续决定不能在后续screen之前下结论 | 部分完成，科学终态仍pending |

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

当前不修改训练超参、reward、actor pooling、难度或科学source。后续checkpoint需检查capture/ring变化、entropy、boundary/global分量和critic/KL/ratio，再作bounded gate结论。即使100k无信号，也不能直接写“MAPPO不能围捕”；是否延长Stage1必须据此作独立后续决定。
