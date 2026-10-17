# Relative-Z / R归一化独立科学与实现审查

**结论：REQUEST_CHANGES。禁止据本报告启动正式smoke或长训；没有生成independent_review_gate.json。**

审查日期：2026-10-18（Asia/Shanghai）。本报告针对提交 `381f9d7008f0a41f3f3f9509017808dead3b1f35`。审查代理实际配置为 `gpt-6.1-sol`、`xhigh`，canonical task ID为 `/root/independent_scientific_review`。本人独立阅读源码、原始合同和旧/新差异，亲自执行测试，未委派审查或仅采信主Agent结果。

当前实现的 `reviewed_source_hash()` 为 `badf7ca8cbc52fe616ca82046b948cbd5d3dac1171e8c84b49d2755b07c22035`；该哈希存在下述覆盖缺口，因此这里只记录，不能当作批准。

## 范围与证据

已读全部五份科学/执行合同、归一化调用点CSV、`algorithm_old_new.patch`、三阶段resolved YAML、三份contract diff、map_preflight和原始测试；已核对从 `4aeb8f0ef7edc1c2e19fc098a7063e9bfb3af628` 到被审提交的生产差异，并读取原Z05、旧Support训练/测试/smoke入口及新版入口。适用AGENTS.md中的旧路线、预算和模型限制由本轮明确用户合同覆盖。旧工作树和旧训练进程均未修改或启动。

独立运行命令：

```bash
python -m pytest -q tests/test_relative_z_rnorm_20261017.py tests/test_z_guided_support_20261016.py tests/test_ce_coverage.py tests/test_ce_energy_schedule.py tests/test_iqn_z_unified_decay_curriculum_contract.py tests/test_z_support_full3stage_20261017.py --junitxml=artifacts/2026-10-17_relative_z_rnorm/independent_unit_tests.xml
python -m pytest -q artifacts/2026-10-17_relative_z_rnorm/independent_diagnostics.py --junitxml=artifacts/2026-10-17_relative_z_rnorm/independent_diagnostics.xml
```

| 独立执行内容 | 实际结果 | 证据 |
|---|---|---|
| 原始相关六个测试文件 | **95 passed，32.92秒** | `independent_unit_tests.txt` / `.xml` |
| 新增审查诊断24项 | **13 passed，11 failed，8.625秒** | `independent_diagnostics.py` / `.txt` / `.xml` |
| 新版QueueTrainer实际小型CPU优化、与原Trainer对照、6+6步恢复对照 | PASS；model/target/optimizer/RNG/Z/下一观测/友机快照逐位一致 | 诊断脚本第116行起 |
| Stage1/2/3、strict/loose、R=100/173，RMS与Max阈值两侧 | 12项PASS；判定与旧物理标准相同，且边界内成功/外失败 | 诊断脚本第129行起 |
| 被审源码、resolved配置与三个归档contract diff | PASS；逐结构比较一致 | `independent_review_evidence.json` |

新增Supervisor测试使用假进程、隔离临时runtime和真实SQLite队列，没有启动真实Supervisor/worker/正式作业。新增QueueTrainer检查是缩小网络和预算的独立单元测试，不构成正式CUDA或四Replay smoke。**未执行 `tools/smoke_relative_z_rnorm_20261017.py`，未启动本线正式训练，未生成正式验收证据。**

特别说明：95项中的 `tests/test_z_support_full3stage_20261017.py:9` 导入的是旧 `tools.z_support_full3stage_20261017`。它通过不能替代对新版QueueTrainer和Supervisor的检查，本次新增诊断直接导入新版模块。

## A–L逐项判定

下表的PASS指本次静态与离线动态审查范围；CUDA、完整正式smoke、训练健康和300k签收仍是后续独立门禁，不能由本表代替。

| 项目 | 判定 | 源码和独立依据 |
|---|---|---|
| A 旧Capture邻居资格门控 | **PASS** | `voronoi_adjacency.py:590` 的新模式在原Capture优先之后直接走可观测Z选择，不调用旧 `_support_pursuing_friend_ids` 作为资格。新矩阵无Capture邻居仍能Support。 |
| B 仅动作前policy可观测相对Z | **PASS** | 同文件 `1418`、`1439`、`1462`、`1569`、`1575` 先按原physical排序和cap打包，snapshot只含实际有效友机float32 token；`2195`使用该快照。`z_guided_support.py:10`屏蔽无效token，最高Z集内选最近单个友机，精确同距用可观测几何、最后才ID。隐藏/截断/NaN、逆序与几何tie测试通过。 |
| C 隐藏active_targets限定 | **PASS** | `voronoi_adjacency.py:2395`在Relative-Z模式不以active_targets限制Support；真实事件、Capture奖励和任务终止分支保留active_targets语义。失活目标的.5/.25/.125真实Recovery步测试通过。 |
| D Capture优先与角色覆盖 | **PASS** | `590`先inactive、再Capture、再Relative-Z二选一；self同值或更大、无友机、全零均Coverage。没有额外Z>=.5、Z=1或直接检测者要求。 |
| E reward/obs/metadata/replay一致 | **PASS** | `2178`固定动作前位置；`2277`用动作前角色；`2381`和`3122`统一task_label；`3465`先打包下一观测，再生成next角色及task_switched。`trainer.py:710`沿用四Replay与phase优先级，无新增类。新单位/真实残留Z步均检查Support奖励和metadata；新QueueTrainer实际更新/恢复通过。 |
| F Z-lag、失活、Recovery | **PASS** | `voronoi_adjacency.py:215`传播只读previous向量，lambda=eta=.5、hardzero=.1；`2785`保留真实捕获/释放清理，`2838`自然更新Z，没有因无目标强制清零。同步多跳、多源保留、峰值四个.5秒步归零和真实Recovery测试通过；没有增加终端后的transition。 |
| G 全部地图归一化真实R | **PASS** | `map_normalization.py:24`读取最终env.width/height并断言有限、正数、方形；`voronoi_adjacency.py:748`、`969`、`1327`统一CE/观测尺度。self障碍/边界/centroid、friend/target位置与distance、obstacle输入半径/位置/distance均走R；速度、朝向、Z、mask不变。三个stage/三个scene实际120×120；80/120/173辅助检查及物理动力学回归通过。base后备CE/成功路径同样适配。 |
| H 正式链无遗留L consumer | **PASS** | `_map_diagonal`保留为历史几何接口但新VorAdj consumer不调用；CE潜势 `1591`、几何 `1681`、观测 `1299`、独立native factory均走共享R。trainer历史 `voradj_v1_to_v2`对角线权重适配仅在旧迁移模式；新Stage1无pretrained，Stage2/3强制shape_compatible。网格hypot、机器人半径、renderer世界坐标不是归一化遗留。 |
| I 原物理成功条件等价 | **PASS** | `map_normalization.py:37`转换旧阈值L/R；`voronoi_adjacency.py:1703`同时用于strict/loose RMS、Max，并在`1738`记录物理量和阈值。新距离为旧sqrt(2)倍、平方CE为2倍；正式阈值是distance而非square。本人新增三阶段、两尺寸、strict/loose、RMS/Max边界两侧检查全部通过。 |
| J finite、奖励尺度、PBRS边界 | **PASS（离线）** | `coverage_ce.py:211`实现完整CE；`voronoi_adjacency.py:2365`仅对实际进入奖励的w=1-alpha消去下一potential且最多一次，Capture权重0不消项，time-limit语义保留。PBRS/真实捕获边界/动态mix账本测试通过。新版QueueTrainer的Q、loss、gradient/reward检查及真实CPU优化通过。R改动将CE/PBRS平方项放大2倍是已披露的科学改动，控制代价未补偿；不能外推为正式GPU或长期无尖峰证明。 |
| K 完整三阶段可训练/执行合同 | **FAIL** | 课程实际4P1E1O/8P2E2O/12P3E3O、2M/700k/700k、AW9、256/8/4网络、四Replay 64/16/32/16、min3000、scratch与fresh handoff结构正确（配置和`140`、`274`核验）。但Supervisor暂停/交接、科学错误停机、源码冻结及正式smoke覆盖存在下述阻断，尚不能授权持久三阶段执行。 |
| L 论文公式与代码隐藏偏差 | **PASS（科学公式）** | `z_guided_support.py:57`使用固定friend-before锚点和clip3；`voronoi_adjacency.py:2640`以selected Z作alpha，`2661`实际只乘一次capture scale，`128`预检要求scale=allyweight=1。CE完整项、控制项、gamma和weighted reset与公式吻合。未引入平均锚点、额外网络、显式目标ID或全局分配器。K中的执行错误仍阻止整体批准。 |

## 阻断项

### B1 / P1：Supervisor多个状态忽略暂停，暂停期间仍能交接

位置：`tools/zsupport_relative_rnorm_20261017.py:562`、`:606`、`:621`、`:629`。

新建训练子进程的轮询分支在572–579处理hold；接管既有训练时没有同等逻辑。`scientific_failure.json`若在adoption期间出现，不会转换成训练器识别的pause_request，旧child会继续训练。普通operator pause虽然能被child读取，但Supervisor不立刻进入持久暂停状态。

在selection等待和formal/render尾队列期间，worker在pause后退出，Supervisor仍反复重建worker、写running状态和睡眠。若pause恰在BalancedFloor选择过程中到达，621–626无hold检查，本人隔离用例实际产生了 `stages/stage2/handoff.json`，违反暂停时禁止跨阶段交接的明确合同。

证据：诊断第44、53、60、81行起的5个用例全部失败。修复须统一覆盖adoption、每个等待循环、每次重启/重试、新训练启动、选模后的handoff和最终闭合；科学失败须同样生成持久pause；已有trainer须等待真实决策边界完整保存，不能加入60秒杀进程超时。暂停marker必须保留到明确恢复。

### B2 / P1：审核source_hash漏掉实际运行依赖

位置：`tools/zsupport_relative_rnorm_20261017.py:105`。

当前只hash全部src、新目录三份YAML、三个新工具和一个新测试；实际导入执行的 `tools/iqn_token_matched_20260919.py`、`evaluate_iqn_z05_independent_20260923.py`、`run_forward_final_bridge_20260908.py`、旧curriculum/renderer等不在集合内。新YAML通过extends读取的旧full3stage等父YAML也不在集合内。审核后改这些文件，门禁仍接受原哈希；评价、选模、配置甚至训练规模可随之漂移。

证据：诊断第90行起在临时源码树中分别修改三个真实依赖和一个extends父配置，四次哈希均未变化；未修改生产文件。修复应覆盖完整活动代码/配置依赖或直接冻结足够全面的src/tools/configs集合，并在实际执行入口核验。不能仅以reviewed_commit非空替代内容冻结；报告/registry的正常提交可继续排除。

### B3 / P1：科学断言被普通worker崩溃重试

位置：`tools/zsupport_relative_rnorm_20261017.py:457`、`:465`。

fatal识别只匹配少数日志片段。真实runtime assertion以 `ValueError: runtime assertion ...` 抛出，friend-token和cached artifact等合同断言以AssertionError抛出；这些不在当前匹配集合内。worker把它们重新置为pending并继续普通重试，而非立即科学暂停。训练侧600–601同样依赖片段，应一并审查ValueError型合同失败。

证据：诊断第152行起两个模拟子进程使用科学断言日志，实际worker逻辑均没有生成scientific_failure/pause marker，两个用例失败。修复应让科学/数值合同异常具备可靠分类，持久化失败证据并暂停；普通I/O/进程崩溃才走最多3次重试。需要增加针对新版worker和trainer/Supervisor的回归。

### B4 / P1：官方smoke尚未测试真实Supervisor路径

位置：`tools/smoke_relative_z_rnorm_20261017.py:116`。

当前supervisor_pause_resume_queue段启动的是该smoke自己的 `--child` QueueTrainer，直接调用 `wait_for_safe_pause` 与 `claim`，整段没有运行 `run.supervise()` 或其真实进程。它能证明child暂停/恢复和单次claim hold，不能证明Supervisor接管、阶段等待、handoff、worker重试及尾队列暂停。B1的五个错误均可在该smoke“PASS”时保留。

正式smoke源码须增强为隔离runtime中的真实Supervisor进程/实际supervise控制路径验证，覆盖adoption、selection等待、暂停到达handoff前、尾队列、暂停期间无worker重建/训练重试、显式恢复与队列持久化。诊断用短预算/替身须显式限定，不能影响正式配置和训练父状态。低Z(.5/.25/.125)触发和alpha分布目前主要由独立单位测试保证，正式smoke也应记录和断言其合同要求的动态证据。仍须在重审APPROVE以后才运行整套正式smoke。

## 非阻断风险与后续验收边界

1. R改动把CE/PBRS平方项放大2倍，并改变原固定控制项的相对权重。这符合本轮用户指定科学改动；尚无收益、非劣性或长期数值稳定结论，不能从Support率上涨推出性能改善。
2. 三阶段根env仍继承4P默认值，真实训练和评价经tasks覆盖为4/8/12；已独立核验。后续报告应使用最终scene/env事实，不能只打印根YAML的num_pursuers。
3. `zsupport_relative_rnorm_reports_20261017.py:35`有两个不重叠50k窗口的明确flag，方向正确；但只有一个50k窗口时provisional仍为false，建议在两个窗口齐备前统一标暂估，避免报告消费者误读。正式评价/GIF service未测得之前须保持未知，后续外推须保留假设。
4. 新版CPU小型优化/恢复通过不能代替完整网络CUDA、四Replay真实采样更新、正式renderer、实际Supervisor smoke。当前训练manifest仅prepared，三阶段progress/selection/fresh_gate均空；没有300k签收，也没有三阶段完成。

## 重新审查要求

主Agent修复B1–B4并提交新SHA后，本独立代理需重新阅读差异，亲自复跑上述失败用例及新增实际控制路径回归，核对新的全依赖source_hash，再出具新结论。当前11个失败用例应全部按合同修复，不能删除检测或改成接受旧行为。通过前保持正式smoke/长训门禁关闭。

本次只新增独立诊断、测试结果、证据JSON和本报告；没有修改生产源码、提交、push或生成批准gate。
