# A3-QA-03 — Core V2 独立正式复审与恢复交付

**结论：`BASE_QA_BLOCK`。不建议 Master 冻结 candidate `863a0cf`。** 独立 CPU 正式套件为 **77 passed / 0 failed / 3 CUDA skipped**，原 V1 大部分问题已修复，T0 数值 parity、256/8/4 正式网络、完整 rollout/resume、775k strict load 通过。但额外独立攻击确认：合法声明的动态模块可从旧 timestamp `.pyc` 执行与锁定源码不一致的代码，随后完成 PPO update 和完整 bound checkpoint 保存/恢复。该缺陷在共同 SourceGuard 层，仍阻断冻结。

本轮没有 V2 CUDA lease，未执行任何 CUDA QA，不能引用 Core/V1 CUDA 证据。本次状态为 BLOCK；CUDA correctness 另为未完成事项，不能用 PENDING_CUDA 覆盖已确认 blocker。五线尚未实现不是 BASE 失败理由，各 ARM 的正式训练仍需独立验收。

## 1. 版本、独立性与中断恢复

| 绑定项 | 核验值 |
|---|---|
| T0 parent | `bb794ca8435f06b9fa5693c0fed98f567320decf` |
| 旧失败 candidate | `40bd91b56cd199e527f7a2bb2317b2c1c5b61e4a` |
| V2 被测科学 candidate / 执行测试 HEAD | `863a0cf55aca0ace8a0aaab36d9166aa4c97268f` |
| V2 canonical lock SHA256 | `885ec7b8617cf88e2537b08ebf041e1d8bfc41f120b7ec4aef1708c4a513a8a4` |
| Core 交付 HEAD | `d49cabbbb514b65c2bd28a1b14cf834eea614a20` |
| V1 QA 已推送 HEAD | `974cdd171898a261b7c862622463537cdc31a487` |
| V2 QA 分支 | `audit/terl-mappo-batch01-v2-recheck-20261009`，从精确 candidate 创建 |

开始时工作树干净，V1 QA remote HEAD 与本地相同；fetch Core 后建立新分支，未覆盖已有分支。96 个共同文件逐 hash 与工作树、candidate、交付对象核对一致。canonical pin 独立重算。candidate→交付只有报告/证据新增，没有科学运行依赖改变。T0 环境、actor/critic、PPO/GAE/ValueNorm learner、动力学、原 runner/evaluator 和 Stage1 配置无 diff。

V1 审计脚本/结果通过 `git show 974cdd...:<path>` 读取，原始 bytes 保存在本轮 `v1_*` 文件中；未覆盖 V1 报告或证据。Core 报告只作为待验证说明，单独保存为 readonly 文档，未把其 77 passed 当作独立测试。

恢复时原正式 suite 和额外 bytecode 攻击都已退出，恢复到的退出码均为0，JUnit完整，未发现测试崩溃；没有残留 QA 测试进程。可观测情况是会话在汇总前中断，客户端/传输的具体原因不可从本地证据确定。保留所有日志、snapshot和未提交结果，没有 reset、清理或覆盖。可靠完成的 suite 和 bytecode attack均未重跑；仅补齐未完成的 generated-source、耗时和历史 seed 检查。

证据：[Git/lock](../artifacts/2026-10-09_batch01_v2_independent_qa/git_and_lock.json)、[恢复记录](../artifacts/2026-10-09_batch01_v2_independent_qa/recovery.json)、[V1→V2完整diff](../artifacts/2026-10-09_batch01_v2_independent_qa/v1_to_v2.patch)、[T0→V2 diff](../artifacts/2026-10-09_batch01_v2_independent_qa/t0_to_v2.patch)、[交付diff](../artifacts/2026-10-09_batch01_v2_independent_qa/candidate_to_delivery.patch)。最终 QA 提交/远程 HEAD见交付回复，避免报告提交SHA自引用。

## 2. 独立 CPU 正式套件

实际在 QA 分支、candidate HEAD运行原五个正式 suite，无生产补丁、无 V1 fixture adapter：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/.runtime-deps:src:. \
TMPDIR="$PWD/runs/a3_v2_cpu/tmp" \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q -p no:cacheprovider \
  test/test_terl_native_mappo_20261008.py \
  test/test_small_step_ac_migration_contract.py \
  test/test_mappo_terminal_rows_20260923.py \
  test/test_terl_mappo_batch01_base_20261009.py \
  test/test_terl_mappo_batch01_qa_repairs_20261009.py \
  --basetemp="$PWD/runs/a3_v2_cpu/pytest" \
  --junitxml="$PWD/artifacts/2026-10-09_batch01_v2_independent_qa/tests.xml"
```

结果 **80 cases：77 passed、0 failures、0 errors、3 CUDA skipped、4 warnings，212.03秒，exit0**。跳过项为原 CUDA update/RNG、正式入口 CUDA、B3 CUDA RNG。warnings为2项protobuf弃用和2项原evaluator多线程fork提示。CPU threads=1，依赖路径只读复用原环境；禁用pytest外部插件/cache属于环境设置，不改变科学实现或source pin。

通过的关键证据：

- native action/reward/event/terminal、horizon/truncation/bootstrap；GAE、旧ValueNorm统计、active mask/loss、optimizer隔离、PPO ratio/KL/clip。
- **正式256 hidden / 8 heads / 4 layers，真实256-decision rollout/update**。与原 build/collect/update逐值比较初始化、batch/episode、模型、两Adam、ValueNorm与RNG，而非只查finite。
- 同一次正式测试保存完整bound checkpoint，再8 decisions/update，和恢复路径逐值比较trainer、环境/APF/观测、RNG、episode和指标；临时大checkpoint按原测试finally删除，不重复生成。
- 原775k selected checkpoint严格加载，来源/配置/source合同通过；恢复后再次只读hash仍为 `590d486876d4fca1cd75311c2dcc1255f847a1f756d91eed202e737193b58db4`。
- formal no-target/fully-masked/padding前后向；3P1E0O4C、4P1E1O6C、7P2E2O8C正式网络shape前后向。
- eval seed域内部隔离、正常/异常RNG恢复；retention计划保护其他run、symlink、pending、best/latest；没有实际越目录删除行为。

本轮不重做历史1200局evaluation，不推断新ARM收益或late collapse成因。前轮可信T0 selected成功与1M late collapse结论不变。T0 CPU parity通过不保证所有任意扩展来源安全，也不代替CUDA验证。

证据：[原始测试输出](../artifacts/2026-10-09_batch01_v2_independent_qa/tests.txt)、[JUnit](../artifacts/2026-10-09_batch01_v2_independent_qa/tests.xml)。

## 3. V1五项问题复验

| 项目 | 独立结论 | 边界 |
|---|---|---|
| B1 | 原lazy import/漂移攻击和13个回归场景通过，但新增bytecode漏洞确认，**整体FAIL** | 无缓存旧代码的普通路径拒绝未声明lazy import；声明依赖可update/save/load；磁盘漂移、晚模块、优化器前检查、spec/origin/symlink和failure latch均有独立suite证据。不能推导实际执行的code必然来自pin源码 |
| B2 | **PASS原缺陷与共同startup gate** | lambda NativeStage1 startup拒绝；谎报capability的整数consumer行为probe拒绝；native categorical完整路径正常。probe不是连续AW原生物理等价证明，C0未实现 |
| B3 | **PASS CPU；CUDA未验** | 密度检查前后Python/NumPy/Torch CPU RNG一致，zero-update ratio约1；错误logprob仍拒绝。isolated_rng保护MC entropy检查，原entropy/learner源码未改；CUDA RNG列表为空，不能声称CUDA通过 |
| B4 | **PASS** | metadata为post-update denormalized V对pre-update GAE raw return targets、active rows；重算EV等于learner指标；原learner无diff，T0数值parity通过 |
| Q1 | **PASS** | 独立分支直接执行全部正式suite，不用V1锁fixture适配。committed lock读取与Core-only生成权限分开，create_lock生产限制和错误pin拒绝保留 |

新增generated-source独立探针：修改Torch唯一授权模板内容被拒绝；同目录新增未声明Python模块在执行前拒绝；失败后collect/update/save/load继续拒绝，0 optimizer updates。合法组合 BASE+R1 source delta来自真实 `SourceGuard.from_base`，包含candidate对象pin，而非从sys.modules生成白名单。

证据：[generated-source与耗时结果](../artifacts/2026-10-09_batch01_v2_independent_qa/remaining_source_probes.json)、[输出](../artifacts/2026-10-09_batch01_v2_independent_qa/remaining_probes_driver.txt)。这些新探针只在ignored QA快照内执行，没有改Core/T0源码。

## 4. 新阻断 B1-V2：已声明源码与执行的timestamp bytecode不一致

**严重度：高，BASE共同层；不是尚未实现的ARM能力。** 受影响位置：`source_guard._audit/check_path/check_loaded`，以及依赖该guard的Runtime和bound checkpoint。audit收到code object，却只用 `co_filename` 指向的当前磁盘文件hash；`_EXECUTED`也记录当前磁盘hash，没有核验执行code与该源码编译产物一致。

最小攻击使用完全合法声明的R1扩展，其reward callback动态import另一个明确声明hash的本地模块，读取 `VALUE=0.4`。准备旧timestamp缓存时编译同长度 `VALUE=0.9`，随后恢复锁定源码与整数秒mtime：

```python
original = module_path.read_text()  # contains VALUE = 0.4
stamp = int(module_path.stat().st_mtime)
module_path.write_text(original.replace('VALUE = 0.4', 'VALUE = 0.9'))
os.utime(module_path, (stamp, stamp))
py_compile.compile(str(module_path), doraise=True)
module_path.write_text(original)
os.utime(module_path, (stamp, stamp))
# source bytes/hash are now exactly the declared source; ordinary .pyc header
# timestamp/size also match. importlib loads the old code VALUE=0.9.
```

攻击没有禁用audit、篡改guard或模块spec。使用标准Python timestamp `.pyc`，可代表快速同长度编辑/恢复产生的陈旧cache风险。`PYTHONDONTWRITEBYTECODE=1`禁止写cache，不禁止读已有cache，本轮实际环境也复现。

独立实测：

- `verify_base`、真实candidate `verify_committed_pin`、`SourceGuard.from_base`、assemble、runtime_manifest均通过。
- 当前源码hash正确、spec/realpath/name正确，`.pyc`头的size与timestamp正确；源码常量0.4，执行code常量0.9。
- R1实际reward读取0.9，较声明实现每agent加量多0.5。
- 完成2 decisions和6 paired PPO minibatches，metrics全部finite。
- 完整 `save_bound/load_bound`通过，恢复steps=2。
- guard.failure未触发，Runtime未锁住；所以不能保证不合格源码不可update/save/load。

证据：[完整独立复现脚本](../artifacts/2026-10-09_batch01_v2_independent_qa/independent_source_probes.py)、[攻击输出](../artifacts/2026-10-09_batch01_v2_independent_qa/pyc_stale.txt)、[攻击结果](../artifacts/2026-10-09_batch01_v2_independent_qa/source_probes.json)、[源码与bytecode/header只读验证](../artifacts/2026-10-09_batch01_v2_independent_qa/bytecode_details.json)。原记录/快照完整保留，未重跑攻击。

复现命令应在candidate的新disposable QA checkout执行，并复制QA脚本；脚本拒绝覆盖现有snapshot：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/.runtime-deps:src:. \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python artifacts/2026-10-09_batch01_v2_independent_qa/independent_source_probes.py pyc_stale
```

建议最小共同修复：本地pin模块从已验证源码bytes编译执行、拒绝未经验证的cached code，或可靠比对将执行code与pin源码编译产物（处理Python版本/compile flags/optimization）。只清空一次cache或设置dontwritebytecode不足。补充fresh-process同timestamp/size旧pyc攻击、声明合法cache路径、提前已加载陈旧code等回归。失败必须在模块/optimizer消费前拒绝并锁住Runtime，checkpoint也不能接受。不要改变TERL/PPO数值来修来源问题。

Master/Core应生成新candidate SHA与canonical lock；不得在本SHA上以补报告替代修复。A3不直接修改Core科学代码。

## 5. guard有界CPU耗时与未来ARM评估隔离

独立短采样（7次force check、20次cached reward-hook）：完整固定集合/module检查中位数 **268.27 ms**，范围263.63–275.23 ms；缓存hook中位数 **0.183 ms**，最大0.431 ms。检查无RNG/optimizer变化，未做正式吞吐压力测试。

频率为collect前后、update前后、每个actor/value optimizer step和checkpoint边界，不是每decision全树扫描。本次6 paired minibatches对应12次optimizer检查，按实测中位数估计仅这些检查约3.22秒，加collect/update四边界约1.07秒；这只是短采样算术预算，不是正式训练实测。缓存hook开销较小，但force检查成本明显，不能宣称性能无影响。Master后续benchmark应量化CPU/CUDA占比，在保持fail-closed合同下优化缓存/模块索引。性能本身未被用作本轮科学BLOCK理由。

独立读取真实历史 `final_000775000.json`，SHA256仍为 `996db51d419ae4e47292d3df70a6b4000e5046e6f5d1e424824f7bb30644069c`。锁定protocol的内部train/screen/selection/final域互斥检查通过，但**future ARM默认final与历史T0 final完全相同**：环境seed2046101800–2046101849，sample动作seed2046201800–2046201849，各50个重叠。screen/selection与历史final均无重叠。

历史T0再现可保留这组seed；它们已经公开、参与此前结果解释，不能作为之后自适应ARM设计的全新blind final。配对环境seed可用于明确标注的历史对照，但不是未见holdout。`seed_manifest`仅检查单protocol内部，不检查历史final，也无ARM line/历史保留域输入；当前delta没有评估protocol分配入口，不能让ARM私自改锁。

**ARM正式训练前门禁：** Master分配新的共同Batch01评估holdout，锁定protocol/ledger，保证未来train/screen/selection及新final的环境/动作流与历史final保留域分离；需要paired comparisons时五线共用同一新域。Core提供经审查的protocol绑定/历史域拒绝路径，保留T0历史复现协议。未分配新holdout属于未完成科学门禁，本报告不把“沿用历史T0协议”误称为已污染训练；没有启动任何ARM。

证据：[seed isolation](../artifacts/2026-10-09_batch01_v2_independent_qa/seed_isolation.json)、[恢复/seed只读脚本](../artifacts/2026-10-09_batch01_v2_independent_qa/verify_recovery_and_seeds.py)。

## 6. 五条ARM剩余门禁与冻结建议

| ARM | 必须在正式训练前独立验收 |
|---|---|
| T1 | 固定跨规模capacity/padding与完整current信息、actor warm-start/critic与Adam策略、partial-agent/multi-target reward与terminal、permutation/focal对齐、阶段切换和完整resume；三个shape已通过不等于课程runner完成 |
| N1 | TERL local信息量逐字段等价、no-target/mask/pooling变化声明、actor容量匹配且固定共同critic；共享hidden_dim会同时改变critic，不能声称actor-only治疗；新actor evaluator |
| R1 | 目标/recipient/几何/量纲单独锁定，原native event/done保持；raw/native components/shaping/shaped独立记录；当前generic batch仍仅保留shaped rewards，episode native return不得混用 |
| P1 | 775k full-state受控分叉、live Adam超参与state、ValueNorm/RNG/env零变化；变config的strict anchor load拒绝不代表fork实现；不可用strict=False或多字段变化假称单变量 |
| C0 | 真实physical AW直接进入原生积分、AW9九点物理/reward/event parity、latent/tanh/scale Jacobian、ratio/clip/KL/entropy、连续evaluator/full resume；B2/B3共同接口通过不等于C0物理/learner验收 |

所有ARM还需新holdout/source delta与共同源码guard的修订验收。原生unmasked pooling、dense/global cooperation/goal reward、partial-agent/boundary等历史语义保持，不借修guard偷偷改原T0；其学习影响属于声明后的后续消融。

当前 **不建议冻结BASE**。必需处理B1-V2实际执行code来源漏洞并提交新SHA/lock；独立验收新的fix且T0数值不回归。B2/B3 CPU/B4/Q1已有通过证据应保留，不把本轮BLOCK解释为所有修复失败。CUDA原update/resume、正式256/8/4 CUDA和B3 CUDA RNG均未验；之后仅在Master明确V2或修订candidate lease下执行有界correctness smoke。没有正式训练、没有中央DAG写入、没有Core/T0科学源码改动。

机器摘要：[qa_result.json](../artifacts/2026-10-09_batch01_v2_independent_qa/qa_result.json)。报告和轻量证据提交到独立QA分支并push，remote HEAD核验后结束本轮；不继续长期运行。
