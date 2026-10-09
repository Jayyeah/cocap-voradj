# A3-QA-02 — TERL-MAPPO Batch01 BASE 独立 CPU 验收

**结论：`BASE_QA_BLOCK`。不建议 Master 冻结当前 BASE。** 原始 T0 成功锚点及其在新 runtime 中的默认路径通过独立 CPU parity；但共同源码锁存在可复现的运行期动态依赖逃逸，连完整 checkpoint 保存/恢复也未拦截。全部指标 finite 不足以证明科学合同正确。Master/Core 应修复共同层、生成新的科学 candidate SHA 和 canonical lock，再提交独立验收。

本轮于 2026-10-09 执行，仅 CPU 离线诊断和有界 rollout/update；未启动正式训练，未修改 Core/T0 科学源码或中央 DAG。Master 未授权 CUDA，用户确认本轮以 CPU QA 交付。CUDA 两项测试跳过，未完成独立 CUDA 验证；这不改变已确认的 BLOCK。

## 1. 精确版本与 Git 审计

| 项目 | 核验值 |
|---|---|
| 仓库 | `Jayyeah/cocap-voradj` |
| 被测科学 candidate / 测试执行 HEAD | `40bd91b56cd199e527f7a2bb2317b2c1c5b61e4a` |
| Core 交付 HEAD / 核验时远程 HEAD | `bdbd502ff2f071b0cac07a0d0d956a2f35015ea1` |
| 原始 T0 | `bb794ca8435f06b9fa5693c0fed98f567320decf` |
| canonical lock SHA256 | `91dc6ea76c0f7b76ffa43441286db40b9e4ea1e9e19885a328df16ab6d024bf3` |
| lock 文件字节 SHA256 | `dff6dc7f8fd37bf7776c9e445ac81ba4aeea164bf5bafaa1361540803a1c0fda` |
| 独立 QA 分支 | `audit/terl-mappo-batch01-base-qa-20261009`，从 candidate 创建 |

canonical hash 按 sorted-key compact JSON 计算，与缩进文件字节 hash 区分。78 个共同文件的工作树 hash 均与 candidate、交付 HEAD 中的内容一致。测试时真实 HEAD 为 candidate；QA 最终提交仅加入报告、脚本和证据，不改共同运行依赖。最终 QA HEAD 与远程核验值见交付回复，避免报告自引用提交 SHA。

独立读取全部 T0→candidate diff（14 个新增文件），原 `vendor/terl`、`src/terl_mappo` 顶层模型/环境/runner/evaluator、`src/cocap_voradj` 和 T0 配置无修改。candidate→交付 HEAD 仅三份交付 JSON 和 Core 报告，未改变科学运行依赖。完整补丁保存在证据目录。

版本链为 T0 → `9a97d392bcf0ab47b01ef8a638dc0b6e57bccafc` → candidate → 交付 HEAD。首次 lock 仅 56 个文件，遗漏实际 startup 传递依赖（包括 logger）；candidate 扩展静态 import closure 至 78 个文件，fresh-process 当前 T0 加载覆盖通过。这修复了首次遗漏，但不能推导运行期 lazy import 也被覆盖，见 B1。

`verify_committed_pin` 校验 candidate 对象中的 lock；`verify_base` 校验可枚举源码与精确 delta；manifest 记录实际 Git HEAD。因此报告中的 candidate 与实际 HEAD没有混称。该 pin 检查本身不要求工作树 HEAD 等于 candidate，这是支持合法 ARM/证据提交所需；本次另外逐个核对 78 文件与两 Git 对象。剩余关键漏洞是 B1 中未被枚举/监控的运行依赖。

证据：[Git 核验](../artifacts/2026-10-09_batch01_base_independent_qa/git_verification.json)、[全部新增文件](../artifacts/2026-10-09_batch01_base_independent_qa/t0_to_candidate_files.txt)、[T0 diff](../artifacts/2026-10-09_batch01_base_independent_qa/t0_to_candidate.patch)、[交付 diff](../artifacts/2026-10-09_batch01_base_independent_qa/candidate_to_delivery.patch)。

## 2. 独立测试结果与复跑方式

| 独立执行 | 结果 | 解释 |
|---|---|---|
| 原 Core 四个 suite，CPU 模式、生产测试不改 | 55 passed / 4 failed / 2 skipped，39.75s | 四个失败全部由 `create_lock()` 强制 Core 分支名导致；独立 QA 分支不允许生成新锁 |
| 使用 committed-lock QA fixture adapter，同样四个 suite | 59 passed / 2 skipped，36.86s | 只替换测试锁生成 fixture；先核对 canonical hash 和真实 inventory；生产检查、runtime、learner 不改 |
| 额外实际完整 256-decision CPU rollout/update | 通过，50.42s（含其它检查） | 正式 256 hidden / 8 heads / 4 layers，768 active transitions，6 paired minibatches；不是仅 16-step smoke |
| bound checkpoint 后续 16 decisions/update | 精确一致 | 全模型、两 Adam、ValueNorm、完整环境/APF/观测、RNG、计数；临时大 checkpoint 删除 |
| 三课程场景正式网络前后向 | 全通过 | 3P1E0O4C、4P1E1O6C、7P2E2O8C；不等于课程 runner 已实现 |
| lazy dependency / continuous interface 专项复现 | 缺陷确认 | 见 B1/B2/B3；候选源码不改，只在忽略的 disposable Git archive 快照中运行 |
| 实际 CLI T0 `prepare` | exit 0 | `OFFLINE_PREPARED_NO_TRAINING`，显式 candidate/pin，真实 HEAD 记录正确 |
| CUDA | 未执行 | 无资源授权；两项 CUDA skip 明确保留 |

不是使用 Core 的 61 passed 作为独立结果。原命令失败与 fixture adapter 结果均完整保留。fixture adapter 是公开的测试适配，不能把未改命令写成全通过。生产 `create_lock` 的分支限制应保留；建议 Core 让读取 committed lock 的回归与“仅 Core 可生成 lock”的测试分开。

CPU 环境：Python 3.12，依赖只读复用原 worktree `.runtime-deps`，`CUDA_VISIBLE_DEVICES=''`，OMP/MKL/OpenBLAS 与 Torch 单线程，`PYTHONDONTWRITEBYTECODE=1`；pytest 自动插件禁用、不写 cache，临时目录位于 QA `runs/a3_qa02`。原 suite 四 warnings 为 protobuf 弃用和原 evaluator fork 多线程提示；适配 suite 两 fork warnings 保留。测试适配提前导入的 protobuf 不再计入 pytest warnings，不能解释为生产修复。

复跑（在从 candidate 创建的同名独立 QA 分支执行；证据输出会重写，仅诊断）：

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/.runtime-deps:src:.
export TMPDIR="$PWD/runs/a3_qa02/tmp"
mkdir -p "$TMPDIR"
python -m pytest -q -p no:cacheprovider \
  test/test_terl_native_mappo_20261008.py \
  test/test_small_step_ac_migration_contract.py \
  test/test_mappo_terminal_rows_20260923.py \
  test/test_terl_mappo_batch01_base_20261009.py \
  --basetemp=runs/a3_qa02/unmodified_suite
python artifacts/2026-10-09_batch01_base_independent_qa/run_cpu_suite.py
python artifacts/2026-10-09_batch01_base_independent_qa/independent_cpu_checks.py
python artifacts/2026-10-09_batch01_base_independent_qa/reproduce_lazy_dependency.py
python artifacts/2026-10-09_batch01_base_independent_qa/reproduce_continuous_guards.py
```

复现脚本拒绝复用已有 snapshot，重跑前可用全新 QA checkout，或将脚本的专用 snapshot 路径改为新的忽略目录；不得清理或覆盖 Core/T0。fixture adapter 断言同名 QA 分支；额外检查脚本要求 science execution HEAD 等于 candidate，最终仅证据提交的 QA HEAD 会使该断言失败，复跑应在 candidate 的新 checkout 执行并复制 QA 脚本。

证据：[原命令输出](../artifacts/2026-10-09_batch01_base_independent_qa/unmodified_suite.txt)、[原 JUnit](../artifacts/2026-10-09_batch01_base_independent_qa/unmodified_suite.xml)、[适配命令输出](../artifacts/2026-10-09_batch01_base_independent_qa/committed_lock_suite.txt)、[独立 JUnit](../artifacts/2026-10-09_batch01_base_independent_qa/committed_lock_suite.xml)、[完整 CPU 检查](../artifacts/2026-10-09_batch01_base_independent_qa/independent_cpu_checks.json)、[CLI prepare](../artifacts/2026-10-09_batch01_base_independent_qa/t0_cli_prepare.txt)。

## 3. T0 科学 parity 与原成功锚点

**原成功锚点仍可信，且新默认 T0 路径没有发现科学 delta。** 正式尺寸下原 `build/collect/update` 与 Batch01 `assemble/collect/update` 初始化、完整 256-decision batch、PPO 指标、actor/critic、两 optimizer、ValueNorm 和 RNG 精确一致。bound resume 的完整对象比较覆盖环境所有 Python 字段、数组、PRNG 与 KDTree 数据/查询，不用 pickle 字节等价替代科学等价。

775k 原完整 checkpoint SHA256 为 `590d486876d4fca1cd75311c2dcc1255f847a1f756d91eed202e737193b58db4`。独立 strict load 通过，读取后 hash 不变。实际包含 actor/value、actor_optimizer/value_optimizer、value_normalizer/update_count；runtime 含 env/APF/observations/evader_obs/return_sum；RNG 含 Python/NumPy/Torch/CUDA。载入实际 Adam lr=3e-5、eps=1e-5。该结果支持 T0 exact resume，不代表 P1 改超参的受控分叉已实现。

前轮已审计的 Stage1 selection/final/late-collapse证据仍成立：775k selected 的独立 final 两模式各 normal45/50、collision5/50；1M last screen 两模式 normal各6/20，collision10/20、12/20。selected 成功与末尾退化同时保留。本轮未重做全部 1200 evaluation episodes，未推断多训练 seed 成功、课程收益或退化原因。late collapse 的成因仍是待科学检验的假设。

本轮 suite 重跑 native action/reward/terminal parity、horizon/bootstrap、Transformer no-target/fully-masked/padding、GAE/旧 ValueNorm stats/active loss、PPO likelihood/KL/ratio/clip、critic 输入和 optimizer 隔离、checkpoint/RNG、eval seed 域/异常恢复和 retention。

原生 unmasked max pooling、no-target sentinel/padding、Global cooperation 分支、重复 dense reward、effective goal reward、partial-agent lifecycle、原 boundary 非terminal行为和 reward scale 保持历史语义；不能把它们的保留说成科学最优或已消除偏差。其影响在多目标/部分退出/改 mask 时可能扩大，必须在 ARM 中声明，不能偷修后仍声称原 T0 对照。actor 信息量改变、target/recipient/count 改变、terminal/horizon 改变均属于独立科学 delta。critic self/focal 与 agent slot 对齐也必须保持；当前单规模 parity 不证明所有跨规模 permutation/partial-agent 场景。

## 4. 已确认共同层缺陷

### B1 — 高：运行期动态依赖绕过源码锁，阻止 BASE 冻结

受影响：`contracts.source_inventory`、`provenance.runtime_manifest/assert_loaded_sources_covered`、`Runtime.collect/update`、`checkpoints.assert_manifest_sources/save_bound/load_bound`。

静态 AST closure 不覆盖 `importlib.import_module`；loaded-source membership 仅 startup 检查。之后 collect/update 和 checkpoint 检查只再次比较可枚举的 inventory，没有再验证新加载源。

最小复现使用合法 R1 reward hook，源文件精确声明在 delta 中：

```python
import importlib
def transform(reward, info):
    m = importlib.import_module("cocap_voradj.models.continuous.box_actor")
    return reward + m.BoxActorConfig().a_max
```

`box_actor.py` 是真实仓库文件，未在 78 lock 中；startup 时尚未 import。`verify_base`、candidate lock pin、assemble、runtime_manifest 全通过。随后仅修改 disposable snapshot 的该依赖默认值 `a_max: 0.4 → 0.35`，未在 delta 声明。运行 2 decisions、6 paired minibatches，指标全部 finite；`assert_manifest_sources`、**完整 save_bound 和 load_bound 也通过**。最后手动调用 coverage guard 才拒绝：`loaded local dependency absent from lock/delta: src/cocap_voradj/models/continuous/box_actor.py`。

此例不是要求 BASE 预先实现 R1：R1 只有合法入口的测试 hook，而共同层承诺的“未声明运行依赖 fail closed”失效。既有78文件覆盖当前 T0 的已加载依赖，不等于覆盖五线所有真实传递依赖。无 delta 的默认 T0 回归也无法发现该例。

建议最小修复：把动态依赖显式纳入可信依赖集合/ARM source delta；在运行期加载、接受 rollout、进入 learner、save/load 时检验已加载本地依赖覆盖及 hash，拒绝 prepare 后新增或修改的未声明依赖。应避免“先用未锁代码更新 optimizer，稍后仅报错”；新增 fresh-process lazy hook 回归覆盖实际 checkpoint 路径。静态闭包和启动检查均保留。Core 必须提交新 candidate/lock，A3 针对新 SHA 重验。

证据：[复现脚本](../artifacts/2026-10-09_batch01_base_independent_qa/reproduce_lazy_dependency.py)、[结果 JSON](../artifacts/2026-10-09_batch01_base_independent_qa/lazy_dependency_repro.json)、[输出](../artifacts/2026-10-09_batch01_base_independent_qa/lazy_dependency_repro.txt)。

### B2 — 中：C0 连续动作到原整数 AW9 路径的启动门禁可绕过

`Hooks.validate_line` 只检查 `env_factory is NativeStage1`。`env_factory=lambda seed: NativeStage1(seed)` 可绕过，实际 adapter 仍为 NativeStage1。使用已完整声明源 hash 的连续 actor/backend，prepare/manifest 全通过，首次 collect 才在 native `int(a)` 报 `TypeError('only length-1 arrays can be converted to Python scalars')`。本次没有 silent coercion，也没有执行 optimizer update。

建议共同层在 assemble/manifest 对**实际构建后的 adapter**检查其 action family/physical AW capability 与 backend 一致，启动即拒绝；不要依赖 callable identity。C0 原生物理连续 AW、AW9 九点 parity 仍属 ARM 验收；共同 guard 绕过属于 BASE 修复项，至少 C0 开训前必须修复。

### B3 — 中：C0 零更新密度 guard 额外消耗 RNG

`Runtime.update` 为比较行为 latent logprob 调用 `actor.evaluate_latent(...)[0]`。仓库真实 box actor 的该方法同时采样 MC entropy；因此仅 likelihood 检查已改变 Torch RNG，再进入 learner。独立 mocked learner 入口确认 RNG 已变、zero-update logprob 校验通过，未执行 optimizer update。与 Protocol 所写“Entropy estimation must not perturb the density check”不一致，会改变 minibatch/entropy随机序列。

建议使用确定性 density-only API 或隔离/恢复检查用 RNG，回归断言 guard 保持 RNG、latent/logprob 不变。实际连续 Jacobian/scale、PPO ratio/entropy 实现仍需 C0 专项验证，不能据此宣布 C0 learner 已完整通过。

B2/B3 证据：[脚本](../artifacts/2026-10-09_batch01_base_independent_qa/reproduce_continuous_guards.py)、[结果](../artifacts/2026-10-09_batch01_base_independent_qa/continuous_guards_repro.json)、[输出](../artifacts/2026-10-09_batch01_base_independent_qa/continuous_guards_repro.txt)。

### B4 — 低：critic EV 元数据与实际 learner 不一致

`evaluation.METRIC_DEFINITIONS['critic_ev']` 写“pre-update raw V”；实际共同 learner `small_step_ac.py` 计算 **post-update denormalized V 对 pre-update GAE return targets** 的 explained variance。不是数值 learner 回归，但报告语义错误会误导稳定性比较。建议修正指标定义和相应契约测试，保持 learner 不变；新 lock 重新验收。

### Q1 — 测试可移植性

四个 source/config/eval 测试无条件调用只允许 Core 分支生成 lock 的生产函数。原命令在独立 QA 分支有4 failures，不应以此断言 learner失效。建议测试读取/验证 committed lock，把生成权限测试单列；不能为了测试放宽生产分支限制。本报告同时交付原始失败和适配通过结果。

## 5. 五条 ARM 的剩余科学门禁

共同接口未完成各 ARM 科学实现不是单独的 BASE blocker。下表必须由各 ARM 在新 BASE 验收后、正式训练前独立通过。

| ARM | 最危险合同差异 | 必需 ARM 证据 |
|---|---|---|
| T1 accelerated curriculum | 场景变化影响 critic容量、current全信息、agent mask、reward count/recipient、多目标 termination；场景切换丢 optimizer/环境状态 | helpers 保留全部 cores，超过容量拒绝，无 current截断；正式三场景 shape已过，但需跨场景固定 padding/capacity、actor warm-start与critic策略、部分agent退出、per-target/joint指标、阶段边界全状态/RNG resume、permutation与focal对齐测试 |
| N1 CoCap backbone | local TERL 信息增加/删减、mask/sentinel改变、容量同时改变critic | 共用 top-level `hidden_dim=128` 允许且实际同时把critic改为128，已复现；actor-only治疗应独立传 actor配置、固定共同critic。逐字段信息等价，no-target/padding/pooling策略显式delta，actor容量/参数数报告与匹配消融，categorical evaluator parity |
| R1 CR-MS | 几何/target选择/recipient/量纲同时改变，raw/shaped 混用或 dense重复 | 只改已声明reward；native event/done/collision/timeout严格不变；固定几何与目标选择，按agent记录raw/native components/shaping/shaped。当前 batch仅`rewards`为shaped，丢raw数组；episode info保留native，需明确两个量纲/来源，不能同名汇总混用。B1修复后动态依赖也要锁定 |
| P1 stability | optimizer/config旧值覆盖分叉，policy-only伪装full resume | 775k完整状态已有且strict T0载入通过；fresh lr1e-5实际生效，但带改config的anchor load正确拒绝`resume contract/source mismatch`。需要显式受控fork：先完整恢复，再只覆盖批准超参、检查live Adam lr/eps/状态、ValueNorm/RNG/env保持、同随机序列零更新parity。不得strict=False。任意多个ppo字段同时变化不能叫单变量 |
| C0 continuous AW | 连续物理AW误入整数路径，latent/logprob测度错误，额外entropy RNG影响 | 先修B2/B3；连续adapter直接进入原原生物理积分，AW9九点物理/奖励/event parity；scaled tanh Jacobian和latent一致性、zero-update ratio=1、有限差分/density检查、PPO clip/KL、entropy测度、连续evaluation与full resume |

可留待后续科学消融：原 unmasked pooling 对学习的影响、masked pooling/模型容量匹配、reward密度与scale、Global合作/goal/helper语义、多目标奖励归因、late-collapse原因、PPO稳定性超参。若某问题造成未来ARM定义不成立，应先解决合同而非作为训练后的解释。不得把原行为修复和新科学治疗捆绑成未声明的单一变量。

## 6. 修订 BASE 验收清单、PASS 证据与 BLOCK 条件

- [x] candidate 与远程对象存在、chain/diff全读、交付HEAD科学依赖未变、canonical pin独立重算。
- [x] 全78源码与candidate/交付逐hash核验，现行T0 startup import覆盖；实际HEAD与candidate分开记录。
- [x] 原/native 与新默认runtime action/reward/event/terminal/horizon/bootstrap parity。
- [x] 正式网络 no-target/fully-masked/padding、PPO/GAE/旧ValueNorm统计/active loss、critic隔离与严格resume/RNG。
- [x] 实际完整256-decision CPU rollout/update与后续完整state resume，不只finite。
- [x] 三场景256/8/4正式网络前后向；未来跨规模生命周期未宣称通过。
- [x] 775k selected真实文件strict加载与读后hash保持；eval seed域与异常RNG恢复；retention跨目录/symlink/alias/pending防护通过（只生成计划，无删除）。
- [ ] **B1动态/lazy依赖覆盖、hash与运行期/checkpoint拒绝；当前失败。**
- [ ] B2实际adapter连续动作语义门禁、B3密度检查RNG不变；当前共同实现有缺陷。
- [ ] B4正确EV定义及Q1独立分支可重复测试交付。
- [ ] Master指定物理GPU/步数或时限后独立有界CUDA rollout/update/resume与资源峰值验证；当前未授权。
- [ ] 五线各自的未来ARM合同/evaluator/source delta与实际live消费者检查。

`QA_PASS` 必需：新的真实candidate/lock pin；共同改动完整影响说明；B1最小复现变为在更新前fail-closed且save/load也拒绝；无delta T0精确parity不退化；共同suite可独立执行并保留完整输出；实际正式CPU/CUDA路径、strict checkpoint/775k/RNG/eval/retention验收证据；B2/B3/B4处理明确。若共同BASE全部通过而ARM实现未完成，可给 `BASE_QA_PASS_WITH_ARM_GATES`，允许Master冻结共同BASE、各ARM仍待验。

`QA_BLOCK` 条件：未声明科学源码/动态依赖能被消费；manifest/candidate/hash绑定失效；动作/奖励/terminal/observation潜在静默变化；raw/shaped评价混用未声明；旧ValueNorm统计或horizon bootstrap回归；恢复缺optimizer/env/RNG或超参在live消费者不生效；checkpoint/evaluator/retention越界；用finite或Core自测替代独立parity。CUDA未授权单独只是未完成证据，不能豁免已有blocker。

**当前至少B1满足共同BASE阻断条件。** 请Master/Core建立修订candidate，交付新SHA/lock及修复回归输出；不得原SHA上覆盖证据后视为通过。A3本轮停止于独立CPU审计交付，等待新的Core candidate和后续CUDA资源门禁。

机器摘要：[qa_result.json](../artifacts/2026-10-09_batch01_base_independent_qa/qa_result.json)。证据目录包含所有轻量脚本/输出/JUnit/manifest与SHA256索引；未交付大模型或正式训练run。
