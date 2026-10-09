# TERL-MAPPO Batch01 BASE QA blocker 修复 V2

状态：`BATCH01_BASE_CANDIDATE_V2`。Core CPU 修复回归已通过，等待 A3 对新 candidate 独立复审及 Master 确认；未冻结 BASE，未启动正式训练。

分支为 `experiment/terl-mappo-batch01-base-20261009`。新的代码 candidate 是 `863a0cf55aca0ace8a0aaab36d9166aa4c97268f`，canonical lock SHA256 是 `885ec7b8617cf88e2537b08ebf041e1d8bfc41f120b7ec4aef1708c4a513a8a4`。交付报告和证据随后单独提交，交付 HEAD 以该分支的 `git rev-parse HEAD` 与 `git ls-remote` 核验值为准；candidate pin 不随报告 HEAD 移动。

原始成功科学 parent 仍为 `bb794ca8435f06b9fa5693c0fed98f567320decf`。旧 candidate `40bd91b56cd199e527f7a2bb2317b2c1c5b61e4a`、旧交付 `bdbd502ff2f071b0cac07a0d0d956a2f35015ea1`、旧 lock `91dc6ea76c0f7b76ffa43441286db40b9e4ea1e9e19885a328df16ab6d024bf3` 保留为历史 `BASE_QA_BLOCK` 版本，原有报告与证据没有覆盖。A3 事实源为 `974cdd171898a261b7c862622463537cdc31a487` 的 [QA 报告](https://github.com/Jayyeah/cocap-voradj/blob/974cdd171898a261b7c862622463537cdc31a487/docs/TERL_MAPPO_BATCH01_BASE_QA_20261009_ZH.md)。两个复现脚本及三个原始 JSON 均通过只读 Git 对象核对，hash 记录于 `source_audit.json`；没有修改 A3 工作区或报告。

| 问题 | A3 原始行为 | V2 修复和 Core 验证 |
| --- | --- | --- |
| B1 | R1 lazy import 未锁定源码，修改后仍可 collect、6 个 PPO minibatch、save/load | 显式 BASE+delta membership、实际 content hash、模块名字/位置/spec/realpath 检查；audit 在 importlib 模块执行前拒绝未声明源码；collect/update/每个 optimizer step/save/load 完整复核；失败锁住 Runtime 并写诊断。13 个场景含合法声明、preflight、晚加载、磁盘漂移、优化器内导入、checkpoint save/load、symlink、错误位置及已执行源码 hash。原 QA 的 box_actor `.4 -> .35` 修改在 collect 前被拒绝。 |
| B2 | `lambda seed: NativeStage1(seed)` 被当作连续 adapter，首次 collect 才报 TypeError | 检查实际 step consumer 和 `ActionCapabilities`；AW9 与 physical AW 不匹配时 assemble 即拒绝。非原生 adapter 需要严格 validator，并在隔离 RNG 的副本上接受 off-grid AW 行为探针、拒绝错误表示。lambda 绕过与谎报 consumer 均在 startup 拒绝。 |
| B3 | likelihood guard 的 MC entropy 改变进入 learner 前的 Torch RNG | 使用与 learner 相同的 `evaluate_latent()[0]`，只在 guard 中隔离 Python/NumPy/Torch CPU/CUDA RNG。CPU 检查进入 learner 前与退出后的全部 RNG 一致；latent replay 与 density-only 路径一致，zero-update ratio≈1；错误 logprob 仍拒绝。未改变 entropy 学习公式。CUDA 此轮待门禁。 |
| B4 | metadata 错写 pre-update raw V | 改为 post-update denormalized V 对 pre-update GAE raw return targets、仅 active rows 的 EV。数值回归按原 learner 重新计算并精确相等；learner 公式未改。 |
| Q1 | 普通测试在 QA 分支调用 Core-only create_lock，4 项失败 | 普通回归读取已提交 canonical lock；独立权限测试仍要求非 Core 分支拒绝生成。另在 Core 自己的 disposable QA-named repo 执行完整套件，未改 production code、未使用锁 fixture adapter。结果见 `qa_portability.json`。这是 Core 可移植性自测，不替代 A3 独立 QA。 |

新增 `src/terl_mappo/batch01/source_guard.py`；调整共同层 `contracts.py`、`interfaces.py`、`provenance.py`、`checkpoints.py`、`evaluation.py`、offline prepare 工具和两份 Batch01 测试。原始 TERL 环境、actor/critic、PPO/GAE/ValueNorm learner、reward、动力学、动作、原 runner/evaluator 和两个 Stage1 config 均未改动。35 项原始 source_hash/config 与成功 parent 的 Git 对象一致，39 项 CoCap 递归依赖另行与 parent 核对一致，两组可能重叠。

新 lock 覆盖 96 个文件，包含新增 guard、测试及测试的递归依赖。原测试工具递归导入的 `box_actor.py` 现在也被明确锁住；另一个未锁定动态模块用于覆盖原 B1 membership 逃逸，不能只靠扩大静态 closure 通过回归。Torch Adam 的 import 在 TMPDIR 内生成一个临时 Python 模板：lock 增加 `generated_runtime_sources` 的精确内容 pin，guard 同时确认 Torch 当前生成目录与内容，不能授权其它生成文件。标准 Python 安装目录及锚点已有的 Gym runtime 包按外部运行库处理，没有把 sys.modules 自动加入白名单。

完整 Core CPU 回归为 **77 passed / 3 skipped / 0 failed**，80 个参数化测试，217.83 秒。三项 skip 都因本轮主动设置 `CUDA_VISIBLE_DEVICES=''`：原始 CUDA finite、正式入口 CUDA、B3 CUDA RNG。没有取得 Master CUDA 资源门禁，不将 V1 CUDA 结果算作 V2 验证。另有 34 passed / 1 skipped 的针对性回归，正式入口两项留给完整套件。完整套件包含原有 43 项、正式 256/8/4 网络的 256-decision rollout/update、逐张量/指标/优化器/环境/RNG 的 T0 bit-exact parity、完整 checkpoint resume、775k 严格读取恢复、PPO/GAE/ValueNorm、terminal/truncation/active-mask、多尺度、评估 RNG 和 retention。

T0 科学语义差异为零，lock 的 `scientific_delta_from_t0=[]`。C0 相对 V1 的 guard RNG 消耗修复是已声明变化：额外 MC entropy 样本不会再改变正式 learner 的随机流；原 continuous learner 及 entropy 定义未改。B4 只改变报告描述。checkpoint 保存增加经源码复核后的临时文件 promotion，字段和完整恢复合同沿用原版；新文件的序列化 hash 不承诺与旧保存路径相同。

复用的 selected 775k 文件仍在 `/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/runs/terl_mappo_stage1_seed9_1m_continuation/checkpoints/step_000775000.pt`，SHA256 为 `590d486876d4fca1cd75311c2dcc1255f847a1f756d91eed202e737193b58db4`。只读核验和 strict resume 通过，没有复制、覆盖或清理该模型。1M 完成和历史 evaluation 证据沿用 V1 恢复记录；本轮没有重新训练，也不对历史结果作统一有效/无效结论。

五条 ARM 接口保持原设计：T1 用 env_factory/state_encoder/critic_factory；N1 用 actor_factory；R1 用 reward_transform；P1 用显式 resolved config 的 ppo delta；C0 用 actor_factory、physical AW adapter 和声明分布语义的 ActionBackend。C0 adapter 新增 `action_capabilities() -> ActionCapabilities` 与 `validate_actions()` 合同，本轮的测试 double 不是完整 C0 实现。

ARM 必须显式使用 `SourceGuard.from_base(lock, delta, pin)`，其内部执行 `verify_base` 和 `verify_committed_pin`，再传入 `assemble(..., source_guard=guard)`。新增本地依赖即使只通过动态 import 使用，也必须声明 source_changes 的 path/before/after/reason/science_impact；修改共同文件还需 candidate_fix 标记及审查。不要从 live source_inventory/sys.modules 派生信任集合，不要采用新的 Core HEAD 自动替换既有 BASE。prepare 在导入 extension factory 前启用 guard，并预加载能解析的动态依赖。computed imports 仍由运行期 audit 与边界复核保护。

格式仍为 `terl.batch01.base.v1`、`terl.batch01.delta.v1`、`terl.batch01.runtime.v1`、`terl.batch01.checkpoint.v1`。delta.base 明确记录 parent_sha/candidate_sha/lock_sha256；runtime 保留 resolved_config、delta_manifest、common_source_hashes、seed_manifest、active_runtime_values、环境/动作/reward fingerprint、资源与版本，并增加 generated_runtime_source_hashes。checkpoint sidecar 绑定 checkpoint SHA256、BASE 和整个 manifest fingerprint。失败诊断为 `terl.batch01.source_failure.v1`，状态 `RUN_FAILED_SOURCE_CONTRACT`，写在该 run 的诊断目录；同一 guard 的失败不会自动解除。已提交真实 T0 prepare 的 manifest，状态为 `OFFLINE_PREPARED_NO_TRAINING`。

hook 检查按 import/exec 事件刷新模块缓存并检查 hook 本身的文件，未在每个 env decision 扫描整个源码树。完整固定集合 hash 检查位于 collect/update/每个 optimizer step/checkpoint 边界。非法 rollout 不返回给调用者，source failure 后不能继续 update/save/load。此机制面向协作 Python 扩展的 provenance 合同，不是防恶意任意 Python 的安全沙箱；已安装外部库仍按现有 runtime/package version 管理，本轮没有建立全系统逐文件 hash。正式 C0 物理语义、完整连续分布，以及 T1/N1/R1/P1 的各自科学 gate 仍由 ARM 独立验收。

所有测试 run、临时文件、临时 checkpoint 和 portability fixture 均在本 Core 工作区 `/home/yjq` 下；TMPDIR 显式设置，CPU threads=1、CUDA hidden，未修改或停止其它进程。测试的大型 checkpoint 与导出 fixture 已清理，保留轻量 JSON/XML 和诊断证据。

A3 在自己的独立工作区 fetch 后从 **candidate SHA** 建立 QA 分支；无需运行 create_lock、无需改 production code 或使用锁 adapter。CPU 复审命令如下（在 QA 工作区执行）：

```bash
git fetch origin experiment/terl-mappo-batch01-base-20261009
git switch -c audit/terl-mappo-batch01-v2-recheck-20261009 863a0cf55aca0ace8a0aaab36d9166aa4c97268f
mkdir -p "$PWD/runs/a3_v2_cpu/tmp"
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/.runtime-deps:$PWD/src:$PWD"
export TMPDIR="$PWD/runs/a3_v2_cpu/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q \
  test/test_terl_native_mappo_20261008.py \
  test/test_small_step_ac_migration_contract.py \
  test/test_mappo_terminal_rows_20260923.py \
  test/test_terl_mappo_batch01_base_20261009.py \
  test/test_terl_mappo_batch01_qa_repairs_20261009.py \
  --basetemp="$PWD/runs/a3_v2_cpu/pytest" \
  --junitxml="$PWD/runs/a3_v2_cpu/tests.xml"
```

完整证据目录为 `artifacts/2026-10-09_terl_mappo_batch01_base_v2/`：`tests.xml`、`core_regression.json`、`source_audit.json`、原 A3 JSON、`qa_branch_tests.xml`、`qa_portability.json`、`t0_delta_manifest.json`、`t0_runtime_manifest.json`。独立 QA 资格是具备 CPU 复审条件；A3 和 Master 结论仍待给出，CUDA 待资源门禁，不能据此宣布 BASE_FROZEN。
