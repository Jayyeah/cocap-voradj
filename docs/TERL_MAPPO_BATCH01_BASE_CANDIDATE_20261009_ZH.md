# TERL-MAPPO Batch01 Common Base Candidate

状态：`BATCH01_BASE_CANDIDATE`。Core 工程与自身回归已完成，具备提交独立 QA 的条件；独立 QA 尚未执行，只有 Master 在 QA 通过后可以确认 `BATCH01_BASE_FROZEN`。本轮没有启动 T1/N1/R1/P1/C0 正式训练，没有写中央 DAG，没有修改其它 worktree 或运行进程。

## 1. 成功锚点与候选版本

工作分支：`experiment/terl-mappo-batch01-base-20261009`。

科学 parent：`bb794ca8435f06b9fa5693c0fed98f567320decf`，来自 `experiment/terl-backbone-mappo-20261008`；开始工作时实际 remote HEAD 已核验相同。原 TERL 为 `143359b2722d49c29b4fecc0ad1fd8d46326e45a`。候选版本以最终交付证据的 candidate SHA 为准。第一次工程提交 `9a97d39` 被自身 startup gate 拦截，因为历史包初始化的日志等传递依赖尚未完整纳入 source lock；修订补充静态 import closure 和全新进程覆盖回归，原始科学模块未改。后续仅补充交付证据的 HEAD 与最终 candidate SHA 分开记录；各实验必须显式固定 candidate SHA 与 canonical lock SHA256，不跟随 Core HEAD。

Selected checkpoint 为原始服务器文件：

`/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/runs/terl_mappo_stage1_seed9_1m_continuation/checkpoints/step_000775000.pt`

SHA256：`590d486876d4fca1cd75311c2dcc1255f847a1f756d91eed202e737193b58db4`，82,882,019 bytes。实际文件重新 hash、完整来源 config/source/counter、actor/critic finite 和 strict resume 均已核验，没有复制模型到新 worktree。

原报告与已有 1M audit 保留：[Stage1 Final](TERL_MAPPO_STAGE1_FINAL_20261009_ZH.md)、[Implementation](TERL_MAPPO_IMPLEMENTATION_20261008_ZH.md)、[Completion Audit](../artifacts/2026-10-08_terl_mappo_1m/completion_audit_20261009.json)。本轮恢复证据：[recovery.json](../artifacts/2026-10-09_terl_mappo_batch01_base/recovery.json)。恢复报告重算 best/last/held-out selection/final 与最后五个 screen 点稳定性摘要，并记录原评估文件的真实 hash；没有重新训练或重做全部 1200 局评估。

原 1M 为 1,000,000 joint decisions、3,000,000 active transitions、3,920 PPO rollout calls、22,994 paired optimizer minibatches；41 个 screen 点 / 980 局，selection 3 点 / 120 局，final 1 点 / 100 局。775k selected final 两模式均 normal 45/50、collision 5/50；last 1M screen 两模式 normal 均 6/20，collision 10/20、12/20。best 不能覆盖末尾退化。原始 50k CUDA32-step budget-resume 证据和实际 100k→1M lineage 均原样引用。

## 2. 相对 T0 的科学语义差异

**T0 科学语义差异为空。** `vendor/terl/`、`src/terl_mappo/` 顶层原模块、原 actor/critic learner 及 `terl_mappo_20261008` 两份配置没有修改。原 `source_hash()` 只枚举顶层 TERL-MAPPO 模块，因此新增 `batch01` 子包不破坏已存在 checkpoint 的 strict source contract。新 lock 递归覆盖该子包，并补充实际导入的 IQN/encoder/package 依赖，防止新设施未被校验。

T0 `assemble/collect/update` 使用原 `build/collect/MAPPOTrainer`；CPU/CUDA 实际 256/8/4 入口比较初始化、完整 rollout、PPO 指标、actor/critic、两 Adam、ValueNorm 与全部 RNG，逐值一致。完整保存后再续接的环境比较覆盖所有 Python 字段、PRNG、数组以及 KDTree 数据与空间查询。pickle 字节包含存储引用布局，不作为 uninterrupted-vs-resumed 的科学等价判据；Adam 标量 step 在 CUDA 加载后的存储设备可能不同，比较其精确数值与后续参数/指标/RNG。没有修改 anchor 序列化实现。

新增变化仅为工程合同：显式 BASE/config/source delta、独立 RNG context、runtime fingerprint、checkpoint sidecar、保留计划以及带规模/分布标签的报告接口。历史 evaluator 与 checkpoint 文件格式均保留。新 `load_bound` 更严格地绑定 manifest/source/BASE；导入原锚点用 `load_anchor(expected_sha256)`，仍执行原 strict resume。

## 3. 共同模块与五条扩展线

| 模块 | 责任 |
|---|---|
| `batch01/contracts.py` | parent/lock pin、resolved config、精确源码 delta、资源/路径校验 |
| `batch01/interfaces.py` | `Hooks/Runtime/ActionBackend`、T0 复用、multi-scale state/critic shape 入口 |
| `batch01/evaluation.py` | 原指标定义、seed manifest、隔离 RNG、scene 标签、best/last/selected final、late stability |
| `batch01/checkpoints.py` | 原 full-state save/resume + BASE sidecar、只读 retention plan |
| `batch01/provenance.py` | active optimizer/ValueNorm/模式/架构断言、实际导入依赖检查、runtime manifest |
| `tools/batch01_base_20261009.py` | 仅 offline `lock/recover/prepare`；没有训练或自动 supervisor 入口 |

| 实验线 | 应采用的接口 | 必须由该线独立实现/验证 |
|---|---|---|
| T1 Curriculum | `Hooks.env_factory/state_encoder/critic_factory`；`multiscale_state/multiscale_critic` | schedule、native lifecycle、跨场景固定容量 padding、切换/resume；shape smoke 不代表课程已实现 |
| N1 Actor backbone | `Hooks.actor_factory(hidden_dim,heads,layers,seed,masked_pool)` | 返回共享 local actor，提供 categorical `sample/evaluate_indices`；保持 critic/reward/环境不变 |
| R1 Capture reward shaping | `Hooks.reward_transform(raw_reward, deepcopy(info))` | 只返回 finite per-agent shaped reward；保留 native done/collision/capture、原 reward 分量与独立 shaping 记录 |
| P1 PPO fork | manifest `config_changes` 的 `ppo.*` | before/after/reason；两 optimizer、raw-return GAE、ValueNorm、likelihood check 保持共同入口 |
| C0 Action distribution | `ActionBackend` + `ContinuousPolicy` + `Hooks.actor_factory/env_factory` | physical AW 环境 adapter、变换 Jacobian/logprob、概率分布 entropy、连续 evaluator；原 `NativeStage1` 的整数动作路径明确拒绝连续策略 |

新扩展模块放在 `src/terl_mappo/batch01/extensions/`，由 `module:factory` 返回 `Hooks`。新增源文件必须逐一在 `source_changes` 中声明 before=null、真实 after hash、reason、science_impact。各线不得通过修改共享模块绕过其接口；共同文件修改需要 `candidate_fix=true` 和影响说明，不能静默纳入 anchor。

T1 padding 必须让一个 rollout 内 local/global/reward/active 的容量一致；场景切换须在 joint reset/update 边界完成，并保存 schedule/布局/PRNG。当前 multi-scale helper 仅验证 3P1E0O4C、4P1E1O6C、7P2E2O8C 的原生观测和正式网络前后向形状；不是已完成的 T1 curriculum runner。推荐跨规模保持固定物理尺度和最大 current 容量，保留场景标签，不以重新建 critic 来悄悄丢掉 optimizer 状态。

C0 `sample` 返回 physical AW `[P,2]`、变换后 logprob `[P]`、pre-tanh latent `[P,2]`；`evaluate_latent` 返回 logprob、physical MC entropy、base Gaussian entropy、具有 loc/scale 的 distribution、log_std `[P,2]`。`Runtime.update` 在 continuous PPO 前检查同一 latent 的行为 likelihood。连续 deterministic 模式必须明确记录其策略规则，不能把 tanh(mean) 描述为已计算的 density argmax。entropy 单独按概率测度比较，不与 AW9 Shannon entropy 原值混合。

## 4. 统一 runtime 与评估口径

原生 Stage1 继续是 3P1E0O4C、120×120、AW9、0.05×10=0.5秒、capture8m、related18m、native end-of-decision collision、3001 decisions horizon。capture 步 native done 可全 false；wrapper 保留 trainer joint lifecycle，terminal 优先、timeout truncated、pre-reset normalized V bootstrap，learner 按旧 ValueNorm stats 还原 raw-return GAE。

共同约定：on-policy rollout256，25k checkpoint boundary 使用较短尾 rollout；raw reward，active-only advantage/loss，独立 Adam eps1e-5；actor rollout/update 都 eval 且 autograd 开启。完整 checkpoint 包含模型/两 optimizer/ValueNorm/update counters、环境/APF/观测/return/PRNG、Python/NumPy/CPU/CUDA RNG。不得从 selected policy-only 权重伪装 exact resume。

评估协议见 [evaluation.json](../configs/experiments/terl_mappo_batch01_20261009/evaluation.json)。screen 每25k、regular 各10局、指定 milestone 各20局；top3 screen 候选在 selection 各20局；selected final 各50局。排序固定为 pooled normal、低collision、strict、ring3、较早step；final 不参与 selection。screen2026100800、selection2036100800、final2046101800，sample Torch offset100000；训练 env/actor seed 与全部物理和动作 seed 域交叉校验。

原 normal=capture 且无同时 collision。ring2/3/strict 是全 episode visitation，T0 对单 evader 的 near active P `<8m`、最大角 gap≤π 且≤3倍最小 gap。多目标 T1 必须声明其 per-target 和 joint 聚合，保持 8m/角度判据及整局 visitation，不能直接用终止步 snapshot 替代。规模 summary 保留 P/E/O/core/map/horizon/decision_seconds，报告 rates、秒及 capture_time/horizon；不无标签混合场景。

capture time 保持“任意 native capture 条件下”的 mean/median/p90、成功n/删失n；所有 noncapture 仍按 T0 原定义 censored，同时分列 collision failure 与纯 timeout censor，不能将碰撞当作普通独立生存删失作因果解释。collision type 是逐局发生率。critic EV、minibatch/post-update KL/ratio/clip、entropy 同步保留；late stability 报告末五点 normal 的 mean/min/max/std 和 last-minus-best，分别保留 argmax/sample 原结果。

`evaluate_t0` 仅服务原 categorical/native evaluator，并在正常/异常退出后恢复父进程所有 RNG。新 actor/多规模/连续 evaluator 由对应线实现，产出同一 rows/summary 合同；不能把本轮 shape/protocol 接口当作这些 evaluator 已通过 QA。

## 5. manifest 格式与严格检查

`base_lock.json`：schema/status/parent_sha/terl_sha、原 anchor resolved config、common source hashes、固定 evaluation protocol、scientific_delta_from_t0=[]、independent_qa=PENDING。

Delta 格式：

```json
{
  "schema": "terl.batch01.delta.v1",
  "line": "P1",
  "base": {
    "candidate_sha": "<explicit 40-character candidate commit>",
    "parent_sha": "bb794ca8435f06b9fa5693c0fed98f567320decf",
    "lock_sha256": "<canonical lock fingerprint>"
  },
  "config_changes": [
    {"path": "ppo.actor_lr", "before": 0.00003, "after": 0.00001, "reason": "controlled P1 fork"}
  ],
  "source_changes": [],
  "extensions": {}
}
```

`lock_sha256` 是 sorted-key compact JSON 的 canonical SHA256；它与有缩进 lock 文件的 byte SHA256 不同。`prepare --pin` 必须显式提供该 canonical 值。工具同时验证 pinned candidate Git object 中的 lock；不接受 HEAD/分支名替代 candidate SHA，也不创建或更新锁。未知 delta 字段、错误 before、越线 config、未经声明的文件新增/删除/修改、陈旧 source delta、未经声明的已加载本地依赖均拒绝。

Runtime manifest schema `terl.batch01.runtime.v1` 包含 base、line、resolved_config、delta_manifest、common_source_hashes、seed_manifest、evaluation_protocol、metric_definitions、active_runtime_values、environment/action/reward/initial_state fingerprints、versions、资源、PID/source Git HEAD、counter units、checkpoint=null、qa_status。active 值检查真实 PPO 配置、optimizer lr/eps/参数隔离、ValueNorm beta/epsilon、eval 模式及原生 runtime getters；记录实际网络参数数目/层数/heads/dims。

Checkpoint sidecar schema `terl.batch01.checkpoint.v1` 含 checkpoint_sha256、steps、manifest_fingerprint、base。保存/恢复都重新检查 runtime manifest 的当前 common hashes。恢复同一 run 使用其持久化 startup manifest；若改变 operational budget/provenance，需明确评审新的 resume manifest 绑定，不能手改 sidecar。原 anchor 的预算扩展仍使用已验证的原严格加载合同，只豁免 budget 字段。

## 6. 资源、磁盘与保留

所有本轮 run/temp/report/checkpoint 位于 `/home/yjq`。依赖复用原 worktree `.runtime-deps` 只读路径，不复制。诊断设置 physical GPU0 / process-local cuda:0，OMP/MKL/OpenBLAS/Torch 单线程；没有操作 GPU0/GPU1 上其它 PID。资源 validator 限定 1..4 threads、1..2 evaluation workers、声明 GPU allocated budget≤2048MiB，要求 cuda:0 和单个显式 GPU0/1。正式网络 CUDA 测试验证 peak allocated<2048MiB；这是 PyTorch allocated 口径，不能替代驱动含 context 的总显存统计。未来正式 launcher 仍须执行资源 gate 和持续上限监控，本轮工具不提供正式 launcher。

`retention_plan` 只返回当前 run 的删除候选，不删除文件。调用者明确传入 milestones、top3 candidates、pending/evaluating steps；latest/best 的 inode 自动保护。只有当前目录的 `step_<steps>.pt`、完整 argmax/sample rows/summary、checkpoint/metadata/evaluation 三方真实 hash 一致才可入计划。其它实验路径、alias、symlink/目录逃逸均跳过。没有清理原 run 的模型。新增正式测试的约79MiB full checkpoints 已在 finally 删除；仅保留轻量报告与小型 pytest fixtures。

## 7. 测试与独立 QA

修改前先重跑原有基础 suite：43 passed、4 warnings。首次运行仅因 runs 父目录不存在造成5个 fixture setup error，补齐目录后完整重跑通过，没有修改科学实现。第一轮联合回归为60 passed；其后的 fresh-process startup 发现 source lock 传递依赖覆盖遗漏，补充第18项测试。修订候选最终联合回归：**61 passed、4 warnings、53.92秒、无 skip**，见 [JUnit 证据](../artifacts/2026-10-09_terl_mappo_batch01_base/tests.xml)。

覆盖原源字节、Stage1 action/reward/event parity、terminal/truncation/pre-reset bootstrap、PPO/GAE/ValueNorm、no-target/fully-masked、on-policy ratio、optimizer隔离、seed隔离、实际256/8/4 CPU/CUDA完整 update/resume、三场景正式shape前后向、continuous物理动作接口拒绝整数误路由、BASE/source/config漂移、checkpoint sidecar与跨实验 retention。正式resume测试比较16-step更新后保存再8-step续接，科学值逐位一致，不把缩小网络 smoke 当作正式入口。

4个warning为2项 protobuf/Python3.14弃用警告与2项原 evaluator fork 多线程警告。evaluator 原实现不在本轮静默修改；其独立进程启动策略可列为后续候选修复。共同源码/config 与 T0 diff 为空；新增差异格式检查通过。源码锁生成、已提交candidate的 startup prepare 与remote HEAD核验在本轮交付证据中另记。

QA应独立复跑以下命令，复核新 manifest/接口和上述局限后给出决定。Core 本身的通过不等于 independent QA PASS。

```bash
PYTHONPATH=/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/.runtime-deps:src \
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
TMPDIR=/home/yjq/rl/CoCap1/terl-mappo-batch01-base-20261009/runs/batch01_base_tmp \
python -m pytest -q test/test_terl_native_mappo_20261008.py \
  test/test_small_step_ac_migration_contract.py test/test_mappo_terminal_rows_20260923.py \
  test/test_terl_mappo_batch01_base_20261009.py --basetemp=runs/batch01_independent_qa_tmp
```

独立分支从明确的 candidate commit 创建。读取 `base_lock.json` 后生成 T0/某实验线 delta，显式填 candidate SHA 与 canonical pin，再调用 offline `prepare --delta <file> --pin <pin> --output /home/yjq/.../runtime_manifest.json`。已有输出拒绝覆盖。该命令仅构造并核验入口，不采样训练 rollout，也不执行 optimizer update。

## 8. 已知 correctness 风险与冻结边界

原 reward global 分支、goal helper count、驻留 dense 激励及多目标重复计算、enemy token 无半径过滤、unmasked pooling、APF过滤索引、duplicate-current、boundary非terminal/非collision均保留。不能自行认定原结果全部有效或失效，也不能通过修这些语义来声称同一 T0 结果。发现的新科学问题必须先形成 candidate_fix/影响说明并由独立 QA/Master 评审。

原结果只含一个训练 seed；best90% normal仍有10% collision，last退化、晚期critic EV较差。T1通用奖励归因/多目标几何、跨场景容量切换与完整resume，N1不同backbone evaluator，R1shaping组件日志，C0变换概率与物理动作/evaluation必须分别通过该线 QA。本轮只交付可独立实现、测试和锁定的扩展入口，没有宣布五条线已实现或可正式开训。

结论边界：Core 已完成 `BATCH01_BASE_CANDIDATE`；等待独立 QA 与 Master 冻结决定。此后停止本轮工作，不启动任何正式训练。
