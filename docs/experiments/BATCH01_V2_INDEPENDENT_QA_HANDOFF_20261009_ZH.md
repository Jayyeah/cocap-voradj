# Batch01 Core V2：A3 同 SHA 独立 QA handoff

日期：2026-10-09，Asia/Shanghai。Master登记 `2026-10-09T17:38:31.228981+08:00`。可直接转交A3；CPU/static复审可立即进行，CUDA仅在本文件及机器lease的有界窗口内执行。中央状态由Master更新，A3交独立证据与结论。本轮不冻结BASE，不启动T1/N1/R1/P1/C0/P1-control，不批准Storage删除/归档。

## 固定验收对象

| 对象 | 固定值 |
|---|---|
| 科学candidate | `863a0cf55aca0ace8a0aaab36d9166aa4c97268f` |
| Canonical lock SHA256 | `885ec7b8617cf88e2537b08ebf041e1d8bfc41f120b7ec4aef1708c4a513a8a4` |
| Core交付HEAD（仅证据/报告） | `d49cabbbb514b65c2bd28a1b14cf834eea614a20` |
| Core branch | `experiment/terl-mappo-batch01-base-20261009` |
| T0 parent/source | `bb794ca8435f06b9fa5693c0fed98f567320decf` |
| A3当前观察分支 | `audit/terl-mappo-batch01-v2-recheck-20261009` |
| 共同锁清单 | candidate的`configs/experiments/terl_mappo_batch01_20261009/base_lock.json`，96文件 |

Core修复报告：[固定交付报告](https://github.com/Jayyeah/cocap-voradj/blob/d49cabbbb514b65c2bd28a1b14cf834eea614a20/docs/TERL_MAPPO_BATCH01_BASE_QA_REPAIR_V2_20261009_ZH.md)。Master已核验96个源码hash、canonical digest、T0 lineage、交付仅14个报告/证据增量及Core两份JUnit。Core77 passed、3 CUDA skipped、0 failed与其QA分支可移植自测均不是A3验收。锁文件byte digest与canonical digest不同，验收使用上表canonical值。

Master接收回执：[core_v2_verification.json](../../artifacts/2026-10-09_terl_mappo_batch01/master04/core_v2_verification.json)，[原lock快照](../../artifacts/2026-10-09_terl_mappo_batch01/master04/received_core_v2_lock.json)，[修复diff](../../artifacts/2026-10-09_terl_mappo_batch01/master04/reviewed_v2_repair.diff.json)。科学SHA与报告HEAD分别记录；A3报告提交允许只加证据，须证明原96共同文件及lock不变。禁止改Core/QA共同源码后仍声称测了此candidate。

## 独立CPU/static验收（不等待GPU）

以`CUDA_VISIBLE_DEVICES=''`执行candidate中5个测试文件的完整suite，保留命令、环境/依赖版本、完整raw/JUnit和skip理由；在A3独立分支直接读取committed lock，不能再依赖Core分支或补丁fixture才能通过。测试目录包括`test/test_terl_native_mappo_20261008.py`、`test/test_mappo_terminal_rows_20260923.py`、`test/test_small_step_ac_migration_contract.py`、`test/test_terl_mappo_batch01_base_20261009.py`、`test/test_terl_mappo_batch01_qa_repairs_20261009.py`；以candidate锁清单确认实际路径。

| 项目 | 必须由A3独立复验的证据 |
|---|---|
| B1 | immutable committed trusted set；fresh-process lazy import/exec、prepare后文件或module origin/spec/realpath变化、symlink/foreign origin及原V1复现；collect、update、每次optimizer step、full save/load均fail closed，拒绝前无learner/optimizer状态改变，失败latch不能绕过。合法declared delta另验；generated Torch template须精确pin，记录外部依赖版本边界，不能把live inventory自动加入信任。 |
| B2 | 对实际构建adapter验证action capability/物理AW；包装`lambda: NativeStage1(...)`搭配continuous backend在prepare/assemble即拒绝；合法continuous adapter含off-grid动作通过。启动probe不得污染正式env或RNG。 |
| B3 | 真实continuous actor latent/logprob与tanh/affine Jacobian、zero-update ratio；density guard前后以及进入learner前Python/NumPy/Torch CPU/CUDA RNG完全相同。mock finite不能代替真实actor及PPO路径。 |
| B4 | EV实际定义为post-update denormalized V对pre-update GAE raw return targets，active-mask/ValueNorm单位与计算点一致；T0 learner数值不变。 |
| Q1 | 无共同源码修改、无临时fixture适配的独立分支全suite；Core-only生成权限拒绝仍保留。 |
| T0 | 正式hidden256/head8/layer4的真实rollout/PPO与原路径parity；完整Adam/ValueNorm/env/RNG/counters save/resume继续轨迹一致；真实775k parent仅只读strict load/完整状态验证，不做长续训。 |

逐项给PASS/FAIL/WAIT及复现限制。保留V1正式`BASE_QA_BLOCK`报告`974cdd171898a261b7c862622463537cdc31a487`，旧candidate`40bd91b...`不能冻结；V1通过项不自动转为V2 PASS。新源码若再需修复，返回Core再交新SHA/lock，禁止运行期热替换。

## V2专用CUDA correctness lease

[机器lease](../../artifacts/2026-10-09_terl_mappo_batch01/master04/qa_smoke_lease.json)：**`B01-QA-CUDA-V2-863a0cf-20261009T173831`**。授权时间`2026-10-09T17:38:31.228981+08:00`至**`2026-10-09T19:38:31.228981+08:00`**；必须在到期前至少留10分钟并完成fresh preflight才能激活。旧V1 lease已关闭，禁止复用。这里只发conditional grant，Master没有执行CUDA或创建QA进程。

| 限制 | 合同 |
|---|---|
| 物理卡 / 映射 | GPU0，UUID `GPU-fb584eaf-0dbe-5487-d539-695f9124efb0`；`CUDA_VISIBLE_DEVICES=0`，进程内`cuda:0`；GPU1无本轮新lease |
| 并发 / 重试 | 最多1个QA compute进程，2个允许节点串行，CPU线程1，evaluator workers=0；一次attempt，无自动重试/续期 |
| 时长 | CUDA两个节点加补充验证总计≤300秒；active lease含pre/post监控≤600秒 |
| 实际工作量 | joint env decisions≤1024、真实rollout/PPO calls≤8；参考/恢复分支也计数，agent transitions与minibatch/optimizer steps另报 |
| 内存与输出 | Torch peak allocated≤2048MiB，自身driver VRAM≤4096MiB；自身RSS≤8GiB、新输出增量≤2GiB |

允许两个candidate CUDA节点：

1. `test/test_terl_mappo_batch01_base_20261009.py::test_formal_entry_t0_full_update_and_exact_resume[cuda:0]`
2. `test/test_terl_mappo_batch01_qa_repairs_20261009.py::test_b3_density_guard_preserves_all_rng_and_replays_exact_latent[cuda:0]`

原`test_cuda_finite_update_and_rng_resume`使用32/4/1小网络，**不在本轮CUDA授权内**；其Core CUDA skip继续原样登记。允许的正式节点静态预算合计528 joint decisions、4次真实PPO；B3 mock调用不算真实PPO。**额外要求真实V2 pin的CUDA保存/恢复和source guard负例**：正式256/8/4、`SourceGuard.from_base`读取实际committed lock、manifest使用上表真实SHA/digest、真实rollout/update后full save/load及继续轨迹/RNG一致；错误pin/source、动态依赖变化在状态更新或checkpoint promotion之前拒绝。补充≤32 joint decisions、≤2次真实PPO，与两节点共享300秒和全部硬上限。现有formal-entry test使用占位`a*40/b*64` manifest，只可称engineering replay，不能单独证明真实V2绑定。

补充诊断写入A3自有fresh artifact/临时fixture，记录test-only harness/hash delta，不能改变96个共同文件或放宽guard。故障注入限新建disposable fixture/复制快照，不能污染Core/QA原源码、T0 parent或正在运行的实验。

设置`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`，Torch intra/inter-op均1并核验live值；无dataloader/evaluator子进程。激活前重新10×1秒采样；GPU0无其它compute PID、UUID匹配、free VRAM≥12GiB、util均值<60%且峰值<90%，load1<128逻辑CPU的50%、MemAvailable≥24GiB；实际output/tmp mount共享可用≥32GiB（2GiB输出上限+30GiB余量），自有小文件write/fsync通过。quota可查询时须额外证明≥2GiB；未知时记录UNKNOWN，仅准此有界smoke，不能用共享df代替私人quota。核对候选SHA、canonical digest及96文件；已有只加证据的提交必须证明共同代码完全一致。

每秒记录自身PID、driver/Torch峰值、RSS、磁盘增量与计数，结束后再做10次GPU样本并交lease release。OOM、NaN/nonfinite、source drift、任何硬上限、超时或其它任务进入GPU0时，终止**自身QA进程组**，TERM后10秒仍未退出才KILL自身组，保留部分证据并返回FAIL/WAIT；不得操作其它进程。到期或preflight不足返回WAITING_LEASE_REISSUE，不自行换卡/扩预算/重试。CPU/static不受GPU WAIT阻断。

## Storage与评估种子门禁

A0原件位于`/home/yjq/storage-audit-A0-20261009`，报告/TSV/保护清单hash已再次匹配。home占用261.60GiB是A0 closing审计值；本轮实际共享可用**136.73GiB**、MemAvailable**103.58GiB**，两卡10样本未见compute PID、util均0；其它用户CPU任务仍在，受保护。见[资源快照](../../artifacts/2026-10-09_terl_mappo_batch01/master04/resource_snapshot.json)、[A0复核](../../artifacts/2026-10-09_terl_mappo_batch01/master04/storage_recheck.json)。个人quota UNKNOWN；共享容量不构成独占配额，也不证明正式训练吞吐。5线74GiB/8并发98GiB含30GiB余量的既有审计继续有效，正式启动前仍须动态复核，无删除/归档授权。

| seed用途 | T0历史/Core当前配置 | 未来ARM中央提案（未冻结） |
|---|---|---|
| screen | 2026100800 | 2056100900 |
| selection | 2036100800 | 2066100900 |
| final | 2046101800 | 2076100900 |

两组仍不同；T0历史合同不改。A3核查未来各ARM/scene实际seed集合（scene offset提案100000）、用途间及已观察T0 final的disjointness、同scene physical pairing和action-sampling RNG隔离，并返回明确reconciliation/冻结提案。不得把T0 final改名为新held-out，不得在当前candidate上无声改配置/lock。共同BASE QA与未来ARM seed合同分别报告；正式ARM运行前必须由Master显式冻结seed并经QA核查。

## A3返回接口与下一次Master唤醒

交独立QA branch/报告HEAD、tested scientific SHA和canonical lock、source manifest、命令/环境版本、完整raw/JUnit、B1–B4/Q1逐项结论、T0 parity/775k/全状态resume、CUDA实际节点与补充证据、PID/UUID/pre/peak/post/计数/时长/输出增长及lease release；无法CUDA时交明确WAIT条件，不阻断CPU结果。最终verdict须明确scope与剩余arm gates，不能由Core77通过推得QA_PASS。

用户转交上述push后的A3交付，或preflight失败/lease到期/硬上限事件时唤醒Master。中央当前**CORE_V2_DELIVERED / WAITING_QA_V2 / BASE_FREEZE_BLOCKED**，`QA_SAME_SHA=PENDING`、`BATCH01_BASE_SHA=null`；T0 COMPLETE，五线及P1-control WAITING_BASE。只有A3明确对同一V2 SHA/lock验收通过，才进入后续Master冻结评审；本轮不自动冻结、不发正式训练lease。
