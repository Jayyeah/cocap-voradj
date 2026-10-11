# Batch02 E1 独立验证（2026-10-11）

结论：**E1_PASS：PASS；independent_validation：true。** 审核对象为固定提交 `8827c11ef3534e5e9894bfb6b223405f24544be3`，配置 resolved hash 为 `6955ecf28383f7e146a0ee485ef3e33cb1d1782b197a51ceadd2ef93968d3b5a`。审核者是 Codex 独立 QA agent `/root/e1_independent_qa`，未参与 E1 实现。

审核者检查了 verifier 的断言范围，在单独、干净的 worktree 中运行固定 verifier，并独立重算 tested source manifest 的 19 个源码 hash；9 个成功 IQN provider 源码与原 IQN worktree 字节相同，7 个复用的 learner 源码与固定基线相同。验证复用了已提交 verifier，因此本结论是独立审核和独立执行签字，不声称使用另一套实现做了科学复现或性能重复试验。

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| SHA | PASS | QA HEAD 与 `8827c11ef3534e5e9894bfb6b223405f24544be3` 一致；manifest 19/19 源文件 hash 与实际源码相符；基线差异只有 19 个新增文件，没有修改或删除旧文件。 |
| config | PASS | IQN YAML 文件 hash `297c44fce7a96137c042e3aefc34de7dd391d4499a9a85a7f04c537d445933f0`，resolved hash `6955ecf28383f7e146a0ee485ef3e33cb1d1782b197a51ceadd2ef93968d3b5a`，并与成功 IQN 配置相同。MAPPO PPO 配置文件 hash `385de99786be3307a1c07269306526591dc59c832b8fe4c0a679cf0a576c51b7`，与固定基线相同；实际运行配置逐项相等。架构仅有两个获授权的输入宽度变化：actor self 4→9、critic self 25→9；模块路径、层结构和参数键不变。 |
| replay | PASS | mixed/capture/coverage 共 64 条 scripted transitions、11 个边界 fixture，以及 256 个 PPO 实采决策都精确重放到成功 IQN provider；观测、奖励、done、下一状态匹配，Z 每步只更新一次。 |
| update | PASS | 256 决策、1024 agent transitions；原 PPO 3 epochs × 2 minibatches，共 6 次 minibatch update；参数有限、optimizer 确有更新。行为策略 log-prob 最大误差 `2.384185791015625e-07`。 |
| resume | PASS | QA 新 checkpoint 在 fresh process 中精确续跑 16 步。原交付 checkpoint `e69025598651c38c01eb2600ca6d476ee472d3c550b7d2a9efcfeac5e0775190` 使用原 expected continuation 也在 fresh process 中精确续跑 16 步；batch、episode、trace、metrics、参数、Adam、ValueNorm、runtime 和 RNG 均匹配。 |

原工程 verifier 的 `result.json` 保留 `E1_PASS=false`、`independent_validation=false`，没有被改写。本报告的独立 reviewer sign-off 单独记录为 PASS。新生成 checkpoint 的二进制 hash 不要求与原 checkpoint 相同；原交付 checkpoint 的 hash 与输入前一致。

### 执行命令

主 verifier：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.:/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/.runtime-deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /usr/local/anaconda3/bin/python tools/verify_batch02_e1_20261011.py --output runs/e1_independent_20261011/evidence > runs/e1_independent_20261011/evidence/stdout.log 2> runs/e1_independent_20261011/evidence/stderr.log
```

原交付 checkpoint 恢复：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.:/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/.runtime-deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /usr/local/anaconda3/bin/python tools/verify_batch02_e1_20261011.py --restore-child --checkpoint /home/yjq/rl/CoCap1/terl-mappo-batch02-e1-20261011/runs/e1_cpu_closed_loop_20261011/resume.pt --expected /home/yjq/rl/CoCap1/terl-mappo-batch02-e1-20261011/artifacts/2026-10-11_batch02_e1/expected_continuation.json --output runs/e1_independent_20261011/existing_checkpoint_recovery > runs/e1_independent_20261011/existing_checkpoint_recovery/stdout.log 2> runs/e1_independent_20261011/existing_checkpoint_recovery/stderr.log
```

### 范围说明

- 此 CPU smoke 不支持 capture/coverage 性能结论。
- clock 保持成功 provider 初始化值 0，没有覆盖 200k reward schedule。
- 原 BASE 独立 QA 仍 pending，但不阻塞本轮 E1 五项检查。
- 没有启动长期训练。

结构化报告位于 `artifacts/2026-10-11_batch02_e1_independent/independent_report.json`。原始 stdout/stderr 与两份 checkpoint 恢复收据的紧凑副本位于同目录 `execution_output/`。QA 运行权重留在 ignored `runs/e1_independent_20261011/`。
