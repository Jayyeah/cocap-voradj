# Batch01 Core V2 定向修复与 A3 独立复验合同

> Master04更新：Core V2已接收，当前CORE_V2_DELIVERED / WAITING_QA_V2；下方是Master03历史修复要求。新的固定SHA/lock与专用CUDA窗口以[V2 A3 handoff](BATCH01_V2_INDEPENDENT_QA_HANDOFF_20261009_ZH.md)为准，不能把下方“无新lease/等待Core”当当前事实。

日期：2026-10-09。中央Master已接收A3第二轮正式 `BASE_QA_BLOCK`；Core修复任务由用户确认已分配，本文件登记后续交付/复验条件，不再次派发Agent，不实现修复，不重新规划实验。

## 当前权威状态

- QA branch：`audit/terl-mappo-batch01-base-qa-20261009`，正式报告HEAD `974cdd171898a261b7c862622463537cdc31a487`。
- 被拒科学candidate：`40bd91b56cd199e527f7a2bb2317b2c1c5b61e4a`；Core文档HEAD `bdbd502ff2f071b0cac07a0d0d956a2f35015ea1`不是科学candidate。
- 被拒canonical lock：`91dc6ea76c0f7b76ffa43441286db40b9e4ea1e9e19885a328df16ab6d024bf3`。
- `CORE_V1_QA_BLOCKED`、`WAITING_CORE_V2`、`BASE_FREEZE_BLOCKED`；`BATCH01_BASE_SHA=null`，五线及P1-control继续 `WAITING_BASE`。
- **Core V1/V2指未冻结candidate的修订轮次**；BASE-v1从未发布，版本账本的released versions/migrations仍空，不虚构BASE-v1→BASE-v2迁移或重训。

固定[QA报告](https://github.com/Jayyeah/cocap-voradj/blob/974cdd171898a261b7c862622463537cdc31a487/docs/TERL_MAPPO_BATCH01_BASE_QA_20261009_ZH.md)和[中央接收回执](../../artifacts/2026-10-09_terl_mappo_batch01/master03/qa_block_receipt.json)。Master独立核验报告/24份证据hash、机器结论、JUnit、candidate/lock以及QA提交仅新增证据；行为复现由A3执行，Master不重跑其rollout。

## Core V2 必需交付

提交并push新的**科学candidate完整SHA**、新的canonical lock/digest、共同源码与依赖清单、实际consumer manifest、B1–B4/Q1逐项修复说明和回归输出、相对被拒candidate及T0的完整diff/科学影响分类。文档交付HEAD单列，不能代替科学SHA。不得原candidate上覆盖报告后宣布通过；不得把旧lock digest标成新版本。新candidate/hash在收到前保持null。

| ID | 修复与独立验收要求 |
|---|---|
| B1 高 | 实际运行期dynamic/lazy依赖的可信覆盖与hash检查；prepare后新增/修改未声明依赖必须在接受受影响rollout或改变learner/optimizer之前fail closed；实际save/load同样拒绝。fresh-process复现、合法declared delta与非法修改分别测试；保留源码锁静态/启动检查。 |
| B2 中 | 检查已构建adapter的动作family/物理AW capability是否匹配backend；wrapped `NativeStage1` factory搭配continuous backend必须在assemble/prepare即拒绝，不能仅依callable identity或等首次collect报错。 |
| B3 中 | density-only检查必须确定性，或隔离/恢复其RNG；行为latent/logprob不变，进入learner前Torch RNG必须保持。使用真实actor路径复验，不能以finite替代。 |
| B4 低 | EV定义改为实际learner的post-update denormalized V对pre-update GAE return targets；修元数据及合同回归，保留原learner数值行为。 |
| Q1 | 在独立QA分支读取/验证committed lock，锁生成权限测试单列；共同suite可以独立复跑，完整保留原始输出/JUnit。不得为了测试放宽Core-only生成权限。 |

机器要求：[core_v2_reqa_requirements.json](../../artifacts/2026-10-09_terl_mappo_batch01/master03/core_v2_reqa_requirements.json)。现有Batch01 fresh held-out seed域与Core历史T0域的reconciliation仍待完成，不改中央既有提案；Common-code修复不能默默混入reward/PPO/observation/terminal或网络容量实验变量。

## 新candidate交付后的 A3 复验

1. Master先核实远程真实candidate/文档HEAD、canonical lock及diff，填写新candidate的待验登记；独立QA必须测试**完全相同的新科学SHA和lock**，报告HEAD另列。
2. A3在固定新candidate执行B1–B4/Q1负测试和正式CPU256/8/4完整rollout/update、T0默认路径parity、完整checkpoint/env/Adam/ValueNorm/RNG及真实775k只读resume、评估seed/RNG/retention回归。V1通过项保留为历史证据，但不自动算作V2 PASS。
3. 有界CUDA是后续证据要求，CPU/static可先行。需对V2实际output/峰值按已有Storage审计动态复核，并fresh检查CPU/RAM/GPU/disk、再由Master登记V2专用短时lease；本轮无新lease、无CUDA执行授权。
4. A3提交push同新candidate的独立报告，分别给PASS/FAIL/WAIT与实际资源证据。共同BASE可判 `BASE_QA_PASS_WITH_ARM_GATES`，但各arm实现与科学门禁仍待独立验证；Core自测不能代替A3。
5. Master接收复验结果后另行明确评审，才决定是否冻结共同BASE；不自动冻结，不启动五线，也不自动创建正式训练lease。

## 保留有效工作与资源收尾

T0 `COMPLETE` 成功结论保持：seed9 Stage1累计1M、selected775k，两模式normal45/50、collision5/50；1M末点normal6/20的退化事实保留。A3正式CPU默认路径parity、256-decision update、完整resume和775k strict load通过，不因共同锁缺陷推翻原历史成功，也不扩大为多seed/课程成功。

A0审计与中央四参照九轴矩阵继续有效，预算/初始化/科学差异未重规划。A0仍是容量审计完成、正式启动前动态复核、quota UNKNOWN；没有删除/归档授权。T2延后、Batch02锁定与Evidence/Persistent N0不变。

旧QA smoke lease `B01-QA-CUDA-40bd91b-20261009T151855` 已**未使用并关闭**，见[closeout](../../artifacts/2026-10-09_terl_mappo_batch01/master03/qa_lease_closeout.json)。原grant记录保留；A3交付实际CPU-only、CUDA未执行，无独立CUDA PASS。旧lease不排他、不可继承V2；本轮不修改/停止任何运行进程。

**NEXT WAKE-UP：用户转交Core V2科学candidate、canonical lock及定向修复证据。此前保持等待，不长期轮询、不创建Agent。**
