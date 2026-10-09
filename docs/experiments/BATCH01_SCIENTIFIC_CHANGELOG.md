# Batch01 Scientific Changelog

Master-only、append-only。科学源码版本与中央文档提交分别登记；文档提交不是BASE。

| Entry | Date | Class | Scope / decision | BASE |
|---|---|---|---|---|---|
| B01-000 | 2026-10-09 | FACT_RECOVERY | 核验T0 seed9 Stage1累计1M、selected775k、两模式45/50 normal及5/50 collision、末点30%退化；T0交付bb794ca、best SHA256590d4868…b58db4 | 历史T0；不填Batch BASE |
| B01-001 | 2026-10-09 | PLANNING_ONLY | 登记T1/N1/R1/P1/C0与P1-control、四参照九轴矩阵；预算/阈值/初始化为提案；全新run WAITING_BASE | UNFROZEN / SHA=null |
| B01-002 | 2026-10-09 | CONTRACT_COORDINATION | Core candidate→独立QA同SHA→Master冻结；共同env/PPO/eval hash、strict consumer assertions、Storage-before-lease、有限并发benchmark、固定进程source与v1→v2迁移 | UNFROZEN |
| B01-003 | 2026-10-09 | PREREGISTRATION_ONLY | BATCH02 E1/R2/N2/V1/M1/M2；T2 deferred；没有实现/训练/lease | UNFROZEN |
| B01-004 | 2026-10-09 | FACT_RECOVERY | Evidence旧supervisor complete报告已存在但corrected目录缺失/time为空；保留corrected-final pending及Persistent N0。其它worktree原件不修改 | 无科学代码变化 |

## 后续记录字段

`entry_id/date/author/incident_id/base_version/base_sha/arm/run_id/class/changed_axes/old_effective_values/new_effective_values/companion_required_changes/core_fix_sha/qa_report_sha/evidence_paths/comparability/retrain_decision/master_decision`。

分类限于 `FACT_RECOVERY`、`PLANNING_ONLY`、`CORRECTNESS_REQUIRED`、`ALGORITHM_REQUIRED`、`EXPERIMENTAL`、`BASE_FREEZE`、`RUN_REGISTERED`、`VERSION_MIGRATION`、`RESULT_CLASSIFICATION`。真实科学变更必须写旧/新值，不能以“bugfix”“配置优化”遮蔽reward/PPO/obs/terminal差异。

## A1-MASTER-02（2026-10-09T15:18:55.622062+08:00）

| Entry | Class | Scope / decision | BASE |
|---|---|---|---|
| B01-005 | FACT_RECOVERY | A0外部原件/TSV/保护清单核验；home261.60/shared136.77GiB，quota UNKNOWN；74/98GiB均含30 reserve；20.66 reclaim未授权，无删除/归档 | UNFROZEN |
| B01-006 | FACT_RECOVERY | A2科学candidate `40bd91b56cd199e527f7a2bb2317b2c1c5b61e4a`；delivery `bdbd502ff2f071b0cac07a0d0d956a2f35015ea1`；canonical `91dc6ea76c0f7b76ffa43441286db40b9e4ea1e9e19885a328df16ab6d024bf3`；78源码hash/T0未变核验；Core61不等于QA_PASS | SHA=null；QA_SAME_SHA=PENDING |
| B01-007 | CONTRACT_COORDINATION | 发放 `B01-QA-CUDA-40bd91b-20261009T151855`，仅GPU0有界QA correctness；300s测试/600slease、128decisions/8PPO、2GiB torch/4GiBdriver、1thread/1GiB新写入；Master未执行，正式lease/launch仍关闭 | UNFROZEN |
| B01-008 | CONTRACT_COORDINATION | Core旧T0评估域与中央Batch新域不一致，登记冻结前待决；QA已有未提交CPU/延迟导入锁逃逸记录，仅观察未验收；不改candidate | QA_SAME_SHA=PENDING |

科学axes/arm预算和初始化提案本轮未改变；以上为交付登记与QA门禁，无BASE发布、版本迁移或正式结果。下一步仅接收独立QA并评审，不自动冻结。

分类补充：继续沿用B01-002的`CONTRACT_COORDINATION`记录共享合同和有界QA授权；只涉及管理门禁，不代表已发布科学BASE。历史条目保留。
