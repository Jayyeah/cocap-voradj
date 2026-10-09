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
