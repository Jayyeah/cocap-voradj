# Batch01 统一 handoff 接口

供用户后续转交五个实验 Agent：T1 Curriculum、N1 Backbone、R1 Reward、P1 Stability、C0 Continuous。也供Core/QA/Storage回交证据。**当前只允许提案/独立CPU准备；不凭本模板启动训练或创建子Agent。** Master唯一写中央DAG/state/Batch附件；各线只在自己branch/worktree/output生成handoff，交用户转中央。

每条线必须从Master已冻结的 `BATCH01_BASE_SHA` 分叉。当前SHA=null、状态WAITING_BASE，不能先各自建立不同common版本。实验名一律 `BATCH01-{ARM}`；本批C0/R1/N1/T0与历史同名节点不共用ID。Core必须交真实candidate SHA；QA交验收的candidate SHA和独立报告HEAD，两者不得混写；Storage交实际容量审计与时效，不代Master grant GPU lease。

机器模板：[handoff_template.json](../../artifacts/2026-10-09_terl_mappo_batch01/handoff_template.json)。必填字段可暂为null，但有null的必要运行字段必须列为blocker，不能宣称READY。

```json
{
  "schema": "terl-mappo-batch01-handoff-v1",
  "task_id": "BATCH01-T1",
  "handoff_kind": "PROPOSAL",
  "branch": null,
  "HEAD": null,
  "remote_HEAD": null,
  "worktree": null,
  "base_version": null,
  "BATCH01_BASE_SHA": null,
  "base_lock_sha256": null,
  "scientific_question": null,
  "exact_changes": {},
  "inherited_contracts": {},
  "algorithm_required_companions": [],
  "unresolved_differences": [],
  "delta_files": [],
  "delta_hash": null,
  "resolved_config_sha256": null,
  "scientific_sources_sha256": {},
  "common_source_hashes": {},
  "runtime_contract_hash": null,
  "evaluator_sha256": null,
  "dependency_lock_sha256": null,
  "initialization": {},
  "seed_manifest": {},
  "budget_and_counter_units": {},
  "tests": [],
  "artifact_paths": [],
  "storage_audit_id": null,
  "benchmark_id": null,
  "gpu_lease_id": null,
  "launch_authorization_id": null,
  "command": null,
  "runtime_status": "WAITING_BASE",
  "physical_gpu": null,
  "gpu_uuid": null,
  "process_local_device": null,
  "PID": null,
  "tmux": null,
  "output_root": null,
  "current_step": null,
  "latest_valid_checkpoint": null,
  "throughput": null,
  "ETA": null,
  "NEXT_WAKE_UP": null,
  "formal_results": null,
  "classification": "NO_SCIENTIFIC_RESULT",
  "blockers": ["WAITING_BASE", "NO_RUN_AUTHORIZATION"],
  "next_gate": "CORE_CANDIDATE -> INDEPENDENT_QA -> MASTER_FREEZE",
  "whether_user_approval_is_required": true
}
```

`exact_changes` 与 `inherited_contracts` 必须覆盖 Environment、Network、Reward、Information、Task、Action、Algorithm、Curriculum、Initialization；每一变更给文件/consumer/旧新有效值及classification。Network维度变化不能代替Information可得性审计。初始化必须列parent checkpoint SHA256和actor/V/Adam/ValueNorm/env/RNG各项restore/reset/迁移，exact/functional/fresh分类。

允许handoff_kind：`PROPOSAL`、`CORE_CANDIDATE`、`QA_REPORT`、`STORAGE_AUDIT`、`PREFLIGHT`、`LEASE_REQUEST`、`RUNTIME_SNAPSHOT`、`RESULT`、`COMMON_BUG`、`MIGRATION_REVIEW`。Core/QA报告另外给 `candidate_sha`、`qa_report_head`、共同source manifest与测试范围；Storage给并发峰值/atomic临时空间/保留保护清单/安全margin/有效期/写入速度。lease request只能在Storage PASS后送Master，未grant时不运行。

五线专有项：T1给完整原生stage schedule及critic/current迁移；N1给19token信息守恒和CoCap actor adapter；R1给distance替换公式/系数/clock/分量重构；P1给target_kl唯一diff、775k完整restore和paired control；C0给action边界、Jacobian、std/entropy、deterministic/sample定义与native物理连续输入parity。任何共同bug交Core统一修，不各自patch共同PPO/env/eval。

RESULT必须区分budget完成、selection完成、final合同完成和科学分类；列每模式normal/collision/geometry/成功时间及failure/time-limit n，实际seeds/fingerprints/hash，无参数更新，保留末点与best。没有运行时PID/GPU字段为null；有运行时必须给实测heartbeat、step、checkpoint、throughput、ETA、NEXT_WAKE_UP，回交后结束交互，不无限poll。
