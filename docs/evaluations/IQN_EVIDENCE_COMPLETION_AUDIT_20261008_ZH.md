# EXP-EVIDENCE-01 completion audit（2026-10-08）

事实快照：`2026-10-08T23:13:30.813798+08:00`。当前分类 **`TRAINING_COMPLETE / SELECTION_PENDING / FINAL_EVAL_PENDING`**。本轮不训练、不启动新的科学实验。

## Training / selection

| representation | stage | actual final step | screening（20 episodes/scene） | selected step |
|---|---|---:|---|---:|
| local_binary | stage1 | 2000000 | 20/20 | 1500000 |
| local_binary | stage2 | 700000 | 7/7 | 300000 |
| local_binary | stage3 | 700000 | 6/7 | PENDING |
| global_oracle | stage1 | 2000000 | 20/20 | 1700000 |
| global_oracle | stage2 | 700000 | 7/7 | 700000 |
| global_oracle | stage3 | 700000 | 7/7 | 300000 |

Local/Global 各 3.4M；六份 final checkpoint 的 payload step、有限参数、training metrics 连续性及完成 log 均通过。训练 seed 依次为 `2026091901/02/03`；S1 scratch，S2/S3 分别使用同表示自己的 selected parent。未发现异常中断、resume 或科学变量 drift；仅 output_root、run_name 和注册 warm-start path 不同。所有已完成选择均为 balanced_floor，fallback=false。完整候选 step、hash、配置差异和 checkpoint provenance 见 [completion audit](../../artifacts/2026-10-06_iqn_evidence_comparison/completion_audit_20261008.json)。

Local S3 已存在六份完整 screening report；700k 的 live progress 尚在运行，selection_report.json 缺失。旧 heartbeat 内存中的 screening_complete 列表可能包括随后被重复队列删除的 report，不能替代磁盘 artifact。

## Final held-out gap

Local selected S3 尚未登记；三表示 `final_heldout_50_corrected/*/report.json` 均缺失。统一合同为 seed base `2026100601`，场景 Coverage/Capture/Mixed 的 seed 分别再加 `0/100000/200000`，每场景 50 局。审计保存的是 expected manifest；尚没有 actual manifest 一致性或环境 fingerprint 配对验证。不得把 screening 或既有 Z05 独立评测作为该正式三臂比较。

## Bounded engineering fixes

- 防止正在运行的 screening checkpoint 再次入队；旧逻辑会重复运行并删除刚完成的 report。此修复仅作用于新进程，当前 PID 14306 没有热替换。
- 从第一个 strict CE success transition 记录 Coverage 时间，并仅统计成功局；旧报告的 time-to-CE 全空是字段读取问题。没有重写旧 screening 数字，也不改变 balanced_floor selection 的输入指标。
- 现有 Global occupancy/shortfall correction 保留；新报告明确记录 diagnostic 与 coverage-timing implementation。旧 supervisor 内存载入的是修正前 evaluator，其未来输出不能作为修正后最终评估。
- `--final-only` 只读取已经完成的 training/selection/checkpoint，不经过 run_stage、训练 resource gate、optimizer 或 resume；缺少任何 selection/screening 即报错。

## 已定合同内下一门禁

待当前 Local S3 screening 真正完成且 selection report/hash 齐全，再使用 fresh 进程执行：

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/supervise_iqn_evidence_comparison_20261006.py --check-final-readiness
PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/supervise_iqn_evidence_comparison_20261006.py --final-only
```

最终评估写入独立 corrected 目录，旧报告保留。当前运行中的旧 evaluator/supervisor 未停止、重启或修改；本轮没有发起重复 rollout。若旧 supervisor 后续生成 final 报告，仍须核对上述实现版本与 Coverage timing，不能仅依 coordinator complete 登记实验 COMPLETE。

## Validation / decision

Evidence、deterministic evaluator、Z curriculum、Z-v2 合同回归 **31 passed**。Read-only readiness 当前按预期拒绝 Local S3 selection pending，没有 trainer launch。当前决策 **N0 / NO_EXTRA_TRAINING_YET**；Z vs Local 的稳定优势、Global 的经验上界、Z>Global 的成因与论文机制因果贡献均须等待 corrected paired held-out，之后仍需独立多训练 seed 验证。
