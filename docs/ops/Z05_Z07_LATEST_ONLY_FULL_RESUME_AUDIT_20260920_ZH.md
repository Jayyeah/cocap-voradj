# Z05/Z07 latest-only full-resume 工程审计（2026-09-20）

## 1. 审计结论

本问题是 checkpoint 编排层的存储问题，不是训练科学合同问题。

审计确认：

- `CoCapTrainer._save_full_resume()` 本身已经只写 `checkpoints/resume_latest.pt`。
- Trainer 使用 `torch.save(temporary)` 后 `temporary.replace(resume_latest)`，原子替换语义正确，full-resume payload 未裁剪。
- 历史 full-resume 膨胀来自 curriculum runner 的 `os.link(resume_latest, resume_step_*.pt)`。这些 hard link 让旧 inode 在后续 `resume_latest` 原子替换后仍被保留。
- supervisor 没有读取或依赖 `resume_step_*.pt`；本修复不需要修改 supervisor，也不修改 storage policy。

修复只改变 runner 的 full-resume 保留和重启遍历语义，不改变 reward、replay ratio、recovery curriculum、environment、network、optimizer、LR、epsilon、stage budget、formal evaluation、balanced selection 或 Stage1→2→3 warm-start 语义。

## 2. 当前与新保存语义

每个 active stage 的持久 full-resume 目标为：

```text
training/checkpoints/resume_latest.pt       # 唯一 rolling full-resume
```

runner 不再创建 `resume_step_*.pt`。ordinary checkpoint 仍全部保留：

```text
step_100000.pt
step_200000.pt
...
latest.pt
```

formal report 和每个 milestone 的 runtime evidence 也全部保留。runtime evidence 仍记录 `global_step`、optimizer updates、replay sizes、recovery pool size、contract hash、latest metrics 和 full-resume bytes。

closure status 现在写入：

```json
{
  "last_full_resume": ".../resume_latest.pt",
  "last_full_resume_global_step": 100000,
  "full_resume_retention": "latest_only"
}
```

现有 live runtime 中的旧 `resume_step_*.pt` 不在本次开发测试阶段删除；它们必须在停训、校验 latest SHA/global_step 且 patch 通过后，按 activation procedure 清理。

## 3. restart state machine

runner 从 `resume_latest.pt.runtime.global_step` 读取 `current`，并对每个 registered milestone `target` 严格执行：

| 条件 | 行为 |
|---|---|
| `target < current` | 不训练、不生成 resume、不重跑 formal eval、不重写 evidence/status；只验证 ordinary checkpoint、formal report、runtime evidence 存在且 evidence step 匹配，否则 fail closed。|
| `target == current` | 合法的 crash-after-checkpoint recovery；验证 latest 的 `runtime.global_step` 和 ordinary checkpoint，缺 report 则补 formal eval，缺 evidence 则从当前 latest 生成，然后写 closure status。|
| `target > current` | 从唯一 `resume_latest.pt` exact resume，训练到 target，验证 latest step，保留 ordinary checkpoint，执行 formal eval/evidence/status closure。|

因此，不能用 1.2M latest resume 冒充 400k 的 exact runtime evidence。历史 milestone 的 evidence 缺失时会 fail closed。

## 4. 变更范围与兼容性

修改文件：

- `tools/iqn_z_unified_decay_curriculum_20260919.py`
- `tests/test_z_resume_latest_only_20260920.py`
- 本审计文档

未修改：

- `src/cocap_voradj/training/trainer.py`
- `tools/supervise_iqn_z_unified_decay_dual_20260919.py`
- Z05/Z07 配置和科学参数

`select_balanced()` 仍只消费 ordinary checkpoint 对应的 formal report；Stage2/Stage3 仍只接收上一阶段选出的 ordinary model checkpoint，并由 Trainer fresh 初始化 optimizer、replay、RNG、env、Z 和 recovery pool。历史 full-resume 删除不影响 selection 或 stage promotion。

## 5. T1–T8 验证

所有测试使用临时目录；没有使用 live runtime。

| 测试 | 结果 | 覆盖内容 |
|---|---|---|
| T1 | PASS | 连续 3 个 fake milestone 后只存在 `resume_latest.pt`；ordinary、formal report、runtime evidence 全部存在。|
| T2 | PASS | 多次真实 Trainer full-resume 保存后 latest 的 global step 为最新值，atomic replace 产生新 inode，未出现 `resume_step_*.pt`。|
| T3 | PASS | CPU 小跑 exact resume；model、target、optimizer、replay、recovery pool、Z state、RNG 和 global step 均匹配。|
| T4 | PASS | `current=300` 重启跳过 100/200，不训练、不重跑 eval、不重写历史 evidence/report。|
| T5 | PASS | `target==current` 且 report/evidence 缺失时，不训练，补齐 eval/evidence 并完成 closure。|
| T6 | PASS | 历史 target evidence 缺失时 fail closed，未用 later latest 伪造历史证据。|
| T7 | PASS | 删除所有历史 full-resume 后 balanced selection 结果不变。|
| T8 | PASS | Stage promotion 只使用 selected ordinary checkpoint，不依赖历史 full-resume。|

定向命令结果：`8 passed`。

## 6. 磁盘空间影响

以下按当前观察到的约 6 GB/full-resume 估算，仅计算 full-resume 部分，不含 ordinary checkpoint、evaluation 和日志：

- 单 arm steady state：约 6 GB。
- Z05 + Z07 steady state：约 12 GB。
- 单 arm 一次 atomic save 的短时峰值：旧 latest 约 6 GB + temporary 约 6 GB = 约 12 GB。
- 两 arm 同时写入的理论 full-resume 峰值：约 24 GB。

因此建议两 arm 运行期间至少保留约 30 GB 的实际空闲空间作为 latest-only 写入安全 floor，并继续服从现有 supervisor 的 200 GiB minimum-free gate。root disk 的 50 GiB warning、20 GiB critical 和其他 storage policy 本次不改。

## 7. Activation procedure（本轮不执行）

本 patch 当前只达到 ready-for-activation，不自动切换 live 训练。正式 activation 必须：

1. 分别等待 Z05/Z07 到已保存且已 closure 的 milestone；确认 `resume_latest.pt`、ordinary `step_TARGET.pt`、formal report、runtime evidence 均存在，且 latest `runtime.global_step == TARGET`。
2. 记录每个 arm 的 PID、stage、global_step、optimizer update count、replay sizes、Z state 摘要、resume SHA256 和 bytes。
3. 停止 coordinator，再停止对应旧 runner；确认没有进程仍使用旧源码。
4. 将 live worktree 切到本修复 commit，验证 live HEAD 与 patch HEAD 一致；运行中的 Python 不会自动加载新源码，因此不能在线 checkout/pull。
5. 在 latest SHA/global_step 再次确认后，删除对应 stage 的 `resume_step_*.pt`。这一步只在进程停止且 latest 已验证后进行。
6. 启动 patched runner/coordinator，并在下一个 milestone 检查 ordinary checkpoint、resume_latest step、zero historical resume、formal report、runtime evidence 和 `du -sh`/`df -h`。

基于本次只读 live snapshot 的建议边界：

- Z05：建议以 1.1M 为切换边界，但必须先补齐/确认 1.1M formal report、runtime evidence 和 status closure；当前 status 曾记录 1.0M，而文件列表已出现 1.1M ordinary/latest，因此不能把 1.1M 视为已 closure。
- Z07：建议以当前记录的 1.0M closure 边界切换；切换前仍需重新确认 PID、latest SHA/global_step 和 evidence/status 未变化。

若主会话观察到进程已越过上述边界，则使用“下一个完整 closure milestone”，不回退或覆盖训练状态。

## 8. Rollback procedure

若 activation 后发现工程问题：停止 patched coordinator/runner，确认无旧源码进程，再将 live worktree 恢复到 activation 前 commit 并重新启动旧 runner。rollback 只回滚编排代码，不回滚或覆盖 `resume_latest.pt`、ordinary checkpoint、formal report、runtime evidence。若旧 runner 重新运行，它可能重新创建历史 hard link；这只会恢复旧存储行为，不改变训练科学状态。

## 9. 状态与 Git

本修复独立于正式训练 branch：

```text
branch: fix/z-resume-latest-only-20260920
base HEAD: 8e0ec7a95d73abacd2dcbad732c44d96bb4141fd
```

live branch/worktree 保持 untouched；本轮不 kill 进程、不删除 live 文件、不 merge 正式训练 branch、不进入 activation。
