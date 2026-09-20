# Z05/Z07 latest-only PRE-ACTIVATION END-TO-END CERTIFICATION

日期：2026-09-20（Asia/Shanghai）  
范围：CoCap Z05/Z07 unified-decay IQN 长训的 patch 分支认证。  
结论只代表工程链路已认证；本轮没有 activation，也没有停止、重启或修改 live 长训。

## 1. Git / base

- formal branch：`experiment/iqn-z-unified-decay-dual-curriculum-20260919`
- formal local/remote HEAD：`8e0ec7a95d73abacd2dcbad732c44d96bb4141fd`
- patch branch：`fix/z-resume-latest-only-20260920`
- patch HEAD at certification start：`a38f770de640f9a5d72b85f2a6be149444c5e5de`
- patch base：直接基于 formal HEAD；`PATCH_BASE_MATCHES_FORMAL_HEAD = true`
- fetch/push proxy：`127.0.0.1:17892`
- formal branch 没有发生 base drift；没有 merge 或 cherry-pick formal branch。

## 2. Automated evidence

执行命令：

```text
pytest -q tests/test_z_resume_latest_only_20260920.py \
  tests/test_iqn_z_unified_decay_curriculum_contract.py \
  test/test_iqn_full_resume_contract.py
```

结果：`30 passed, 2 warnings`。

- T1–T8 regression：PASS；原有基线 `25 passed` 保持，完整组合回归为 `30 passed`。
- T9 continuation equivalence：PASS；CPU deterministic setup 中，uninterrupted 与 save → destroy → reload `resume_latest` → continue 在 bit-exact 条件下比较了 global step、model、target model、optimizer、update counters、replay 内容 digest、recovery pool 内容、task/cursor、observations、histogram、epsilon/LR、Z state 与 Python/NumPy/Torch RNG。仅对运行目录等非训练状态字段作合同归一化。
- T10 stage-restart chain：PASS；覆盖 Stage1 已完成 promotion、Stage2 partial resume、Stage3 fresh promotion，以及 Stage3 partial resume。
- T11 coordinator relaunch：PASS；mock 验证 dead PID + absent tmux 的 `needs_launch`、Z05→GPU0、Z07→GPU1、patched runner、原 output/storage 目录和不 fresh-reset runtime。
- formal evaluation crash safety：PASS；`evaluate_checkpoint()` 以 atomic write 写 `report.json`，`target == current` 时缺 report 重评、完整 report 跳过，不会接受半写 JSON。
- final comparison automation：PASS；只从两个 arm 的 final reports、stage selections 和 formal evaluations 生成 comparison；加入 `resume_step_*` 后输出不变。

T1–T8 的历史 milestone skip、target==current closure、provenance fail-closed、selection independence、stage-promotion independence 仍全部 PASS。

## 3. Stage1 → Stage2 → Stage3 automation

```text
Stage1 partial
  → resume_latest exact resume                         PASS
  → Stage1 complete                                    PASS
  → balanced selection                                 PASS
  → selected ordinary checkpoint                       PASS
  → Stage2 fresh runtime                               PASS
  → Stage2 resume if interrupted                       PASS
  → Stage2 selection                                   PASS
  → Stage3 fresh runtime                               PASS
  → Stage3 resume if interrupted                       PASS
  → final report / Z05_VS_Z07 comparison               PASS
```

Stage promotion 只消费 selected ordinary checkpoint；promotion 后 optimizer、replay、RNG、environment、Z 和 recovery runtime fresh。历史 `resume_step_*` 在任何节点都不是依赖；latest-only runtime 依赖为 `resume_latest.pt`。

## 4. Live coordinator / runner 只读审计

没有发现正在运行的 `supervise_iqn_z_unified_decay_dual_20260919.py` dual coordinator。当前 live 是两个独立 per-arm supervisor：

```text
Z05 PID 3399755:
cd /home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919 && CUDA_VISIBLE_DEVICES=0 python3 tools/iqn_z_unified_decay_curriculum_20260919.py supervise --arm z05 --device cuda:0 --output /home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05

Z07 PID 3399761:
cd /home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919 && CUDA_VISIBLE_DEVICES=1 python3 tools/iqn_z_unified_decay_curriculum_20260919.py supervise --arm z07 --device cuda:0 --output /home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z07
```

当前 live coordinator 参数因此是：

- `--storage`：不适用；当前 per-arm command 未传入。
- `--output`：Z05 为 `...-runtime/z05`，Z07 为 `...-runtime/z07`。
- `--interval`：不适用；当前 per-arm command 未传入。
- `--min-free-gib`：不适用；当前 per-arm command 未传入。

代码默认值只读审计结果为：`--storage /data/disk2/home/yjq/cocap-runs/iqn-z-unified-decay-dual-curriculum-20260919`、`--interval 30`、`--min-free-gib 200`。代码中的 `needs_launch()` 对 `pid_alive=false` 且 `tmux_present=false` 返回 `true`，旧 status PID 不会阻止 relaunch；T11 对此有 mock 覆盖。

当前 live `launch.json` 记录的进程启动 HEAD 是 `fa12ce40ff76fe2c9235c9aead6078283addb661`；live worktree/formal branch 当前为 `8e0ec7a...`。这属于启动记录差异，本轮没有 checkout、重启或修改 live worktree。

## 5. Storage gate compatibility

当前 runtime：

```text
/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime
```

其文件系统为 `/dev/nvme0n1p2`，挂载点 `/`，所以：

```text
CURRENT_STORAGE_ON_ROOT = true
```

只读快照：root free `75,750,371,328` bytes，约 `70.6 GiB`；root warning 为 `50 GiB`，critical 为 `20 GiB`，当前两者均未触发。runtime 目录估算：Z05 `12G`，Z07 `18G`。

因此旧 coordinator gate：

```text
--min-free-gib 200
```

在当前 root runtime 上分类为：

```text
OLD_STORAGE_GATE_INCOMPATIBLE
```

本认证只注册 activation 后的显式 orchestration policy，不修改 supervisor 默认值、不修改科学训练配置：

```text
--min-free-gib 50
```

理由是 latest-only 双线理论 simultaneous atomic-save peak 约 `24 GiB`，建议最低约 `30 GiB`，运行 gate `50 GiB` 保留 margin。精确的 activation/relaunch command（本轮仅文档化，未执行）为：

```text
python3 tools/supervise_iqn_z_unified_decay_dual_20260919.py supervise \
  --output /home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919/artifacts/2026-09-19_iqn_z_unified_decay_curriculum \
  --storage /home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime \
  --interval 30 --min-free-gib 50
```

root critical 逻辑只记录为 known limitation：当前约 `70.6 GiB` 不会触发 warning/critical，显式 `50 GiB` gate 可正常 relaunch；没有扩大 scope 修复。

## 6. Live durable snapshot（只读）

本快照不构成 activation 边界，也不授权停止 live：

- Z05 live heartbeat：`1,193,000 / 1,200,000`；durable resume 为 Stage1 `1,100,000`。
- Z07 live heartbeat：`1,155,000 / 1,200,000`；durable resume 为 Stage1 `1,100,000`。
- 两条 `resume_latest.pt` 与各自 `resume_step_001100000.pt` 当前 inode 相同、link count 为 `2`；本轮没有删除任何 historical hardlink。
- Z05 `resume_latest.pt`：inode `22948662`，`6,174,658,131` bytes。
- Z07 `resume_latest.pt`：inode `22950690`，`6,521,735,699` bytes。

当前 live runner 是 patch activation 之前启动的旧进程，因此上述 live files 仍由旧启动链维护；patch branch 的 latest-only 行为已由 T1–T11 认证，尚未写入 live worktree。

## 7. Activation readiness matrix

| Component | Status |
| --- | --- |
| Full resume payload | PASS |
| T9 continuation equivalence | PASS |
| historical milestone restart | PASS |
| target==current closure | PASS |
| formal eval crash recovery | PASS |
| Stage1 restart | PASS |
| Stage1→2 promotion | PASS |
| Stage2 restart | PASS |
| Stage2→3 promotion | PASS |
| Stage3 restart | PASS |
| final comparison automation | PASS |
| coordinator relaunch | PASS |
| GPU mapping | PASS |
| storage gate compatibility | PASS（显式注册 `50 GiB`；旧默认 `200 GiB` 标记 incompatible） |
| scientific contract unchanged | PASS |

工程认证结论：`ACTIVATION_CHAIN_CERTIFIED`。  
执行状态：`activation_performed = false`；`live_processes_untouched = true`；当前没有执行 activation，也没有在未到安全停机边界时停止 live。

