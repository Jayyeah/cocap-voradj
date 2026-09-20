# Z05/Z07 latest-only full-resume activation 记录

日期：2026-09-20（Asia/Shanghai）  
结论：`LATEST_ONLY_ACTIVATED`

## 版本与架构

- formal branch：`experiment/iqn-z-unified-decay-dual-curriculum-20260919`
- activation 前 formal HEAD：`8e0ec7a95d73abacd2dcbad732c44d96bb4141fd`
- activation 后 formal HEAD：`58baa8143f143d013e32939d3bbb17f9c60bcec4`
- cherry-pick：`a38f770de640f9a5d72b85f2a6be149444c5e5de`、`e2265cfc0e2732b5a5a497d2e18802ce1ce06ba6`
- patch branch：`fix/z-resume-latest-only-20260920`
- Git fetch/push proxy：`127.0.0.1:17892`

本次保持原有独立 supervisor 架构，没有启动 dual coordinator。Z05 使用 GPU0，Z07 使用 GPU1；两条线继续由各自的 `supervise --arm` 自动负责 Stage1→Stage2→Stage3。

## Quiesce、清理与重启

两条线均在 Stage1 的 durable boundary `1,200,000` 停止。Z05 在边界自然退出；Z07 在已完成 durable save、且已通过只读 full-resume load 检查后按授权 TERM。训练步损失为 0。旧 PID、tmux session 和对应 GPU PID 均已确认消失，未触碰无关 GPU 进程。

`resume_latest.pt` 的边界摘要：

| arm | step | SHA256 | optimizer updates | target updates | replay / recovery |
| --- | ---: | --- | ---: | ---: | --- |
| Z05 | 1,200,000 | `611594166946406d222393c1730901b355376d12841c36335282ea3d9d69b6a1` | 298705 | 120 | 643264 / 92624 / 1000000 / 1000000；1000 |
| Z07 | 1,200,000 | `6af2a557c9b878f48e51712ffba6d54b31ce3dbd0241e75b3ac0aa5ec662c7ce` | 298490 | 120 | 783840 / 102892 / 1000000 / 1000000；1000 |

删除历史 `resume_step_*.pt` 前共 6 个，删除后两条 runtime 均为 0。重新 `torch.load` 后两条 `resume_latest.pt` 的 SHA 和 `global_step=1,200,000` 均不变。当前 root filesystem 为 `/dev/nvme0n1p2`，最终 free 约 88 GiB；runtime 当前约 Z05 6.4 GiB、Z07 6.7 GiB。

## Exact resume 与 target==current closure

两条 arm 均从各自 `resume_latest.pt` 的完整 payload 恢复，包含 model/target、optimizer、replay 内容、recovery pool、task cursor、observations/runtime state、Z state、RNG 和训练合同；不是 model-only warm start。patched `target==current` 路径完成了已有 1.2M ordinary checkpoint 的 formal evaluation/runtime evidence closure，然后继续后续训练。

重启后观测：

- Z05：`stage1 / training`，从 1,200,000 继续到 1,208,000，下一 target 为 1,300,000。
- Z07：`stage1 / formal_evaluation`，`current=target=1,200,000`，正在补齐 target==current closure；进程健康，无错误。

由于 target==current closure 路径不构造新的 Trainer，旧 `resume_loaded.json` 不会被伪造刷新；此行为已由源码审计、完整 `torch.load`、heartbeat continuation 和 closure artifact 共同验证。

## 测试与自动线

- T1–T8 regression：25 passed。
- T9 continuation equivalence：bit-exact PASS。
- T10 stage-restart chain：PASS。
- T11 coordinator relaunch contract：PASS。
- 完整相关套件：30 passed；生产 Z contract 套件：24 passed。
- production preflight：PASS；cross-arm scientific/non-alpha contract diff：0。
- formal evaluation crash safety：PASS（report atomic write；缺失则重算，完整则 skip）。
- Stage1 restart、Stage1→2 promotion、Stage2 restart、Stage2→3 promotion、Stage3 restart、final comparison automation：均由 T10/T11 和现有 contract tests PASS 覆盖。
- `AUTO_CURRICULUM_READY=true`：独立 supervisor 会自动完成当前 stage、balanced selection、下一 stage fresh runtime、后续 exact resume 及最终 arm report，不依赖人工 promotion。

历史 `resume_step_*` 在任何自动链节点都不是依赖；只使用当前 stage 的 `resume_latest.pt`、ordinary selected checkpoint、selection report 和 formal evaluation。

## Storage 与 comparison 说明

当前 runtime 在 root filesystem。latest-only steady storage 约 6–7 GiB/arm，双写理论峰值约 24 GiB；未来 dual coordinator 的注册 gate 为 `--min-free-gib 50`。本次没有启动 dual coordinator，因此没有修改其默认 200 GiB，也没有把它引入当前训练架构。

当前尚未人为启动最终 comparison writer；两条 arm 完成后，各自生成 `Z05_CURRICULUM_FINAL_REPORT.json`、`Z07_CURRICULUM_FINAL_REPORT.json`，再由现有 final-comparison writer 生成 `Z05_VS_Z07_FINAL_COMPARISON.json`。对应自动化逻辑测试已 PASS。

原始 launch command 与 durable snapshot 已保存于 activation 工作记录中的 `PRE_ACTIVATION_LAUNCH_COMMANDS.json` 和 `PRE_PATCH_DURABLE_SNAPSHOT.json`；本次未提交任何 resume 文件。
