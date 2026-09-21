# Z05 / Z07 恢复状态（2026-09-21）

## 范围

本记录同步 IQN Z05/Z07 双线的恢复入口、代码修复和运行快照。
训练 checkpoint、full-resume、replay 与 heartbeat 仍保留在本机 runtime 目录，
不上传 GitHub，避免把多 GB 运行态文件写入代码仓库。

代码分支：`experiment/iqn-z-unified-decay-dual-curriculum-20260919`

本次同步提交：`995e7e0 fix: resume Z curriculum selection and evaluation`

## 当前运行

| 线 | 物理 GPU | PID | tmux | Stage / 状态 | 当前进度 |
|---|---:|---:|---|---|---:|
| Z05 | GPU0 | 17097 | `iqn_z05_recovery_20260921` | Stage2 / formal evaluation | 100k / 100k |
| Z07 | GPU1 | 19555 | `iqn_z07_recovery_20260921` | Stage2 / training | 146k / 200k |

Z05 heartbeat 最近报告：`update_steps=21991`、`loss_ema=40.9071`。
Z07 heartbeat 最近报告：`update_steps=31621`、`loss_ema=41.4348`。
两线 status/heartbeat 均为 `running`，当前没有新的 `error`。

## 恢复入口

进程内 GPU 编号遵循 `CUDA_VISIBLE_DEVICES` 映射，因此物理 GPU1 的 Z07 使用
进程内 `--device cuda:0`：

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/iqn_z_unified_decay_curriculum_20260919.py \
  supervise --arm z05 --device cuda:0 \
  --output /home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05

CUDA_VISIBLE_DEVICES=1 python3 tools/iqn_z_unified_decay_curriculum_20260919.py \
  supervise --arm z07 --device cuda:0 \
  --output /home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z07
```

当前阶段的 exact full-resume 文件分别位于：

```text
...runtime/z05/stages/stage2/training/checkpoints/resume_latest.pt
...runtime/z07/stages/stage2/training/checkpoints/resume_latest.pt
```

## 本轮自动线修复

1. Balanced selection 对缺失指标使用 finite penalty，避免 `allow_nan=False` 写出
   `inf` 并触发 fail-closed。
2. formal evaluator 的 central diagnostic padding 扩展到 12 pursuers，覆盖
   4/8/12 pursuer 的 Stage1/2/3，而不改变策略输入或训练合同。

本轮只同步代码和本记录；runtime 产物继续由本机 Stage supervisor 管理。
