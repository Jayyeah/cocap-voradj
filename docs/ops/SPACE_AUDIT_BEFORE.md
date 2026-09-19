# Space Audit Before Cleanup

审计时间：2026-09-19（Asia/Shanghai）

## Filesystem baseline

- 根文件系统 `/dev/nvme0n1p2`：915G total，812G used，57G free，94% used。
- 根文件系统 inode：约 6% used，inode 不是当前瓶颈。
- 已存在 `/data/disk2`：5.5T total，约 4.5T free，15% used；不挂载新盘。

## User-owned top consumers

| Path | Observed size | Classification |
|---|---:|---|
| `/home/yjq/rl/CoCap1/cocap-voradj/artifacts` | 58G | 历史实验产物；其中 replay-like 文件可重建 |
| `/home/yjq/rl/CoCap1/cocap-voradj/runs` | 11G | 历史 runs；checkpoint 用途未统一确认，暂不删除 |
| `/home/yjq/.vscode-server` | 5.1G | 当前 Remote/Codex 相关目录，暂不做粗暴清理 |
| `/home/yjq/.cache` | 320K | 可重建 cache，收益很小 |
| `/home/yjq/rl/CoCap1` | 200G | 多个 worktree；活跃 worktree 不删除 |

## Cleanup candidates

1. Legacy worktree `/home/yjq/rl/CoCap1/cocap-voradj/artifacts` 下的 replay-like 文件：340 files，42,353,669,292 bytes，约 39.44 GiB。
2. 仅清理明确可重建的 `__pycache__`、`.pytest_cache` 和 stale temporary 文件；不触碰当前正式 worktree 的运行状态。

删除前证据：旧 legacy worktree 没有匹配进程；当前 supervisor 使用独立正式 worktree `/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919`，运行存储配置为已存在挂载盘 `/data/disk2`。

## Explicitly retained

- 当前 Z05/Z07 supervisor、课程 coordinator 状态、live replay 和 resume/checkpoint。
- 所有正式 branch source、docs、configs、tests。
- 历史 selected/final/Z200k/milestone checkpoint、effective config、manifest、report 和 machine-readable result。
- `~/.vscode-server` 当前 server/extension 环境，因无法证明旧版本均未使用而跳过。

预计优先回收约 39.44 GiB；实际回收以后续 ledger 和 `df -h` 为准。
