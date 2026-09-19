# Space Reclamation Ledger

日期：2026-09-19（Asia/Shanghai）

| Batch | Path / class | Before | After | Reclaimed | Why safe | Retained evidence |
|---|---|---:|---:|---:|---|---|
| completed | `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/**/*replay*` | 39.44 GiB logical | 0 | included below | legacy worktree 无匹配进程；历史 replay 可重建 | config/report/checkpoint/manifest/runtime_state.pkl |
| completed | `cocap-voradj-speedopt/artifacts/**/*replay* >100M` | 1.94 GiB logical | 0 | included below | inactive 2026-08 worktree；无进程 | checkpoint/manifest/config/report |
| completed | `cocap-voradj-allagent-oldmix/artifacts/**/*replay* >100M` | 18.73 GiB logical | 0 | included below | inactive 2026-08 worktree；无进程 | checkpoint/manifest/config/report/resume_latest manifest |

累计逻辑删除约 60.11 GiB；因重复路径/硬链接/稀疏文件等 filesystem accounting，`df` 观察到根盘约回收 33G：57G free → 90G free。

保护项：当前 Z05/Z07 formal worktree、supervisor/coordinator state、live replay、resume_latest、milestone/selected/final checkpoint、source/docs/config/tests 均不在删除范围。
