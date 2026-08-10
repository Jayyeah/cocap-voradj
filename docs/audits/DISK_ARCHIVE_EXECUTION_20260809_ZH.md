# R1/R2/R3 归档与 Stage4 speedopt 替换执行报告（2026-08-09）

## 执行结果

- 用户确认：执行 R1/R2/R3；关闭原 Stage4A/4C；保留旧 trainer checkpoint，不保留旧 replay；立即启动 speedopt 同配置双线；新线完成自动评估且用户再次同意前不得清理 checkpoint。
- 操作前根盘空闲约 39 GiB、使用率 96%；操作完成后空闲约 101 GiB、使用率 89%。
- 没有使用 sudo、没有更改文件 owner/权限、没有触碰原 IQN 最终模型/rollout/GIF。

## 旧 Stage4 处置

| 线 | 原 PID | 停止时最后 metrics | 最后完整 trainer bundle | replay 处置 | 保留根大小 |
|---|---:|---:|---:|---|---:|
| Stage4A | 527834 | 67k | step=50k | step1/25k/50k 共 3 份已删除 | 约 338 MiB |
| Stage4C | 528005 | 67k | step=50k | step1/25k/50k 共 3 份已删除 | 约 338 MiB |

共删除 6 个旧 replay、5,028,027,007 B（约 4.683 GiB）。旧线所有 `trainer.pt`、effective config、scene config、manifest、runtime state、metrics、diagnostic eval、stdout 和上层 eval20 结果均保留。旧 replay 是本轮唯一不可逆删除项，按用户明确指示执行。

## R1：历史 W&B / 长 JSONL 无损压缩

- 候选排除了旧 Stage4 保护根、speedopt validation 和新 `2026-08-09_200k_speedopt` 根。
- 614 个文件完成 gzip level-1 压缩；每个文件均在删除源文件前完整解压回读，并校验恢复字节数和 SHA-256。
- 原始总量 46,725,106,342 B（约 43.516 GiB）；压缩后 4,806,114,903 B（约 4.476 GiB）；净释放 41,918,991,439 B（约 39.040 GiB）。
- 两个位于 `/home/yjq/rl/save/terl_native_supervised_20260607/`、非 `yjq` 可写的 W&B 文件被记录并跳过，总计 399,411,057 B；未尝试 sudo、chown 或权限绕过。
- 最终 dry-run 只剩上述两个权限跳过项；未遗留 `.gz.tmp`。

Manifest：`docs/audits/R1_GZIP_MANIFEST_20260809.jsonl`。

## R2：replay 字节级去重

- 33 对 standalone/bundle replay 先逐对计算双 SHA-256；33 对全部相同，0 mismatch。
- 仅在同文件系统且 hash 相同时，以原子硬链接替换 standalone inode；standalone 路径和 bundle 路径都继续存在，直接读取/resume 语义不变。
- 净释放 15,627,664,157 B（约 14.554 GiB）。
- 新旧 Stage4、speedopt validation 均在排除范围。

Manifest：`docs/audits/R2_HARDLINK_MANIFEST_20260809.jsonl`。

## R3：VS Code Server 可重装缓存

- 删除 4 个非当前 Server：`125df...`、`1b6a...`、`8a7a...`、`e4c7...`；保留并核验当前 `Stable-df53...` 进程。
- 删除旧 OpenAI 26.727、Claude 2.1.223/2.1.224 和 CachedExtensionVSIXs；保留并核验当前 OpenAI 26.803、Claude 2.1.226 进程。
- `.vscode-server` 由约 5.9 GiB 降至约 1.6 GiB，净释放约 4.3 GiB；被删版本可由 VS Code/扩展市场重装。

## 新 Stage4 speedopt 双线

| 线 | PID | GPU | tmux | tag |
|---|---:|---|---|---|
| Stage4A | 1326505 | cuda:0 | `stage4a_speedopt_200k` | `stage4a_speedopt_200k_20260809` |
| Stage4C | 1326499 | cuda:1 | `stage4c_speedopt_200k` | `stage4c_speedopt_200k_20260809` |

共同根：`artifacts/2026-08-09_200k_speedopt/`。两线 seed 2026080801、total steps 200k、nice=10；实际 step1 effective config 已核验 `checkpoint_interval_env_steps=25000`、`diagnostic_eval_interval_env_steps=25000`。每 25k 保存完整 trainer+replay bundle 并自动 diagnostic eval；最终自动生成 report/eval。新线任何 checkpoint/replay 均不会自动清理，必须等待训练完成、评估复核和用户再次确认。

## ETA

- 2026-08-09 20:06：Stage4A/4C 最新约 77k/69k，均 `mean_finite=1`。
- 纳入 25k checkpoint 保存和诊断评估后的近期稳态吞吐约为 A **13.6–13.7k steps/h**、C **13.3k steps/h**；早期 9k–10k 的 15.38k/h 偏乐观，现以长窗口为准。

| 里程碑 | Stage4A 预计 | Stage4C 预计 |
|---|---:|---:|
| 75k | 已于 19:54 完成 | 08-09 20:30–20:45 |
| 100k | 08-09 21:45–22:05 | 08-09 22:00–22:20 |
| 150k | 08-10 01:25–01:50 | 08-10 01:50–02:20 |
| 200k + 最终内部评估 | 08-10 05:10–05:50 | 08-10 06:00–06:40 |

保守按 **08-10 07:00 前两线完成** 规划。ETA 已计入每 25k 保存完整 checkpoint/replay 和内部诊断评估的开销；同机外部负载变化可能造成偏移。训练完成后不会自动清理 checkpoint 或 replay，须先完成自动评估并取得用户确认。

## 20:06 行为信号与提前并行结论

| checkpoint | eval20 capture | eval20 collision | 结论 |
|---|---:|---:|---|
| Stage4A 50k | 95% | 5% | 强 PASS；当前最佳，永久保留 |
| Stage4A 75k | 25% | 75% | 明显退化；追捕速度上升但碰撞坍缩 |
| Stage4C 50k | diagnostic 0/4 | 0/4 | 尚无 moving+obstacle 晋级信号 |

A50 证明连续 `(a,ω)` stationary capture 已形成可靠 anchor；A75 同时证明表现非单调，最终必须逐 checkpoint 选优。当前可提前并行 Stage4B moving/no-obstacle 25k gate，但不能跳到 Stage4D/4E/5。资源上只考虑较冷 GPU0 的一条低优先级短线；GPU1 90–91°C 不叠加。当前 runner 缺少 MASAC actor-only 跨配置初始化，须先实现严格加载/审计/测试，不能通过 full resume 混入 A replay。
