# Z05/Z07 双线 Runtime 交接

更新时间：2026-09-19（Asia/Shanghai）  
正式分支：`experiment/iqn-z-unified-decay-dual-curriculum-20260919`  
本次状态快照 commit：`fa12ce40ff76fe2c9235c9aead6078283addb661`

## 当前状态

两条线均已启动并持续写 heartbeat，当前都在 Stage 1 的 `100000` milestone formal evaluation：

| 线 | 状态 | GPU | PID | tmux | stage/phase | step/target | replay | update steps |
|---|---|---:|---:|---|---|---:|---:|---:|
| Z05 | running | 0 | 3399755 | `iqn_z05_unified_decay_20260919` | stage1 / formal_evaluation | 100000 / 100000 | 400000 | 23705 |
| Z07 | running | 1 | 3399761 | `iqn_z07_unified_decay_20260919` | stage1 / formal_evaluation | 100000 / 100000 | 400000 | 23490 |

双线 coordinator：`iqn_z_unified_decay_dual_20260919`。训练输出使用用户可写 fallback：
`/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/{z05,z07}`。

当前 heartbeat 还显示：Z05 recent capture `0.0674`、collision `0.9888`；Z07 recent capture `0.0337`、collision `0.9888`。这些是运行遥测，不是课程合同或选择规则的修改。

## 科学与训练参数

两线共同参数：

- `train_mode=voradj_mixed_coverage`；Z state enabled，更新语义为 `synchronous_previous_decision_state`。
- `hard_zero_threshold=0.10`；`NormSense-V2=forward-final-density-normalized-sensing-v2`；`friend_ordering_mode=physical_only`。
- 观测包含 Z state，不包含 `is_pursuing`；`self_feature_dim=9`、`pursuer_feature_dim=7`。
- IQN：single-head，9 actions，hidden 256，4 layers，8 heads，32 quantiles，batch 128，`gamma=0.99`，gradient clip 0.5，train frequency 4，target update 10000，replay capacity 1,000,000。
- full resume 开启；checkpoint 每 `100000` steps。
- 每个 milestone 自动运行 coverage/capture/mixed 三个场景，每场景 20 episodes；deterministic quantiles=`midpoint32`。
- 两线唯一科学差异是 Z05 `lambda=eta=0.5` 与 Z07 `lambda=eta=0.7`。

| 阶段 | 任务形状 | 总步数 | epsilon | learning rate | active pursuers | post-capture window |
|---|---|---:|---|---|---:|---:|
| Stage 1 | 4 pursuers / 1 evader / 1 obstacle | 2,000,000 | 0.60 -> 0.05 / 500k | 1e-4 at 0, 3e-5 at 200k | 4 | 500 |
| Stage 2 | 8 pursuers / 2 evaders / 2 obstacles | 700,000 | 0.28 -> 0.05 / 420k | 3e-5 | 8 | 600 |
| Stage 3 | 12 pursuers / 3 evaders / 3 obstacles | 700,000 | inherits Stage 2 | 3e-5 | 12 | 700 |

Stage 2/3 的 mixed task 使用 map-random；coverage task 使用 inner-random-cluster。Stage 2 cluster radius 为 11m，Stage 3 为 13m；pursuer/evader map-random 最小间隔为 15m，coverage pursuer 最小间隔为 7m。

## 自动化行为与测试

- 启动前共享 preflight 为 `PASS`：验证双线非 alpha 差异为 0、initial model bit-exact、fresh-stage gate、checkpoint exact load、full-resume exact、formal evaluator smoke 和每决策一次 Z update。
- 没有常驻的全量 `pytest` 进程；自动测试入口是启动 preflight/smoke 和本仓库 contract tests。里程碑 formal evaluation 是 runner 内建自动任务，不依赖 Git push。
- Stage 1 从 scratch 开始。每 100k 到 2M 自动训练、保存 checkpoint/full-resume、正式评估并生成 evidence。
- Stage 2 和 Stage 3 自动从同一 arm 上一阶段的 `selected_checkpoint` warm-start；每次晋级重新开始 optimizer、replay、RNG、env、Z 和 recovery runtime，保留 selected model weights。
- 每一阶段结束后自动执行 `select_balanced()`：优先 BalancedFloor，按注册 tie-break 选择；若 strict coverage 全为 0，使用 capture-retention + capture/CE-RMS/area-CV maximin fallback。
- coordinator 的 arm 状态读取已修复为 `status.json -> heartbeat.json -> launch.json` fallback。此前 arm runner 在训练阶段只写 heartbeat，导致 coordinator 错误显示 `missing`；该显示问题不代表训练中断。

## 磁盘与 watchdog

- 根盘 `/dev/nvme0n1p2`：本次复核约 `88G` free，inode 约 `6%` used。
- 之前已清理明确可重建的历史 replay，观察到约 `57G -> 90G` 的可用空间恢复；未删除当前 live replay、resume、milestone/selected/final checkpoint、source、docs、configs、tests 或 reports。
- 当前 live runtime 约 `3.81 GiB`；`resume_latest.pt` 与 milestone resume 使用 hard link 复用块，不应重复估算。
- `/data/disk2` 未新挂载；因 `/data/disk2/home` 权限不足，当前继续使用用户可写 fallback runtime。
- root disk watchdog：warning `<50 GiB`，critical `<20 GiB`；critical 时 coordinator `failed_closed`，不自动删除 live replay，不改变 replay capacity、stage budget、LR、epsilon 或任何科学参数。

## Proxy 与 Git

- stale 小写变量仍可能指向 `127.0.0.1:17891`；mihomo 当前 mixed-port 为 `127.0.0.1:17892`，监听已确认。
- 未修改 `~/.bashrc`、global Git config、系统 proxy；永久配置修改：**NO**。
- fetch 使用命令级 17892 覆盖已成功；push 也必须只使用命令级 17892，最多尝试两次。

```bash
env http_proxy=http://127.0.0.1:17892 https_proxy=http://127.0.0.1:17892 \
  HTTP_PROXY=http://127.0.0.1:17892 HTTPS_PROXY=http://127.0.0.1:17892 \
  all_proxy=socks5://127.0.0.1:17892 ALL_PROXY=socks5://127.0.0.1:17892 \
  git -c http.proxy=http://127.0.0.1:17892 \
      -c https.proxy=http://127.0.0.1:17892 push
```

训练与 Git 同步解耦：push 失败不能阻止本地 coordinator 或两条训练线继续运行。

