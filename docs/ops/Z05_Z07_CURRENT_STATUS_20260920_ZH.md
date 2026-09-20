This document supersedes previous runtime-status snapshots for current live-state reporting only.

# Z05 / Z07 Runtime Status Auditor & Handoff

## 1. Snapshot

- captured at: `2026-09-20T11:38:24+08:00` (Asia/Shanghai)
- worktree: `/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919`
- branch: `experiment/iqn-z-unified-decay-dual-curriculum-20260919`
- local HEAD: `661378561ccd74cee2945851b420297975ce1c1a`
- remote HEAD: `661378561ccd74cee2945851b420297975ce1c1a`
- local/remote一致：`true`
- worktree：审计开始时 clean；本轮只新增本 status 文档与 JSON
- live runtime: `/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime`
- remote fetch：已用命令级 `127.0.0.1:17892` 成功；未修改 Git/global/system proxy

运行事实优先级为 live PID/GPU/tmux、heartbeat、metrics、status/checkpoint/evaluation；旧 Markdown 只作辅助证据。

## 2. Scientific Contract

当前正式分支的 resolved config 与 effective runtime config 一致于以下合同：

| 项目 | Z05 | Z07 |
|---|---:|---:|
| `z_state.lambda` | `0.5` | `0.7` |
| `z_state.eta` | `0.5` | `0.7` |
| `hard_zero_threshold` | `0.10` | `0.10` |
| Stage 1 budget | `2,000,000` | `2,000,000` |
| Stage 2 budget | `700,000` | `700,000` |
| Stage 3 budget | `700,000` | `700,000` |

共同合同已核对为：

- NormSense-V2: `forward-final-density-normalized-sensing-v2`
- `perception.include_z_state=true`、`iqn.include_z_state=true`
- `perception.include_is_pursuing=false`、`iqn.include_is_pursuing=false`
- `pursuing_embed_dim=0`、`pursuing_late_fusion=false`
- `friend_ordering_mode=physical_only`
- `train_mode=voradj_mixed_coverage`
- `checkpoint_freq=100000`、`checkpointing.full_resume=true`
- formal evaluation：coverage/capture/mixed，各 `20` episodes，间隔 `100k`
- replay batch：`pursuing=64`、`pre_capture_cover=16`、`post_capture_real=32`、`recovery_pure=16`
- recovery：capacity `1000`、captured-state ratio `0.75`、non-capture map-random ratio `0.5`

Z-v2 实现核对：每个 environment decision 在动作推进后、下一决策 observation 前同步更新一次；physics substep 不更新。公式为：

```text
z_raw_i = max(direct_i, lambda * z_i(previous), eta * max_neighbor(z_j(previous)))
z_i = 0 if z_raw_i < 0.10 else z_raw_i
```

传播读取 immutable previous-Z。checkpoint schema 为 `cocap-z-state-v2`，保存并校验 values、source、lineage、neighbor diagnostics、update count、lambda/eta/threshold。当前代码的 preflight/contract diff 逻辑显示：每个 stage 的 Z05/Z07 差异仅为 `z_state.lambda` 与 `z_state.eta`，non-alpha difference count 为 `0`。

## 3. Z05 Live State

- status：`running`
- arm：`z05`；alpha=`0.5`
- stage/phase：`stage1 / training`
- current global step：`871000`
- current target milestone：`900000`
- latest heartbeat JSON time：`2026-09-20T11:38:09+0800`；capture 时约 `15s` 前
- PID：`3399755`；process alive=`true`
- parent：`3399754`；tmux=`iqn_z05_unified_decay_20260919`
- GPU：physical GPU `0`；`CUDA_VISIBLE_DEVICES=0`；nvidia-smi PID 映射一致
- runtime duration：约 `13h50m`
- metrics latest step：`871000`
- optimizer updates：`216455`
- target update count：`87`
- epsilon：`0.05`
- learning rate：`3e-5`
- latest loss：`28.034395217895508`
- loss EMA：`28.05422328054203`
- replay total：`2226316`
- replay：`pursuing=478253`、`pre_capture_cover=59163`、`post_capture_real=688900`、`recovery_pure=1000000`
- recovery pool：`462`
- action histogram（training telemetry）：`[568266,377616,606443,303964,254965,272146,437501,308392,354707]`
- 最新 training telemetry（非 formal eval）：success=`0.35`、capture=`0.99`、strict coverage=`0.35`、collision=`0.01`、area CV=`0.120885`
- current runtime size：约 `25G`
- latest ordinary checkpoint：`.../z05/stages/stage1/training/checkpoints/step_800000.pt`；`latest.pt` 同为约 800k 保存点
- latest full-resume：`.../z05/stages/stage1/training/checkpoints/resume_latest.pt`，其保存 global step=`800000`
- latest formal evaluation：`.../z05/stages/stage1/evaluations/step_000800000/report.json`，status=`complete`
- latest fully closed milestone：`800000`

Z05 进程、tmux、heartbeat、metrics 均证明它仍在继续训练；`status.json` 仍显示上一个已经写入 status 的 800k transition，这是 runner 的覆盖式 status 设计，不是 stale PID 状态。

## 4. Z07 Live State

- status：`running`
- arm：`z07`；alpha=`0.7`
- stage/phase：`stage1 / formal_evaluation`（800k report 已完成，runner 当时仍在写 runtime evidence）
- current global step：`800000`
- current target milestone：`800000`；下一训练段应为 `900000`
- latest heartbeat JSON time：`2026-09-20T11:37:17+0800`；capture 时约 `67s` 前
- PID：`3399761`；process alive=`true`
- parent：`3399759`；tmux=`iqn_z07_unified_decay_20260919`
- GPU：physical GPU `1`；`CUDA_VISIBLE_DEVICES=1`；nvidia-smi PID 映射一致
- runtime duration：约 `13h50m`
- metrics latest step：`800000`
- optimizer updates：`198490`
- target update count：`80`
- epsilon：`0.05`
- learning rate：`3e-5`
- latest loss：`27.411945343017578`
- loss EMA：`27.528415282354207`
- replay total：`2225288`
- replay：`pursuing=599509`、`pre_capture_cover=68571`、`post_capture_real=557208`、`recovery_pure=1000000`
- recovery pool：`430`
- action histogram（training telemetry）：`[398650,291442,462131,341262,236871,342976,369846,322931,433891]`
- 最新 training telemetry（非 formal eval）：success=`0.78`、capture=`0.98`、strict coverage=`0.78`、collision=`0.03`、area CV=`0.128803`
- current runtime size：约 `26G`
- latest ordinary checkpoint：`.../z07/stages/stage1/training/checkpoints/step_800000.pt`
- latest full-resume：`.../z07/stages/stage1/training/checkpoints/resume_latest.pt`，其保存 global step=`800000`
- latest formal evaluation：`.../z07/stages/stage1/evaluations/step_000800000/report.json`，status=`complete`，完成时间 `2026-09-20T11:37:19+0800`
- latest fully closed milestone：`700000`
- 800k closure：checkpoint、full-resume、formal report 已有；截至本 snapshot `runtime_evidence/step_000800000.json` 尚未出现，`status.json` 仍停在 700k，因此 800k 尚未视为完整闭环

这不是 `STALE_STATUS`：PID、tmux、heartbeat 和进程 CPU 均表明 runner 仍存活；这里只是 formal report 完成到 evidence/status 写入之间的短暂 live transition。

## 5. Latest Formal Evaluations

以下均为 deterministic formal evaluation（每场景 20 episodes），不是 training heartbeat。

### Z05 / Stage 1 / 800k

- Pure Coverage：strict CE=`0.40`；CE RMS mean=`0.056991`；area CV mean=`0.119476`；collision=`0`；time-to-CE mean=`89.6875s`（8 个成功样本）
- Pure Capture：normal=`1.00`；stationary=`0`；ring2=`1.00`；ring3=`1.00`；collision=`0`；capture time mean=`80.025s`
- Mixed：capture=`1.00`；collision=`0.05`；post-capture CE=`0.25`；safe-complete=`0.25`；capture time mean=`75.925s`；recovery time mean=`94.4s`（5 个样本）；mission time mean=`159.1s`（5 个样本）
- Z：direct-visible-Z-one=`1.00`；support Z mean=`0.481029`；coverage Z mean=`0.000481`；pure coverage exact zero=`true`；neighbor-dominant count=`1924`；neighbor Z mean=`0.489995`；max lineage hop=`2`；post-capture release mean=`1.5s`；never-release=`0`；update checks=`51625`；violations=`0`
- formal action histogram：`[983,152448,7416,10671,515,2053,7017,23129,2268]`

### Z07 / Stage 1 / 800k

- Pure Coverage：strict CE=`0.05`；CE RMS mean=`0.077423`；area CV mean=`0.194674`；collision=`0`；time-to-CE mean=`61s`（1 个成功样本）
- Pure Capture：normal=`1.00`；stationary=`0`；ring2=`1.00`；ring3=`1.00`；collision=`0`；capture time mean=`42.675s`
- Mixed：capture=`1.00`；collision=`0.05`；post-capture CE=`0`；safe-complete=`0`；capture time mean=`37.25s`；recovery time=`not available from saved formal result`；mission time=`not available from saved formal result`
- Z：direct-visible-Z-one=`1.00`；support Z mean=`0.675857`；coverage Z mean=`0.000778`；pure coverage exact zero=`true`；neighbor-dominant count=`2092`；neighbor Z mean=`0.690363`；max lineage hop=`2`；post-capture release mean=`3s`；never-release=`0`；update checks=`69836`；violations=`0`
- formal action histogram：`[1002,367,123,130847,33,133530,3317,7777,2348]`

不要把以上 800k 结果外推为最终 alpha 优劣，也不据此 early-stop；预注册 Stage 1/2/3 课程仍继续。

## 6. Milestone Completeness

- Z05：`100k, 200k, 300k, 400k, 500k, 600k, 700k, 800k` 均有 ordinary checkpoint、full-resume、formal `report.json` 和 runtime evidence。当前在 800k 之后的 900k training segment。
- Z07：`100k…700k` 均有上述四类证据；800k 的 checkpoint、full-resume、formal report 已完成，但截至 snapshot runtime evidence 与 status transition 尚未写出，故 latest fully closed milestone 仍为 700k。
- 每个 stage 的持久 status 是单个 `status.json` 覆盖写入；没有 append-only status-transition history 文件。因此历史 transition 的直接保存证据不可用，只能以当前 status、runner source、checkpoint/eval/evidence 文件组合核实。
- 当前两条线都只有 `stage1/` runtime 目录；没有 Stage2/Stage3 selected checkpoint 或 final report。

## 7. Training Telemetry

Training telemetry 与 formal deterministic evaluation 严格分开。当前 telemetry 显示：

- Z05 已从 800k 继续到 871k，loss 与 loss EMA 为有限值，replay、update count、heartbeat 持续增长；最新窗口 capture 很高但 latest mixed task 的 coverage success 仍低，不能替代 formal conclusion。
- Z07 已完成 800k rollout 并进入 formal report/evidence transition；其最新 training telemetry 停在 800k，不能将其 800k telemetry 当作 formal 800k summary。
- 两条线 action histogram 都不是单一固定 action；正式评估中的动作分布仍较集中，但本轮不做策略优劣判断。

## 8. Automation

代码审计结论：`AUTO_CURRICULUM_READY=true`。

- Stage 1：4v1、2M、scratch；当前 runner 的 Stage1 无 pretrained path，并写入 fresh-stage gate。
- Stage 1 完成后：本 arm 的 `select_balanced()` 读取该 arm 全部 100k milestone report，选择 own-line selected checkpoint。
- Stage 2：8v2、700k；只继承同线 Stage1 selected model weights。
- Stage 3：12v3、700k；只继承同线 Stage2 selected model weights。
- 每次 promotion：fresh optimizer、fresh replay、fresh RNG、fresh env、fresh Z、fresh recovery pool；代码无 performance promotion gate。
- Balanced selection：`CaptureScore=min(pure normal capture,mixed capture)`，`CoverageScore=pure strict CE`，最大化 `BalancedFloor=min(CaptureScore,CoverageScore)`；tie-break 顺序为 mixed safe-complete、mixed post-capture CE、harmonic、worst collision、CE RMS、area CV、capture time、later checkpoint。strict CE 全零时仍使用 registered capture-retention + normalized capture/CE-RMS/area-CV maximin fallback。

## 9. Runtime Health

- 两个 arm PID 都 alive；两个 arm tmux 都存在；没有发现重复 trainer、孤儿 trainer 或错误 GPU 映射。
- nvidia-smi：GPU0 只有 PID `3399755` 训练进程，GPU1 只有 PID `3399761` 训练进程；各约 790 MiB。
- 两条线 loss 均 finite；checkpoint/full-resume schema 与 Z-v2 证据存在。
- dual coordinator：PID `3437106`，tmux=`iqn_z_unified_decay_dual_20260919`，alive；branch/head 记录为当前 `6613785…`。
- coordinator 当前 status：`queued / waiting_for_storage`，更新时间约 `11:36:55`，root free 低于 `--min-free-gib 50`，所以 persisted `arms` 字段为 `null`。直接读取 arm live evidence 仍能看到两条线；这不是 arm missing，也没有停止当前 trainer。

## 10. Known Issues

1. `COORDINATOR_WAITING_FOR_STORAGE`：根盘当前约 37G free、96% used，低于 dual coordinator 的 50 GiB storage gate，但仍高于 20 GiB critical threshold。coordinator 因此持续 queued；本轮不清理、不修改 watchdog、不重启训练。
2. `Z07_800K_EVIDENCE_STATUS_LAG`：800k formal report 已完成，但 snapshot 时 runner 尚未写出 800k runtime evidence/status transition；PID 仍 alive，属于 transition lag，不是 stale/dead training。
3. `RUNTIME_LAUNCH_HEAD_LAG`：两个 arm 的 launch.json 记录启动 HEAD=`fa12ce40…`，当前仓库 HEAD=`6613785…`。两者差异仅为 ops runtime 文档、contract test 和 coordinator heartbeat fallback 代码；未发现训练科学变量差异，但 launch provenance 不是当前 HEAD。
4. `PREFLIGHT_PROVENANCE_MISMATCH`：共享 `startup_sanity.json` 内容为 PASS，但其记录 head=`81be35…`、branch=`experiment/iqn-z-token-scratch-20260918`，与正式 worktree/runtime launch head 不同。因此它只能作为历史 preflight evidence，不能单独当作当前 HEAD 的 bit-exact proof；当前合同另由最新源码、effective_config、contract diff 与 live runtime 交叉核对。
5. `STATUS_HISTORY_NOT_PERSISTED`：runner 只覆盖写 `status.json`，没有每个 milestone 的 append-only transition log；历史 transition 需依靠现有 checkpoint/eval/evidence 和 source semantics 重建。

## 11. Next Expected Automatic Event

- Z05：继续 `stage1` training，达到 `900000` 后自动写 ordinary checkpoint、full-resume、formal evaluation、runtime evidence/status，再继续下一个 100k segment。
- Z07：先完成 800k runtime evidence/status transition；随后自动 resume 到 `900000`，再按同一 100k 流程推进。
- 两条线各自 Stage1 达到 2M 后自动 balanced selection；没有 early-stop 或 performance promotion gate，然后进入本线 Stage2。

## 12. Storage Note

当前：root `/dev/nvme0n1p2` 约 `37G` free、`96%` used、inode `6%` used；`/data/disk2` 约 `4.5T` free、inode `3%` used。live runtime 约 Z05 `25G` + Z07 `26G`。下一次计划开启新的长期训练前，必须重新检查存储余量，并依据当前 replay/full-resume/checkpoint 增长率估算所需空间。本轮不清理磁盘。

## 13. Commands

```bash
# live status
jq . /home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05/heartbeat.json
jq . /home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z07/heartbeat.json

# heartbeat / training log tail
tail -f /home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05/heartbeat.json
tail -f /home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z07/heartbeat.json

# attach arms / coordinator
tmux attach -t iqn_z05_unified_decay_20260919
tmux attach -t iqn_z07_unified_decay_20260919
tmux attach -t iqn_z_unified_decay_dual_20260919
```
