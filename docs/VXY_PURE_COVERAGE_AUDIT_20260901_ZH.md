# IQN-VXY Pure-Coverage isolation 审计（2026-09-01）

## 1. 目的

Stage2 8v2 selected VXY `step_600000.pt` 的 formal20 为 capture `.90`、standalone coverage CE `.15`、mixed capture `.95`、mixed CE `.05`。因此本线不再用capture指标做gate，而是隔离回答：

1. 已学会capture的Stage2 VXY，在纯coverage训练中能否快速恢复/强化CE；
2. 同一VXY9是否能从scratch独立学出稳定coverage；
3. 由 warm-start 与 scratch曲线区分 skill difficulty、phase/replay interference和mixed recovery exposure不足。

## 2. 历史合同与预算选择

仓库没有唯一命名为“Final Pure-Coverage”的独立stage；Final成功合同是 integrated mixed curriculum。故本线以 Final VXY Stage2 的 reward、CE判定、horizon、observation、IQN网络、optimizer、epsilon、target update、batch/replay容量和更新节奏为主合同，只把训练任务分布切成pure coverage。

历史 `ce13_pure8_scratch_ce8schedule_500k.yaml` 是明确的8-agent pure-CE正例，预算500k且最终CE成功；因此 warm/scratch都用500k，禁止缩成短smoke。纯任务不再有capture/coverage class mix，replay自然变成仅含pure-coverage transition的单一uniform buffer；容量、batch、warmup和update cadence保持Final Stage2。

## 3. Paired配置

共同合同：8 pursuers、0 evaders、2 obstacles、inner-random-cluster半径11/最小间距7、120×120地图、episode horizon 3000、legacy-end-step collision、centroid-energy-v0、CE RMS≤`.05`且max≤`.10`连续30步、settling disabled、body-frame rate-limited VXY9、IQN以及Stage2优化/探索参数。

```text
warm_start:
  pretrained = Stage2 selected step_600000.pt
  sha256 = 1388ac6813fa6c6ee9f0cc6041446e04556a01ef611a07d0afe3071340a86940
  optimizer/replay/env/RNG = fresh

scratch:
  pretrained = null

其余训练语义与seed 2026090101相同
```

配置位于 `configs/experiments/iqn_vxy_pure_coverage_20260901/`。每25k保存轻量checkpoint并做coverage-only deterministic20；完整rolling resume写 `/dev/shm`，避免根盘保存约9 GiB replay包。

## 4. 评估与指标补全

筛选不使用capture指标。每个checkpoint记录：CE success、final/best Voronoi CV、centroid RMS/max、active speed mean/RMS/max、collision/boundary、episode length、CE success step、per-agent/team return以及reward component decomposition。

selection按以下词典序：CE success最大；collision、boundary、centroid RMS/max、CV、final velocity RMS最小；最后才偏好较晚step。选中点做coverage formal20与前10个paired GIF，并要求无failures、20 records、10 GIF和 `FORMAL_DONE`。

## 5. GPU0只读队列安全合同

`tools/supervise_vxy_pure_coverage_20260901.py` 在Stage3期间只读观察，不持有、kill、restart或修改Stage3任何进程/config。只有同时满足以下三项才启动warm-start：

1. Stage3 selection存在且其 `formal_output_root/FORMAL_DONE` 已落盘；
2. `stage3_12p3e3obs` 相关train/screen/finalizer PID全部退出；
3. GPU0 compute-app列表为空。

这避免旧Stage3 supervisor在selection先写、formal仍运行时产生竞态。warm训练完成而25k screening尚未追平时，也只继续watcher，不会误重启已完成trainer。顺序固定为：

```text
现有 Stage3 natural completion + formal/GIF
  -> pure coverage warm-start 500k + formal
  -> pure coverage scratch 500k + formal
```

## 6. 验证与存储

- strict config diff、warm/scratch paired差异、8P0E2obs/VXY9/CE合同与selected SHA：PASS；
- warm CUDA 1→2 full-resume：PASS；
- scratch CUDA 1→2 full-resume：PASS；
- watcher coverage-only场景参数、Python/shell syntax、focused regression：PASS。

为保证当前Stage3与两条新长训不触发磁盘guard，已先核验Stage1/2 terminal milestone、selected checkpoint、formal marker和无人持有文件，然后删除仅用于中断恢复的两个已完成rolling resume：8,562,739,695与9,224,988,015 bytes。所有轻量milestone、selected/final、评估和GIF保留；Stage3活跃 `/dev/shm` resume未动。清理后根盘约47 GiB可用。

实际Stage3和队列runtime状态在本文件后续提交中追加。

## 7. 队列启动状态（2026-09-01 21:58 CST）

```text
GPU0 STAGE3: ACTIVE, step 305k / 700k
TRAIN PID/TMUX: 109572 / cocap_vxy_full_s3_12p3e3obs_train
SCREEN PID/TMUX: 109587 / cocap_vxy_full_s3_12p3e3obs_screen
FINALIZER PID/TMUX: 109602 / cocap_vxy_full_s3_12p3e3obs_finalize
CHECKPOINT: 12个25k milestone，latest step_300000.pt
SCREEN: DONE through 275k；300k screening active
FORMAL/GIF: 尚无selection与FORMAL_DONE；finalizer仍等待
FINITE: true；305k loss 33.788、EMA 34.151、recent capture .95、collision .06
```

Pure-Coverage supervisor已启动于 `cocap_vxy_pure_coverage_queue_gpu0`，PID 264842，状态严格为 `observing_stage3_read_only`。没有warm/scratch train或screen child，没有修改/重启/kill现有Stage3。GPU0此刻compute app仍仅为Stage3 train和300k screening workers。

队列顺序已持久化为warm-start→scratch；除Stage3三重放行外，每个新entry/restart前还会再次确认GPU0没有其他compute app。启动commit为 `4f7216fb8ec786a58be2e922a7ccae6dcd720267`。
