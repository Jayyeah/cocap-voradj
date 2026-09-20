# Z05 / Z07 Unified-Decay IQN 当前训练状态

当前 live runtime 快照；formal 性能只读取 evaluations/step_*/report.json，训练 telemetry 不替代 formal 结果。

## Provenance

- captured_at: 2026-09-20T23:03:36+08:00 (Asia/Shanghai)
- repo: Jayyeah/cocap-voradj
- branch: experiment/iqn-z-unified-decay-dual-curriculum-20260919
- local HEAD / remote HEAD: 83a67bc3294930604a9a1047aa93bb822aaf09d9
- local/remote consistent: true
- worktree clean before update: true
- runtime: /home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime
- latest-only activation 已完成；历史 activation/certification 文档保留
- 两条 launch HEAD 均为 58baa8143f143d013e32939d3bbb17f9c60bcec4；与当前 HEAD 的差异仅为 ops 文档，未发现科学代码/config drift

## Scientific contract

| 项目 | Z05 | Z07 |
|---|---:|---:|
| alpha / lambda / eta | 0.5 / 0.5 / 0.5 | 0.7 / 0.7 / 0.7 |
| hard-zero threshold | 0.10 | 0.10 |

共同 effective config：NormSense-V2=forward-final-density-normalized-sensing-v2、friend_ordering=physical_only、include_z_state=true、include_is_pursuing=false、pursuing_embed_dim=0、pursuing_late_fusion=false、每100k checkpoint/formal evaluation、每场景20 episodes。两 arm 非 alpha 差异数为0；每个 environment decision 一次同步 Z update。

## Live health and training

| arm | stage / phase | current → target | PID / GPU / CVD | heartbeat | health |
|---|---|---:|---|---|---|
| Z05 | stage1 / training | 1,404,000 → 1,500,000 | 3948961 / GPU0 / 0 | 23:03:23 fresh | HEALTHY |
| Z07 | stage1 / training | 1,400,000 → 1,400,000 | 3948953 / GPU1 / 1 | 23:03:36 fresh | HEALTHY |

两个 PID、PPID、CPU activity、tmux 均存在，GPU mapping 一致。Z05 正在1.4M后续训练段；Z07正在完成1.4M transition。

| arm | optimizer / target updates | epsilon / LR | loss / EMA | replay total | recovery pool |
|---|---:|---:|---:|---:|---:|
| Z05 | 349,705 / 140 | 0.05 / 3e-5 | 36.8313 / 36.8707 | 2,809,072 | 1,000 |
| Z07 | 348,490 / 140 | 0.05 / 3e-5 | 34.5905 / 33.1417 | 2,984,624 | 1,000 |

Replay 分项：Z05=[pursuing 703095, pre_capture_cover 105977, post_capture_real 1000000, recovery_pure 1000000]；Z07=[864546,120078,1000000,1000000]。

当前 action histogram：Z05=[817987,532255,829012,492820,375875,447727,819302,645804,655218]；Z07=[654057,516784,713621,554915,403222,617557,717008,621424,801412]。

telemetry（非 formal）：Z05 strict coverage=0.99、capture=0、collision=0、area CV=0.0671；Z07 strict coverage=0.82、capture=1.00、collision=0.02、area CV=0.0843。

## Formal milestone closure

相对上一版 status（Z05=1.2M、Z07=1.1M）新增：Z05 1.3M、1.4M；Z07 1.2M、1.3M。当前最新完整 formal：Z05=1.4M，Z07=1.3M。

### Z05 / Stage1 / 1.4M

- Pure coverage：strict CE=0.25；CE RMS=0.0651；area CV=0.1494；collision=0；time-to-CE=235.4s。
- Pure capture：normal=1.00；stationary=0；ring2=1.00；ring3=1.00；collision=0；capture time=32.875s。
- Mixed：capture=1.00；collision=0；post-capture CE=0.15；safe-complete=0.15；capture=31.05s；recovery=143s；mission=169.333s。
- Z：direct-visible Z=1=1.00；support mean=0.4809；coverage mean=0.000389；pure exact zero=true；neighbor-dominant count=1901、mean=0.4892；max hop=2；release=1.5s；never-release=0；update violations=0。

vs 1.3M：strict CE 下降0.75；CE RMS/area CV 上升0.0236/0.0760；pure/mixed capture 持平；capture time 下降2.675s；mixed post-CE/safe 下降0.75；collision 持平。

### Z07 / Stage1 / 1.3M

- Pure coverage：strict CE=0.15；CE RMS=0.0599；area CV=0.1427；collision=0；time-to-CE=52.5s。
- Pure capture：normal=1.00；stationary=0；ring2=1.00；ring3=1.00；collision=0；capture time=41.25s。
- Mixed：capture=1.00；collision=0；post-capture CE=0.20；safe-complete=0.20；capture=35.575s；recovery=59.5s；mission=86.375s。
- Z：direct-visible Z=1=1.00；support mean=0.6737；coverage mean=0.000835；pure exact zero=true；neighbor-dominant count=1897、mean=0.6914；max hop=2；release=3s；never-release=0；update violations=0。

vs 1.2M：strict CE 上升0.10；CE RMS/area CV 下降0.0076/0.0394；pure/mixed capture 持平；capture time 上升1.575s；mixed post-CE/safe 上升0.20；mixed collision 下降0.05。

## Latest three-milestone trend

- Z05 (1.2M→1.3M→1.4M)：coverage strict CE 0.95→1.00→0.25；mixed safe 0.80→0.90→0.15；pure capture 1.00→1.00→1.00。
- Z07 (1.1M→1.2M→1.3M)：coverage strict CE 0.95→0.05→0.15；mixed safe 0.85→0.00→0.20；pure capture 1.00→1.00→1.00。

单个20-episode formal milestone 波动明显；不据此宣判最终 winner 或 early-stop。

## Automation

两条 arm 仍在 Stage1，budget=2,000,000，尚未到 Stage1→2 gate。Stage1 fresh_stage_gate 均 PASS；当前无 selection_report、Stage2/Stage3 selected checkpoint 或 final arm report，这是阶段未完成的正常状态。runner 的 milestones、balanced selection、same-line warm-start、promotion 后 fresh optimizer/replay/RNG/env/Z/recovery 语义已核对。

AUTO_CURRICULUM_READY = true

## Latest-only and storage

| arm | resume step | bytes | SHA256前12位 | links | resume_step count |
|---|---:|---:|---|---:|---:|
| Z05 | 1,400,000 | 6,472,289,363 | 5afc4400c371 | 1 | 0 |
| Z07 | 1,400,000 | 6,771,837,075 | 2da917425c44 | 1 | 0 |

全 runtime 树 resume_latest.pt=2，temporary full-resume=0。Z07 status.json 仍落后到1.3M，但真实磁盘 retention 只有 resume_latest.pt，无 resume_step 文件；不标记 LATEST_ONLY_RETENTION_REGRESSION。

root /dev/nvme0n1p2：可用约87G、已用91%、inode使用6%；Z05 runtime=6.7G，Z07=6.9G。分类 STORAGE_HEALTHY；未删除/清理文件，未发现 latest-only storage regression。

## Next automatic event

Z05继续至1.5M；Z07完成1.4M transition后进入下一100k段；两线到Stage1 2M后才执行balanced selection和Stage2 fresh promotion。本轮未启动 dual coordinator、未新跑 evaluation、未改训练合同、未 kill/restart。
