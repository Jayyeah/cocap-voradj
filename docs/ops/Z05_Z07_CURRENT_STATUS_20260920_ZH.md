# Z05 / Z07 Unified-Decay IQN 当前训练状态

本文件是当前 live runtime 状态快照；旧 status snapshot 只作历史基线。formal 性能只读取 evaluations/step_*/report.json，不把训练窗口 telemetry 当作正式结论。

## 1. Provenance

- captured_at: 2026-09-20T20:07:13+08:00 (Asia/Shanghai)
- repo: Jayyeah/cocap-voradj
- branch: experiment/iqn-z-unified-decay-dual-curriculum-20260919
- local HEAD: c3fda33094704d1d90f7fdf50d06fc4989d5d92c
- remote HEAD: c3fda33094704d1d90f7fdf50d06fc4989d5d92c
- local/remote consistent: true
- worktree clean before this status update: true
- runtime: /home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime
- latest-only activation: 已完成；activation commit=c3fda330...，未修改 activation/certification 历史文档

两条 live runner 的 launch.json HEAD 都是 58baa814...；当前 HEAD 与其差异仅为 latest-only ops activation 文档，runner/config 科学文件未发生差异。该 launch provenance lag 已记录，但不构成科学合同 drift。

## 2. Scientific contract

| 项目 | Z05 | Z07 |
|---|---:|---:|
| alpha / lambda / eta | 0.5 / 0.5 / 0.5 | 0.7 / 0.7 / 0.7 |
| hard-zero threshold | 0.10 | 0.10 |

共同合同 live effective config 已核对：NormSense-V2=forward-final-density-normalized-sensing-v2、friend_ordering=physical_only、include_z_state=true、include_is_pursuing=false、pursuing_embed_dim=0、pursuing_late_fusion=false、每 100k checkpoint/formal eval、每场景 20 episodes。Z05/Z07 之间允许的差异只有 lambda/eta，非 alpha 差异数为 0。每个 environment decision 一次同步 Z update；正式 Z diagnostics 的 update-count violations 均为 0。

## 3. Live health and training state

| arm | stage / phase | current → target | PID / GPU / CVD | heartbeat | health |
|---|---|---:|---|---|---|
| Z05 | stage1 / training | 1,251,000 → 1,300,000 | 3948961 / GPU0 / 0 | 20:07:13, fresh | HEALTHY |
| Z07 | stage1 / formal_evaluation | 1,200,000 → 1,200,000 | 3948953 / GPU1 / 1 | 20:06:14, about 1 min | HEALTHY |

两个 PID、PPID、CPU activity、tmux 均存在；tmux 分别为 iqn_z05_unified_decay_20260919 与 iqn_z07_unified_decay_20260919。GPU mapping 与 CUDA_VISIBLE_DEVICES 一致。Z07 的 1.2M 是当前 formal transition，尚未把它计为完整 formal milestone。

| arm | optimizer updates | target updates | epsilon | learning rate | latest loss / EMA | replay (pursuing / pre-cover / post-real / recovery) | recovery pool |
|---|---:|---:|---:|---:|---:|---|---:|
| Z05 | 311,455 | 125 | 0.05 | 3e-5 | 32.6298 / 32.8970 | 658,399 / 96,617 / 1,000,000 / 1,000,000 (total 2,755,016) | 1,000 |
| Z07 | 298,490 | 120 | 0.05 | 3e-5 | 31.0918 / 29.6966 | 783,840 / 102,892 / 1,000,000 / 1,000,000 (total 2,886,732) | 1,000 |

当前训练 action histogram：

- Z05: [757688,494804,773695,443445,347877,390379,698141,543567,554404]
- Z07: [580998,454666,648823,500298,351528,517724,586749,503637,655577]

训练窗口 telemetry（非 formal）：Z05 latest strict coverage=0.99、capture=0、collision=0.01、area CV=0.0662；Z07 latest strict coverage=1.00、capture=0、collision=0、area CV=0.0752。两者都仍在学习/推进，不能替代下述正式结果。

## 4. Formal milestone closure

上一版 authoritative status 之后新增并已闭环的 formal reports：

- Z05: 900k, 1.0M, 1.1M, 1.2M；最新完整 formal=1.2M。
- Z07: 800k, 900k, 1.0M, 1.1M；最新完整 formal=1.1M。1.2M 当前仍在 formal evaluation。

### Z05 / Stage1 / 1.2M

| 场景 | 指标 |
|---|---|
| pure coverage | strict CE 0.95; CE RMS 0.0381; area CV 0.0521; collision 0; time-to-CE 176.895s (n=19) |
| pure capture | normal 1.00; stationary 0; ring2 1.00; ring3 1.00; collision 0; capture time 33.425s |
| mixed | capture 1.00; collision 0; post-capture CE 0.80; safe-complete 0.80; capture 33.25s; recovery 118.156s; mission 151.188s |
| Z diagnostics | direct-visible Z=1 1.00; support Z mean 0.4805; coverage Z mean 0.001440; pure coverage exact zero true; neighbor-dominant 1872, mean 0.4884; max hop 2; release 1.5s; never-release 0; update violations 0 |

vs same-arm previous complete formal 1.1M：strict CE 上升 +0.95；CE RMS 下降 -0.0607；area CV 下降 -0.1445；pure capture 持平；capture time 下降 -4.275s；mixed capture 持平；mixed post-capture CE 与 safe-complete 上升 +0.80；mixed collision 持平。

### Z07 / Stage1 / 1.1M

| 场景 | 指标 |
|---|---|
| pure coverage | strict CE 0.95; CE RMS 0.0310; area CV 0.0665; collision 0; time-to-CE 59.211s (n=19) |
| pure capture | normal 1.00; stationary 0; ring2 1.00; ring3 1.00; collision 0; capture time 37.625s |
| mixed | capture 1.00; collision 0.15; post-capture CE 0.85; safe-complete 0.85; capture 33.10s; recovery 65.441s; mission 97.382s |
| Z diagnostics | direct-visible Z=1 1.00; support Z mean 0.6746; coverage Z mean 0.006223; pure coverage exact zero true; neighbor-dominant 1949, mean 0.6926; max hop 2; release 3s; never-release 0; update violations 0 |

vs same-arm previous complete formal 1.0M：strict CE 上升 +0.45；CE RMS 下降 -0.0269；area CV 下降 -0.0720；pure capture 持平；capture time 下降 -7.800s；mixed capture 持平；mixed post-capture CE 与 safe-complete 上升 +0.40；mixed collision 上升 +0.10。

## 5. Latest three-milestone descriptive trend

- Z05 (1.0M → 1.1M → 1.2M): pure coverage strict CE 0.00 → 0.00 → 0.95；mixed safe-complete 0.00 → 0.00 → 0.80；pure capture 1.00 → 1.00 → 1.00。
- Z07 (0.9M → 1.0M → 1.1M): pure coverage strict CE 0.00 → 0.50 → 0.95；mixed safe-complete 0.00 → 0.45 → 0.85；pure capture 0.95 → 1.00 → 1.00。

当前 milestone 的 descriptive observation（不是最终 winner 判定）：两条线 pure capture 均为 1.00，Z05/Z07 pure coverage strict CE 均为 0.95；Z07 当前 mixed post-capture CE/safe-complete 略高，但 mixed collision 也为 0.15，高于 Z05 的 0。课程尚未结束，不 early-stop、不宣布最终优劣。

## 6. Stage promotion and automation

- 当前两条 arm 都仍在 Stage1，Stage1 budget=2,000,000；尚未到 Stage1→2 gate。
- 两条 Stage1 fresh_stage_gate.json 均 PASS：global step/replay/optimizer/recovery/Z 为 fresh，model/target equal，warm-start=null。
- 当前尚无 selection_report.json、Stage2/Stage3 selected checkpoint 或 final arm report；这是因为 Stage1 尚未完成，不是 automation failure。
- 当前 runner 源码已核对 milestones、balanced selection、same-line warm-start、promotion 后 fresh optimizer/replay/RNG/env/Z/recovery 语义。

AUTO_CURRICULUM_READY = true

## 7. Latest-only full-resume and storage

| arm | resume step | bytes | SHA256 前12位 | inode links | resume_step_*.pt |
|---|---:|---:|---|---:|---:|
| Z05 | 1,200,000 (rolling; 当前已训练至 1,251,000) | 6,302,043,859 | 611594166946 | 1 | 0 |
| Z07 | 1,200,000 (rolling; 当前正在该 milestone formal) | 6,643,473,875 | 6af2a557c9b8 | 1 | 0 |

全 runtime 树 resume_latest.pt 共 2 个；temporary full-resume 文件=0。Z07 覆盖式 status.json 仍指向历史 resume_step_001100000.pt，但磁盘真实 retention 为 latest-only，未发现该文件；因此本轮不标记 LATEST_ONLY_RETENTION_REGRESSION，而将其记录为 status-file lag。

磁盘：root /dev/nvme0n1p2 总 915G、已用 90%、可用约 88G；inode 使用 6%；Z05 runtime 6.4G，Z07 runtime 6.7G。分类：STORAGE_HEALTHY。当前未删除、未清理任何文件；未发现 +5G/+6G per milestone 的 latest-only retention regression。

## 8. Next automatic event

- Z05：继续 Stage1 training，达到 1.3M 后自动 checkpoint/full-resume/formal evaluation/runtime evidence，再继续下一个 100k milestone。
- Z07：完成当前 1.2M formal evaluation/runtime evidence/status transition 后，自动进入 1.3M training。
- 两条线达到 Stage1 2M 后才执行各自 balanced selection 和 Stage2 fresh promotion。

本轮未启动 dual coordinator，未新跑 evaluation，未改变训练/环境/reward/network/replay/curriculum/checkpoint policy，未 kill/restart 训练。
