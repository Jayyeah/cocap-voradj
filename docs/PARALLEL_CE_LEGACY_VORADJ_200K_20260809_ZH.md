# Pure-CE 与 Legacy-VorAdj 并行 200k 线（2026-08-09）

## 当前执行状态

- 两份正式配置、扩展评估指标、滚动 replay 保存和安全启动器已经完成；32-step 双线端到端预检 finite，核心新合同和相关整组回归通过。
- 昨夜服务器停机导致等待队列与旧训练 tmux 全部丢失。按用户 08-10 最新指令，不再等待旧线结束：17:11 已立即启动两条新线。
- Pure-CE：PID 224391、tmux `parallel_pure_ce_200k`、cuda:0；Legacy-VorAdj：PID 224386、tmux `parallel_legacy_voradj_200k`、cuda:1。两线 step-1 checkpoint 与 rolling resume 均已落盘。
- 同机还从 A150/C125 恢复旧 A/C，现为每张 GPU 两条本项目训练加既有外部负载。17:17 GPU 温度约 80°C/92°C，当前吞吐尚未形成稳定窗口；旧 ETA 已失效，首个 1k/25k 后重算。

### 2026-08-10 20:34 快照

- Pure-CE 已到 25k 并落盘：strict 0/4、CV<0.20 0/4、collision 2/4，平均 CE energy progress +0.0108；尚无成功信号，但部分 episode 的 CE energy 有下降。
- Legacy-VorAdj Baseline A 已到 35k，25k checkpoint 已落盘：pure-CE strict 1/4、CV<0.20 2/4、collision 1/4、平均 CE progress +0.0608；capture 与 mixed 均 0/4 capture、detected 4/4、collision 0/4，平均 distance progress 分别 -5.82/-3.99。
- 两线所有最新训练窗口 `mean_finite=1`。25k 只有 4 episodes，继续按原合同训练至 200k，不据此提前停止。
- 对照 Baseline B 尚未启动；自动 supervisor 已武装，等待 Stage4A/C 任一 clean 200k 释放 GPU 后执行 CUDA/2k Gate 并自动启动。

### 2026-08-10 22:13 快照

- Legacy-VorAdj Baseline A 已到 52k，50k checkpoint/诊断完整落盘；Pure-CE 已到 36k，最新 checkpoint 仍为 25k。两线最新训练窗口均 finite。
- Baseline A 50k pure-CE 继续给出正向 coverage 信号：strict 2/4、CV<0.20 3/4、平均 CE progress +0.0660。
- Baseline A 50k capture/mixed 尚未围捕成功，但平均 distance progress 已转正为 +14.79/+15.85、最小敌距降到 10.84/8.75；代价是 collision 达 4/4 与 3/4。这是“接敌增强但碰撞坍缩”的混合信号，不能视为 capture 成功。
- Baseline B 尚未启动；Stage4A/C 仅到 181k/169k，supervisor 继续等待首条 clean 200k。

## 线一：Pure CE

- 配置：`configs/experiments/parallel_ce_legacy_voradj_20260809/pure_ce_4p0e1obs_200k_aw.yaml`
- 4 pursuers、0 evader、1 obstacle，连续 body-frame `(a,w)`，scratch 200k。
- 训练只轮转 `pure_ce`；coverage 使用 CE centroid-energy + PBRS。
- 速度项仅 `0.0001`，不进入 CE-strict gate；主成功为 RMS<=0.05、max<=0.10 连续保持 30 步。
- 同时独立记录 CV<0.15、CV<0.20、配置 loose gate、area CV、CE center RMS/max 和 CE progress。

## 线二：Legacy VorAdj capture + CE coverage

- 配置：`configs/experiments/parallel_ce_legacy_voradj_20260809/legacy_voradj_oldmix_4p1e1obs_200k_aw.yaml`
- 4 pursuers、1 evader、1 obstacle，连续 body-frame `(a,w)`，scratch 200k。
- capture 恢复 A3 legacy reward：approach=1、old mean-shift=2、front=0.5；`k_required=3`、stationary minimum pursuers=2；evader 保持 APF-v2_fixed 与旧 5 档转向集合。
- capture 图使用 legacy VorAdj：全部活动 pursuer/evader 共同参与 Voronoi 分割，旧 capture obstacle assign；Voronoi 邻接直接决定可见集合，不使用 VCT-LS 的固定 20m 截断。VCT-LS 和 support blend 关闭。
- K10 只平滑敌人仍存活时的短暂邻接丢失；所有敌人被捕获后立即清空 release counter、直接切 coverage。coverage/post-capture 使用 CE 和 free-mask coverage 图。
- episode 比例 `mixed_crms:pure_ce=1:1`；focal replay 比例为 pursuing/pre-cover/post-real/recovery-pure=`64:16:32:16`。
- 每 25k 同时评估 `capture`、`pure_ce`、`mixed_crms`，除 capture/碰撞/CE 成功外，记录 detected rate、首次发现步、初末/最小敌距、distance progress、CE progress、动作与速度。

## 保存与清理合同

- 每 25k 的 `checkpoints/step_*` 保留 trainer、配置、manifest、metrics 和该节点自动评估，作为可选最优模型；这些 milestone 不重复保存 replay。
- 每 25k 自动诊断为每场景 4 次、最多 400 步；200k 结束自动执行每场景 20 次完整 horizon 正式评估，并写入 final report 的 `screening`。
- `resume_latest/` 以原子替换方式只保留最新的完整 trainer+replay+runtime，保证随时续训。
- 200k final 保存完整 bundle；`resume_latest` 与 final 用硬链接共享大文件，不重复占盘。
- 训练完成后不自动清理任何 milestone。先完成自动评估，再由用户确认哪些 checkpoint 无用；确认前不得删除。

## 验证证据

- 新配置/评估/保存核心合同与 CTDE、阶梯、legacy/VCT-LS 拓扑整组回归均通过；全敌人捕获后直接切 coverage 有独立合同测试。
- CPU 双线 32-step preflight 位于 `/tmp/cocap_parallel_preflight_20260809/`：两线 `all_finite=true`，配置场景轮转、scene hash、checkpoint/replay 均正确。
- 启动器：`tools/launch_parallel_ce_legacy_voradj_when_safe.sh`；队列/资源 gate 日志：`artifacts/2026-08-10_parallel_ce_legacy_voradj_200k/launcher.log`。
