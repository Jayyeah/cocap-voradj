# Baseline B — Standard All-Agent MASAC on Legacy-VorAdj Old-Mix

更新日期：2026-08-11

## 2026-08-11 17:14 接手事实快照（停训前）

- Git：worktree `/home/yjq/rl/CoCap1/cocap-voradj-allagent-oldmix`，branch `ablation/all-agent-oldmix-20260810`，HEAD `1353193`；台账已有 supervisor 自动追加，artifact 目录未跟踪。
- Baseline B：PID `666602`，tmux `allagent_oldmix_ablation_200k`，`cuda:0`；supervisor PID `542373` / tmux `allagent_oldmix_ablation_supervisor`。同机另有其他用户 PID `589205` 占用 `cuda:1`，不得触碰。
- 停训前最后一条完整 metrics 窗口：env step `104000`、update count `24751`、replay size `104000`、`mean_finite=1`、critic loss `43.3365`、actor loss `30.1598`、alpha `0.04010`、update wall-time `1.0855 s`、窗口吞吐 `3.3235 step/s`。
- 最新完整可恢复点不是 104k，而是 `resume_latest @ 100000`：trainer `16:52:09`、replay `16:52:33`、runtime/manifest/metrics `16:52:35` 写完；对应 milestone `step_000100000`。runner 没有 signal-triggered checkpoint，因此 100k 后未落盘约 4k 只作 pre-stop 事实记录，不冒充可恢复进度。
- 资源：`cuda:0` 中 B 占 `23156 MiB`，整卡约 `23181/49140 MiB`；RAM `23/125 GiB` used、swap `1.2/8 GiB`；根盘 `831/915 GiB`、仅余 `38 GiB`。最新指标记录的 PyTorch peak allocated 为 `22461.934 MiB`。
- 其他正式线：进程表中未发现 Baseline A 或 Pure-CE 训练；Stage4A 命令残留是 supervisor 的父 tmux shell，不是 Stage4A Python 训练。只停止上述 Baseline B，不处理 PID `589205` 或其他项目。

## 2026-08-11 10:08 A/B 25k/50k早期对照

- Baseline B当前54k、12,251次更新、all finite；最近窗口约1.860 env step/s，50k里程碑约1.957 env step/s（约7.0k/h）、1.829 s/update、289 active-agent loss terms/s。
- B25k：capture 0/4、collision 4/4、distance progress +4.17；pure-CE strict 0/4、CV<0.20 0/4；mixed capture 0/4、collision 4/4、CV<0.20 2/4。
- B50k：capture 0/4、collision 4/4、distance progress +9.83；pure-CE strict 0/4、CV<0.20 3/4、collision 0/4、CE progress +0.0642；mixed capture 0/4、collision 3/4。
- 同step Focal A：25k pure-CE strict 1/4、CV<0.20 2/4；50k strict 2/4、CV<0.20 3/4。50k capture同样0/4、collision 4/4，但distance progress +14.79。
- 早期判断：B的coverage从25k到50k有积极改善，但strict sample efficiency暂落后A；capture两者都失败，B并未解决碰撞。样本仅4 episodes且A后续明显非单调，继续完整200k，不作提前裁决。
- ETA：B75k约今日13:15--13:45；B200k训练约08-12 07:00--09:00，final report约09:00--12:00。Baseline A约今日11:40--12:00完成训练、12:30--14:00完成report。

## 2026-08-11 04:00 A/B实时状态

- Baseline B已于01:35:28在cuda:0正式启动；CUDA32与2k Gate通过，2k共469次更新、all_finite=true、约1.167 env step/s、3.145 s/update、512 agent loss terms/update、峰值VRAM约22,462 MiB。
- 当前Baseline B为14k、2,251次更新、all finite；最近窗口约1.109 env step/s（约4.0k/h），uniform joint batch为63条pre-capture与65条pure-CE transition，role metadata未参与采样。
- B当前没有25k行为评估，积极信号仅限数值稳定、自然mixed:pure近1:1、512个有效agent loss terms/update以及当前窗口collision=0；不能提前宣称all-agent性能优于focal。
- Baseline A control当前107k；其100k pure-CE为strict 1/4、CV<0.20 4/4，capture仍0/4且collision 3/4。A已表现出coverage和接敌信号，但还没有可靠capture。
- ETA：B首个25k约07:00--07:30；B 200k训练约08-13 03:00--07:00，final report约08-13 05:00--09:00。A 200k训练约今日13:00--14:00、report约14:00--15:30。

## 1. 研究问题与唯一实验变量

本实验只回答：在完整 Legacy-VorAdj capture→coverage 与 pure-coverage 混合 MASAC 中，旧 IQN 训练路径遗留的 focal role-balanced replay/update，是否比标准 joint-transition all-agent MASAC 提供实际、可复现的训练收益？

- Baseline A：当前正在运行的 Legacy-VorAdj old-mix Focal MASAC。
- Baseline B：相同 Legacy-VorAdj old-mix，使用 uniform joint replay，并对每条 sampled joint transition 中的全部 active agents 做 critic、actor 与 alpha update。

唯一算法差异：

```text
Baseline A: focal role-balanced item sampling + one focal agent loss/item
Baseline B: uniform joint-transition sampling + all active-agent losses/transition
```

本实验不测试、也不得修改：

- capture/coverage reward；
- tau、alpha/entropy、learning rate、batch size、update interval、grad clip；
- mixed:pure episode 比例；
- observation、Legacy-VorAdj topology、APF；
- 动作空间、Actor、central twin critics 或网络结构；
- teacher、snapshot、encoder initialization 或 focal checkpoint warm-start。

## 2. A/B 冻结合同

共同合同：4 pursuers、1 evader、1 obstacle，连续 body-frame `(a,w)`；200k env transitions；seed `2026080902`；每 25k 保存模型并对 capture、pure_ce、mixed_crms 做同合同诊断；`mixed_crms:pure_ce=1:1` episode cycle；其余环境、reward、observation、VorAdj、APF、MASAC 超参数全部从 Baseline A effective config 继承。

Baseline A 保持原 run、配置、manifest、checkpoint、sampler 与 artifacts 不变；禁止用本分支代码恢复或重启 Baseline A。

Baseline B 从 scratch 开始，manifest 必须包含：

```text
optimizer_unit: joint_transition_all_active_agents
replay_sampling: uniform_joint
focal_training: false
scene_cycle: mixed_crms,pure_ce
```

Baseline B 的 role/phase metadata 只用于诊断统计，不进入 sampler，也不决定任何 transition 或 agent 是否进入 loss。uniform replay 按 stored joint transitions 的自然分布采样；episode 仍为 1:1，但 batch 不强制 mixed:pure=1:1。

## 3. 优化器与梯度合同

- critic target 对每个 active agent 独立计算，critic loss 为 active mask 上的 mean；inactive padding 不进入 loss。
- actor 对每个 active agent `i` 分别获得 decentralized gradient：`a_i` 可微，其他 agent action detach；最后在全部 active agent loss terms 上取 masked mean。
- alpha loss 同样只在 active agents 上取 mean。
- active-only actor-Q 优化只能跳过全 batch 均 inactive 的 padded slot，必须与旧 all-agent reference 的 Q、loss 和 actor gradients 数值等价。

## 4. Sampling 与效率观测

Baseline B 每次 update 采样 128 条标准均匀 joint transitions；replay 至少有 128 条时不放回采样。每个 batch 记录但不据此重采样：

- sampled pre-capture、post-capture、pure-CE transition 数；
- scene/phase 自然分布；
- pursuing、coverage、support active-agent 数；
- unique joint transitions 与 replacement 数；
- active agent loss terms/update、loss terms/s、updates/s、env steps/s；
- update wall time、critic time、actor-Q time 与峰值 VRAM。

## 5. Checkpoint / Replay 存储合同

Baseline B 显式使用 `periodic_checkpoint_replay_mode: rolling_latest`：

- 历史 `checkpoints/step_xxx/` 为 evaluation model-only，不包含 `replay.pkl`；
- `resume_latest/` 原子维护唯一一份最新完整 trainer+replay+runtime bundle；
- 200k final bundle 完整可恢复，并通过 hardlink/link-or-copy 暴露最终文件，避免重复物理存储；
- 不得删除当前最新、已验证可恢复的 replay。

## 6. 启动 Gate

正式 200k 启动前必须完成：配置 effective diff；focal regression；uniform joint sampler；all-active-agent/active-mask/gradient equivalence；manifest A/B 隔离；rolling-latest regression；32-step CPU/CUDA smoke；约 2k 的吞吐、显存和共存影响检查。

资源 Gate：Baseline A、Pure-CE、Stage4A/C 不得被 kill、重启或明显降速。Baseline B 固定放在 Pure-CE 所在的 `cuda:0`；只等待同卡旧 Stage4A 自然结束，不使用 Stage4C 释放的 `cuda:1`，因此不与 Baseline A 抢资源。

2026-08-10 19:58 启动前证据：

- Baseline A/B effective config diff 仅包含 run metadata 与 optimizer/replay/focal mode 三项预期变量；
- all-agent loss/gradient、active mask、uniform joint sampler、focal regression、manifest 隔离、rolling-latest 与旧 final 安全迁移共 34 项测试通过；
- Python 编译与 `git diff --check` 通过；CPU 32-step smoke 已通过；
- CUDA 32-step 与约 2k 吞吐/显存检查不在两张满载 GPU 上抢跑，由 supervisor 在 Stage4A/C 首条自然完成、资源 Gate 通过后执行；
- 旧 final 只有在 trainer、replay、runtime 三者 200k 步数一致且能够完整 load 时，才会原子建立 `resume_latest`；随后仅清理由该完整 bundle 覆盖的旧 milestone replay。

## 7. 正式运行台账

状态：实现与本地回归完成；尚未启动正式 Baseline B。自动 supervisor 已于 2026-08-10 19:58 武装，当前等待 Stage4A/C 首条 clean 200k。

代码：本地 branch `ablation/all-agent-oldmix-20260810`，commit `18255b2`。

supervisor：tmux `allagent_oldmix_ablation_supervisor`，PID `454469`；正式训练 tmux 预留名 `allagent_oldmix_ablation_200k`。

计划 artifact root：`artifacts/2026-08-10_allagent_oldmix_ablation/`

计划 run：`legacy_voradj_oldmix_allagent_4p1e1obs_200k_aw_20260810`

每 25k 在本节追加 A/B 的 sample efficiency、wall-clock efficiency、best/final performance，以及 sampler distribution 对照。25k 仅作早期诊断，不作为提前停止 Gate。

### 2026-08-10 20:34 自动启动状态

- Baseline B：未启动；未出现正式训练 PID、tmux 或 run artifact。
- supervisor：PID 454469 持续健康轮询，状态为等待 Stage4A/C 首条 clean 200k。
- 资源释放候选：Stage4A 170k（最新 checkpoint 150k），Stage4C 153k（150k checkpoint 已落盘）。
- Baseline A control：35k，最新正式诊断为 25k；Pure-CE：25k。
- Baseline A 25k：capture/mixed capture 0/4，但两场景 detected 4/4、collision 0/4；pure-CE strict 1/4、CV<0.20 2/4。该结果只作为 B 后续同 step A/B 比较基准。
- GPU0/GPU1 均为 100% utilization，温度 84°C/92°C；因此 supervisor 正确地没有提前执行 CUDA/2k 或启动 B。

### 2026-08-10 22:13 自动启动状态

- Baseline B：仍未启动；正式 tmux `allagent_oldmix_ablation_200k` 与正式 run artifact 均不存在。
- supervisor：PID 454469 持续健康，每分钟确认 Stage4A/C 仍存活；当前未执行 CUDA32/2k Gate。
- 资源释放候选：Stage4A 181k（175k checkpoint、capture 3/4、collision 1/4），Stage4C 169k（最新 checkpoint 150k）。
- Baseline A control：52k，50k checkpoint 已落盘；Pure-CE：36k。
- Baseline A 50k 对后续 A/B 的关键基准：pure-CE strict 2/4、CV<0.20 3/4；capture/mixed distance progress +14.79/+15.85，但 capture 0/4 且 collision 4/4、3/4。
- GPU0/GPU1 均为 100% utilization，温度 85°C/92°C；根盘余约 57 GiB，故继续等待是符合资源保护合同的行为。

### 2026-08-10 22:17 资源调度合同修订

- 用户指定 Baseline B 与 Pure-CE 同驻 `cuda:0`，不得使用 Baseline A 所在 `cuda:1`。
- 唯一释放 Gate 改为旧 Stage4A/cuda:0 clean 200k；Stage4C 是否先结束不再触发 Baseline B。
- 旧 supervisor 已单独停止，四条训练线均未受影响；新调度合同及相关 SAC/replay/storage 回归共 35 项通过。
- 新 supervisor 已于 22:17:55 以同名 tmux 重新武装，PID `542373`；实时状态仅含 `Stage4A: true`，消息为 `waiting for clean Stage4A completion on Pure-CE GPU`。
- 正式 Baseline B tmux/run artifact 仍不存在，没有提前启动。

## 8. 最终解释 Gate

- B 明显优于 A：后续主线默认 all-agent，focal 保留作历史消融。
- A 明显优于 B：继续比较 focal role balancing 与 phase-balanced joint replay + all-agent update，不宣称当前 quota 最优。
- 两者都差：focal 不是主要瓶颈，转向 reward、observation、credit、task interference、critic stability、curriculum。
- 两者都好：优先采用更标准、简单且易扩展的 all-agent 主线。

<!-- AUTO_ALLAGENT_FORMAL_LAUNCH -->
## Baseline B — Standard All-Agent Old-Mix 已启动

- 时间：2026-08-11T01:35:28+08:00
- 释放资源的旧线：Stage4A
- branch/worktree：`ablation/all-agent-oldmix-20260810` / `/home/yjq/rl/CoCap1/cocap-voradj-allagent-oldmix`
- seed/GPU/tmux：`2026080902` / `cuda:0` / `allagent_oldmix_ablation_200k`
- artifact root：`/home/yjq/rl/CoCap1/cocap-voradj-allagent-oldmix/artifacts/2026-08-10_allagent_oldmix_ablation`
- 唯一算法差异：focal role-balanced item update → uniform joint-transition all-active-agent update。
- 保持不变：mixed:pure=1:1、reward、Legacy-VorAdj、APF、network、SAC hyperparameters、seed、200k budget。
- 2k preflight：`{"cuda32_tag": "allagent_oldmix_cuda0_smoke32_20260810", "preflight_tag": "legacy_voradj_oldmix_allagent_preflight_2k_cuda0_20260810", "report": "/home/yjq/rl/CoCap1/cocap-voradj-allagent-oldmix/artifacts/2026-08-10_allagent_oldmix_ablation/preflight_2k/legacy_voradj_oldmix_allagent_preflight_2k_cuda0_20260810/legacy_voradj_oldmix_allagent_preflight_2k_cuda0_20260810_report.json", "updates": 469, "env_steps_per_second": 1.1669494495633304, "update_wall_time_s": 3.145146484375, "critic_time_s": 0.8741939086914062, "actor_q_time_s": 0.750358642578125, "updates_per_second": 0.3179502147095444, "active_agent_loss_terms_per_update": 512.0, "active_agent_loss_terms_per_second": 162.79050993128672, "peak_vram_mib": 22461.93408203125, "gpu_profile": {"elapsed_s": 1719.469720097004, "samples": 341, "gpu_utilization_mean_percent": 99.99413489736071, "gpu_utilization_max_percent": 100, "temperature_max_c": 86, "memory_used_max_mib": 43354, "memory_free_min_mib": 5786}, "all_finite": true}`
- 下一个正式 checkpoint：25k。

<!-- AUTO_ALLAGENT_STEP_25000 -->
### Baseline B 自动里程碑 25,000

- 时间：2026-08-11T05:42:47+08:00
- checkpoint：`/home/yjq/rl/CoCap1/cocap-voradj-allagent-oldmix/artifacts/2026-08-10_allagent_oldmix_ablation/legacy_voradj_oldmix_allagent_4p1e1obs_200k_aw_20260810/checkpoints/step_000025000`
- diagnostic：`/home/yjq/rl/CoCap1/cocap-voradj-allagent-oldmix/artifacts/2026-08-10_allagent_oldmix_ablation/legacy_voradj_oldmix_allagent_4p1e1obs_200k_aw_20260810/checkpoints/step_000025000/diagnostic_eval.json`
- storage：`evaluation_model_only`，contains_replay=False
- 指标摘要：`{"step": 25000, "update_count": 5001, "mean_finite": 1.0, "mean_critic_loss": 16.239204359054565, "mean_actor_loss": 17.235658317565917, "mean_alpha": 0.1228210374712944, "mean_update_wall_time_s": 1.9219128017578122, "mean_updates_per_second": 0.5210293972269913, "mean_active_agent_loss_terms_per_update": 512.0, "mean_active_agent_loss_terms_per_second": 266.7670513802195, "env_steps_per_second": 1.8571332086226953, "collision_count": 1}`
- sampler 摘要：`{"sampler": "uniform_joint", "requested_batch_size": 128, "actual_batch_size": 128, "unique_joint_transitions": 128, "replacement_count": 0, "pre_capture_transition_count": 48, "post_capture_transition_count": 0, "pure_ce_transition_count": 80, "sampled_phase_counts": {"pre_capture": 48, "pure_coverage": 80}, "sampled_scene_counts": {"mixed_crms": 48, "pure_ce": 80}, "sampled_active_agent_role_counts": {"pursuing": 135, "support": 55, "coverage": 322}, "active_agent_loss_terms": 512, "role_metadata_used_for_sampling": false}`
- GPU 摘要：`{"index": 0, "temperature_c": 85, "utilization_percent": 100, "memory_used_mib": 33063, "memory_total_mib": 49140, "memory_free_mib": 16077}`

<!-- AUTO_ALLAGENT_STEP_50000 -->
### Baseline B 自动里程碑 50,000

- 时间：2026-08-11T09:26:03+08:00
- checkpoint：`/home/yjq/rl/CoCap1/cocap-voradj-allagent-oldmix/artifacts/2026-08-10_allagent_oldmix_ablation/legacy_voradj_oldmix_allagent_4p1e1obs_200k_aw_20260810/checkpoints/step_000050000`
- diagnostic：`/home/yjq/rl/CoCap1/cocap-voradj-allagent-oldmix/artifacts/2026-08-10_allagent_oldmix_ablation/legacy_voradj_oldmix_allagent_4p1e1obs_200k_aw_20260810/checkpoints/step_000050000/diagnostic_eval.json`
- storage：`evaluation_model_only`，contains_replay=False
- 指标摘要：`{"step": 50000, "update_count": 11251, "mean_finite": 1.0, "mean_critic_loss": 25.8559407081604, "mean_actor_loss": 25.056128120422365, "mean_alpha": 0.0671182989180088, "mean_update_wall_time_s": 1.8292255302734375, "mean_updates_per_second": 0.5651204524466106, "mean_active_agent_loss_terms_per_update": 512.0, "mean_active_agent_loss_terms_per_second": 289.3416716526646, "env_steps_per_second": 1.9569574220841743, "collision_count": 2}`
- sampler 摘要：`{"sampler": "uniform_joint", "requested_batch_size": 128, "actual_batch_size": 128, "unique_joint_transitions": 128, "replacement_count": 0, "pre_capture_transition_count": 40, "post_capture_transition_count": 0, "pure_ce_transition_count": 88, "sampled_phase_counts": {"pre_capture": 40, "pure_coverage": 88}, "sampled_scene_counts": {"mixed_crms": 40, "pure_ce": 88}, "sampled_active_agent_role_counts": {"pursuing": 119, "support": 41, "coverage": 352}, "active_agent_loss_terms": 512, "role_metadata_used_for_sampling": false}`
- GPU 摘要：`{"index": 0, "temperature_c": 83, "utilization_percent": 100, "memory_used_mib": 33063, "memory_total_mib": 49140, "memory_free_mib": 16077}`

<!-- AUTO_ALLAGENT_STEP_75000 -->
### Baseline B 自动里程碑 75,000

- 时间：2026-08-11T13:11:20+08:00
- checkpoint：`/home/yjq/rl/CoCap1/cocap-voradj-allagent-oldmix/artifacts/2026-08-10_allagent_oldmix_ablation/legacy_voradj_oldmix_allagent_4p1e1obs_200k_aw_20260810/checkpoints/step_000075000`
- diagnostic：`/home/yjq/rl/CoCap1/cocap-voradj-allagent-oldmix/artifacts/2026-08-10_allagent_oldmix_ablation/legacy_voradj_oldmix_allagent_4p1e1obs_200k_aw_20260810/checkpoints/step_000075000/diagnostic_eval.json`
- storage：`evaluation_model_only`，contains_replay=False
- 指标摘要：`{"step": 75000, "update_count": 17501, "mean_finite": 1.0, "mean_critic_loss": 31.32510574531555, "mean_actor_loss": 27.56184755706787, "mean_alpha": 0.0430303825289011, "mean_update_wall_time_s": 1.9056682998046874, "mean_updates_per_second": 0.525597536520836, "mean_active_agent_loss_terms_per_update": 512.0, "mean_active_agent_loss_terms_per_second": 269.105938698668, "env_steps_per_second": 1.8552578344404043, "collision_count": 1}`
- sampler 摘要：`{"sampler": "uniform_joint", "requested_batch_size": 128, "actual_batch_size": 128, "unique_joint_transitions": 128, "replacement_count": 0, "pre_capture_transition_count": 36, "post_capture_transition_count": 0, "pure_ce_transition_count": 92, "sampled_phase_counts": {"pure_coverage": 92, "pre_capture": 36}, "sampled_scene_counts": {"pure_ce": 92, "mixed_crms": 36}, "sampled_active_agent_role_counts": {"pursuing": 98, "support": 46, "coverage": 368}, "active_agent_loss_terms": 512, "role_metadata_used_for_sampling": false}`
- GPU 摘要：`{"index": 0, "temperature_c": 84, "utilization_percent": 98, "memory_used_mib": 33063, "memory_total_mib": 49140, "memory_free_mib": 16077}`

### 2026-08-11 14:15 运行状态与 ETA

- Baseline B 正常运行于 cuda:0，最新81k/200k，loss finite；75k pure-CE 为 strict 1/4、CV<0.15 2/4、CV<0.20 3/4，存在coverage积极信号，但 capture/mixed均0/4且collision均4/4，尚无围捕成功信号。近25k稳态约6.7k steps/h，训练到200k ETA 为 2026-08-12 08:00左右，含最终20回合screening约08:30--09:15。
- Pure-CE 正常运行于同一 cuda:0，最新174k/200k；150k为strict 0/4、CV<0.15 2/4、CV<0.20 3/4、collision 0/4，CE energy平均改善0.05669，属于明确宽松coverage/energy积极信号。稳态约10k steps/h，训练到200k ETA 为当日16:45--17:00，含最终screening约17:15--17:45。
- Stage4A、Stage4C、Baseline A已完成。完结线20-rollout/5-GIF任务使用CPU低优先级后台tmux，不使用训练GPU；Stage4A 50k/200k paired结果已完成，剩余Stage4C与Baseline A四个任务会自行退出并落盘。

## 9. 2026-08-11 代码审计：显存占用、断点续训与修复方案

### 9.1 断点续训合同验证 ✅

对 Baseline B 的 checkpoint/resume 合同进行逐项核查，结论：**正确实现，可正常断训续训**。

| 检查项 | 路径/文件 | 结果 |
|--------|-----------|------|
| 里程碑 checkpoint | `checkpoints/step_000025000/` (×3: 25k/50k/75k) | ✅ `evaluation_model_only`，不含 replay.pkl（138MB trainer.pt only） |
| 滚动 resume bundle | `resume_latest/` | ✅ `rolling_latest_full_resume`，含完整 replay.pkl（2.4GB）+ trainer.pt（138MB） |
| 存储模式 | `checkpoint_storage.json` | ✅ milestone=`evaluation_model_only`/contains_replay=false；resume=`rolling_latest_full_resume`/contains_replay=true |
| 原子保存 | `_replace_dir_atomic(tmp, bundle_dir)` | ✅ tmp → rename，防半写 |
| Resume step 校验 | `_verify_resume_steps()` | ✅ trainer.transition_count == replay.transition_count，不匹配拒绝加载 |
| Manifest 隔离 | manifest.json 含 `optimizer_unit`/`replay_sampling`/`focal_training` | ✅ A/B 线 manifest 不同，A 线 checkpoint 无法误加载到 B 线 |
| step-1 初始点 | `checkpoints/step_000000001/` | ✅ 含 trainer.pt（63MB，结构初始化参数），可做初始化对照 |
| 75k 全景 | `resume_latest/` @ 75k | ✅ 2.5GB 总量：trainer 138MB + replay 2.4GB + 元数据 ~300KB |

**结论**：任意时刻断电/被 kill 后，用 `resume_latest/trainer.pt` + `resume_latest/replay.pkl` + `--resume-step <step>` 即可精确续训。milestone 的 trainer.pt 可独立用于 eval，不依赖 replay。

### 9.2 显存占用根因分析 🔍

Baseline B 峰值显存 **22,462 MiB**（preflight 记录），稳态 nvidia-smi 显示 **23,156 MiB**。根因定位：

**罪魁祸首：`_focal_actor_q` 在 all-agent 路径中循环执行 8 次完整 critic 前向传播**

位置：[`src/cocap_voradj/training/continuous/central_sac.py:205-227`](../src/cocap_voradj/training/continuous/central_sac.py#L205-L227)，由 all-agent update 路径第 298 行调用。

```python
# central_sac.py:298 — all-agent 更新路径中的 actor Q 计算
actor_q = self._focal_actor_q(central_obs, policy_actions, active)

# _focal_actor_q 内部（L205-227）：
# 对 4 个 active agent slot 各执行 critic1 + critic2 前向：
for focal in range(actions.shape[1]):   # max_agents=12
    if focal not in active_slots:        # 8 个 inactive 跳过
        continue
    focal_actions = actions.detach().clone()
    focal_actions[:, focal] = actions[:, focal]
    q1 = self.critic1(central_obs, focal_actions)[:, focal]  # ← 完整前向
    q2 = self.critic2(central_obs, focal_actions)[:, focal]  # ← 完整前向
    # 共 4 agents × 2 critics = 8 次完整 transformer 前向
```

**为何消耗 22GB**：

每次 critic 前向处理 `[B×max_agents, 26 tokens, 256 dim] = [1536, 26, 256]`，经过 4 层 transformer。每层 FFN 中间张量 `[1536, 26, 1024]` ≈ 156 MB。8 次前向的 autograd 图**全部在 `actor_loss.backward()` 之前保持存活**（因为需要用 `torch.no_grad` 但此处不能——actor 需要 Q 对 action 的梯度），仅 actor update 阶段的 autograd 图就占用 ~10–12 GB。

**为何 Focal Baseline A 无此问题**：Focal 路径用 `_focal_actor_q_for_ids`（L229-242），一次 critic 前向覆盖所有 batch item（每个 item 的 focal agent 不同，通过 gather 取对应的 Q 值），**只需 2 次 critic 前向**（c1+c2），而非 8 次。

**这是设计问题，不是内存泄漏**：语义正确、梯度正确、所有 metric finite。`_focal_actor_q` 是为 focal path（每个 batch item 只有一个 agent 进入 loss）设计的，被直接复用到 all-agent path（需要所有 active agent 的 Q 值），产生了 4× 的冗余计算和 ~3–4× 的显存放大。

### 9.3 修复方案

**方案（推荐）**：增量 backward，一次一个 active agent，立即释放 autograd 图。

```python
# 替换 central_sac.py:298（all-agent update 中）
# 旧代码：
actor_q = self._focal_actor_q(central_obs, policy_actions, active)
actor_loss = masked_mean(self.alpha.detach() * log_prob - actor_q, active)
...
actor_loss.backward()

# 新代码：逐个 agent 计算 loss 并累积梯度，每步释放计算图
self.actor_optimizer.zero_grad(set_to_none=True)
n_total = active.sum().clamp_min(1)
for a_id in sorted(set(
    int(s) for s in torch.nonzero(active.any(dim=0), as_tuple=False).flatten().tolist()
)):
    a_id_tensor = torch.full((actions.shape[0],), a_id, dtype=torch.long, device=self.device)
    q_i = self._focal_actor_q_for_ids(central_obs, policy_actions, a_id_tensor)
    mask_i = active[:, a_id].to(dtype=torch.float32)
    loss_i = (mask_i * (self.alpha.detach() * log_prob[:, a_id] - q_i)).sum() / n_total
    loss_i.backward()  # 立即释放该 agent 的计算图
# 梯度已累积在 actor 参数中，接续原有的 grad clip + optimizer.step()
```

**效果预估**：

| 指标 | 当前 | 修复后 |
|------|------|--------|
| Actor update 阶段 critic 前向次数 | 8（4×2） | 8（4×2，但逐个 backward 释放） |
| Peak autograd 图同时存活 | 8 个完整的 critic 计算图 | **1 个** agent 的 2 个 critic 图 |
| Actor update 阶段峰值显存 | ~10–12 GB | ~2–3 GB |
| **总峰值显存** | **~22 GB** | **~12–14 GB** (↓ ~40%) |
| 训练语义 | 参考 | **完全等价**（masked mean 可分解为逐 agent 加权和） |

**验证方法**：在同一 `resume_latest` checkpoint 上跑 1k steps A/B smoke，确认 (a) loss 曲线一致、(b) 梯度数值一致（max abs diff < 1e-6）、(c) `torch.cuda.max_memory_allocated` 显存下降。

**执行时机**：当前 Baseline B 的 200k 训练**不中断**。修复在 200k 完成后、下一条 all-agent 主线启动前实施。当前不影响训练正确性。

### 9.4 显存账本（供后续参考）

| 组件 | 大小 | 说明 |
|------|------|------|
| 模型参数（Actor + 4 Critics） | ~65 MB | 16.3M params × float32 |
| 优化器状态（Adam） | ~233 MB | momentum + variance |
| PyTorch CUDA context/cuDNN | ~1 GB | 固定开销 |
| Critic update（2 critics backward） | ~4–6 GB | 双 critic 前向+反向 |
| Actor update（8 critic forwards） | ~10–12 GB | ⚠️ 当前冗余 |
| PyTorch caching allocator | ~1–3 GB | 碎片/缓存 |
| **合计** | **~22 GB** | |

修复后 Actor update 降至 ~2–3 GB，总峰值降至 ~12–14 GB，Pure-CE（~10 GB）可与其更舒适地共存于 48 GB A6000。

## 10. 2026-08-11 All-Agent 修复、no-clip 切换与 P0/P1

### 10.1 Baseline B 定向停训与恢复点

- 17:15 只向 Baseline B tmux `allagent_oldmix_ablation_200k` 发送正常 Ctrl-C；pre-stop PID `666602` 随后退出，B supervisor 自然报停并退出。其他用户 PID `589205` / cuda:1 和其他项目均未处理。
- 停训前最后完整 metrics 是 104k / update 24,751；runner 无 signal checkpoint，所以 100k 后约4k只是不可恢复观察，不计入续训进度。最新原子完整 bundle 是 `resume_latest @100000` / update 23,751。
- CPU 严格加载确认 trainer/replay/runtime 均为 100k、uniform joint、all active-agent、含 runner/replay/model RNG 与 optimizer/target/alpha state。原 bundle 以 hardlink 冻结为 `resume_frozen_pre_fix_step_000100000`。

### 10.2 Actor-Q root cause 与修复 Gate

- root cause：旧 all-agent Actor update 在一次 shared stochastic Actor forward 后，为4个 active slots 各保留 twin critic 完整 autograd graph，直到最终 `actor_loss.backward()`；约8个 central Transformer graph 同时驻留。数学语义正确，但显存冗余。
- fix `bounded_critic_vjp_v1`：joint policy action只采样一次；逐 active slot 计算 `Q_i(S,a_i,a_-i.detach)` 对 `a_i` 的精确 VJP并立即释放 critic graph，再将汇总 `d(-mean Q)/da` 与 entropy term一次回传共享 Actor。focal path不变、teammates detach不变、inactive slots为零。
- 同 checkpoint、同固定 replay batch、同 checkpoint torch/CUDA RNG：critic/actor/alpha loss abs diff 均 `0`；per-agent Q max abs diff `0`；全 Actor parameters gradient max abs/relative-L2 diff `0`，direction一致；inactive action gradient nonzero=`0`。
- 单次实测 peak allocated `22462.04 → 8045.24 MiB`，reserved `22828 → 8324 MiB`。5次稳态同批次中位数（排除1次warmup）peak allocated `17770.38 → 8217.79 MiB`、reserved `18100 → 8538 MiB`，update wall `0.9517 → 0.9476 s`，active terms/s约`1.004×`，没有时间退化。
- 等价/benchmark：`artifacts/2026-08-11_allagent_actorq_fix/equivalence/`。相关回归 47/47；全仓 178 passed、2 failures，两个 failure 都是本 worktree 未复制历史 `runs/...` checkpoint/effective-config，和本补丁无关。
- 修复/no-clip commit：`f19dac6cc3cd505415985a745e784cb9f84e10e8`。

### 10.3 no-clip bundle migration 与 exact switch

```text
step < 100000:
all-agent + pre-fix retained Actor-Q graphs + grad_clip_norm=0.5

step >= 100000:
all-agent + bounded_critic_vjp_v1 + grad_clip_norm=None
```

- migration只允许3个 effective-config diff：`training.grad_clip_norm 0.5→null`，以及新增 Actor-Q implementation / gradient-clipping 两个审计标记；所有模型结构、LR、MSE、tau、UTD、batch、reward、action、scene cycle、optimizer/replay unit均相同。
- 新 `resume_latest` 重新绑定当前 manifest；再次严格加载确认 trainer/replay/runtime 100k、update 23,751，Actor/critics/targets/alpha、三个optimizer、全部 RNG 均保留。迁移证据为 `resume_latest/resume_contract_migration.json`。
- 原位续训于 17:34 启动：PID `1138338`、tmux `allagent_oldmix_b_noclip_resume`、cuda:0、从 100k继续200k预算；没有 reset seed/replay/optimizer/alpha/targets/schedule。
- no-clip safety：101k/102k均 finite，peak allocated稳定 `8045.25 MiB`；update wall `1.055/0.961 s`，吞吐 `3.37/3.78 step/s`。101k critic loss/TD/grad一度升至 `355.84/4.69/4073.66`，102k回落至 `82.96/2.37/1641.45`；无NaN/OOM/持续显存增长，因此保留 no-clip，不启用5.0 fallback。

### 10.4 P0 Health Audit

- 独立文档：`docs/SAC_HEALTH_AUDIT_20260811_ZH.md`。
- 固定100k/32×128 batch下，旧`.5`对critic和actor触发率均100%；critic norm p50/p95/p99=`381/2511/3246`、典型clip scale `0.00131`，actor=`1.53/2.87/4.86`、scale `0.327`；alpha从不触发。
- pursuing TD p50/p95/p99=`1.94/9.19/23.12`，collision小样本 TD p50=`66.69`；post-capture样本为0。
- `|alpha log pi|/|Q|` p50约0.2%--0.4%、p99约1%，entropy相对Q偏弱。
- Q ranking 800 agent-state items中，严格 `Qseek>Qpolicy>Qrandom` 仅21.9%；visible 21.3%、invisible 24.5%。当前最直接瓶颈是critic没有稳定学出seek/policy/random动作排序，而非已证实的observability问题。

### 10.5 P1 no-clip scratch

- formal config：`legacy_voradj_oldmix_4p1e1obs_200k_aw_allagent_noclip_scratch.yaml`；seed `2026080902`；独立 artifact `artifacts/2026-08-11_allagent_noclip_scratch/`。
- manifest明确 `optimizer_unit=joint_transition_all_active_agents`、`replay_sampling=uniform_joint`、`focal_training=false`、`grad_clip=none`、`initialization=scratch`、`actor_q_implementation=bounded_critic_vjp_v1`；没有 checkpoint warm start。
- CUDA32 scratch smoke 已通过：32/32 transitions、all finite、0 updates（正式warmup仍为5k）、0 collision、无OOM，manifest合同正确。
- 2k optimizer preflight 使用独立 config 将 warmup临时缩到128，仅用于覆盖no-clip update安全 Gate；正式P1仍保持原5k warmup。2k/469 updates完成且 all finite，末次 critic/actor/alpha loss=`5.679/4.881/-2.860`、alpha=`0.1909`、TD=`0.929`；raw=post grad，peak allocated稳定`8045.14 MiB`，无OOM/leak。与B并行的端到端吞吐约`1.971 step/s`（约7.10k/h）。
- P1正式200k已于 `2026-08-11 18:03:54+08:00` 从 scratch 启动；PID `1171022`、tmux `p1_allagent_noclip_scratch_200k`、cuda:0。命令没有任何 `--resume-*` 或 initialization checkpoint，独立run目录为 `artifacts/2026-08-11_allagent_noclip_scratch/legacy_voradj_oldmix_allagent_noclip_scratch_4p1e1obs_200k_aw_20260811/`。
- 18:05实时：P1为1k/200k、update=0（原合同5k warmup），warmup吞吐22.78k step/s；ETA不使用该虚高值，而使用与B同卡且真实执行updates的P1 preflight `7.10k step/h` 与B双线窗口 `6.72k step/h`，保守取`6.7--7.1k step/h`。P1 25k ETA `2026-08-11 21:28--21:40+08:00`，200k ETA `2026-08-12 22:05--23:50+08:00`。

### 10.6 双线实时资源与 ETA（2026-08-11 18:05+08:00）

- Baseline B：104k/200k、PID `1138338`、tmux `allagent_oldmix_b_noclip_resume`、cuda:0；最近双 optimizer 线共存窗口 `1.866 step/s=6.72k/h`、finite、update wall `1.979 s`、peak allocated `8045 MiB`。按`6.7--7.1k/h`，125k ETA `2026-08-11 21:03--21:13+08:00`，200k ETA `2026-08-12 07:36--08:25+08:00`。
- P1：1k/200k、PID `1171022`、tmux `p1_allagent_noclip_scratch_200k`、cuda:0；真实update稳态以上述2k preflight为准。25k / 200k ETA 如10.5。
- 当前P1仍在warmup，因此两进程显存约`8.66+0.40 GiB`；2k并行optimizer smoke已验证steady约`8.66+8.05=16.71 GiB`，远低于A6000 48 GiB。系统RAM约20/125 GiB、swap 0.15/8 GiB，根盘余35 GiB；不使用高温cuda:1上的其他用户进程。

<!-- AUTO_ALLAGENT_STEP_100000 -->
### Baseline B 自动里程碑 100,000

- 时间：2026-08-11T16:52:38+08:00
- checkpoint：`/home/yjq/rl/CoCap1/cocap-voradj-allagent-oldmix/artifacts/2026-08-10_allagent_oldmix_ablation/legacy_voradj_oldmix_allagent_4p1e1obs_200k_aw_20260810/checkpoints/step_000100000`
- diagnostic：`/home/yjq/rl/CoCap1/cocap-voradj-allagent-oldmix/artifacts/2026-08-10_allagent_oldmix_ablation/legacy_voradj_oldmix_allagent_4p1e1obs_200k_aw_20260810/checkpoints/step_000100000/diagnostic_eval.json`
- storage：`evaluation_model_only`，contains_replay=False
- 指标摘要：`{"step": 100000, "update_count": 23751, "mean_finite": 1.0, "mean_critic_loss": 31.23241411781311, "mean_actor_loss": 29.80998917388916, "mean_alpha": 0.03932702188193798, "mean_update_wall_time_s": 1.2904385898437498, "mean_updates_per_second": 0.797520158482335, "mean_active_agent_loss_terms_per_update": 512.0, "mean_active_agent_loss_terms_per_second": 408.33032114295554, "env_steps_per_second": 2.786929419352983, "collision_count": 1}`
- sampler 摘要：`{"sampler": "uniform_joint", "requested_batch_size": 128, "actual_batch_size": 128, "unique_joint_transitions": 128, "replacement_count": 0, "pre_capture_transition_count": 40, "post_capture_transition_count": 0, "pure_ce_transition_count": 88, "sampled_phase_counts": {"pure_coverage": 88, "pre_capture": 40}, "sampled_scene_counts": {"pure_ce": 88, "mixed_crms": 40}, "sampled_active_agent_role_counts": {"pursuing": 110, "support": 49, "coverage": 353}, "active_agent_loss_terms": 512, "role_metadata_used_for_sampling": false}`
- GPU 摘要：`{"index": 0, "temperature_c": 70, "utilization_percent": 100, "memory_used_mib": 33063, "memory_total_mib": 49140, "memory_free_mib": 16077}`

### 10.7 双线后台快照与积极信号边界（2026-08-11 19:30+08:00）

- Baseline B continuation：PID `1138338`、tmux `allagent_oldmix_b_noclip_resume`、cuda:0；最新落盘 `114000/200000`、update `27251`、replay `114000`，近窗口 `1.862 step/s`（约 `6.70k step/h`）。`finite=1`，critic/actor loss=`40.91/34.00`、alpha=`0.04206`、TD abs mean=`1.536`，critic/actor raw grad norm=`862.80/2.03`，peak allocated=`8045.25 MiB`。no-clip初期101k的 loss/TD瞬时冲高已回落；112k TD的局部反弹 `2.229` 也在114k回落至 `1.536`，暂无持续数值发散。
- P1 no-clip scratch：PID `1171022`、tmux `p1_allagent_noclip_scratch_200k`、cuda:0；最新落盘 `14000/200000`、update `2251`、replay `14000`，近窗口 `1.859 step/s`（约 `6.69k step/h`）。`finite=1`，critic/actor loss=`7.89/9.42`、alpha=`0.16240`、TD abs mean=`0.564`，critic/actor raw grad norm=`41.77/0.445`，peak allocated=`8045.14 MiB`。相比5k首次update的 critic loss/TD=`105.99/1.273`，早期价值拟合已快速回落且保持finite。
- 行为信号只能给出谨慎的“已有单机接敌”：P1 12k窗口曾有 `d1_min=9.15 m`、`max_num_in_ring=1`、`fraction_closing=0.611`；B 108k窗口也曾有单机入ring。但两线 `fraction_steps_2plus_in_ring=0`、`fraction_steps_3plus_in_ring=0`，且截至本快照 replay的 `post_capture_coverage=0`，因此**没有 formal k=3 capture 突破证据**。P1出现过非碰撞 termination，但在无post-capture转移的前提下不将其误报为capture，更可能是Pure-CE结束。
- 资源：GPU0 `17355/49140 MiB`、85°C、100% utilization；两条正式线各占约 `8662--8664 MiB`。系统RAM available `94 GiB`、swap `149 MiB/8 GiB`、根盘available `179 GiB`；无OOM、无持续VRAM/RAM泄漏证据。cuda:1的其他用户训练未处理。
- 按两线最近真实optimizer稳态约 `6.7k step/h`，并为同时25k checkpoint/eval保留写盘波动：B 125k预计 `2026-08-11 21:05--21:20+08:00`，artifact完成预计 `21:15--21:35`；B 200k预计 `2026-08-12 08:20--09:20`。P1 25k预计 `2026-08-11 21:05--21:20`，artifact完成预计 `21:15--21:35`；P1 200k预计 `2026-08-12 23:15--2026-08-13 01:00`。

### 10.8 Baseline B 200k完成与P1 125k里程碑（2026-08-12 10:24+08:00）

#### Baseline B：fixed/no-clip continuation已完整结束

- B于 `2026-08-12 08:45:30+08:00` 完成 `200000/200000`，进程和tmux已正常退出。final bundle为 `checkpoints/step_000200000`，同时完整写出 `resume_latest/{trainer.pt,replay.pkl,runtime_state.pkl}`；trainer约145 MB，replay约6.73 GB。
- final contract：replay=`200000`、updates=`48751`、`all_finite=true`，final window critic/actor loss=`46.86/40.52`、alpha=`0.04676`、TD abs mean=`1.529`，raw critic/actor grad norm=`821.47/2.61`，peak allocated=`8045.25 MiB`。从100k切换至200k无NaN/OOM/显存增长，证明fixed Actor-Q + no-clip continuation工程与数值稳定。
- 负面主结论：全线replay `post_capture_coverage=0`，200k diagnostic eval的capture/mixed均 `capture_rate=0`，capture/mixed collision rate均 `1.0`；strict coverage rate也均为0。因此B没有学出formal moving/local/obstacle/k=3 capture，去掉`.5` clip并继续100k steps没有带来capture突破。
- 有限正面信号：200k Pure-CE eval的 `coverage_cv020/loose=0.75`，但strict=0；capture/mixed的mean minimum distance分别 `14.68/12.06 m`，但collision=1且capture=0。这只支持“relaxed coverage与接敌几何存在”，不支持真正多机围捕。

#### P1：no-clip from scratch仍在运行

- P1正式125k checkpoint/eval已于 `10:18+08:00` 完整落盘，随后训练已继续到 `126000/200000`、update `30251`；PID `1171022`、tmux `p1_allagent_noclip_scratch_200k`、cuda:0。B结束后普通训练窗口吞吐稳定为 `3.71--3.80 step/s`，约 `13.4--13.7k step/h`，相比双线并行阶段提高约2倍；126k窗口因刚完成checkpoint/eval仅为2.56 step/s，不用于稳态ETA。
- 125k window `finite=1`，critic/actor loss=`24.15/34.15`、alpha=`0.03884`、TD abs mean=`1.090`，raw critic/actor grad norm=`339.39/1.70`，peak allocated=`8045.14 MiB`；无NaN/OOM/leak。
- 几何上最明显的新信号是反复单机深度接敌：114k--117k窗口 `d1_min=5.93/9.02/8.41/6.26 m`，122k/124k达 `3.63/4.24 m`；但 `max_num_in_ring<=1`、`fraction_steps_2plus_in_ring=0`、`fraction_steps_3plus_in_ring=0`，replay `post_capture_coverage=0`。接近evader还没有转化为多机同时围捕。
- 125k diagnostic eval仍为 capture/mixed `capture_rate=0`；capture/mixed collision rate=`1.0/0.5`，Pure-CE `coverage_cv020/loose=0.5`、strict=0。capture/mixed mean minimum distance=`19.52/9.99 m`，mixed有接敌但未capture。P1 50k曾有capture/mixed collision rate同时降至`0.25`和Pure-CE loose coverage=`1.0`，后续未稳定保持，不能视为稳健改善。
- 按最近独占GPU0的真实吞吐：150k预计 `2026-08-12 12:08--12:20+08:00`；175k预计 `13:58--14:15`；200k训练step预计 `15:48--16:10`，包含final eval与6+ GB replay落盘的完整artifact预计 `16:00--16:30`。

#### 当前证据解读

- fixed Actor-Q显存修复与no-clip路径已通过工程稳定性验证；B 200k证明“单纯取消强clip + 多训100k”不足以解决formal capture。
- P1从step 0 no-clip的干净线仍有75k预算，但到125k的中期证据也只到“单机接敌”，未出现多机ring或capture。如200k仍为0 capture，应按既定路线基于P0 Q-ranking结果讨论UTD或Formal Capture-Only，而不继续将“训练步数不足”作为主解释。
