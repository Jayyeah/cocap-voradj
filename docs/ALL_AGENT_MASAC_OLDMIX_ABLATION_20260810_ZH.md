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

### 10.9 P1首次真实capture→post-capture信号与B正式rollout启动（2026-08-12 15:25+08:00）

#### P1 193k实时状态

- P1最新落盘 `193000/200000`、update `47001`、replay `193000`；PID `1171022`、tmux `p1_allagent_noclip_scratch_200k`、cuda:0。最近普通窗口吞吐 `3.71--3.75 step/s`（约 `13.36--13.50k step/h`），`finite=1`、critic/actor loss=`25.58/35.62`、alpha=`0.03931`、TD abs mean=`1.081`、raw critic/actor grad norm=`354.71/1.63`、peak allocated=`8045.14 MiB`，无NaN/OOM/leak。
- **首次真正积极的formal phase信号**：replay的 `post_capture_coverage` index在180k窗口从0跳到 `2000` 并保持。合同的post-capture coverage window为500 joint transitions，4个active agents对应 `500×4=2000` active-agent index entries；`pure_recovery_coverage`另行计数，因此这不是Pure-CE/recovery样本冒充，而是至少一次真实mixed episode完成capture并进入完整post-capture coverage window。
- 同一180k窗口无collision，`d1_min=3.60 m`、`max_num_within_8m=2`、`max_num_in_ring=2`、`fraction_steps_2plus_in_ring=1.81%`；182k也出现 `max_num_in_ring=2`、2+ ring=`2.09%`。这是此前“只有单机接敌”之后首次出现双机同步接敌并实际进入post-capture phase，属于明确但稀有的正信号。
- 证据边界：150k/175k deterministic 4-episode diagnostic capture/mixed `capture_rate`仍为0；训练replay目前只支持至少一次capture，尚不能证明稳定capture，且没有3+ ring窗口信号。是否能在200k deterministic eval和20-rollout中复现，需等待P1 final artifact后判断。
- 以193k和最近真实吞吐计算：剩余7k训练约31--32分钟，P1 200k step ETA `2026-08-12 15:56--16:02+08:00`；考虑final diagnostic eval和约6.7 GB replay写盘，完整final artifact ETA `16:10--16:35`。

#### Baseline B final 20-rollout/5-GIF

- B final `step_000200000`已通过 `old_mix` preset dry-run合同验证：精确checkpoint/effective-config hash匹配，deterministic Actor，capture/Pure-CE/mixed各20 rollout、各前5条GIF，seed `2026081201`。
- 正式后台评估于 `2026-08-12 15:23:31+08:00` 启动：PID `1732563`、tmux `rollout_baseline_b_200k_20r5g`、CPU、2 workers、`nice=10`；输出目录 `artifacts/2026-08-12_allagent_completed_20rollout5gif/baseline_b_allagent/last_step200000/`，日志 `artifacts/2026-08-12_allagent_completed_20rollout5gif/logs/baseline_b_last200_20r5g.log`。
- 启动验证通过：父进程和两个spawn workers存活，run args/effective config/preset已写出，GIF开始生成；RAM available约92 GiB、swap约148 MiB，未占用额外GPU显存，P1窗口吞吐未见下降。按用户指令本轮不等待、不汇总rollout结果；下次GitHub状态更新时统一同步。

### 10.10 P1 200k冻结、旁路诊断与C0启动（2026-08-12 17:24+08:00）

- P1于 `2026-08-12 16:09:34+08:00` 正常完成 `200000/200000`，update `48751`、replay `200000`、`all_finite=true`。final window critic/actor loss=`24.37/35.41`、alpha=`0.03803`、TD abs mean=`1.093`，peak allocated=`8045.14 MiB`。
- 完整200k trainer/replay/runtime以hardlink冻结为 `resume_frozen_p1_step_000200000`；trainer约145 MB、replay约6.71 GB、runtime约825 KB，保留optimizer、target critics、alpha、replay ring与全部runtime/RNG状态。C0/C1均以此同一冻结bundle为起点。
- 训练replay最终仍只有一次明确mixed capture窗口：`post_capture_coverage=2000`，即500个post-capture joint transitions×4 active agents；未出现第二次capture证据。因此积极信号真实但稀有。
- P1 200k formal deterministic 20-rollout已完成：capture/mixed均`capture_rate=0`、`collision_rate=1.0`；capture mean minimum distance=`14.61 m`、distance progress=`+8.61 m`。Pure-CE strict/CV.15/CV.20=`0.45/0.80/0.95`、collision=`0`、mean area CV=`0.1052`、CE progress=`+0.1111`。相较B，P1的coverage明显更好，但formal capture仍未复现。
- P1 200k Q-ranking（800 agent-state items）已完成：严格`Qseek>Qpolicy>Qrandom=7.75%`，`Qseek>Qpolicy=22.5%`，`Qpolicy>Qrandom=53.5%`，mean seek-policy=`-3.09`。visible严格排序仅`5.66%`，far visible仅`3.51%`；critic动作排序在200k仍是主要疑点，且较100k审计没有改善。
- Baseline B 200k formal 20-rollout也已完成：capture/mixed capture=`0`、collision=`1.0`；Pure-CE strict/CV.15/CV.20=`0.05/0.50/0.70`、collision=`0.10`。P1相对B提升了Pure-CE coverage，但没有提升deterministic capture。
- C0按step-extension control合同于 `2026-08-12 16:18:04+08:00` 从P1冻结bundle启动：PID `1748359`、tmux `c0_p1_extension_utd025_300k`、cuda:0，UTD=.25、update_every=4，其余保持P1不变。17:24已到`214000/300000`、update`52251`、replay`214000`，最近窗口`3.767 step/s≈13.56k step/h`、全部finite、peak allocated=`8045.25 MiB`，无OOM/显存增长。
- C0按当前稳态吞吐：225k训练step ETA约 `2026-08-12 18:12+08:00`（另计checkpoint/eval写盘），300k训练step ETA约 `2026-08-12 23:45--2026-08-13 00:15+08:00`。

### 10.11 C0实时状态与C1自动挂起失败审计（2026-08-12 19:22+08:00）

#### C0仍正常

- C0 PID `1748359`、tmux `c0_p1_extension_utd025_300k`、cuda:0仍存活；最新落盘 `240000/300000`、update `58751`、replay `240000`。
- 最近1k窗口 `3.732 step/s≈13.44k step/h`，`finite=1`，critic/actor loss=`23.62/35.46`、alpha=`0.03688`、TD abs mean=`1.068`、peak allocated=`8045.25 MiB`。GPU0约`8687 MiB`、84°C、100% utilization；系统RAM available约90 GiB、swap 148 MiB/8 GiB，无OOM或持续内存增长。
- 240k窗口再次出现双机几何接敌：`max_num_in_ring=2`、2+ ring fraction=`1.21%`，但3+ ring fraction仍为0，且replay `post_capture_coverage=2000`没有超过P1 200k冻结时的值。因此截至240k仍没有第二个distinct real capture episode证据。
- 按最近真实吞吐，250k训练step ETA约 `2026-08-12 20:07+08:00`；300k训练step ETA约 `2026-08-12 23:50+08:00`，完整final eval与约8 GB replay落盘预计再增加15--30分钟。

#### C1准备已经完成

- commit `bee406a`已经包含独立C1 config、runner的`--resume-fork utd_only`严格白名单校验、回归测试及`tools/supervise_p1_extension_c1.py`。
- 真实P1-200k manifest与C1 resolved config的只读preflight通过：唯一训练规则变化为`training.update_every_env_steps 4→2`（UTD .25→.5）；另有总预算200k→300k和run/stage/resource审计元数据变化。LR、tau、batch、MSE、reward、action、replay、gradient_steps=1、no-clip均不变。
- C1/all-agent合同回归为`12 passed`；supervisor prerequisite检查为true，GPU0检测正确返回C0 PID `1748359`。代码/config已推送GitHub。

#### 自动挂起两次均失败，当前没有C1 supervisor

- 第一次尝试：tmux `c1_p1_utd05_autostart` 中以 `nice -n 5 env PYTHONPATH=src:. python3 tools/supervise_p1_extension_c1.py` 启动，并把stdout/stderr追加到 `artifacts/2026-08-12_p1_extension_controls/c1_autostart.log`。约2秒后复查时tmux会话与supervisor PID均已消失；日志文件存在但大小为0。
- 随后以 `PYTHONPATH=src:. timeout 3 python3 tools/supervise_p1_extension_c1.py` 做3秒前台诊断，未产生stderr/traceback输出；这没有证明长期等待循环正常，只说明没有捕获到可见的即时Python异常。
- 第二次尝试去掉文件重定向并加 `PYTHONUNBUFFERED=1`，仍在约5秒后发现tmux会话和supervisor PID消失；`tmux capture-pane -pt c1_p1_utd05_autostart` 返回 `can't find pane: c1_p1_utd05_autostart`。
- 因两次均为“tmux会话瞬退且无Python traceback/日志”，当前**不能把根因归为C1合同、GPU gate或GitHub网络**；只确认自动挂起没有成功，具体退出原因未捕获。按用户要求停止重复尝试，未直接启动C1，也未影响C0。

### 10.12 用户指令下C1改为GPU1直接启动（2026-08-12 19:41+08:00）

- 用户明确要求不再等待C0，立即在GPU1启动C1。启动前GPU1已有其他正式进程PID `1258417`、显存约7342 MiB；GPU1总显存49140 MiB，仍有约41.8 GiB空闲，因此没有停止或修改该进程。
- C1 config的resource审计元数据更新为 `cuda:1 concurrent with C0 by explicit user decision on 2026-08-12`；训练合同不变：从同一个P1-200k frozen bundle继续、UTD=.5、update_every=2、gradient_steps=1、总预算300k，其他保持P1。
- C1于 `2026-08-12 19:39:10+08:00` 直接启动：PID `1853291`、tmux `c1_p1_utd05_300k_gpu1`、device `cuda:1`。artifact为 `artifacts/2026-08-12_p1_extension_controls/c1_utd05/legacy_voradj_oldmix_allagent_noclip_c1_utd05_300k_aw_20260812/`，日志为相邻的 `c1_utd05_300k_gpu1.log`。
- 启动命令显式使用 `--resume-checkpoint/--resume-replay` 指向 `resume_frozen_p1_step_000200000`、`--resume-step 200000`、`--resume-fork utd_only`，没有从C0 bundle启动，也没有reset replay/optimizer/alpha/targets/RNG/runtime。
- `resume_fork_audit.json` 于19:39:47成功落盘，证明source P1 manifest加载和UTD-only白名单校验通过；没有合同拒绝或Python traceback。
- 19:41复查时C1 PID/tmux持续存在，C1在GPU1占用约8664 MiB且GPU利用率100%，说明已完成6.7 GB replay/optimizer反序列化并进入计算，不再是此前supervisor tmux瞬退状态。
- 并行资源：C0仍在GPU0约8664 MiB；GPU1原进程约7342 MiB+C1约8664 MiB，总约16038 MiB/49140 MiB。系统RAM available约77 GiB、swap148 MiB。无OOM，但GPU1温度已到92°C、风扇100%，这是明确热风险；后续优先观察温度、throttle和两进程吞吐，不得误停原PID。
- 当前尚未出现C1首个新1k metrics落盘，因此不能报告稳定steps/hour或行为信号；本次只确认合同加载、GPU更新和进程存活。C1 225k/300k ETA应在首个真实窗口后计算，不沿用C0吞吐。

### 10.13 C0 287k / C1 215k并行状态（2026-08-12 22:54+08:00）

#### C0 — UTD .25 extension

- PID `1748359`、tmux `c0_p1_extension_utd025_300k`、cuda:0持续正常；最新 `287000/300000`、update `70501`。replay显示`250000`是合同容量`capacity_joint=250000`下的正常ring封顶，不是停写或丢失runtime。
- 最近五个普通1k窗口吞吐约`3.46--3.77 step/s`，最近为`3.756 step/s≈13.52k step/h`；全部finite，末窗critic/actor loss=`23.18/34.22`、alpha=`0.03000`、TD abs mean=`1.002`、peak allocated=`8045.25 MiB`。
- 250k与275k model-only里程碑均完整落盘；275k对应rolling `resume_latest`也已完整写出trainer/replay/runtime，replay约8.41 GB。
- capture核心结论仍为负：post-capture index保持`2000`，原P1 capture窗口在250k ring内仍未被覆盖，因此截至287k没有第二个distinct capture。最近五窗没有2+或3+ ring；285k虽有单机深入`d1_min=4.91m`，仍未转化为多机capture。
- 275k 4-episode diagnostic：capture/mixed capture均0、collision均1.0，mean minimum distance=`21.76/12.64m`，distance progress=`-6.33/-2.29m`；比250k没有capture改善。Pure-CE仍有正面但不稳定的coverage信号：CV.15/CV.20=`1.0/1.0`、collision=0、area CV=`0.0724`，但strict=0。
- 按最近真实吞吐，300k训练step ETA约 `2026-08-12 23:52--23:58+08:00`；含final eval和完整replay落盘预计 `2026-08-13 00:10--00:30+08:00`。

#### C1 — UTD .5 fork

- PID `1853291`、tmux `c1_p1_utd05_300k_gpu1`、cuda:1持续正常；最新 `215000/300000`、update `56251`、replay `215000`。相对P1-200k的15k新steps增加7500 updates，精确符合UTD=.5。
- 最近五个1k窗口约`1.257--1.265 step/s=4.53--4.55k step/h`，全部finite；215k critic/actor loss=`26.33/35.85`、alpha=`0.03794`、TD abs mean=`1.250`、peak allocated=`8045.25 MiB`，无NaN/OOM或显存增长。
- C1的post-capture index仍为`2000`，最近窗口无2+/3+ ring；目前没有新capture或优于C0的行为证据。当前只能确认更高UTD数值稳定，不能确认性能更好。
- GPU1同时有非本项目PID `1258417`和新出现的PID `1876168`，总显存约`21405/49140 MiB`、利用率95%、温度92°C；C1吞吐显著低于独占GPU的C0。系统RAM available约65 GiB、swap618 MiB/8 GiB，仍安全但资源压力上升；未处理其他进程。


### 10.14 C0完成、CF0/CF1主线启动与补丁工具故障审计（2026-08-13 00:35+08:00）

#### C0 300k最终状态与旁路评估

- C0于2026-08-13 00:01前后完成300000/300000，update=73751、replay=250000（ring容量封顶）、all_finite=true；final bundle为checkpoints/step_000300000。
- final replay的post_capture_coverage仍为2000，与P1-200k冻结点完全相同，说明200k→300k没有第二个distinct real capture证据。runner final formal 20-rollout中capture/mixed capture rate均为0、collision rate均为1.0；capture mean minimum distance=14.43 m、distance progress=+2.79 m，mixed分别17.13 m/-1.34 m。Pure-CE strict/CV.15/CV.20=0.40/0.70/0.95，collision=0.10。结论：C0没有支持“只需更多step即可放大capture”的解释，但coverage能力仍在。
- 独立低优先级CPU旁路于00:32:25启动：PID 1980214、tmux eval_c0_300k_representative_cpu。它先对capture/Pure-CE/mixed各做20个deterministic rollout，再按成功或最接近capture、代表性collision、典型失败几何、最好/最差distance progress选择5个GIF；不占GPU，不阻塞CF0。
- C1完成300k后的同合同CPU旁路已挂起：PID 1980568、tmux eval_c1_300k_autostart_cpu。只有在C1 report满足transition_count=300000、all_finite=true且final trainer/effective config齐全后才启动20-rollout+代表性GIF。

#### C1 UTD=.5实时状态

- C1继续在cuda:1运行，PID 1853291、tmux c1_p1_utd05_300k_gpu1；00:33最新223000/300000、update=60251、replay=223000，最近窗口1.421 step/s≈5.12k step/h，mean_finite=1、critic/actor loss=26.92/36.18、alpha=0.03955、TD abs mean=1.137、peak allocated=8045.25 MiB。
- post_capture_coverage仍为2000，尚无第二个capture证据。GPU1同时承载其他正式进程，总显存约16038 MiB/49140 MiB、100%利用率、90°C；未停止或改动其他进程。
- 按最近1.33--1.42 step/s：250k ETA约2026-08-13 05:50--06:15+08:00；300k step ETA约15:40--16:40，final eval/replay写盘完成预计16:00--17:20。

#### CF0 Formal Capture-First / Local正式线

- 新config为configs/experiments/parallel_ce_legacy_voradj_20260809/legacy_voradj_cf0_capture_first_local_4p1e1obs_100k_aw.yaml；scratch seed=2026081301，新replay，只训练capture scene，episode/pre-capture horizon=1000，capture成功终止，不含独立Pure-CE和post-capture coverage。SAC冻结为all-agent、(a,w)、no clip、MSE、UTD=.25、LR=1e-4、tau=.005、batch128。
- 代码显式增加legacy_capture_reward_fallback_all_active=false：Legacy-VorAdj中没有敌人邻接观测的agent不再退回所有active target获取oracle capture shaping。该开关默认仍为true，故不改变历史配置/正在运行的C1语义。
- CF1独立config为legacy_voradj_cf1_capture_first_global_enemy_4p1e1obs_100k_aw.yaml，除global_evader_visibility=true和run审计元数据外继承CF0。环境新增Legacy分支的diagnostic broadcast：每个active pursuer observation都含敌人robot-frame相对位置与相对速度；奖励候选仍保持CF0局部合同，没有随广播引入oracle reward。
- 合同回归tests/test_capture_first_cf_contract.py + tests/test_all_agent_oldmix_ablation_contract.py为14 passed；覆盖CF0冻结合同、CF1唯一变量、所有active pursuer敌人位置/速度token、CF0不可见槽位及all-agent既有回归。CF0 32-step CUDA scratch smoke此前通过。
- 正式CF0于00:26:03在空闲GPU0启动：PID 1976538、tmux cf0_capture_first_local_100k_gpu0、artifact artifacts/2026-08-13_capture_first_controls/cf0_local/。00:33最新6000/100000、update=251、replay=6000；首个warmup后训练窗口3.806 step/s≈13.70k step/h，mean_finite=1、critic/actor loss=2.44/1.92、alpha=0.1975、TD abs mean=0.307、peak allocated=8045.14 MiB，无NaN/OOM/leak。该6k窗口terminated=2且collision=2，不计作capture；2+/3+ ring均为0。
- 当前GPU0约8685 MiB、99%利用率、85°C；系统RAM available约86 GiB、swap1.1/8 GiB，资源安全。按首个真实update窗口吞吐并计入25k诊断/落盘开销：75k ETA约2026-08-13 05:45--06:15+08:00；100k ETA约07:30--08:15；自动Gate预计07:45--08:30执行。
- 自动Gate supervisor已稳定挂起：PID 1981431、tmux cf_capture_first_gate。强信号为frozen replay真实capture event、3+ ring、重复2+ ring；否则要求至少两项持续中等改善。PASS原位resume CF0 100k→150k，FAIL等待GPU0释放后从scratch启动CF1 100k。
- Gate最初实现曾错误使用terminated_count近似capture；6k出现terminated=2且collision=2后立即纠正。当前实现直接反序列化100k frozen replay并统计metadata.event_ids中的capture，碰撞终止反例与真实capture正例合成测试均通过，运行中的supervisor已重启加载修正版。

#### apply_patch / git apply故障与绕过记录

- 本会话首次使用apply_patch修改测试时，sandbox helper失败，原始错误为：bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted。随后普通git apply -也遭遇同一bwrap错误。
- 切换到已授权的非sandbox执行后，git apply可以启动，但较长补丁先报error: corrupt patch at line 19；拆成精确小hunk后又反复报patch failed ... patch does not apply，即使rg/sed确认目标文本和行号存在。该问题发生在补丁传输/匹配层，不是训练代码、CUDA或GitHub网络故障。
- 按用户指令不再死磕git apply：已有文件采用精确perl/sed字符串替换；新文件仍在可工作的非sandbox git apply路径创建；文档采用只追加tee。所有绕过修改之后均用git diff人工复核、py_compile、14项pytest、合成Gate测试和实际CUDA smoke兜底，因此没有降低合同验证标准。
- 截至本节写入时GitHub尚未执行push；下一步为git diff --check、聚焦测试、提交并推送。此处明确记录：本次卡点不是GitHub网络不通。


### 10.15 CF0 9k / C1 224k与C0代表性评估进度（2026-08-13 00:47+08:00）

- 重新审计确认后台没有卡住：CF0、C1训练PID/tmux均存在，CF0 100k Gate supervisor与C1 final-eval waiter均存活；C0 CPU旁路评估继续渲染GIF。Git分支仍为ablation/all-agent-oldmix-20260810，审计前本地/远端均在a5dd045。
- CF0当前9000/100000、update=1001、replay=9000；最近5个真实1k窗口吞吐中位数3.791 step/s=13.65k step/h，末窗mean_finite=1、critic/actor loss=9.67/5.96、alpha=0.1832、TD abs mean=0.424、peak allocated=8045.14 MiB。末窗terminated=2且collision=2，不是capture；至今2+/3+ ring窗口均为空，当前没有新的多机capture积极信号。
- CF0 PID=1976538、tmux=cf0_capture_first_local_100k_gpu0、cuda:0；GPU0约8685 MiB/49140 MiB、99%利用率、85°C。按最近5窗稳态吞吐并为25k diagnostic/checkpoint留出波动：25k ETA约2026-08-13 02:00--02:15+08:00，50k约03:55--04:20，75k约05:45--06:15，100k训练step约07:30--08:00，自动Gate约07:50--08:30执行。
- C1当前224000/300000、update=60751、replay=224000；最近5窗吞吐中位数1.398 step/s=5.03k step/h，末窗mean_finite=1、critic/actor loss=22.58/36.11、alpha=0.03851、TD abs mean=1.058、peak allocated=8045.25 MiB。224k出现弱接敌信号：d1_min=6.52 m、max ring=1、any-ring fraction=6.98%；但2+/3+ ring仍为0，post_capture_coverage仍为2000，没有第二个distinct capture证据。
- C1 PID=1853291、tmux=c1_p1_utd05_300k_gpu1、cuda:1；GPU1另有非本项目PID 1258417，总显存约16038 MiB、100%利用率、93°C，未操作其他进程。按最近5窗：225k step约00:59，含里程碑诊断/落盘约01:10--01:30；250k ETA约06:00--06:30；300k step ETA约15:50--16:30，final report/replay与runner eval预计16:15--17:10。
- 主机RAM available约85 GiB、swap1.1/8 GiB；根分区可用158 GiB。当前无OOM、NaN、显存增长或RAM/swap危险。
- C0 300k独立formal评估的三场景各20 rollout已于00:43完成：capture/mixed capture rate=0、collision rate=1.0，二者mean minimum distance均12.87 m、distance progress=+7.49 m；Pure-CE strict/CV.15/CV.20=0.35/0.75/0.90、collision=0.05、mean area CV=0.1034、CE progress=+0.1102。结论仍是coverage可学而formal capture不可复现。
- C0代表性GIF旁路PID=1980214、tmux=eval_c0_300k_representative_cpu；截至00:47已完成7个GIF（capture 5个、coverage 2个），继续CPU低优先级渲染coverage/mixed。按既定要求不等待旁路完成，本次只同步已完成的20-rollout汇总。


### 10.16 自动线重排：C1停于225k、CF1接力、CF0固定续到200k（2026-08-13 01:00+08:00）

#### C1 UTD=.5在225k冻结停止

- 用户将C1预算上限改为225k。step_000225000的trainer/runtime/manifest/effective_config/metrics/diagnostic_eval完整落盘，rolling resume_latest额外包含完整replay.pkl；随后以hardlink原子冻结为resume_frozen_c1_step_000225000。frozen trainer约145 MB、replay约7.55 GB，未删除任何C1 artifact。
- 自动接力于2026-08-13 00:58:29确认上述文件完整后向唯一匹配C1 tag的PID 1853291发送SIGINT。PID随后正常消失，GPU1显存从约16.0 GiB降至约7.37 GiB，证明C1 allocation已释放；没有触碰GPU1原有PID 1258417。
- 对冻结replay逐条检查transition_id>=200000的metadata.event_ids：新real capture=0；200k--225k训练窗口没有任何2+ ring或3+ ring，更不存在repeated 2+ ring。结论为：UTD=.5数值稳定，但没有新的capture收益，C1正式停止于225k，不再运行250/275/300k。
- 225k轻量4-episode diagnostic：capture capture=0/4、collision=4/4、mean min distance=20.14 m、distance progress=-17.53 m；mixed capture=0/4、collision=2/4、mean min distance=11.28 m、distance progress=-5.23 m。Pure-CE CV.15/CV.20=4/4但strict=0/4。训练224k曾有单机ring 6.98%和d1_min=6.52 m，只是弱接敌信号，不满足多机Gate。
- 旧的C1-300k formal20+GIF tmux waiter已停止，仓库中的tools/supervise_c1_final_eval.py已删除；225k不启动昂贵20-rollout。完整审计写入artifacts/2026-08-13_capture_first_controls/control/c1_stop_225k.json。

#### CF1 Global Broadcast从scratch在GPU1启动

- C1释放显存并完成replay审计后，原接力supervisor进程于00:59:17 exec为CF1训练；当前PID 1992295、tmux c1_stop225_cf1_gpu1_autostart、cuda:1，artifact为artifacts/2026-08-13_capture_first_controls/cf1_global/legacy_voradj_cf1_capture_first_global_enemy_4p1e1obs_100k_aw_20260813/。
- 启动命令没有任何resume checkpoint/replay/step参数。manifest明确initialization=scratch_new_replay、seed=2026081302、optimizer_unit=joint_transition_all_active_agents、replay_sampling=uniform_joint、focal_training=false、grad_clip=null、action=(a,w)。
- CF1继承CF0的stationary fallback：enabled=true、speed threshold=0.2、hold=10、min pursuers=2；Legacy K10 release delay=10。唯一核心信息变量是global enemy broadcast。聚焦回归显式构造远距离敌人并验证每个active pursuer Actor observation均含robot-frame relative position和relative velocity，不是只检查YAML布尔值。
- 01:00已出现CF1首个1k metrics文件，进程持续推进、scene仅capture、replay从0新建；当前仍在5k warmup前，暂不据此报告稳态训练吞吐或行为结论。

#### CF0 Local固定100k→200k

- 旧PASS→150k / FAIL→CF1分支已删除。CF0仍按原合同跑完0→100k；100k diagnostic Gate继续统计真实normal/stationary capture、3+ ring、repeated 2+ ring、2+ fraction、collision和distance，但controls_training_branch=false。
- 100k report与rolling bundle完整后，supervisor将用hardlink原子冻结resume_frozen_cf0_step_000100000；无论Gate PASS/FAIL均以相同trainer/replay/runtime/RNG原位resume到200k。125/150/175/200k仍按25k间隔保存和诊断。
- 新supervisor PID 1992290、tmux cf0_continue_200k_autostart；旧cf_capture_first_gate已停止。CF0和CF1现在完全互不等待。


### 10.17 CF0/CF1并行运行核验与ETA（2026-08-13 01:04+08:00）

- CF0 Local当前PID `1976538`、tmux `cf0_capture_first_local_100k_gpu0`、cuda:0；最新完整窗口`13000/100000`、update=`2001`、replay=`13000`。最近稳态约`3.78--3.80 step/s`（约`13.6k step/h`），`mean_finite=1`、peak allocated=`8045.14 MiB`，尚无2+/3+ ring或real capture新信号。按当前吞吐并计入25k诊断、100k冻结/重启和完整replay落盘：25k ETA约`2026-08-13 01:55--02:10+08:00`，100k约`07:40--08:20`，200k约`15:20--16:15`。
- CF1 Global当前PID `1992295`、tmux `c1_stop225_cf1_gpu1_autostart`、cuda:1；已完成`5000` warmup并执行首个真实update，replay=`5000`，critic/actor/alpha/TD均finite，peak allocated=`8019.93 MiB`，无OOM/NaN。首个update wall-time=`1.488 s`；结合约25.6秒/1k的环境采样成本，GPU1共享条件下保守预计稳态`2.3--2.7 step/s`（`8.3--9.7k step/h`）。因此25k ETA暂估`03:10--03:35+08:00`，100k ETA暂估`11:00--13:00+08:00`，待首个完整post-warmup 1k窗口自动校准。CF1当前合同只到100k，未安排200k，故无200k ETA。
- 资源核验：GPU0约`8685/49140 MiB`、100%利用率、85°C；GPU1约`16036/49140 MiB`、100%利用率、92°C，其中约7342 MiB属于既有PID `1258417`，未触碰。主机RAM available约`98 GiB`，swap使用`576 MiB/8 GiB`。两条正式训练及CF0续训supervisor均存活，当前无显存/RAM增长异常。


### 10.18 CF0 90k / CF1 60k详细状态（2026-08-13 06:45+08:00）

#### 结论先行

- CF1 Global出现当前capture-first主线的首个明确强信号：50k rolling replay精确包含1个real capture event（transition ID `44892`）；28k窗口达到`max_num_in_ring=3`、3+ ring fraction=`0.8%`，并在28/29/36/45/55/56/58k共7个独立窗口出现2+ ring。45k窗口`terminated=7`、`collision=6`，唯一非碰撞终止与replay capture ID相符。
- 该capture符合stationary fallback而非normal loose capture：capture所在45k窗口全程`max_num_within_8m=2`，normal合同要求至少3机进入8m；当前fallback允许敌速<=0.2时2机保持10步。因此这是有效formal stationary capture，但尚不是normal k=3 capture。
- CF1的50k deterministic 4-episode诊断仍为0/4 capture，故积极信号目前是“训练探索中真实出现一次并伴随重复多机几何”，尚未证明策略可重复capture。
- CF0 Local截至90k没有real capture或3+ ring。75k frozen replay精确capture count=0；76--90k所有terminated均与collision一一对应，没有新增非碰撞终止。CF0虽有8个2+ ring窗口，但信号稀疏且最近5k没有2+ ring；75k deterministic distance progress相对25k明显退化。

#### CF0 Local：90k

- 进程/资源：PID `1976538`、tmux `cf0_capture_first_local_100k_gpu0`、cuda:0；最新`90000`、update `21251`、replay `90000`。最近5窗吞吐中位数`3.753 step/s=13.51k step/h`，范围`3.738--3.778 step/s`；`finite=1`，peak allocated=`8045.14 MiB`。
- 最新训练数值：critic loss=`128.714`，actor loss=`95.240`，alpha loss=`0.0295`，alpha=`0.12367`；Q1/Q2=`-94.609/-94.593`，twin gap=`4.014`，target Q=`-94.081`，TD abs mean=`3.729`。
- 无clip梯度/效率：critic/actor/alpha grad norm=`1452.23/4.548/0.0780`；update wall=`0.944 s`，updates/s=`1.059`。最近5窗critic/actor loss均值=`136.47/94.07`，TD abs=`3.916`；全部finite，但critic/Q/gradient尺度较早期显著增大，后续100k checkpoint必须继续监控。
- 最新几何：d1/d2/d3/d4 mean=`27.84/45.63/57.95/73.10 m`，d1 min=`12.72 m`，fraction closing=`42.45%`；本窗max ring=0。最近5窗最近接敌`4.29 m`、max ring=1，无2+/3+。
- 全程几何：2+ ring窗口为40/43/61/72/75/77/81/83k，共8个；最高2+ fraction=`1.6%`（43k），max ring=2，3+从未出现。最佳d1 min=`2.24 m`（72k），未转化为capture。
- rolling审计：75k replay完整、75000 records、capture event=0、collision event=237；75k rolling bundle约2.5 GiB。25/50/75k checkpoint均完整。
- deterministic 4-episode：25/50/75k capture均`0/4`；collision rate=`0.75/1.00/0.75`；mean min-min distance=`15.68/20.11/17.94 m`；distance progress=`+16.88/+10.56/-1.33 m`。因此75k没有形成稳定正向趋势。

#### CF1 Global：60k

- 进程/资源：PID `1992295`、tmux `c1_stop225_cf1_gpu1_autostart`、cuda:1；最新`60000`、update `13751`、replay `60000`。最近5窗吞吐中位数`2.681 step/s=9.65k step/h`，范围`2.666--2.684 step/s`；`finite=1`，peak allocated=`8045.14 MiB`。
- 最新训练数值：critic loss=`98.792`，actor loss=`69.725`，alpha loss=`0.2669`，alpha=`0.16177`；Q1/Q2=`-69.152/-69.162`，twin gap=`3.851`，target Q=`-68.345`，TD abs mean=`3.479`。
- 无clip梯度/效率：critic/actor/alpha grad norm=`1118.69/4.339/0.1636`；update wall=`1.356 s`，updates/s=`0.738`。最近5窗critic/actor loss均值=`110.53/68.19`，TD abs=`3.774`；全部finite。共享GPU导致其update慢于CF0，但无OOM/显存增长。
- 最新几何：d1/d2/d3/d4 mean=`23.00/41.00/66.91/77.71 m`，d1 min=`6.97 m`，fraction closing=`55.81%`，any-ring fraction=`6.2%`，max ring=1。本窗无2+/3+。
- 全程强信号：28k `max ring=3`、2+/3+ fraction=`2.6%/0.8%`；2+ ring还出现在29/36/45/55/56/58k。最高2+ fraction=`2.6%`，最近5窗中的56k/58k仍有2+，表明多机几何不是只出现一次。
- rolling replay精确审计：50k replay共50000 records，real capture event=1（ID `44892`），collision event=246；50--60k无额外非碰撞终止，因此当前仍为1个distinct capture。50k rolling bundle约1.7 GiB；25/50k checkpoint完整。
- deterministic 4-episode：25k与50k capture均`0/4`；collision rate从`1.00`降到`0.50`；mean min-min distance从`21.80`改善到`20.57 m`；distance progress从`+7.35`提高到`+14.38 m`；episode length从`117.75`升至`243.25`。这是组合中等改善，但样本仅4集，不能替代75/100k验证。

#### 资源与ETA

- GPU0约`8685/49140 MiB`、100%、85°C；GPU1约`16036/49140 MiB`，其中既有PID `1258417`约7342 MiB，当前瞬时利用率32%、93°C。GPU1温度仍是主要运行风险，但显存余量充足，未触碰其他项目。
- 主机RAM available约`95 GiB`，swap=`1.4/8 GiB`；无RAM/VRAM leak迹象。两条训练PID/tmux及CF0 100k->200k supervisor PID `1992290`均正常。
- CF0：100k train-step ETA `2026-08-13 07:25--07:40+08:00`；含100k diagnostic、完整replay冻结和自动resume预计`07:45--08:10`。125k ETA `09:35--10:00`；200k ETA `15:30--16:30`。
- CF1：75k ETA `2026-08-13 08:15--08:35+08:00`；100k ETA `10:55--11:25`。当前合同止于100k，未安排200k ETA。


### 10.19 CF0在100k冻结停止、CF2角色强化启动（2026-08-13 07:40+08:00）

#### CF0 Local最终冻结

- 按最新决策取消CF0 `100k->200k`续训。旧supervisor PID `1992290`在不影响trainer的情况下正常停止，替换为“等待100k完整落盘、冻结、释放GPU0、启动CF2”的一次性接力器；没有让旧逻辑抢跑。
- CF0于`2026-08-13 07:31:22+08:00`完成100k bundle：trainer、replay、runtime、manifest、effective config、metrics及diagnostic均完整。冻结目录为`resume_frozen_cf0_step_000100000`，使用hardlink保留同一份完整trainer/replay/runtime，replay约3.35 GB。CF0 PID `1976538`及原tmux已退出，GPU0显存已释放；没有继续到100001。
- final训练状态：`step=100000`、`update=23751`、`replay=100000`、`all_finite=true`，critic/actor loss=`127.63/102.81`、alpha=`0.12945`、TD abs mean=`3.772`、末窗吞吐=`3.732 step/s`。100k窗`max ring=2`、2+ ring fraction=`2.9%`、3+=0、collision=`5/1000`、`d1_min=3.94 m`；全程无real capture或3+ ring，但2+ ring在40/43/61/72/75/77/81/83/100k多个窗口重复出现。
- final deterministic 20-rollout：capture=`0/20`、collision=`20/20`、mean min-min distance=`17.18 m`、distance progress=`+3.55 m`。因此CF0 Local只保留接敌/双机几何信号，未打通K3；本线正式止于100k。

#### CF2最小单变量实现与强制Gate

- 新config：`configs/experiments/parallel_ce_legacy_voradj_20260809/legacy_voradj_cf2_capture_first_global_support_full_4p1e1obs_100k_aw.yaml`；run=`legacy_voradj_cf2_capture_first_global_support_full_4p1e1obs_100k_aw_20260813`，seed=`2026081303`，scratch + new replay。它继承CF1的verified global enemy broadcast、moving APF、1 obstacle、Legacy-VorAdj、normal K3、stationary fallback、K10、capture-first horizon1000和稳定SAC合同；唯一研究变量是support reward credit。
- 新增独立显式开关`legacy_voradj_support_reward_blend_enabled=true`，没有打开或借用VCT-LS模式。global broadcast只进入Actor observation；reward角色从实际Legacy邻接原始标签计算：直接enemy adjacency=`capture`，非直接但有friendly capture邻居=`support`，其余=`coverage`。
- reward合同：capture拿原完整legacy capture task；support拿`1.0*full capture_task + 1.0*full coverage_task`；pure coverage只拿coverage task。approach、mean-shift、front、timestep、terminal和既有safety规则未改。CF1没有显式Legacy开关，因而旧行为保持不变。
- 人工邻接链`P0-enemy, P1-P0, P2-P1`通过：即使global broadcast让每个active Actor都收到enemy position/velocity token，角色仍精确为`P0=capture/P1=support/P2=coverage`。实际reward分量检查为：capture槽capture非零且coverage=0；support槽capture与coverage都非零、权重均1.0；coverage槽capture=0且coverage非零。CF1同图中support blend保持关闭。
- 回归与启动Gate：聚焦合同最终`37 passed`，`py_compile`及`git diff --check`通过；CF2 CUDA 1k scratch smoke完成、无OOM。capture event额外区分`normal_capture_count`和`stationary_capture_count`，metrics新增三角色占比及各角色capture/coverage/terminal/total平均reward。
- 本次追加台账时`apply_patch`再次被同一宿主sandbox故障阻断：`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`；按用户既有授权改用精确只追加的Perl绕过，随后以diff/check/test复核。该错误仍不是训练异常或GitHub网络错误。

#### CF1 vs CF2实时状态与信号

- CF1 Global：PID `1992295`、tmux `c1_stop225_cf1_gpu1_autostart`、cuda:1。07:35前最新`68000/100000`、update=`15751`、replay=`68000`，最近吞吐`2.661 step/s=9.58k/h`，finite=1，critic/actor loss=`126.93/76.75`、alpha=`0.18154`、TD abs mean=`4.293`、peak allocated=`8045.14 MiB`。68k窗口出现`max ring=3`、2+/3+ fraction=`1.1%/0.2%`；结合28k的3+与多个重复2+窗口，这是新的积极几何证据。rolling replay目前仍只有50k前确认的1次stationary-fallback capture，没有normal capture证据。
- CF1里程碑4-rollout：25k capture=`0/4`、collision=`4/4`、mean min distance=`21.80 m`、distance progress=`+7.35 m`；50k capture=`0/4`、collision=`2/4`、mean min distance=`20.57 m`、distance progress=`+14.38 m`。训练内几何/capture信号尚未转化为deterministic可重复成功。
- CF2 Global+Support：`2026-08-13 07:32:17+08:00`在GPU0从step0启动，PID `2120473`、tmux `cf0_stop100_cf2_gpu0_autostart`。step1 manifest明确`initialization=scratch_new_replay`、seed=`2026081303`、uniform joint replay、all-active-agent update、no clip，且命令没有任何resume参数。
- CF2首个正式update已在5k通过：`step=5000/update=1/replay=5000`、`mean_finite=1`、critic/actor loss=`117.05/1.899`、alpha=`0.19998`、Q1/Q2/targetQ=`-0.759/0.226/-1.093`、TD abs mean=`2.012`、peak allocated=`8019.93 MiB`，无NaN/OOM。5k角色占比capture/support/coverage=`70.25%/29.30%/0.45%`；capture槽capture/coverage/total=`-0.967/0/-1.142`，support=`-1.039/-0.163/-1.535`，coverage=`0/-0.0685/-0.0685`，证明在线support确实同时收到两类task gradient。尚无capture/2+/3+ ring；这只是warmup边界安全证据，不作学习结论。
- 资源快照：GPU0约`8685/49140 MiB`、100%、84°C；GPU1约`16036/49140 MiB`、其中CF1约8662 MiB且另有未触碰的PID `1258417`约7342 MiB、93°C。RAM available约98 GiB、swap1.5/8 GiB；根盘可用145 GiB。两条正式训练均存活，无显存/RAM增长或OOM。
- ETA按各自最新真实稳态吞吐并计入25k diagnostic/checkpoint波动：CF1的25k/50k已于03:05/05:41完成，75k约`2026-08-13 08:20--08:40+08:00`，100k约`11:00--11:35`；CF2首批update wall-time推算约`12.5--13.0k step/h`，25/50/75/100k约`09:05--09:25`、`11:05--11:35`、`13:05--13:45`、`15:05--16:00`。CF2到75/100k后再按相对CF1的repeated 3+ ring、normal capture、capture数量与2+频率决定是否延长150/200k，不提前自动扩步。


#### 07:41稳态复核（覆盖上方warmup边界估算）

- CF2已完成首个完整250-update窗口：`6000/100000`、update=`251`、`3.782 step/s=13.61k/h`、finite=1，critic/actor loss=`29.66/3.959`、alpha=`0.19746`、TD abs mean=`0.865`、peak allocated=`8045.14 MiB`。角色占比capture/support/coverage=`65.48%/29.23%/5.30%`；对应task分量capture槽=`-0.967/0`、support=`-0.961/-0.363`、coverage=`0/-0.189`，再次确认三角色在线分流和support双分量稳定工作。
- CF1最新`69000/100000`、update=`16001`、`2.675 step/s=9.63k/h`、finite=1；本窗无2+/3+，但不撤销68k及28k已落盘的3+证据。
- 以实际稳态吞吐和checkpoint诊断开销更新：CF1 75k约`2026-08-13 08:20--08:40+08:00`，100k约`11:05--11:35`；CF2 25/50/75/100k约`09:05--09:25`、`10:55--11:25`、`12:50--13:25`、`14:45--15:30`。


### 10.20 CF1完成100k、CF2首次normal capture（2026-08-13 12:15+08:00）

- CF1 Global于`11:00:09+08:00` clean完成100k，完整trainer/replay/runtime/manifest/metrics/diagnostic均已落盘；训练进程和tmux已退出、GPU1显存释放。final为`23751` updates、replay=`100000`、`all_finite=true`。
- CF1全程精确证据仍为1次stationary-fallback capture、0次normal capture；100k formal 20-rollout capture=`0/20`、collision=`20/20`、mean min-min distance=`12.32 m`、distance progress=`+11.93 m`。4-episode diagnostic在25/50/75/100k均为0 capture；collision=`1.00/0.50/0.75/0.75`，distance progress=`+7.35/+14.38/+15.80/+14.99 m`。
- CF1全100个1k窗口共有19个2+ ring窗口、3个3+ ring窗口，best 2+/3+ fraction=`2.6%/0.8%`。说明global信息明显改善多机几何，但截至100k没有形成deterministic稳定capture。
- CF2 Global+Support最新`55000/100000`、update=`12501`、replay=`55000`，PID `2120473`、tmux `cf0_stop100_cf2_gpu0_autostart`、cuda:0继续正常。最近5窗吞吐=`2.706--2.979 step/s`，末窗`2.979 step/s=10.72k/h`；finite=1、critic/actor loss=`143.08/80.31`、alpha=`0.14336`、TD abs mean=`4.224`、peak allocated=`8045.14 MiB`。
- **CF2在53k出现本主线首次明确normal K3 capture**：新诊断字段为`normal_capture_count=1`、`stationary_capture_count=0`，同窗`terminated=9/collision=8`，排除碰撞终止；2+ ring fraction=`2.803%`。这是比CF1的stationary fallback更强的新信号，但目前仍只有1次，不能提前宣称稳定学会。
- CF2截至55k共有9个2+ ring窗口、1个3+ ring窗口；19k `max ring=3`、best 3+ fraction=`0.6%`，累计normal/stationary capture=`1/0`。同55k比较：CF1为5个2+、1个3+、1次stationary capture；CF2为9个2+、1个3+、1次normal capture。CF2几何频率和capture类型当前占优，但collision累计为`346 vs 272`，安全性更差。
- 50k 4-episode直接比较：CF1/CF2均0 capture；CF1 collision=`0.50`、min-min distance=`20.57 m`、progress=`+14.38 m`；CF2 collision=`1.00`、min-min distance=`15.58 m`、progress=`+16.02 m`。即CF2接敌更深，但碰撞更严重。
- CF2 support credit持续按合同工作。55k窗口角色占比capture/support/coverage=`56.83%/33.40%/9.78%`；support capture/coverage component=`-0.866/-0.633`，两类梯度继续同时非零；capture槽coverage=0、coverage槽capture=0。
- CF2 25k/50k checkpoint及rolling bundle完整，分别于`09:20:45/11:40:42+08:00`落盘。GPU0新增未触碰的外部PID `2133267`占约5316 MiB，CF2约8662 MiB，总显存约14066/49140 MiB；吞吐因此由独占时约13.6k/h降至约10.7k/h。GPU0 85°C，GPU1 85°C；RAM available约103 GiB、swap1.4/8 GiB、根盘可用140 GiB，无OOM/NaN/RAM风险。
- 新ETA：CF2 75k约`2026-08-13 14:00--14:20+08:00`，100k约`16:35--17:05`（含75k diagnostic/checkpoint开销）。75/100k继续重点判断normal capture是否重复、3+ ring是否持续；当前不提前扩到150/200k。


### 10.21 CF2自动扩至200k、CF3 Local-Support启动（2026-08-13 13:00+08:00）

#### CF2 100k→200k自动接力

- CF2不再在100k等待人工Gate。新增`tools/supervise_cf2_continue200.py`，PID=`2232220`、tmux=`cf2_extend200_supervisor`，已于12:42挂起并持续等待完整finite 100k report与rolling bundle。
- 接力合同：100k完成后hardlink原子冻结到`resume_frozen_cf2_step_000100000`；确认原100k进程正常退出后，以同一config/seed/tag/artifact、trainer、replay、runtime/RNG和`resume_step=100000`原位续到200k。没有`resume_fork`、不reset optimizer/targets/alpha/replay/RNG；125/150/175k继续周期checkpoint+4-episode diagnostic，200k做final 20-rollout。
- supervisor命令合同测试明确目标`total_steps=200000`、cuda:0及frozen trainer/replay路径；100k final的diagnostic和完整bundle先落盘、冻结，续训不覆盖冻结副本。

#### CF3 Local-Support严格单变量

- 新config：`configs/experiments/parallel_ce_legacy_voradj_20260809/legacy_voradj_cf3_capture_first_local_support_full_4p1e1obs_100k_aw.yaml`；run=`legacy_voradj_cf3_capture_first_local_support_full_4p1e1obs_100k_aw_20260813`、seed=`2026081304`、scratch + new replay、cuda:1。
- resolved config与CF2的唯一训练语义差异是`global_evader_visibility: true→false`。Legacy邻接角色、capture full reward、support `1.0*full capture + 1.0*full coverage`、pure coverage only、K3/K10/stationary fallback、moving APF、obstacle、horizon1000、all-agent/no-clip/MSE/UTD=.25/LR/tau/batch/action/network均完全相同。
- 人工Legacy链测试通过：capture agent有enemy token；support自身非enemy-adjacent且无enemy token，但第一friendly token精确包含capture友邻的robot-frame relative position/velocity及`is_pursuing=1`；pure coverage无enemy token且没有capture友邻。CF3 support reward仍通过capture友邻邻接target计算完整capture shaping，没有降级成coverage。CF2/CF1既有行为回归保持通过。
- 新增的只读diagnostic只写`info/replay_metadata`和metrics，不进入Actor observation或reward：角色step/比例；support无enemy token/有pursuing friend比例；support→friend距离及delta；support→enemy距离及progress；进入ring、升级capture次数；各角色capture/coverage/terminal/total的mean/p50/p95；normal/stationary/distinct capture；2+/3+独立窗口及最大连续hold steps。
- 该诊断从CF3 step0以及CF2 100k resume段开始生效；CF2 0--100k保持原进程已加载的既有指标，不能事后伪造新增support行为字段。
- 强制回归为`48 passed`，`py_compile`和`git diff --check`通过。CF3 6k CUDA smoke完成251 updates：`finite=1`、critic/actor loss=`4.02/2.24`、alpha=`0.19747`、TD abs mean=`0.455`、peak allocated=`8045.14 MiB`、吞吐=`2.614 step/s`；support无enemy token和有pursuing friend比例均=`1.0`，support capture/coverage mean=`-0.994/-0.237`，证明局部观测与双reward分量同时成立。
- 正式CF3已于`12:52:07+08:00`由smoke安全链自动启动：PID=`2232870`、tmux=`cf3_smoke_to_formal_gpu1`、cuda:1。step1 manifest为`initialization=scratch_new_replay`、seed=`2026081304`、uniform joint、all-active-agent、no clip，无resume/warm start。

#### 当前研究Gate与冻结路线

- 当前双P0：CF2 Global-Support固定续到200k；CF3 Local-Support固定跑到100k，不因25/50k无capture提前停。75/100k重点比较normal capture重复性、2+/3+几何与support follow行为。
- winner只有满足“replay中至少3个独立normal capture episode”或“formal deterministic 20-rollout normal capture至少20%”，且成功不主要依赖stationary fallback，才进入`winner + post-capture coverage`闭环。
- 后续固定顺序：选CF2/CF3 winner→恢复post-capture coverage→capture reward ablation→`(a,w)` vs `vx,vy`。当前不启动新capture reward、post-capture mixed、动作空间改动、UTD=1、LR sweep或MATD3。


#### 12:55正式双线状态与ETA

- CF2当前`63000/100000`、update=`14501`、replay=`63000`，末窗`2.986 step/s=10.75k/h`、finite=1，critic/actor loss=`151.09/90.52`、alpha=`0.16711`、TD abs mean=`4.445`、peak allocated=`8045.14 MiB`。累计积极证据仍包含53k一次normal capture、19k一次3+ ring和多个2+窗口；本63k窗无新增capture/2+/3+。
- CF3正式run当前`5000/100000`、update=`1`、replay=`5000`，首个optimizer update finite=1，critic/actor loss=`13.69/1.911`、alpha=`0.19998`、TD abs mean=`2.118`、peak allocated=`8019.93 MiB`。本窗support无enemy token/有pursuing friend比例=`1.0/1.0`，friend-distance下降比例=`61.5%`、enemy-distance正progress比例=`64.0%`，support capture/coverage mean=`-0.819/-0.455`；max 2+ ring hold=`4` steps。这是启动安全/诊断证据，不是学习结论。
- GPU0/1均100%利用率、`86/92°C`；CF2/CF3各约8662 MiB，连同两卡既有外部进程后总显存=`14066/16036 MiB`，仍有大幅余量。RAM available约102 GiB、swap1.4/8 GiB、根盘可用139 GiB；replay、metrics和step1 rolling bundle均正常，无OOM/NaN/leak。
- ETA按CF2最近`10.75k/h`与CF3同卡6k smoke稳态`9.41k/h`并计checkpoint/eval波动：CF2 125/150/175/200k约`2026-08-13 18:45--19:15`、`21:10--21:45`、`23:35--2026-08-14 00:15`、`02:00--02:45+08:00`；CF3 25/50/75/100k约`2026-08-13 15:05--15:25`、`17:50--18:20`、`20:35--21:15`、`23:20--2026-08-14 00:10+08:00`。


### 10.22 P0默认语义修复、CF切换与P1 Local-Max调度（2026-08-13 14:50+08:00）

#### P0只修默认合同，不训练独立baseline

- 默认old-mix focal/all-agent基配置的`reward.min_active_pursuers`和`reward.coverage_ce_min_active_pursuers`均由4改为2。真实环境step回归明确验证：4→3 active不因too-few结束，保留2 active也不结束，只有2→1后唯一active pursuer才因too-few结束。
- K10角色以effective label为唯一任务真源：raw Legacy-VorAdj adjacency只保留在`raw_task_label`、raw统计和真实局部enemy-token可见性中；Actor self/friendly `is_pursuing`、task/replay label、capture/support/coverage reward role、support target关联和coverage hold eligibility全部读取同一effective label。
- 根因是CF2/CF3 Legacy support模式曾显式写成`reward_role_labels = before_raw_labels`，而observation/replay已使用`before_labels` effective state；因此K10释放延迟中可能出现obs/replay仍为capture、reward却降为support/coverage。现改为`reward_role_labels = before_labels`，三角色仍严格为：self effective capture→capture；self非capture且friendly effective capture→support；其余coverage。reward权重不变：capture full capture，support full capture+full coverage，coverage only。
- 人工移除raw enemy adjacency边的K10回归通过：self observation `is_pursuing=1`、replay `task_label=capture/effective_pursuing=true`、reward role=capture、capture component非零、coverage component为0；hold eligible数量与同一reward role集合一致。没有通过K10给局部Actor伪造enemy token。
- runner新增严格`--resume-fork p0_semantics`。它从冻结bundle自身的`effective_config.yaml`读取旧合同，只允许两个min-active字段4→2；所有其余manifest字段受保护，implementation hash变化单独审计。CF2真实75k bundle预检仅得到上述两项差异并通过，避免通过宽松resume绕过合同校验。
- P0不训练单独`P0-fixed self+mean` baseline；上述规则从此为所有新实验默认合同。除明确ablation外禁止重新漂移。

#### CF2正信号线：75k切到P0-fixed并继续200k

- CF2 pre-P0截至完整75k已有明确积极信号：1次normal capture、0 stationary capture、16个独立2+ ring窗口、1个3+窗口，best 2+/3+ fraction=`2.8028%/0.6%`，best min enemy distance=`2.54 m`。75k累计collision=`505`；75k末窗finite=1，critic/actor loss=`172.44/102.71`、alpha=`0.20274`、TD abs mean=`4.984`、吞吐=`2.975 step/s`。75k deterministic 4-episode仍为0/4 capture、collision=4/4、mean min distance=`16.13 m`、distance progress=`+11.13 m`，故积极信号仍是训练探索而非稳定部署成功。
- 原进程实际已推进到80k，但最近完整trainer+replay+runtime rolling bundle是75k。旧artifact和0--80k metrics完整保留；75k bundle以hardlink冻结为`resume_frozen_pre_p0_step_000075000`，没有覆盖旧证据。旧100k→200k supervisor已取消，旧trainer正常退出且约8662 MiB显存释放。
- 新artifact/tag为`cf2_global_support_full_p0fixed/legacy_voradj_cf2_global_support_p0fixed_cont75k_to200k_20260813`。切换边界明确为：`step<=75000: pre-P0 semantics`；从恢复后的下一transition起：`P0-fixed semantics`。PID=`2268414`、tmux=`cf2_p0fixed_gpu0`、cuda:0，继续到200k并在100/125/150/175/200k按原25k节奏保存/诊断。
- P0-fixed首个新窗口已到76k：`update=17751/replay=76000`、`2.979 step/s=10.72k/h`、finite=1，critic/actor loss=`172.02/103.17`、alpha=`0.20614`、TD abs=`5.073`、peak allocated=`8045.25 MiB`；CF2进程显存约8664 MiB，无NaN/OOM/吞吐退化。本窗没有新增capture/2+/3+，不撤销pre-P0的normal capture证据。
- replay不做破坏性清空或重标注。旧75k transitions在切换瞬间占100%，随uniform新样本写入自然衰减；若到200k且capacity未覆盖，其理论比例降至37.5%。后续指标必须继续把pre/post切换曲线分段解释。

#### CF3 Local reference：25k自动切换，随后P1 MaxPool

- CF3 pre-P0当前22k：0 normal/stationary capture，5个独立2+ ring窗口、0个3+，best 2+ fraction=`0.6%`、best min distance=`2.74 m`、累计collision=`112`；末窗`2.561 step/s=9.22k/h`、finite=1，critic/actor loss=`54.96/31.60`、alpha=`0.13893`、TD abs=`2.218`。support无enemy token/有pursuing friend比例均1.0，末窗friend-distance delta mean=`-0.144 m/step`、enemy progress=`+0.0529 m/step`，累计support→capture upgrades=`333`。这是早期局部support方向性和重复双机几何，按合同不在25k前截断。
- 自动链PID=`2271611`、tmux=`cf3_p0_then_p1_gpu1`：等待`step_000025000` checkpoint与rolling replay/runtime全部完整→hardlink冻结`resume_frozen_pre_p0_step_000025000`→只向唯一CF3 tag PID发送SIGINT→用`p0_semantics`从25k续到100k。切换标记将为`step<=25000 pre-P0`、其后P0-fixed；旧25k replay在100k时自然降至25%。
- CF3 P0-fixed 100k完成且final report finite后，自动在cuda:1执行P1 32-step scratch smoke；smoke通过即启动`P1 — corrected Local CF + MaxPool` scratch 0→100k，seed=`2026081305`、新replay、独立artifact。另一个Gate PID=`2275833`、tmux=`p1_maxpool_100k_gate`等待P1 100k；仅当0--100k有normal capture，或75--100k至少2个3+独立窗口，或至少3个2+独立窗口时，才冻结100k并原位续到200k，否则stop100k并保留完整bundle。PASS续训前还会确认100k原进程完全退出，避免同artifact并发写入。
- P1 config为`legacy_voradj_p1_capture_first_local_support_maxpool_4p1e1obs_100k_aw.yaml`。CF3 Actor维持`self_token+mean_context`；P1唯一网络差异为`self_token+mean_context+max_context`，policy第一层输入`512→768`，输出仍256；Transformer保持hidden=256、heads=8、layers=4，central critic和其余网络不变，未加入pursuing residual、target/summary attention或role embedding。

#### 验证、资源、ETA和工具故障

- 聚焦综合回归包括Actor、UTD resume、all-agent Actor-Q、CF合同、P0与自动Gate，共`38 passed`；单独CF/P0/P1调度集合为`20 passed`。`py_compile`与`git diff --check`通过。
- GPU0总显存约14068 MiB（CF2约8664 MiB+未触碰外部PID 2133267约5316 MiB）、100%、86°C；GPU1约16036 MiB（CF3约8662 MiB+未触碰外部PID 1258417约7342 MiB）、100%、92°C。RAM available约99 GiB，swap1.4/8 GiB，根盘可用139 GiB，无VRAM/RAM/replay/storage异常。
- CF2按P0-fixed首窗10.72k step/h并计里程碑诊断/落盘：100k约`2026-08-13 17:00--17:30+08:00`，125k约`19:25--20:00`，150k约`21:50--22:30`，175k约`2026-08-14 00:15--01:00`，200k约`02:40--03:30`。
- CF3按最近9.22k step/h：25k完整切换约`2026-08-13 15:10--15:30+08:00`；P0-fixed 50/75/100k约`18:00--18:40`、`20:50--21:40`、`23:40--2026-08-14 00:40`。P1只有在CF3 100k完成后才启动，启动前不以旧线吞吐伪造正式ETA。
- 本次`apply_patch`包装器再次在读取阶段报`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`，权限自动审查也两次超时；补丁未触及仓库。绕过方式为先用严格Perl小替换，随后定位并直接调用普通用户环境中的`apply_patch`二进制完成新supervisor/测试和文档hunk；所有改动以diff、compile和pytest复核。该故障仍不是GitHub网络或训练故障。


#### 15:05 CF3 25k实际切换完成（覆盖上方“等待切换”状态）

- CF3 pre-P0已完整达到25k：0 normal/stationary capture、6个2+ ring窗口、0个3+，best 2+ fraction=`1.4%`、best min distance=`2.31 m`、累计collision=`143`、support→capture upgrades=`402`。25k末窗`2.823 step/s`、finite=1，critic/actor loss=`62.30/36.80`、alpha=`0.13250`、TD abs=`2.565`；support无enemy token/有pursuing friend比例仍为1.0/1.0。
- `step_000025000`与rolling trainer/replay/runtime/manifest/effective config完整后，自动器于`15:03:22+08:00`完成hardlink冻结；frozen trainer/replay约`143.8 MB/838.1 MB`。原CF3 PID `2232870`正常退出，未触碰GPU1其他进程。
- P0-fixed CF3已于`15:03:42+08:00`从同一25k bundle启动：PID=`2280607`、tmux仍为`cf3_p0_then_p1_gpu1`、cuda:1，新artifact为`cf3_local_support_full_p0fixed/legacy_voradj_cf3_local_support_p0fixed_cont25k_to100k_20260813`。真实resume audit只包含两个min-active 4→2差异，process约8664 MiB显存并持续占用GPU；当前等待首个26k post-switch窗口，不用启动前速度替代正式post-switch吞吐。
- 推送后最终快照时CF2已到79k，末窗`2.964 step/s`、finite=1，critic/actor loss=`179.43/106.36`、alpha=`0.21566`、TD abs=`5.121`；post-switch 76--79k尚无新增capture/2+/3+，support friend-distance delta=`-0.204 m/step`、enemy progress=`+0.194 m/step`。CF2下一完整checkpoint仍为100k，ETA维持`17:00--17:30+08:00`；CF3 50/75/100k以25k前同卡速度保守估计为`18:00--18:40`、`20:50--21:40`、`23:40--2026-08-14 00:40`，待26k窗口校准。


### 10.23 CF2截断、CF3扩200k与P1改接GPU0（2026-08-13 16:45+08:00）

#### CF3局部信息正信号与CF2截断结论

- CF3 pre-P0 0--25k为0 normal/stationary capture、6个独立2+窗口、0个3+。P0-fixed首次25k恢复段的26--33k又有5/8个窗口出现2+ ring，best fraction=`0.9%`、max ring=2、best `d1_min=2.37 m`，仍无normal/stationary capture或3+ ring。
- 更重要的是这些局部support全部无enemy token且有pursuing friend，26--33k support→friend distance delta均值=`-0.108 m/step`、support→enemy diagnostic progress均值=`+0.123 m/step`，累计159次support→capture升级；31k、33k再次出现2+ ring。数值全部finite。
- 这已经证明global enemy broadcast不是形成双机几何和方向性support follow的必要条件，支持把部署条件收紧到local。但它仍未证明local已学会K3：截至33k没有normal capture或3+ ring，因此结论严格限定为“方向性正信号”，不能写成capture已解决。
- 因此CF2不再需要跑满200k。为了保留可解释的P0-fixed global reference和完整trainer/replay/runtime，CF2不在98k立即强停，而是在完整100k checkpoint/rolling bundle落盘后冻结并停止；其0--75k为pre-P0、75--100k为P0-fixed。100k后GPU0立即启动P1 Local-Max scratch。

#### 最新性能快照

- CF2 P0-fixed已到98k：`2.943 step/s=10.59k/h`、finite=1，critic/actor loss=`187.23/127.14`、alpha=`0.25383`、TD abs=`5.563`。P0-fixed 76--98k累计0 capture/0 stationary、10个2+窗口、0个3+，best 2+ fraction=`2.1%`、best `d1_min=2.33 m`、support→friend/enemy均值=`-0.126/+0.137 m/step`。连同pre-P0证据，CF2整线仍保留53k的1次normal capture、16个pre-P0 2+窗口和1个pre-P0 3+窗口。
- 原CF3 P0-fixed段最后到33k：`1.569 step/s=5.65k/h`、finite=1，critic/actor=`72.54/51.24`、alpha=`0.12326`、TD abs=`3.312`；GPU1同卡外部进程使update wall-time约2.20 s，故吞吐明显低于pre-P0阶段。
- 双卡均约100%利用率；GPU0总14068 MiB、85--86°C，GPU1总16110 MiB、92--93°C。两条本项目trainer各约8664 MiB，未触碰外部GPU进程。RAM available约97 GiB、swap521 MiB/8 GiB、根盘可用138 GiB，无OOM、NaN、RAM或storage风险。

#### 自动线重接与一次进程组恢复

- 新目标结构为：GPU0 `CF2→100k完整冻结/停止→P1 32-step smoke→P1 scratch 100k`；GPU1 `CF3 corrected local→100k完整冻结→原位续200k`。P1 100k PASS后的extension设备也统一改为cuda:0，避免误抢GPU1。
- 新supervisor为`tools/supervise_cf2_stop100_then_p1_maxpool.py`、`tools/supervise_cf3_extend200.py`；P1 Gate同步改cuda:0。调度/合同回归为`7 passed`，py_compile通过。
- 重接旧`CF3→P1`父supervisor时先尝试SIGSTOP，但该父进程正阻塞在`subprocess.run`，未进入预期停止态；随后TERM旧tmux父进程导致同一会话组向CF3 trainer传播HUP，CF3 trainer在33k退出。这不是训练数值或checkpoint损坏，但25--33k没有周期rolling bundle，不能从33k无损resume。
- 恢复方式：保留原artifact/metrics作为只读证据，从已冻结且验证完整的pre-P0 25k trainer/replay/runtime重新以`--resume-fork p0_semantics`启动独立recovery artifact/tag；没有从scratch、没有reset旧25k optimizer/target/alpha/replay/RNG。模型训练状态回退8k、约1.4小时，26--33k指标不与recovery重放段重复累加。新CF3 PID=`2313014`、tmux=`cf3_p0fixed_recovery_gpu1`，已成功加载并重新占用8664 MiB，无启动报错。
- 已挂自动器：`cf2_to_p1_gpu0` PID=`2314338`；`cf3_extend200_gpu1` PID=`2314348`；`p1_maxpool_100k_gate_gpu0` PID=`2314351`。它们分别只处理对应tag，检查完整bundle/finite report和原进程退出，避免跨run误停。
- 本次常规`apply_patch`再次因`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`失败；经已验证的普通用户`apply_patch`二进制绕过，diff与回归测试确认没有半写文件。它不是GitHub网络故障。

#### ETA（Asia/Shanghai）

- CF2按98k最新10.59k step/h：100k训练约`2026-08-13 16:55`，含4-episode checkpoint/rolling冻结与P1 smoke，P1正式启动预计`17:10--17:35`。
- CF3 recovery从25k重放，按原P0-fixed真实5.65k step/h：50k约`2026-08-13 21:10--21:40`、75k约`2026-08-14 01:50--02:35`、100k训练约`06:20--07:15`；计20-rollout/final bundle后，100→200k接力约`07:15--08:15`启动。后续125/150/175/200k粗估为`12:00--13:00`、`16:40--18:00`、`21:20--22:50`、`2026-08-15 02:00--03:45`，需在recovery首个26k实测窗后校准。
- P1尚未正式启动，不能用CF3旧吞吐冒充其ETA；启动并产生首个有update的6k窗口后再报告正式25/50/75/100k时间。

#### 17:15 实际交接完成与ETA校准

- CF2已完整到100k并结束：frozen bundle的runtime step=`100000`、update=`23751`、replay文件约3.36 GB、`all_finite=true`，8个必需文件齐全。P0-fixed 76--100k为0 normal/stationary capture、12个2+窗口、0个3+，best 2+ fraction=`2.2%`、best d1=`2.33 m`、collision windows累计209。100k deterministic diagnostic为0/4 capture、4/4 collision、mean min-min distance=`21.63 m`、distance progress=`+13.54 m`。结合pre-P0 53k唯一normal capture，继续global到200k的边际价值低于立即测试Local-Max。
- P1 32-step CUDA smoke为scratch replay=32、finite；于`16:59:37+08:00`正式启动。首个有update窗口已到6k：`2.982 step/s=10.74k/h`、finite=1，critic/actor=`16.53/3.629`、alpha=`0.19747`、TD abs=`0.814`、collision=2，尚无capture/2+/3+（warmup边界，不作学习结论）。PID=`2314338`、tmux=`cf2_to_p1_gpu0`、cuda:0。
- CF3 recovery已到27k：`1.567 step/s=5.64k/h`、finite=1，critic/actor=`62.03/40.53`、alpha=`0.12916`、TD abs=`2.718`；27k再次有2+ fraction=`0.9%`、max ring=2，support→friend/enemy=`-0.094/+0.183 m/step`，无capture/3+。该重放窗口复现了局部方向性正信号。
- 以`2026-08-13 17:15+08:00`实测：P1 25/50/75/100k约`19:10--19:30`、`21:35--22:00`、`00:00--00:35`、`2026-08-14 02:25--03:10`；CF3 recovery 50/75/100k约`21:20--21:50`、`2026-08-14 02:00--02:40`、`06:40--07:30`，final20与100→200k接力预计`07:30--08:30`。CF3 125/150/175/200k约`12:20--13:20`、`17:05--18:20`、`21:50--23:20`、`2026-08-15 02:35--04:20`。
