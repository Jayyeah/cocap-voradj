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

<!-- AUTO_ALLAGENT_STEP_100000 -->
### Baseline B 自动里程碑 100,000

- 时间：2026-08-11T16:52:38+08:00
- checkpoint：`/home/yjq/rl/CoCap1/cocap-voradj-allagent-oldmix/artifacts/2026-08-10_allagent_oldmix_ablation/legacy_voradj_oldmix_allagent_4p1e1obs_200k_aw_20260810/checkpoints/step_000100000`
- diagnostic：`/home/yjq/rl/CoCap1/cocap-voradj-allagent-oldmix/artifacts/2026-08-10_allagent_oldmix_ablation/legacy_voradj_oldmix_allagent_4p1e1obs_200k_aw_20260810/checkpoints/step_000100000/diagnostic_eval.json`
- storage：`evaluation_model_only`，contains_replay=False
- 指标摘要：`{"step": 100000, "update_count": 23751, "mean_finite": 1.0, "mean_critic_loss": 31.23241411781311, "mean_actor_loss": 29.80998917388916, "mean_alpha": 0.03932702188193798, "mean_update_wall_time_s": 1.2904385898437498, "mean_updates_per_second": 0.797520158482335, "mean_active_agent_loss_terms_per_update": 512.0, "mean_active_agent_loss_terms_per_second": 408.33032114295554, "env_steps_per_second": 2.786929419352983, "collision_count": 1}`
- sampler 摘要：`{"sampler": "uniform_joint", "requested_batch_size": 128, "actual_batch_size": 128, "unique_joint_transitions": 128, "replacement_count": 0, "pre_capture_transition_count": 40, "post_capture_transition_count": 0, "pure_ce_transition_count": 88, "sampled_phase_counts": {"pure_coverage": 88, "pre_capture": 40}, "sampled_scene_counts": {"pure_ce": 88, "mixed_crms": 40}, "sampled_active_agent_role_counts": {"pursuing": 110, "support": 49, "coverage": 353}, "active_agent_loss_terms": 512, "role_metadata_used_for_sampling": false}`
- GPU 摘要：`{"index": 0, "temperature_c": 70, "utilization_percent": 100, "memory_used_mib": 33063, "memory_total_mib": 49140, "memory_free_mib": 16077}`
