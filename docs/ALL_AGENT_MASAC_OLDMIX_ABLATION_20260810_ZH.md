# Baseline B — Standard All-Agent MASAC on Legacy-VorAdj Old-Mix

更新日期：2026-08-10

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
