# CoCap-VorAdj 项目接手状态

更新时间：2026-08-02 17:17（Asia/Shanghai）

> **状态优先级说明（2026-08-06）：** 本文只提供旧 CR-MS/VCT-LS/IQN baseline 背景，不是当前连续线动态状态。当前唯一状态、六条 500k 长训、AW bridge、IQN replay 可行性和后续 TODO 以 [`CONTINUOUS_ACTION_MARL_EXPERIMENT_TRACKER_20260804_ZH.md`](./CONTINUOUS_ACTION_MARL_EXPERIMENT_TRACKER_20260804_ZH.md) 为准。

## 术语与工作边界

- 仓库路径：`/home/yjq/rl/CoCap1/cocap-voradj`
- 核心文档路径：`/home/yjq/rl/CoCap1/cocap-voradj/docs`
- 用户口述的 `VST-LS` 在当前仓库、配置和文档中统一写作 `VCT-LS`；本接手文档视为同一实验线。
- 当前分支：`feature/vct-ls-v0-20260730`。
- 工作树包含大量未提交的实验源码、配置、文档与 artifacts；这些均视为已有项目资产，禁止清理、回退或覆盖。
- 不做持续人工监控。后台训练与 watcher 自行运行；用户说“更新状态”时再做一次进程、日志、checkpoint、screening 和 GPU 对账。

## 当前 solid 基线

### A3

A3 是进入三条并行线之前的最近 solid 主线。它保留旧 mix capture、APF-v2_fixed、K10、CenterSqrtN、hard boundary/death 和四缓冲区 mixed/recovery 骨架；后续 CE、VCT-LS 与 CR-MS 都是在开关隔离下演进，不应破坏 A3 默认语义。

### CE

CE 的 pure coverage 已 solid：

- CE13，8-agent pure coverage，20-rollout：strict CE success `1.00`，collision `0.00`。
- CE-OldMix-v1 已完成 4v1 -> 8v2 -> 12v3 课程。
- 最终选中 stage3 12v3 step 600k；正式 20-rollout 中 capture/mix capture/mix CE strict 均为 `0.90`，pure coverage CE strict `1.00`，collision `0.10`。
- 最新可复用 coverage 训练设置：replay `64/16/32/16`，captured snapshot ratio `0.75`，非 capture 部分 map-random/synthetic-cluster 各半，4v1/8v2/12v3 post window `500/600/700`；CE speed weight 在 200k 从 `0` 切到 `0.0005`。

### VCT-LS

VCT-LS 局部感知与友方 Voronoi 通信拓扑可行，但 support 梯度是关键：

- 4v1 support scratch 1M 正式 20-rollout：capture `1.00`、mix capture `1.00`、collision `0.00`。
- 同一点 mix CE strict `0.35`、coverage CE strict `0.45`；loose CV<0.15 分别为 `0.90/1.00`。
- plain warm/plain scratch 的确定性 capture 均为 `0`，不能依据训练窗口的偶发 capture 晋级。
- 当前结论是“support 可行”，不是所有 VCT-LS 组合都 solid。

### CR-MS 首批

CR-MS 首批 direct ring-MS + CE、无 support：

- warm500k 与 scratch1M 的正式 capture/mix capture 均为 `0`。
- 失败不能单独归因于 ring-MS，因为同批无 support 的 VCT-LS plain 线也均为 `0`。
- 因此首批结论是“缺少 support 梯度时失败”，下一批必须同时保留新 MS 并给 support 明确信用分配。

## CR-MS 二批：已实现口径

直接 detector 的 MS 侧基本不改：

- `capture_reward_mode=ring_importance_ms_v0`
- compact ring：inner surface `d_safe+0.5`、preferred center radius `8.0`、outer center radius `10.5`
- velocity importance `0.75..1.25`
- direct enemy-body local visibility trigger
- direct capture timestep penalty `0`
- 旧 direct `approach/mean_shift/front` 权重仍为 `0`

VCT-LS 侧沿用：

- pursuer-only Voronoi communication graph
- enemy/obstacle surface sensing radius `20m`
- K10 release delay
- support observation 不增加 enemy token，不增加全局敌人坐标
- support target 只来自正在 pursuing 的一阶邻居直接看到的 evader id

support 奖励改为：

```text
r_support = 0.5 * r_legacy_approach_only + 0.5 * r_CE
```

其中 `r_legacy_approach_only` 只保留旧 mix-capture 的距离进度吸引项；不包含 timestep、旧 MS、front 或 direct ring-MS。这样 support 可以通过“跟随正在围捕的邻居会接近敌人”获得方向信号，进入自身 20m 感知范围后再切换为 direct ring-MS。

实现开关：

```yaml
voradj:
  support_reward_blend_enabled: true
  support_reward_capture_weight: 0.5
  support_reward_coverage_weight: 0.5
  support_reward_capture_target_mode: neighbor_visible
  support_reward_capture_component_mode: approach_only
  support_reward_approach_weight: 1.0
  support_reward_approach_clip: 3.0
```

配置目录：

```text
configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/
  common.yaml
  stage1_4p1e1obs_scratch2m.yaml
  stage2_8p2e2obs_700k.yaml
  stage3_12p3e3obs_700k.yaml
```

stage1 必须训满 2M，不因中途 screening 提前停止；100k 至 2M 每 100k 独立跑 capture/coverage/mix 各 10 rollout。stage2/3 暂按最新 CE 快课程口径各 700k，分别 8v2 与 12v3；只有 stage1/2 的历史最佳独立 screening 点通过人工复核后，才向下一阶段注入 `pretrained.path`。

## 当前后台状态

2026-08-02 15:55 已启动：

```text
训练 tmux: crms_sa_ce_s1_train_gpu0_0802
训练 GPU: cuda:0
训练 run: runs/crms_supportapproach_ce_curr_stage1_4p1e1obs_scratch2m_20260802_run1
训练日志: logs/train_crms_supportapproach_ce_s1_4v1_2m_20260802.log

screening tmux: crms_sa_ce_s1_screen_gpu1_0802
screening GPU: cuda:1
screening 输出: artifacts/2026-08-02_cr_ms_support_approach_ce_curriculum/screening/stage1_4p1e1obs
watcher 日志: logs/watch_crms_supportapproach_ce_s1_20260802.log
```

启动检查：

- 两个 tmux 会话存活。
- 正式训练进程 PID `632844`，占用 GPU0；GPU1 watcher 正确等待 `step_100000.pt`。
- 正式 effective config 已确认 actual replay 为 `64/16/32/16`、snapshot ratio `0.75`、support mode 为 `approach_only`。
- 2k smoke 自然完成，生成 init/latest/final checkpoint、metrics 与 episodes。
- 全量测试：`38 passed`；仅有 protobuf/Python 3.14 未来兼容 deprecation warning，无运行错误。
- 交接前已到 14k 并跨过采样门槛：`updates_voradj_mixed_coverage=175`，`loss=0.3726`、`loss_ema=1.9929`；batch 中 support approach 与 support CE 统计均非零，确认新奖励实际进入训练更新。
- 随机策略最初数千步 collision 高是冷启动现象，不作为模型质量结论；是否有学习信号应等四个 replay 类满足采样门槛后再看 `loss/loss_ema`。

## 用户说“更新状态”时的检查清单

1. `tmux ls`：训练与 watcher 会话是否还在。
2. `pgrep -af`：训练、watcher、短时 rollout worker 的真实 PID。
3. `nvidia-smi`：GPU0 训练占用；GPU1 仅在 checkpoint screening 时占用。
4. 训练日志：最后 global step、异常 traceback、CUDA/OOM、是否自然退出。
5. `metrics.jsonl`：loss/loss_ema、四 replay size、update count、support blend capture/coverage batch stats、recent capture/collision。
6. checkpoint：应按 100k 递增；final 为 `final_step_2000000.pt`。
7. screening：每个 `step_<N>/DONE`、`all_summaries.json`、worker/rollout 日志是否完整。
8. 汇报时分开写“训练窗口指标”和“确定性独立 screening 指标”；晋级只以后者为主。

## 后续 TODO

1. stage1 2M 全程完成后，从全部 100k screening 点选历史最佳；主序为 capture、mix capture、collision，随后先比较 mix/coverage CE strict，再比较 mix/coverage CV<0.15 与 capture steps。2026-08-04 已将 finalizer 修正为该顺序，避免低 CV 但 CE 中心覆盖失败的 checkpoint 被优先选中。
2. 对候选点补正式 20-rollout/10-GIF，固定 IQN 分位中点；不得用训练窗口偶发高点直接晋级。
3. 将选中 stage1 checkpoint 注入 stage2 `pretrained.path`，跑 8v2 700k warm 课程；screening 上限建议 capture/coverage/mix 为 `1000/1500/2500`。
4. stage2 通过后同样注入 stage3，跑 12v3 700k；上限建议 `1000/1800/2800`。
5. 若 support-approach 二批仍 capture=0，先检查 support capture 分量在 batch stats 中是否持续非零、neighbor-visible target 命中率、first detection/miss-evader；再考虑 ring 目标、omega 或半径 ablation。
6. 若 capture 成立但同侧绕圈明显，再开 CR3b phase-sector occupancy；若 capture 成立但 coverage 失败，按 CE post-capture/recovery 分布归因，不改 direct MS。

## 2026-08-02 17:16：三线并行训练与自动展示流水线

用户确认并行开启 VCT-LS+CE 8v2 support warm/scratch，同时要求三条线训练期间均做 screening，并在当前大阶段结束后自动选择历史最佳 checkpoint、跑对应的 20-rollout/10-GIF。

当前三条正式线：

| 线 | 训练配置 | 训练 GPU | stage cap |
| --- | --- | ---: | ---: |
| CR-MS support-approach 4v1 | cr_ms_support_approach_ce_curriculum_20260802/stage1_4p1e1obs_scratch2m.yaml | 0 | 2M |
| VCT-LS+CE support 8v2 warm | vct_ls_ce_support_8v2_parallel_20260802/warm_from_4v1_support_700k.yaml | 0 | 700k |
| VCT-LS+CE support 8v2 scratch | vct_ls_ce_support_8v2_parallel_20260802/scratch_700k.yaml | 1 | 700k |

VCT-LS 两线共同使用：

- VCT-LS/A3 legacy capture，不使用 CR-MS ring reward；
- 一阶 support 0.5 * capture_task + 0.5 * CE，target 为 neighbor_visible；
- 最新 CE replay 64/16/32/16、captured snapshot ratio 0.75、8v2 post window 600；
- CE speed weight 在 200k 从 0 切到 0.0005；
- 每 100k 独立 deterministic screening，capture/coverage/mix 各 10 rollout，固定 IQN midpoint quantiles；
- 8v2 screening 上限为 capture 1000、coverage 1500、mix 2500。

新增通用流水线：

- tools/watch_screened_run.sh：等待稳定 checkpoint，用 4 worker 并行做 screening；
- tools/finalize_screened_run.py：等待 stage cap 与全部 screening 完成，按 capture/mix-capture 优先、碰撞次优、mix/pure CE strict、mix/pure CV<0.15 与步数依次作 tie-break 的规则选择历史最佳；
- 历史最佳自动跑 capture/coverage/mix 各 20 rollout / 10 GIF；GIF 默认显示友方邻接与局部感知圈，但默认不绘制 faded 尾迹。只有显式向 finalizer 传入 `--draw-trails` 时才开启尾迹；
- 三线统一展示根目录：artifacts/2026-08-02_three_line_stage_best_20rollout10gif/。

正式后台：

~~~text
VCT warm train:    vctls_ce8_support_warm_train_gpu0_0802
VCT warm screen:   vctls_ce8_support_warm_screen_gpu1_0802
VCT warm finalize: vctls_ce8_support_warm_finalize_gpu1_0802

VCT scratch train:    vctls_ce8_support_scratch_train_gpu1_0802
VCT scratch screen:   vctls_ce8_support_scratch_screen_gpu1_0802
VCT scratch finalize: vctls_ce8_support_scratch_finalize_gpu1_0802

CR-MS finalize: crms_sa_ce_s1_finalize_gpu1_0802
~~~

验证：

- 两条 2k GPU smoke 均自然结束并生成 final_step_2000.pt；
- warm smoke 成功 shape-compatible 加载 4v1 support scratch 1M checkpoint，首个 8v2 episode capture 成立且无碰撞；
- warm 正式配置已同时显式设置 learning_rate 与 step-0 schedule 为 3e-5；最初约 7k 的零-update 预启动已保留为 pre_lrscalarfix 归档，正式 run 从 0 重启后 metrics 确认为 3e-5；
- 20-step 展示 smoke 成功生成 GIF，诊断图层与 fixed_midpoint_32 口径均落盘；
- 全量测试 43 passed，无 traceback、CUDA/OOM、NaN/Inf。

CR-MS 第二批首个 100k 独立 screening 已完成：capture 0.80、mix capture 0.80、capture/mix collision 0.20；pure coverage CV<0.15 为 1.00，mix CV<0.15 为 0.50，CE strict 暂为 0。这是早期正信号，但不提前结束 2M；仍从 20 个 screening 点选择历史最佳。
