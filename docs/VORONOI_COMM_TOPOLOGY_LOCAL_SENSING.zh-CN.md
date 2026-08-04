# VCT-LS v0：友方 Voronoi 通信拓扑 + 敌障局部感知

更新时间：2026-07-30

这是一个尚未设为默认的新并行实验线，目标是在原 A3/mix 任务基础上重新定义 observation 中的邻接与感知关系。本线与 ZoneDemo B1、CE-Coverage 并行推进，互不覆盖。

核心思想：

- 只用我方 pursuer 计算 Voronoi 分割，并由共享 cell 边的 pursuer 构成动态通信拓扑。
- enemy 和 obstacle 不作为通信节点，不参与友方通信图。
- enemy/obstacle token 只来自本地感知半径。
- friend token 仍包含邻居的 `is_pursuing`，使一阶邻居可通过通信知道“邻居正在追捕”，但不直接获得敌人坐标。

建议第一版状态定义：

```text
direct_detect_i = 自己本地感知到至少一个 evader
neighbor_detect_i = 一阶通信邻居中至少一个 direct_detect=true
support_i = (not direct_detect_i) and neighbor_detect_i
```

self token 的 `is_pursuing` 只表示 `direct_detect_i`。支援个体通过 friend token 的 `is_pursuing` 学习靠近、补位或继续搜索，第一版不直接泄露 enemy token。

参考依据：

- Gosrich et al., `Coverage Control in Multi-Robot Systems via Graph Neural Networks`, arXiv:2109.15278 / ICRA 2022。该文明确把共享 Voronoi cell 边的机器人定义为 Delaunay graph 邻居，并在同一图上叠加信息交换网络。
- Cortes, Martinez, Bullo, `Spatially-distributed coverage optimization and control with limited-range interactions`, ESAIM COCV 2005。该文基于 Voronoi partitions 与 proximity graphs 处理有限感知/通信 coverage。
- Luo and Sycara, `Voronoi-based Coverage Control with Connectivity Maintenance for Robotic Sensor Networks`, MRS 2019。该文提示 coverage 展开时通信连通性需要额外监控。
- Wang et al., `ViPER: Visibility-based Pursuit-Evasion via Reinforcement Learning`, CoRL 2025。该文证明有限感知 pursuit-evasion 中用图注意力进行分布式协同是相关路线。

TODO 摘要：

1. 增加配置隔离：`perception_topology_version=legacy_voradj|friendly_voronoi_comm_v0`。
2. 写几何/observation smoke，确认 evader 不再参与 Voronoi sites，敌人不可见时不会进入 enemy token。
3. 用 A3 checkpoint 做少量零训练 rollout，检查 observation shift 后是否崩溃。
4. 优先开 warm-start 适应训练，例如 A3 `8v2` best -> 新 VCT-LS `8v2 mix` 500k。
5. 若 warm-start 能闭环，再开 4v1 scratch 2M 与后续慢课程。

更完整的历史 TODO 原先保存在同机 TERL 工作区，本仓库发布不依赖该外部文件；当前可复现语义以本文和最终发布说明为准。

## 实施状态（2026-07-30 18:16）

当前分支：`feature/vct-ls-v0-20260730`。

已完成 v0 开关式接入，默认仍是 `legacy_voradj`；只有配置 `voradj.perception_topology_version=friendly_voronoi_comm_v0` 时才启用 VCT-LS。新逻辑只改 observation/拓扑来源：友方通信图使用 pursuer-only Voronoi 邻接，enemy/obstacle token 使用本地半径感知；旧 A3、ZoneDemo、CE 配置保持各自开关，不被默认启用。

新增精简实验配置：

```text
configs/experiments/vct_ls_20260730/vct_ls_a3_stage3_8v2_warm_500k.yaml
```

它继承 A3 stage3 `8p2e2obs` 配置，只覆盖 500k warm-start、A3 stage3 best checkpoint、VCT-LS 开关、sensing 半径，并显式设置 `vct_ls_apply_release_delay: true` 使用 K10。该文件是当前推荐的首条适应训练入口。

验证结果：

```text
python3 -m py_compile src/cocap_voradj/envs/voronoi_adjacency.py tests/test_vct_ls_topology.py
PYTHONPATH=src pytest -q tests/test_vct_ls_topology.py  # 5 passed
PYTHONPATH=src pytest -q tests                         # 24 passed
```

500 step smoke 已通过，输出在：

```text
runs/smoke_vct_ls_8v2_warm_500_20260730
```

首条后台训练已启动：

```text
tmux session: vctls_k10_8v2_warm500k_0730
run_dir: runs/vct_ls_a3_stage3_8v2_k10_warm_500k_20260730_run1
GPU: cuda:1
```

后续若要上传 GitHub，建议只纳入 VCT-LS 源码、测试、thin config 和本说明文档；不上传 smoke run、训练 run、大量 checkpoint 或大批 GIF。

### K10 修订

按最新口径，当前有效 VCT-LS 训练显式启用 K10：`vct_ls_apply_release_delay: true`，`is_pursuing_release_delay_steps=10`。K10 只平滑“自己直接感知到 enemy 后短暂丢失”的 pursuing 状态；邻居报告仍只通过 friend token 的 `is_pursuing` 传递，不把自己的 self `is_pursuing` 直接置 1，也不泄露 enemy token。

此前短暂启动过的非 K10 run 已停止并标记为 superseded：

```text
runs/vct_ls_a3_stage3_8v2_warm_500k_20260730_run1
```

### 感知半径

当前有效 K10 配置使用 `enemy_sensing_radius=20.0`、`obstacle_sensing_radius=20.0`，并开启 `local_sensing_uses_surface_distance=true`。也就是按边界距离判断可见，而不是纯中心距离：`center_distance - pursuer_radius - target_radius <= 20m`。

这个 20m 是首轮 warm-start 的保守选择，目的是和 A3/旧配置中的 `perception.range=20.0` 对齐，只隔离 VCT-LS 的拓扑变化。若后续发现主要瓶颈是敌人长期不可见、first-detection 太慢或 support 信号太少，优先做 `enemy_sensing_radius=25/30` ablation；障碍感知半径首轮暂不改。

### Rollout 兼容

已修复 VCT-LS thin config 对 rollout/GIF 工具的兼容：`tools/rollout_voradj_visual.py` 现在复用 trainer 的 `load_config()` 展开 `extends`；可视化快照按 phase 选择 `_capture_voronoi_map()` 或 `_coverage_voronoi_map()`，因此 VCT-LS GIF 中 Voronoi sites 保持 pursuer-only，不会把 evader 画回通信图。

验证：

```text
PYTHONPATH=src pytest -q tests  # 26 passed
artifacts/vct_ls_smoke_visual_20260730/k10_smoke_mix/rollout_mix_seed_2026073051.gif
```

50k checkpoint 后推荐先做 `mix coverage` 两场景各 20 rollout / 10 GIF，输出到本地 artifacts，不纳入 GitHub 精简上传。

### 50k Watcher

已启动 `vctls_50k_rollout_0730`，等待 `step_50000.pt` 出现后自动跑 `mix coverage` 两场景各 20 rollout / 10 GIF。脚本与日志在：

```text
artifacts/2026-07-30_vct_ls_k10_warm_8v2/run_step50000_rollout_watcher.sh
artifacts/2026-07-30_vct_ls_k10_warm_8v2/step_50000_rollout_watcher.log
```

该 watcher 使用 CPU，不影响 GPU1 上的 VCT-LS 训练。

### 50k Warm-Start 首检口径修订

按最新实验口径，VCT-LS 本线的主问题不是 coverage settle，而是“局部感知敌人 + 友方 Voronoi 邻接通信”是否能支撑协同围捕，以及是否出现漏敌。coverage 奖励和判定的重构由并行 CE-Coverage 线推进，因此本线评估时：

- 主指标：`capture_success_rate`、`fully_capture_rate`、`collision_rate`、漏敌/不可见步数、capture 用时。
- 次指标：capture 后 coverage/settle 只记录，不作为 VCT-LS v0 是否成立的主要判据。

已完成的 A3 stage3 8v2 warm-start `step_50000.pt` 的 mix 20-rollout 首检为：

```text
capture_success_rate: 1.00
collision_rate: 0.00
coverage_geometric_rate: 0.80
coverage_settled_rate: 0.15
coverage_settle_timeout_rate: 0.65
```

该结果说明：warm-start 下 VCT-LS 的局部敌人感知与一阶友邻通信没有破坏围捕入口，短线已基本证明围捕闭环可成立；coverage settle 弱只作为附注，后续交给 CE 线解决。

基于该信号，已开启 VCT-LS 4v1 scratch capture-focused 并行线：

```text
config: configs/experiments/vct_ls_20260730/vct_ls_a3_4v1_scratch_mix_2m_capturefocus.yaml
tmux: vctls_scratch4v1_2m_0730
run_dir: runs/vct_ls_a3_4v1_scratch_mix_2m_capturefocus_20260730_run1
watcher: vctls_scratch4v1_50k_capture_rollout_0730
```

scratch 线仍沿用 A3 mixed/recovery 训练骨架，但筛选和汇报优先看 capture，不因 coverage settle 偏弱提前否定本线。

warm-start 线在 `step_100000.pt` 已补启动 capture-focused 复检：

```text
tmux: vctls_warm100k_capturemix_rollout_0730
script: artifacts/2026-07-30_vct_ls_k10_warm_8v2/run_step100000_capturemix_watcher.sh
output: artifacts/2026-07-30_vct_ls_k10_warm_8v2/step_100000_capturemix_capture30_mix20_5gif
scenarios: capture 30 rollout / 5 GIF; mix 20 rollout / 5 GIF
visual layers: --draw-neighbor-edges --draw-sensing-circles
```

该复检的读取顺序：先看 `capture30/capture/batch_summary.json` 中的 capture success、collision、avg_steps；再看 `mix20/mix/batch_summary.json` 中的 capture success、fully captured/碰撞与是否存在明显漏敌。coverage settle 字段只作为背景。

### VCT-LS GIF 绘制约定

后续 VCT-LS 的 `mix` GIF 必须打开诊断图层：

```bash
--draw-neighbor-edges --draw-sensing-circles
```

含义：

- `--draw-neighbor-edges`：画 pursuer-only Voronoi 通信邻接线，用于观察友方通信拓扑是否覆盖发现敌人的个体及其支援邻居。
- `--draw-sensing-circles`：给每个 active pursuer 画本地感知圆。当前 VCT-LS 默认 `enemy_sensing_radius=20.0`、`obstacle_sensing_radius=20.0`，因此只显示一圈；若后续敌人/障碍半径不同，敌人感知圆为紫色实线，障碍感知圆为灰色虚线。
- 感知圆按配置半径绘制，用作“本地传感器尺度”的直观参考；当前可见性判定仍是 surface distance，即实际 token 进入条件为 `center_distance - pursuer_radius - target_radius <= sensing_radius`。因此当目标有半径时，目标中心可能略在圆外但表面已经进入感知范围。

默认不把该诊断图层强加给 CE/Zone/普通 coverage GIF，避免其他实验线可视化被额外元素污染。

### Support Reward Blend 对照线

为排查从零训练时 support agent 是否缺少追捕方向的奖励信号，新增一条 4v1 scratch 对照线。除下列开关外，其余配置继承当前 VCT-LS scratch 2M：

```text
config: configs/experiments/vct_ls_20260730/vct_ls_a3_4v1_scratch_mix_2m_supportblend.yaml
run_name: vct_ls_a3_4v1_scratch_mix_2m_supportblend_20260730_run2
control_against: vct_ls_a3_4v1_scratch_mix_2m_capturefocus_20260730_run1
```

触发条件：agent 自己的有效 `is_pursuing=0`，但一阶 Voronoi 友邻中至少一个 agent 的有效 `is_pursuing=1`。此时 observation 和 task label 仍保持 coverage，不把 support agent 强行改成 capture；只把 dense task reward 改为：

```text
r_support = 0.5 * r_capture + 0.5 * r_coverage
```

若自己直接发现敌人并进入 `is_pursuing=1`，则回到原本纯 capture reward；若友邻也不再 pursuing，则回到原本纯 coverage reward。capture shaping 的目标暂用 `neighbor_visible`：只取 pursuing 一阶友邻直接感知到的 evader id 作为 support capture reward 的目标，避免完全使用全局敌人信息。

日志新增字段：`support_reward_blend_active_count`、`support_reward_blend_active_ratio`、`reward_support_blend_capture_*`、`reward_support_blend_coverage_*`。本线仍以 capture success、collision、漏敌和 capture 用时为主，coverage settle 只作背景。

`step_50000.pt` 已完成 capture 30 rollout / 5 GIF 快评：

```text
capture_success_rate: 0.00
collision_rate: 0.70
avg_steps: 593.83
avg_final_active_pursuers: 3.03
avg_radius_growth: -0.04
```

对照原 VCT-LS scratch 50k：`capture_success_rate=0.00`、`collision_rate=0.90`、`avg_final_active_pursuers=2.47`、`avg_radius_growth=+2.31`。因此 support blend 早期尚未带来 capture 成功，但碰撞和队形崩散略有改善，可继续观察 100k/150k 后是否转化为 capture 信号。

### VCT-LS + CE Coverage 融合实验（2026-07-31）

由于 VCT-LS 的局部敌人感知与一阶友邻通信已经能证明围捕闭环成立，但旧 coverage 在探索扩散与收敛稳定性上拖累 mix，新增一组独立融合实验：保留 VCT-LS/A3 capture 主体、K10 release delay、局部感知和 pursuer-only 通信拓扑；coverage 奖励与成功判定切换为 CE-Coverage。

公共配置：

```text
config_common: configs/experiments/vct_ls_ce_20260731/vctls_ce_4v1_common.yaml
任务场景: 旧 mix/A3 4v1，非 ZoneDemo，非 CE standalone
拓扑: friendly_voronoi_comm_v0，enemy/obstacle surface sensing radius = 20m
coverage 成功: CE center RMS <= 0.05，max <= 0.10，连续 30 步
loose 统计: rollout 同步记录 legacy area CV < 0.15
batch: pursuing 64, pre_capture_cover 16, post_capture_real 16, recovery_pure 32
recovery_pure 初始化: 0.5 capture snapshot，0.5 ordinary map_random，与 mix 我方初始化一致
post/pure coverage 限制: post window 300，env episode max 900；检测 rollout 中 coverage max steps 500
```

四条并行线：

```text
plain warm-start 500k:
configs/experiments/vct_ls_ce_20260731/vctls_ce_4v1_plain_warm500k.yaml
warm checkpoint: runs/vct_ls_a3_4v1_scratch_mix_2m_capturefocus_20260730_run1/checkpoints/step_2000000.pt

support warm-start 500k:
configs/experiments/vct_ls_ce_20260731/vctls_ce_4v1_support_warm500k.yaml
warm checkpoint: runs/vct_ls_a3_4v1_scratch_mix_2m_supportblend_20260730_run2/checkpoints/step_1900000.pt

plain scratch 1M:
configs/experiments/vct_ls_ce_20260731/vctls_ce_4v1_plain_scratch1m.yaml

support scratch 1M:
configs/experiments/vct_ls_ce_20260731/vctls_ce_4v1_support_scratch1m.yaml
```

scratch 两条线每 100k checkpoint 自动跑轻量检测，场景为 capture、mix、pure coverage，各 10 rollout、默认不产 GIF；输出位于：

```text
artifacts/2026-07-31_vct_ls_ce_coverage_4v1/screening/
```

读取重点：本实验优先看 VCT-LS 局部感知条件下 capture 是否稳定、support blend 是否继续提高从零学习；coverage 读取 CE success，同时用 `coverage_cv015_rate`/`coverage_cv015_best_rate` 作为旧 CV loose 对照。

## 2026-08-02：VCT-LS + CE 融合线训练完成与正式评估归档（deepseek 执行）

2026-08-01 凌晨至上午，四条融合线全部自然训满。本归档由 deepseek 执行，正式评估产物位于：

```text
artifacts/2026-08-02_final_20rollout10gif/
```

评估口径：每线 capture/coverage/mix 三场景各 20 rollout / 10 GIF，固定 IQN 分位中点（policy_quantile_mode=fixed_midpoint_32），步数上限 capture 1000 / coverage 1200 / mix 2200，seed 20260802xx。

### 训练完成情况

| 线 | 总步长 | final checkpoint |
| --- | ---: | --- |
| plain warm-start | 500k | final_step_500000.pt |
| support warm-start | 500k | final_step_500000.pt |
| plain scratch | 1M | final_step_1000000.pt |
| support scratch | 1M | final_step_1000000.pt |

### 正式 20-rollout 结果

| 线 | capture | mix capture | mix CE strict | mix CV<0.15 | coverage CE strict | coverage CV<0.15 | 碰撞 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| plain warm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 | 0.00 |
| support warm | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| plain scratch | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.95 | 0.00 |
| support scratch | 1.00 | 1.00 | 0.35 | 0.90 | 0.45 | 1.00 | 0.00 |

补充：support warm capture 平均 232.8 步，support scratch capture 平均 88.5 步，plain scratch coverage 平均 119.3 步。

### 结论

1. support reward blend 是 4v1 上唯一能稳定学出围捕的组合：support scratch 从零训练即可达到 capture 1.00、零碰撞，并同时保持 CV<0.15 loose coverage 0.90-1.00，为本组最优候选。
2. 非 support 线（plain warm、plain scratch）在确定性独立评估下 capture 全部为 0，与训练窗口数字（plain warm 曾有 0.24）差距极大；说明无 support blend 时 4v1 训练分布下无法形成稳定围捕闭环，训练窗口 capture 不可作为晋级依据。
3. support warm 保留了 capture（1.00）但 coverage 全部超时（900 步），warm 起点 supportblend 1900k 模型的 coverage 扩散在融合训练中未恢复。
4. coverage 以 CV<0.15 loose 为主达标，CE strict（中心误差 hold）在 mix 中最高仅 0.35-0.45，与 CE14B 结论一致。

## 未来方向

1. 以 VCT-LS support scratch 1M 为 4v1 最优候选，优先开 8v2 support 课程（warm + scratch）。
2. plain 线暂缓；support warm 需补 coverage 恢复诊断（warm 起点 supportblend 模型 coverage 弱）。
3. 4v1 正式评估统一口径：capture 1000 / coverage 1200 / mix 2200，20-rollout，确定性分位中点，训练窗口指标不作晋级依据。

### 2026-08-02：8v2 support warm/scratch 已启动

上述未来方向第 1 条已执行。新增配置目录：

~~~text
configs/experiments/vct_ls_ce_support_8v2_parallel_20260802/
~~~

- warm：从 vctls_ce4v1_support_scratch1m_20260731_run1/checkpoints/final_step_1000000.pt 迁移，700k；
- scratch：8v2 从零训练，700k；
- 两线均沿用 VCT-LS legacy capture 与 support 0.5 capture_task + 0.5 CE，并升级到最新 CE replay/recovery/post-window 设置；
- 每 100k 自动 screening；stage cap 后自动选历史最佳并跑 capture/coverage/mix 各 20-rollout/10-GIF。

展示产物统一写入：

~~~text
artifacts/2026-08-02_three_line_stage_best_20rollout10gif/
~~~
