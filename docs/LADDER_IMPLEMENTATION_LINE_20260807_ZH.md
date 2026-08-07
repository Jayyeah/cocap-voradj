# CoCap-VorAdj 连续 MARL 正反馈阶梯实现线

> 建立：2026-08-07  
> 目标分支：`ladder/implementation-20260807`（基于 `main`，合并连续线阶梯提交）  
> 本文件用于记录：完整预期阶梯、当前位置、历史结果/结论、实现线边界，以及用户确认的并行规则。

## 1. 完整预期阶梯

```text
Stage 0 旧 IQN 基线复现
  -> Stage 1 连续 (a,ω) bridge 等价
  -> Stage 2 MASAC+(a,ω) 极简单任务（1p1e stationary/global/no-obs）
  -> Stage 3A pure coverage（4p0e0obs）
  -> Stage 3B pure coverage（4p0e1obs）
  -> Stage 4A capture（stationary/global/no-obs）
  -> Stage 4B capture（moving evader）
  -> Stage 4C capture（1 obstacle）
  -> Stage 4D capture（local visibility）
  -> Stage 4E capture（support/pre-capture coverage 角色）
  -> Stage 5A episode-level capture/pure-CE 交替
  -> Stage 5B mixed capture->coverage
  -> Stage 6 body-frame [ax,ay]
  -> Stage 7A world-frame observation+action
  -> Stage 7B robot-obs+yaw 的 world action 方案
```

成功分级：Level A（完整 world-frame mixed）/ Level B（body-frame）/ Level C（可靠 continuous `(a,ω)`）/ Level D（明确乐观信号）。

## 2. 当前进度（2026-08-07）

| Stage | 状态 | 证据 |
|---|---|---|
| 0 旧 IQN 基线 | PASS | 20-episode capture 1.0 / coverage 1.0 / mix 0.9 |
| 1 连续 (a,ω) bridge | PASS | 9/9 fixed-seed parity，state/reward diff=0 |
| 2 极简单任务 | PASS | seed1 6/20、seed2 20/20 capture |
| 3A pure coverage（0 obs） | PASS | seed1/2/3 均 CE energy 改善、collision 可控 |
| 3B pure coverage（1 obs） | IN_PROGRESS | r3 已修复 active-only 索引 bug，25k 运行中 |
| 4A-4E | READY | 配置/基线已就绪，待 3B 后启动 |
| 5A/5B | READY | 配置已就绪 |
| 6A-6C body-frame | READY | 实现/配置/20-step smoke 通过 |
| 7A1-7A3 / 7B1-7B3 world-frame | READY | 实现/配置/20-step smoke 通过 |

当前唯一 formal config：`configs/experiments/positive_feedback_ladder_20260807/stage3b_pure_ce_obs_aw.yaml`。

## 3. 历史结果与结论

### 3.1 旧 world-frame `[ax,ay]` 25k（历史，SUPERSEDED）

- `ctde_25k`（2026-08-06）：capture/pure_ce/mixed 全 0，collision 0.45–0.55。
- 结论：一次性改变过多变量，不能归因于 MASAC；转入阶梯重建。

### 3.2 Stage 2（PASS）

- seed1 eval20：capture 6/20（30%），collision 0/20，min-distance 15 vs baseline 40–50。
- seed2 eval20：capture 20/20，collision 0/20。
- random/no-op baseline：capture 0；oracle 100%。

### 3.3 Stage 3A（PASS，3/3 seeds）

- seed1 eval20：CE progress 0.097（~96%），collision 0/20。
- seed2 eval20：CE progress 0.0155（~15%），collision 0/20。
- seed3 eval20：CE progress 0.0546（~54%），collision 1/20。
- random baseline：progress 0.0138 / collision 100%；noop：progress 0。

## 4. 实现线边界

### 4.1 已具备

- 环境/动力学：`continuous_aw_v1` 与旧 IQN 严格等价；body/world frame 动作合同。
- 训练链：CTDE central twin critic、focal joint replay、25k bundle、diagnostic eval、resume。
- 配置矩阵：Stage2–7B 全部 YAML 已存在。
- 工具：bridge parity、Stage2/3A/3B/4A baselines、25k analyzer、里程碑监督。
- 测试：阶梯合同测试 10 passed，CTDE 回归通过。

### 4.2 当前边界

- 训练速度受外部资源影响极大；25k 单 seed 常需数小时。
- Stage3B 曾暴露 active-only 索引 bug（已修复并加回归测试）。
- Stage4 之后尚未训练；Stage5-7 仅实现/冒烟就绪，无训练结果。

## 5. 用户确认规则

- Stage4A 启动后，Stage4B（moving evader）与 Stage4C（1 obstacle）可作为两个独立单变量分支并行开启；二者互不叠加。
- 算法正式切换 MATD3/MADDPG、改 reward/observation/map 等仍需用户确认。

## 6. 分支与台账

- 阶梯实现分支：`ladder/implementation-20260807`（基于 `main`，合并连续线提交）。
- 专用台账：`docs/CONTINUOUS_MARL_POSITIVE_FEEDBACK_TRACKER_20260807_ZH.md`。
- 阶段报告目录：`artifacts/2026-08-07_positive_feedback_ladder/`。
