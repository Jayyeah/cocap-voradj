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
  -> Stage 4C capture（moving evader + 1 obstacle）
  -> Stage 4D capture（local visibility）
  -> Stage 4E capture（support/pre-capture coverage 角色）
  -> Stage 5A episode-level capture/pure-CE 交替
  -> Stage 5B mixed capture->coverage
  -> Stage 6 body-frame [ax,ay]
  -> Stage 7A world-frame observation+action
  -> Stage 7B robot-obs+yaw 的 world action 方案
```

成功分级：Level A（完整 world-frame mixed）/ Level B（body-frame）/ Level C（可靠 continuous `(a,ω)`）/ Level D（明确乐观信号）。

## 2. 当前进度（2026-08-09）

| Stage | 状态 | 证据 |
|---|---|---|
| 0 旧 IQN 基线 | PASS | 20-episode capture 1.0 / coverage 1.0 / mix 0.9 |
| 1 连续 (a,ω) bridge | PASS | 9/9 fixed-seed parity，state/reward diff=0 |
| 2 极简单任务 | PASS | seed1 6/20、seed2 20/20 capture |
| 3A pure coverage（0 obs） | PASS | seed1/2/3 均 CE energy 改善、collision 可控 |
| 3B pure coverage（1 obs） | PASS | r3 eval20 CE -51%、collision 15% |
| 4A capture | PASS | 2 seeds（capture 5%/15%、min-dist 22 < baselines） |
| 4B capture（moving） | FAIL（历史） | 2 seeds × 25k eval20 capture 0/20、min-dist ≥ baseline；oracle seek 10/10 证明合同可学 |
| 4C capture（moving+1obs） | FAIL（历史） | 2 seeds × 25k eval20 capture 0/20、min-dist ≥ baseline；400 步 cap 掩盖慢接近 |
| A 批 reward 变体（A1/A2/A3） | FAIL/参考 | A1/A2 碰撞坍缩、A3 安全但无信号；探索批 Base=A3，GPU 释放后再启动 |
| 4A/4C 200k 原合同对照 | IN_PROGRESS | IQN 200k 参照已完成（100k 首现 capture、125k 峰值 75%、200k 40%）；4A @50k capture 5/20（25%）、4C @50k 0/20 slow approach；当前 61k/61k（2026-08-09 10:45）；速度瓶颈已定位为自身 focal replay 采样 |
| 4D-4E | READY | 配置已就绪；4B 有积极信号才提前启动 4D（当前未满足） |
| 5A/5B | READY | 配置已就绪 |
| 6A-6C body-frame | READY | 实现/配置/20-step smoke 通过 |
| 7A1-7A3 / 7B1-7B3 world-frame | READY | 实现/配置/20-step smoke 通过 |

当前位置（2026-08-09）：Stage4A/4C 200k 原合同对照训练中（seed 2026080801，每 25k 评估并回填 `artifacts/2026-08-08_200k_reference/LEGACY_200K_PROGRESS.md`）；训练速度实测与瓶颈归因见该文件（外部线非主因，`FocalReplaySampler` 池重建为主因）；完成后输出与 IQN 200k 的完整三线对照结论，再按 gate 决定 4D/4E/5 的推进。

当前 formal config：`stage4a_capture_aw.yaml` / `stage4c_capture_aw.yaml`（`configs/experiments/positive_feedback_ladder_20260807/`，200k 对照用原合同，奖励不改）。

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

## 4.5 [2026-08-08 PATCH] Stage4 诊断补丁（主线不变）

- Stage4 mainline（A1/A2/A3、4B/4C、4D 预备）保持不变，不重排。
- E1 target_entropy -2→-4 暂停（方向未定），替换为 E0 entropy calibration（只记录指标）。
- 新增诊断：action 拆分、pursuit 几何（d1–d4/closing/bearing/ring 访问）、reward 密度、replay state coverage、critic Q-ranking、deterministic/stochastic + paired seeds 评估。
- 新增 legacy IQN scratch 25k/50k early-training 参照（原成功合同不改），产出 25K/50K DIAGNOSTIC。
- 完整设计：`docs/LADDER_STAGE4_REWARD_BATCH_ATTRIBUTION_20260808_ZH.md`；台账 §0.2/§0.3。

## 5. 用户确认规则

- Stage4A 启动后，Stage4B（moving evader）与 Stage4C（moving evader + 1 obstacle）可作为两个独立分支并行开启以加速。用户确认 2026-08-07：4C 就是动态敌人分支（原为 4B 之后串行）；4B/4C 不得再叠加其他未授权改动。
- Stage4D 提前启动（用户确认 2026-08-08）：若 4B seed2 先结束且积极，直接启动 4D 两 seed（4D = moving evader + 1 obstacle + local visibility，相对 4C 唯一改动 global_evader_visibility false）；4C 后结束积极则 4D 正常维护，4C 无积极信号则 4C 与 4D 并行维护。
- 历史表述（2026-08-07 用户确认修正）：此前把 4C 写成“stationary+obstacle 单变量”，有误；4C 正确合同 = 动态敌人 + 1 obstacle。
- Stage4A 起，性能验证 seed 数由 3 改为 2（至少 1 个明显优于 random/no-op 即可晋级），加速验证。
- 算法正式切换 MATD3/MADDPG、改 reward/observation/map 等仍需用户确认。

## 5.2 4B/4C seed2 无乐观信号备案（2026-08-07 制定，2026-08-08 生效条件）

触发条件：4B 与/或 4C 的两个 seed 均完成 25k，且 eval20 满足以下全部“无信号”判据：
capture=0/20、avg min-distance ≥ min(random, noop)、训练窗口 capture 终止率无明显上升趋势。
（任一项不满足即视为有弱信号，直接按弱信号走。）

决策顺序（先证据后动作，绝不自动跳 Stage gate）：

1. 深层次分析（只读/低成本，训练前必做）：
   - 训练轨迹解剖：metrics.jsonl 每 1k 窗口的 terminated/truncated/collision、Q/target-Q/td_error、action_norm/speed/log_std/alpha 趋势；capture 事件是否集中在后期。
   - 行为探测：对 25k checkpoint 用 --max-steps 1000/1500 重评估（排除 400 cap 掩盖慢接近）；对比 4A 同条件结果。
   - 可行性上限（oracle）：用简单 seek 控制器（不训练）在同一场景跑 20 集；oracle 也 capture≈0 ⇒ 合同不可学，需重新定标（用户确认）。
   - 数据链/合同核对：effective_config vs 期望、seed 无冲突、warmup/replay/focal 正常。

2. 分支决策：
   - 若存在潜在乐观信号（capture 后期集中 / min-dist 改善 / Q 分化）⇒ 续训到 50k（唯一变量=steps，从 25k bundle 热启动，25k 保存节奏不变），50k 后重新 gate。这是首选。
   - 若 oracle 可学但训练完全无信号 ⇒ 超参/机制调整候选需用户确认（不自动改）：例如 warmup 5000→10000、update_every 4→2、alpha_init 微调；只改一个变量，单 seed 5k 冒烟→25k。
   - 若 oracle 也不可学或合同不匹配 ⇒ 不调参，回合同层报告（evader max_speed / 400 cap / capture 半径），由用户定标。

3. 纪律：4B/4C 各自独立判定，不互相背书；同一轮只做一个动作；所有诊断产物写入台账。

## 5.1 双线证据 TODO 修订（2026-08-07）

- `ctde_pure_random_25k_v2`：world `[ax,ay]` pure-CE 25k 4/4 positive signal → Stage6/7 不再严格串行。
- Stage5 `(a,ω)` anchor 达到 PASS/OPTIMISTIC_PARTIAL 后，Stage6A 与 Stage7A1 并行启动。
- Stage7A1 首轮目标：复现原线 world-axay pure positive signal（actor-prior warmup、1 obstacle、local VCT-LS、parity drag），不作为新 yaw 设计实验。
- uniform-disk warmup 降为 Stage7A1b 二级诊断。
- Teacher-assisted：失败触发式；snapshot 优先用于 post-capture/mixed；encoder 不再默认。
- Action-translation BC：FAILED GATE / NOT ACTIVE TODO（radial saturation 98.68%、terminal velocity error 2.436、position error 1.307）。
- yaw=0/world-axis 不再视为 world action 不可学习的必要解释，改为样本效率/泛化/capture/mixed 难度的潜在因素。

```text
ACTIVE MAINLINE:
Stage 4A -> 4B/4C -> 4D -> 4E -> 5A -> 5B -> continuous (a,w) anchor

THEN PARALLEL:
BODY: 6A pure -> 6B capture -> 6C mixed
WORLD: 7A1 reproduce ctde_pure_random_25k_v2 -> 7A2 capture -> 7A3 mixed

TEACHER: failure-triggered only
ACTION TRANSLATION BC: FAILED GATE / inactive
ALGORITHM SWITCH: not active
```

## 6. 分支与台账

- 阶梯实现分支：`ladder/implementation-20260807`（基于 `main`，合并连续线提交）。
- 专用台账：`docs/CONTINUOUS_MARL_POSITIVE_FEEDBACK_TRACKER_20260807_ZH.md`。
- 阶段报告目录：`artifacts/2026-08-07_positive_feedback_ladder/`。
