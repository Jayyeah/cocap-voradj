# Stage 4B/4C 调整提案（2026-08-08）

> 状态：全部为提案，未应用。应用前需用户确认；应用时一次只改一个变量，先 5k 冒烟再 25k。

## 定量证据摘要

- 真实 capture 事件（25k replay 逐条统计）：4B s1=0、4B s2=2、4C s1=1、4C s2=0；碰撞事件 34/79/39/56。
- 奖励：p50=0、mean 为负；非 capture 最大稠密奖励 ≤1.88（ring 进度 clip=3.0 但实际几乎不达上限）；capture 大奖励（≥50）仅 4B s2 出现 2 条、4C s1 出现 1 条。
- reward 配置核对（2026-08-08）：capture_reward_mode=ring_importance_ms_v0、capture_timestep_penalty=0.0（capture 场景无 -1 时间惩罚）、omega_ring_ms=2.0、ring_ms_progress_clip=3.0（单步最大 ±6）、k_required=1、capture_stationary_enabled=true → reward 机制接线正确，capture 条件反而宽松；问题不在接线，而在“策略训练中从未稳定接近”。
- 手写 seek oracle：4B/4C 各 10/10 capture（<50 步）→ 合同可学。
- 结论：训练几乎从未体验 capture 正反馈 → 策略坍缩（游荡/冲撞）；需要让稠密接近信号更强或让探索能发现 capture。
- 跨线佐证（2026-08-08 复核参考分支）：world [ax,ay] 的 `ctde_capture_random_25k` 同样 Q1→-14.2、TD≈1.77（Q 坍缩与 body (a,w) 一致），training collision 仅 1/25000 但无 final eval → Q 坍缩非 body/world 特因，支持“训练信号/探索”方向。

## 提案 A：reward 侧强化稠密接近信号（最直接，需明确批准）

原理：capture 场景无时间惩罚，但 ring 进度信号在移动目标下实际幅度小（max≈+1.2/步，未达 ±6 上限），尾部被 -160 碰撞/边界惩罚主导；提高进度 clip 放大“接近”梯度，直接针对“稠密信号太弱”假设。

```diff
- reward.ring_ms_progress_clip: 3.0
+ reward.ring_ms_progress_clip: 6.0
```

单变量、可回退。风险：可能放大对移动目标的噪声（before_d-after_d 含目标移动分量）；若噪声主导，可改回。
验证：5k 冒烟看 Q 是否停止单调下降、接近事件（distance progress）是否出现；再 25k + eval20。

## 提案 B：提高探索熵（target_entropy 单变量）

原理：capture 正反馈需策略在训练中“碰到”capture；当前熵在 15k 后收缩（log_std -0.3→-0.5），探索停止。提高 target_entropy 让策略维持更多随机性，提高发现 capture 的概率。

```diff
- masac.target_entropy: -2.0
+ masac.target_entropy: -4.0
```

单变量。风险：收敛变慢、碰撞可能更多；若冒烟显示 capture 事件出现（>0），说明方向正确。

## 提案 C：提高初始 alpha（alpha_init 单变量）

原理：同 B 的探索方向，但只抬高早期探索权重，alpha 仍按目标熵自适应回落。

```diff
- masac.alpha_init: 0.2
+ masac.alpha_init: 0.5
```

单变量。风险：早期不稳定；与 B 二选一，不叠加。

## 提案 D：学习密度（update_every 单变量）

原理：同一条数据链上每 4 环境步更新 1 次改为每 2 步更新 1 次，让 critic/actor 更充分利用 ring 进度奖励。

```diff
- training.update_every_env_steps: 4
+ training.update_every_env_steps: 2
```

单变量。风险：wall-clock 训练时间增加约一倍；若稠密信号本身太弱则帮助有限。

## 提案 E：50k 续训（无配置改动）

从 25k bundle 热启动，`--total-steps 50000`，25k 保存节奏不变。证据显示无上升趋势，预期收益低；仅作为“确认无信号”的成本最低手段。若用户要求，可只跑 4B seed2（训练期曾出现 2 次 capture）。

## 建议顺序

1. 若允许动 reward：先 A（clip 3→6），单 seed 5k 冒烟；
2. 若不动 reward：先 B（target_entropy -4），单 seed 5k 冒烟；
3. C/D 作为备选；E 不建议首选。

## 纪律

- 一次一个变量；4B/4C 用同一修改（同根因），各自独立判定；
- 冒烟通过后跑 2-seed 25k；每 25k 保存完整 bundle 并回填台账；
- 任何冻结项（reward 结构、map、perception、a_max/ω_max/v_max、drag、focal、grad clip、update ratio）改动均以本文件为批准记录。
