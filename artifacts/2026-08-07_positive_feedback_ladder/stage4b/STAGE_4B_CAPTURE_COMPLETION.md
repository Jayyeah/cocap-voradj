# Stage 4B Completion：Capture（moving evader / 0 obstacle）

## 1. 状态

FAIL（2-seed 25k 验证均无“明显优于 random/no-op”信号）

## 2. 本阶段目标

验证多智能体 `(a,ω)` MASAC 在 moving evader、global visibility、0 obstacle 下能形成 capture 接近行为。

## 3. 唯一核心改动（相对 4A）

stationary evader → 原 IQN APF moving evader。

## 4. 结果

| seed | eval20 capture | collision | avg min-distance | 训练 terminated/collision |
|---|---:|---:|---:|---:|
| 1 | 0/20 | 5% | 14.45 | 34 / 34 |
| 2 | 0/20 | 100% | 17.15 | 81 / 79 |

baseline：random capture 0%、min-dist 13.99；noop min-dist 13.59。两 seed min-distance 均劣于 baseline。

## 5. 深层次分析（2026-08-08）

- 轨迹：seed1 学成“低速安全游荡”（6k 后零事件、speed≈0.1、Q 单调降至 -4.9）；seed2 学成“高速冲撞”（后期 term≈coll 最高 10/1k、Q 降至 -9.8、eval 100% collision）。
- 1000 步上限重评估：seed1 0/10 capture、distance_progress 多为负（远离 evader）；seed2 0/10、100% collision、60–351 步撞停。
- 可行性上限（手写 seek 控制器）：10/10 capture，平均 23–49 步 → 任务可学，问题在 RL 训练信号。
- terminated≈collision 表明训练中真实 capture 事件几乎为零（稀疏奖励），+120 capture reward 未被策略体验。

## 6. 结论

Stage4B 在 25k 合同下不可学出接近行为；无潜在乐观信号；不满足“至少 1 seed 明显优于 random/no-op”。

## 7. 是否晋级

否。Stage4D 提前启动条件（4B 积极）未满足，不启动。

## 8. 下一步

进入机制/超参调整分支，候选（warmup / update_every / alpha / reward 侧）提交用户确认；或按用户指示执行 50k 续训等。
