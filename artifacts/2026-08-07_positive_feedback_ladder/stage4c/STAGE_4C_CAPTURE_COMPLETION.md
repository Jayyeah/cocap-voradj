# Stage 4C Completion：Capture（moving evader + 1 obstacle）

## 1. 状态

FAIL（2-seed 25k 验证，eval20 均无“明显优于 random/no-op”信号；4ep 诊断小样本曾出现 PASS 假象，已排除）

## 2. 本阶段目标

验证多智能体 `(a,ω)` MASAC 在 moving evader + 1 obstacle、global visibility 下能形成 capture 接近行为。4C 即动态敌人分支（用户确认 2026-08-08）。

## 3. 唯一核心改动（相对 4B）

num_obstacles 0 → 1。

## 4. 结果

| seed | eval20 capture | collision | avg min-distance | 训练 terminated/collision |
|---|---:|---:|---:|---:|
| 1 | 0/20 | 0% | 14.96 | 40 / 39 |
| 2 | 0/20 | 35% | 15.48 | 56 / 56 |

baseline：random capture 10%、min-dist 13.44；noop min-dist 13.68。两 seed eval20 min-distance 均劣于 baseline。seed2 的 4ep 诊断 min-dist 12.92 < baseline 使自动分析器判 PASS，但 20 集 eval 无稳定改善（15.48），按 eval20 gate 判 FAIL。

## 5. 深层次分析

- 轨迹：seed1 学成低速游荡（Q→-4.6）；seed2 中期碰撞密集、Q→-10.35，eval20 35% collision。
- 可行性上限（手写 seek 控制器）：10/10 capture，平均 22–47 步 → 合同可学。
- terminated≈collision，训练中真实 capture 几乎为零。

## 6. 结论

Stage4C 在 25k 合同下无乐观信号；与 4B 同根因（稀疏 capture reward + 碰撞主导探索）。

## 7. 是否晋级

否。4D 不因 4C 启动（用户规则：4B 积极才提前启动 4D；4C 自身也未达 gate）。

## 8. 下一步

与 4B 合并决策：机制/超参调整（用户确认）或 50k 续训；4B/4C 各自独立判定，不叠加其他改动。
