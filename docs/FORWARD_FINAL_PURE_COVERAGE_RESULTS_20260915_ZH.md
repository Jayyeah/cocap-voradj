# Forward-Final Pure-Coverage Corrected Scratch MAPPO 结果（2026-09-15）

## 结论

Pure-Coverage corrected Scratch MAPPO 单 seed 已完成预声明的 `0 → 200k` budget。结果分类为：

`PURE_COVERAGE_LEARNABLE`

这不是单点 best：sample 从 50k 到 200k 的 strict CE success 连续为 100%，argmax 从 75k 开始保持 80%–100%，并且 CE RMS、area CV、strict hold 在多个相邻 checkpoint 上同步改善。200k 终点 argmax/sample 都是 20/20 strict CE success、0/20 collision、30-step strict hold。

由于 baseline 已显示清晰学习，按 gate 立即停止 Coverage reward 修改；Probe A（random-init supervised actor）、Probe B（reward separability）和 Coverage-R2 shaping×2 均没有启动。不能把未启动的 probe 写成失败结果，也不能把本结果外推为 Full-Task 已解决。

## 合同与运行完整性

- branch：`experiment/small-step-ac-migration-20260828`
- result commit：以本文件提交为准；训练 runtime 使用共同 contract commit `25dd0f8` 之后的 runner
- seed：`2026091501`
- budget：`200,000` joint environment decision steps
- checkpoint/eval：0、25k、50k、75k、100k、125k、150k、175k、200k；每点 argmax/sample 各20局
- 初始化：Actor、central V、Adam、ValueNorm 全部 fresh random-init；`teacher_dependency=0`
- transition：`terminal-priority-truncation-bootstrap-weighted-ce-v2`
- collision：`synchronized_swept_v1`
- reward：当前 Final CE centroid-energy + PBRS + speed/control/safety；没有 success bonus、recovery pool 或 post-capture
- 评估 RNG 与训练 RNG 隔离；checkpoint 保留 optimizer、ValueNorm、环境和 RNG 状态

完整逐 episode 数据在 [Coverage artifact](../artifacts/2026-09-15_single_task/coverage/)，最终评估为 [eval_step_200000.json](../artifacts/2026-09-15_single_task/coverage/eval_step_200000.json)，合同 parity 为 [parity.json](../artifacts/2026-09-15_single_task/parity.json)。

## Checkpoint 趋势

下面的 RMS、CE max、area CV、strict hold、time-to-CE 和 discounted return 均为20局均值；`success` 和 `collision` 是20局比例。Coverage 的主 gate 使用 sample，argmax 独立保留。

| step | argmax success | argmax RMS / max / CV | argmax hold / time(s) | argmax collision / return | sample success | sample RMS / max / CV | sample hold / time(s) | sample collision / return |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0% | .2179 / .2739 / .3503 | 0.0 / — | 100% / −60.160 | 0% | .2061 / .2659 / .3644 | 0.0 / — | 100% / −77.586 |
| 25k | 0% | .2011 / .2954 / .3667 | 0.0 / — | 0% / −44.016 | 0% | .2121 / .2660 / .2977 | 0.9 / — | 95% / −53.947 |
| 50k | 0% | .2011 / .2954 / .3667 | 0.0 / — | 0% / −44.016 | 100% | .0406 / .0551 / .0902 | 30.0 / 291.4 | 0% / −37.308 |
| 75k | 80% | .0381 / .0538 / .0753 | 25.9 / 43.5 | 0% / −9.251 | 100% | .0350 / .0500 / .0884 | 30.0 / 72.7 | 0% / −16.508 |
| 100k | 100% | .0251 / .0372 / .0615 | 30.0 / 48.3 | 0% / −9.971 | 100% | .0344 / .0502 / .0673 | 30.0 / 65.2 | 0% / −13.263 |
| 125k | 95% | .0445 / .0613 / .0890 | 28.5 / 62.8 | 5% / −12.371 | 100% | .0323 / .0448 / .0814 | 30.0 / 56.0 | 0% / −12.903 |
| 150k | 100% | .0247 / .0398 / .0512 | 30.0 / 46.2 | 0% / −9.325 | 100% | .0255 / .0383 / .0567 | 30.0 / 52.0 | 0% / −10.986 |
| 175k | 100% | .0267 / .0431 / .0635 | 30.0 / 47.6 | 0% / −9.534 | 100% | .0290 / .0455 / .0649 | 30.0 / 52.4 | 0% / −10.728 |
| 200k | 100% | .0208 / .0332 / .0486 | 30.0 / 83.3 | 0% / −9.554 | 100% | .0246 / .0347 / .0676 | 30.0 / 48.8 | 0% / −9.398 |

`RMS / max / CV` 分别是 CE RMS、CE max error、area CV。成功局的 time-to-CE 只在 strict success episode 上统计；step0、25k 的无成功局记为删失，不当作完成时间。

### 200k 终点分布

| mode | strict success | CE RMS mean / P50 / P90 | CE max mean / P50 / P90 | area CV mean / P50 / P90 | hold mean | time-to-CE mean / P50 / P90 | collision / boundary |
|---|---:|---:|---:|---:|---:|---:|---:|
| argmax | 20/20 | .02076 / .02102 / .03381 | .03322 / .02898 / .06585 | .04859 / .03588 / .10025 | 30.0 | 83.35 / 45.75 / 157.35 s | 0/20 / 0/20 |
| sample | 20/20 | .02459 / .02184 / .03467 | .03473 / .02989 / .05387 | .06761 / .05355 / .10679 | 30.0 | 48.75 / 42.50 / 75.85 s | 0/20 / 0/20 |

200k sample 的 strict CE success、hold 和 collision 均稳定；argmax 125k 的 1/20 collision 是中间波动，之后恢复为0/20，不改变持续窗口判定。

## Reward 分量与 return

最终20局的未折扣分量均值如下；Coverage 没有 capture、support 或 terminal success reward。

| mode | CE center | CE PBRS | CE control | safety | capture | terminal | total return | discounted return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| argmax | −13.8681 | +.5832 | −.1668 | −5.1250 | 0 | 0 | −18.5766 | −9.5541 |
| sample | −11.8654 | +.5632 | −.0688 | 0 | 0 | 0 | −11.3709 | −9.3985 |

折扣分量中 argmax/sample 的 CE center 分别为 −9.1292/−9.7961，CE PBRS 均为 +.4446，CE control 为 −.0786/−.0470，safety 为 −.7909/0。评估中的奖励函数与训练完全相同，没有为结果单独重算或修改 shaping。

## PPO 训练健康

`learning.jsonl` 共记录781个完整 rollout update（最后完整 update 为199,936 step；200k checkpoint 保留64步 partial rollout）。全程指标 finite，ValueNorm、raw/normalized advantage 和梯度均有记录。

| 指标 | 全程范围 | 最后一次完整 update |
|---|---:|---:|
| entropy | 1.6876 – 2.1972 | 1.8345 |
| exact full-batch KL | 1.38e−5 – .09139 | .00659 |
| clip fraction | 0 – .2124 | .02214 |
| explained variance | −.6775 – .9934 | .8895 |
| value loss | 8.57e−5 – 1.8447 | .000354 |
| raw advantage mean / std | −38.25 – 16.83 / .474 – 69.44 | −.1253 / .9241 |
| normalized advantage mean / std | 约0 / 1 | −3.7e−9 / 1.0000 |
| actor grad norm | .3076 – 7.1058 | 1.6191 |
| value grad norm | .0047 – 11.9011 | .0338 |

没有触发非finite、错误恢复或 checkpoint 污染。KL/clip 的历史峰值被保留在完整日志中；它们没有与 CE 学习趋势同步恶化到停止条件。

## Gate 裁决与后续

Coverage gate 的持续证据为：

- sample：50k、75k、100k、125k、150k、175k、200k 连续 strict success=100%，RMS/CV 相对 step0 大幅下降，strict hold=30；
- argmax：75k–200k 多个相邻 checkpoint 保持高 success，100k、150k、175k、200k 为100%；
- collision：sample 在75k之后全为0%，argmax 除125k的5%外为0%；
- 200k 终点双模式均20/20 success、0 collision。

因此本线是 `PURE_COVERAGE_LEARNABLE`，并执行以下 gate 行为：

1. 停止所有 Coverage reward 修改；
2. 不启动 Probe A random-init supervised representation；
3. 不启动 Probe B reward separability；
4. 不启动 Coverage-R2 shaping×2；
5. 下一核心问题转为 Full-Task capture/coverage gradient interference，但要等待 Pure-Capture 线完成后再形成总因果结论。

本结果支持“Coverage 单任务 RL 可学”，不支持“Coverage reward scale 是当前瓶颈”，因为 R2 根本没有被需要或测试；也不支持“Full-Task 已可学”。总体 MASTER 结论继续等待 Capture 500k 分类。
