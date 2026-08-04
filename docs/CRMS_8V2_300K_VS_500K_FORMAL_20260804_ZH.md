# CR-MS 8v2：300k 与 500k 正式对比（2026-08-04）

## 目的与口径

旧 finalizer 在 capture 与 collision 打平后先比较 `CV<0.15`、再比较 CE strict，
因此历史课程选中了 500k。2026-08-04 已将通用排序修正为先比较 mix/pure CE strict、
再比较 mix/pure `CV<0.15`；完整七点 screening 在新规则下选中 300k。

为验证这一变化，300k 使用与既有 500k 完全相同的正式口径复评：

- capture、coverage、mix 各 20 rollout、各 10 GIF；
- seed 均为 `2026082201..2026082220`；
- capture/coverage/mix 上限为 `1000/1500/2500`；
- 固定 IQN midpoint quantiles；
- 邻接边与局部感知圈开启，faded trail 关闭。

历史 500k selection 与 12v3 注入记录不回写，以保留真实实验链路。

## 正式结果

| 场景 | 指标 | 300k | 500k | 300k - 500k |
| --- | --- | ---: | ---: | ---: |
| capture | capture | 1.00 | 1.00 | 0.00 |
| capture | collision | 0.00 | 0.00 | 0.00 |
| capture | avg steps | 90.70 | 83.95 | +6.75 |
| coverage | CE strict | 1.00 | 0.45 | +0.55 |
| coverage | CV<=0.15 | 0.30 | 0.25 | +0.05 |
| coverage | collision | 0.00 | 0.00 | 0.00 |
| coverage | avg final CV | 0.1911 | 0.1829 | +0.0082 |
| coverage | avg steps | 172.65 | 990.25 | -817.60 |
| mix | capture | 1.00 | 1.00 | 0.00 |
| mix | CE strict | 1.00 | 0.45 | +0.55 |
| mix | CV<=0.15 | 0.25 | 0.25 | 0.00 |
| mix | collision | 0.00 | 0.00 | 0.00 |
| mix | avg final CV | 0.2283 | 0.1901 | +0.0382 |
| mix | avg steps | 296.85 | 540.30 | -243.45 |

## 结论

300k 与 500k 的捕获率和安全性相同，均为 capture/mix capture 100%、全场景零碰撞。
300k 的 pure 与 mix CE strict 均从 500k 的 45% 提升到 100%，同时 pure coverage
平均步数减少 817.60，mix 平均步数减少 243.45。500k 的平均末态 CV 略低，但这种
面积诊断优势没有转化为 CE 中心覆盖成功。

因此，300k 是更合理的 8v2 综合展示 checkpoint，也直接支持“CE strict 必须排在
CV 诊断之前”的新选优顺序。后续自动课程使用新排序；历史 12v3 最终结果仍保留，
因为它实际由旧规则选中的 8v2 500k 注入并最终在 700k 收敛。

## 产物

- 300k：`artifacts/2026-08-02_three_line_stage_best_20rollout10gif/crms_supportapproach_8v2_s2_step_300000_cefirst_comparison/`
- 500k：`artifacts/2026-08-02_three_line_stage_best_20rollout10gif/crms_supportapproach_8v2_s2_step_500000/`

300k 的 30 个主 GIF 已全部校验可读，共 5852 帧；显示元数据统一为邻接边开启、
感知圈开启、尾迹关闭。
