# Batch02 五线状态与历史信号｜2026-10-12

观测时间：2026-10-12T09:02:18.723391+08:00。五线均保持 **PROVISIONAL**，每线预算 1,000,000 joint decisions，每 25,000 保存并 screen。

## 当前运行与 ETA

| 线 | PID / 存活 | GPU | 当前阶段 | 当前 decisions | 最近 checkpoint | 吞吐 decisions/s | 1M ETA | 全流程估计 |
|---|---|---:|---|---:|---:|---:|---|---|
| N2 | 179017 / alive | 0 | TRAINING | 805,120 / 1,000,000 | 800,000 | 17.73 | 3.1h（12:05） | 14:50–15:22 |
| R2 | 179417 / alive | 1 | TRAINING | 805,888 / 1,000,000 | 800,000 | 17.76 | 3.0h（12:04） | 14:30–14:58 |
| N3 | 214952 / alive | 0 | checkpoint_evaluation | 675,000 / 1,000,000 | 675,000 | 16.59 | 5.4h（14:28） | 17:25–17:59 |
| R3 | 216982 / alive | 1 | checkpoint_evaluation | 725,000 / 1,000,000 | 725,000 | 17.96 | 4.3h（13:17） | 15:04–15:24 |
| V3 | 219016 / alive | 1 | TRAINING | 644,456 / 1,000,000 | 625,000 | 15.97 | 6.2h（15:13） | 18:13–18:48 |

`1M ETA` 是当前 runner 对决策预算完成时间的估值，排除预算后的 selection/final。全流程估计另按最近三次 screen 的中位耗时，加 21–25 个 screen 等效时长（3 个 final、3–5 个 selection checkpoint）。评估耗时会随 episode 长度和 GPU 竞争变化。

五个 PID 均仍存活，命令对应已登记的 runner；没有 failure 文件。N2/R2 使用 `8ce5593`，N3/R3/V3 使用 `3575b74`。最近 PPO 参数和梯度均有限。config/evaluator hash、seed、完整 checkpoint hash/清单及当前评估子进度见 `runtime_snapshot.json`。

## 历史 screen 信号

screen 为每个 checkpoint 的固定 seed 评估：argmax 与 sample 各 10 个 episode。表中 rate 是两种模式均值，总计 20 个 episode，单次变化 0.05 即 5 个百分点。best screen 是描述性峰值，正式选择仍待独立 selection/final seed 域。

| 线 | 初始 capture | 历史 best（step / capture / collision） | 最近完整 screen（step / capture / collision / mean reward） | last−best capture | 信号判断 |
|---|---:|---|---|---:|---|
| N2 | 0.00 | 575,000 / 0.25 / 0.70 | 800,000 / 0.20 / 0.75 / -785.82 | -0.05 | 相对 step 0 有少量 capture completion；best 到 last 略回落，碰撞仍高，信号偏弱且未稳定。 |
| R2 | 0.00 | 550,000 / 0.65 / 0.45 | 800,000 / 0.35 / 0.70 / -429.84 | -0.30 | screen 峰值高于初始值，但到 last 回落且碰撞上升；表现为学习后退化/高波动。 |
| N3 | 0.00 | 550,000 / 0.25 / 0.85 | 650,000 / 0.10 / 0.90 / -370.89 | -0.15 | representation 线有低幅 capture 信号，best 后回落且最近碰撞率高；目前没有稳定提升证据。 |
| R3 | 0.00 | 550,000 / 0.65 / 0.45 | 700,000 / 0.00 / 1.00 / -96.33 | -0.65 | screen 在 550k 有较高峰值，700k 已归零且碰撞全覆盖；呈明显学习后退化信号。 |
| V3 | 0.00 | 550,000 / 0.25 / 0.85 | 625,000 / 0.20 / 0.80 / -528.17 | -0.05 | last 略低于 screen 峰值，碰撞仍高；仅有小幅、波动的 task signal。 |

R2/R3 的 screen 峰值约 0.65（550k）；last 分别回落到 0.35（R2，800k）和 0.00（R3，700k），R3 last collision 为 1.00。N2 从 0.00 到最近 0.20、峰值 0.25（575k）；N3 最近 0.10、峰值 0.25（550k）；V3 最近 0.20、峰值 0.25（550k）。这些 screen 的每模式样本量只有 10，峰谷可能含有明显抽样波动。

reward 与任务完成并非总是同向：R3 的 700k reward 均值比 550k 不那么负，但 capture 从峰值跌至 0，collision 到 1.00。不能只看 episode reward 判能力。逐 checkpoint、分模式历史信号均保存在 `screen_history.json`；训练 update 曲线仍在各 run 原目录的 `training_curve.csv` 和 `metrics.jsonl`。

当前证据更像“曾出现 task-level 学习信号，随后有退化/高波动”，尚不能确定稳定学习或泛化。单训练 seed、固定 4P1E1O screen 和非严格单变量线间差异都限制结论。等五线完成 1M 与独立 selection/final 后再评估；暂不将 best screen 升格为正式 checkpoint。

本次只读取运行数据并归档状态；训练代码、配置、预算、checkpoint 和进程均未修改。下一次查看约 30 分钟后或任一新 25k checkpoint。
