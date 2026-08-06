# CR-MS support-approach 自动课程（2026-08-03）

当前课程已接通为：`4v1 2M -> 8v2 700k -> 12v3 700k`。

`tools/supervise_crms_support_approach_curriculum.py` 接管当前独立运行的 stage1 4v1
流水线。4v1 完成 2M 训练、20 个 checkpoint screening 和历史最优选择后，自动把
selection 中的 checkpoint 注入 stage2 `pretrained.path`，并启动 8v2 训练、每 100k
screening 与阶段最优 20-rollout/10-GIF finalizer。8v2 完成同一闭环后自动启动 12v3。

正式展示 GIF 默认关闭 faded pursuer trail，保留邻接边、局部感知圈及原有其他显示项。
只有显式向 `tools/finalize_screened_run.py` 传入 `--draw-trails` 时才绘制尾迹；自动课程
不传该参数，因此后续 8v2/12v3 阶段最优 GIF 均默认无尾迹。

2026-08-04 起，阶段历史最优采用以下词典序：capture/mix capture、最大碰撞率、
mix/pure CE strict、mix/pure `CV<0.15`、末态 CV、步数、较晚 checkpoint。CE strict
明确排在面积 CV 之前，避免“面积偶然均衡但未满足 CE 中心覆盖”的模型被优先选中。
已经完成的历史 selection 文件不回写，以保留实际课程注入链路。

课程采用阶段训满后晋级口径，不提前终止。screening 的 coverage/mix 上限分别为
8v2 `1500/2500`、12v3 `1800/2800`。训练使用 GPU0，screening 与正式展示使用
GPU1。

后台与状态：

- tmux：`crms_sa_ce_curriculum_supervisor_0803`
- 日志：`logs/supervise_crms_supportapproach_curriculum_20260803.log`
- 状态：`logs/crms_supportapproach_curriculum_status.json`
- 统一展示：`artifacts/2026-08-02_three_line_stage_best_20rollout10gif/`

后续“更新状态”汇报应为每条训练线提供基于近期实际吞吐的训练 ETA，并将尚未完成
的 screening/正式 rollout 时间单列，避免把训练完成时间误写成整条流水线完成时间。

8v2 300k 与历史所选 500k 的同 seed 正式对比见
`docs/CRMS_8V2_300K_VS_500K_FORMAL_20260804_ZH.md`。结果确认 300k 的 pure/mix
CE strict 均为 100%，500k 均为 45%，而捕获与碰撞表现相同；因此新 CE-first
排序与正式 20-rollout 结果一致。
