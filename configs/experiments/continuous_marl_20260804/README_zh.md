# 连续动作实验配置 namespace

这是连续动作 PSA-MASAC 重构线的独立配置目录。P0 只建立 namespace，不启动训练。

> **当前状态优先级（2026-08-06）：** 本 README 只说明配置 namespace；当前六条 500k 长训、AW bridge、实际 report 复盘和后续 TODO 以 [`docs/CONTINUOUS_ACTION_MARL_EXPERIMENT_TRACKER_20260804_ZH.md`](../../../docs/CONTINUOUS_ACTION_MARL_EXPERIMENT_TRACKER_20260804_ZH.md) 为准。README 中的“P0 不启动训练”是历史 namespace 建立规则，不表示当前没有训练。

约束：

- 旧 `unicycle_discrete` 配置和 `CoCapIQN` 路径不得从本目录反向修改；
- 连续线配置必须显式写 `algorithm`、`action_mode`、`yaw.mode`、`yaw.init` 和 `a_max`；
- 训练 run 使用 `runs/continuous_marl_<run_id>/`；
- 产物使用 `artifacts/2026-08-04_continuous_marl_refactor/`；
- 每个配置从 step 0 开启 screening，GIF faded trails 默认关闭。

历史首个学习型配置只能在实验台账 P0--P4 Gate 全部通过后建立，顺序为：

`local SAC + pure CE smoke -> local SAC + capture -> local SAC + mixed CR-MS -> PSA-MASAC`

当前实际运行还包含独立 AW bridge namespace/CLI 分支；AW 不是 ax/ay 合同的替代品，结果须分支复盘。
