# R3 native reward decomposition

PROVISIONAL。代码历史 parent=N3实现28a3337；算法父线=R2/8ce5593。R3不启用N3 actor，不改变R2 TERL self4/19-slot输入、原TERLActor、central V、PPO、action或terminal。成功IQN Z05/1a9da05是唯一CoCap reward来源。新增reward.py只读provider replay_metadata，PPO继续直接消费provider的reward数组，不使用重建值替换。

| 来源 | 实际reward | R3差异 |
|---|---|---|
| TERL MarineEnv | timestep penalty + 固定近距离/指数远距离dense + global nearby group/boundary + emergency/collision + capture formation goal | 不复用这些公式，不注入CoCap任务 |
| 当前成功 CoCap | native capture + coverage/CE + safety + terminal，原角色/phase/200k CE控制schedule | 不改变任何系数、scale、shaping、时点或done |
| R3 | 上述同一原数组 + 可重建六分类ledger | 数值delta=0，新增telemetry，不是新奖励优化实验 |

分类规则来自provider前动作reward_role。auxiliary_shaping包含原CE control/PBRS及speed_repeat/early_speed/deceleration/motion_penalty字段；CE terminal correction已经包含在CE PBRS，settled_terminal已包含在terminal，support blend是capture/coverage子项，均不重复求和。

非support行：capture=native capture；geometry=native coverage−auxiliary；support=0。support行：support=native capture+coverage−auxiliary，capture/geometry=0。safety=native safety，terminal=native terminal。合计重建原total。terminal独立列出，避免把mix completion误标为capture；geometry是原coverage排除已命名auxiliary后的余额，保留原primitive账本供检查。

这是一种MAPPO-compatible accounting view，并非新的CoCap task definition。没有奖励归一化/团队共享/重新加权/额外辅助奖励。与R2同初始化、动作和训练随机路径时，策略优化应等价；新增R3训练只提供可归因日志，不可声称独立reward改善。

验证复用既有mixed固定transition输入（含capture/boundary/timeout等），不新增coverage或lifecycle场景实验；同时64步scripted parity、256步PPO与16步fresh-process exact resume。残差只作为reward数据合同核验，浮点误差容许度1e-10×(1+|reward|)，没有新增性能门槛。

pilot1024，long固定1M/25k，scratch；任何未来改变reward数值的refinement需要明确科学设计，本任务未自动引入。
