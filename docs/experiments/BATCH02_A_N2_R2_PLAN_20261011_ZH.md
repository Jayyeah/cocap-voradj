# Batch02-A：N2 / R2 消费核验与短 pilot

**后续授权更新：用户已授权 N2/R2各1M、每25k checkpoint/evaluate 的 PROVISIONAL capability validation；R2现明确使用原TERL observation。** 见 [固定长训计划](BATCH02_A_CAPABILITY_PLAN_20261011_ZH.md)。下文保留当时共享消费 pilot 的范围与预算记录，不代表当前长训授权。

用户已确认：先由一个独立 QA reviewer 复验 E1 `8827c11`，仅 SHA/config/replay/update/resume，不修改代码、不扩大门禁；PASS 后冻结 bridge。**该复验已 PASS 并冻结**，见 [独立报告](BATCH02_E1_INDEPENDENT_VALIDATION_20261011_ZH.md)。Batch02-A 只推进 N2 observation 和 R2 reward，不启动 V1/T1/C0，不自动长训。

用户同时确认本批按“先做消费核验与短 pilot”执行。**1024步 shared pilot 已完成，消费资格通过，不进入 long run**，见 [结果](BATCH02_A_N2_R2_PILOT_REPORT_20261011_ZH.md)。E1 已原样接入完整 observation/reward，**N2/R2 相对 E1 的任务合同 delta 为零**，所以本批是迁移消费资格确认，不能用来估计 observation/reward 各自的因果增益。没有自行构造旧 TERL 合同在 CoCap 上的反事实 control。

| 线 | 相对 TERL baseline 的目标 delta | 相对成功 IQN-CoCap 的 task delta | 相对冻结 E1 的实现 delta |
|---|---|---|---|
| N2 | self4→self9，friend5→8，19→22 slots；原 native 信息→原 VorAdj/NormSense/Z05/CE-centroid/boundary-clearance tokens 与原 mask/type/order | 零；直接使用成功 Z05 完整 observation，λ=η=.5、floor=.10、physical_only、原 global-friendly | 零 task/input 变换；核验原 observation 到 rollout、actor 和 PPO batch 的完整消费，新增只读账本 |
| R2 | native time/distance/global/emergency/collision/goal → 原 CR-MS/cartesian_disk/support/CE/PBRS/safety/event reward | 零；原 reward 系数/分配/pre-action role/phase/terminal correction 不变 | 零 reward hook；核验 raw reward 到 rollout/GAE/PPO 的消费，新增组件账本，不替换 R1 shaping |

TERL→CoCap 的 physics/current/APF/collision/boundary/spawn/terminal 变化已经由公共 E1 承担，不能归因给 N2 或 R2。IQN expected-Q/replay → 既有 TERLActor categorical MAPPO/on-policy 是 **MAPPO adaptation**；self input 与4P capacity 的必要绑定已在 E1 经用户授权。网络层拓扑、PPO 默认、action AW9、reward、observation、reset/lifecycle 不在本批变化。

短 pilot 共用一条1024个新增 decisions 的真实 on-policy trajectory，按原 rollout256完成4次 PPO 调用，分别输出 N2/R2 消费报告。共同起点是已独立复验的 E1 第256步 checkpoint，完整恢复 actor/V/Adam/ValueNorm/env/APF/RNG；这是 CoCap 同合同 resume，不是旧 TERL 权重跨尺寸 transfer。总预算只计1024，不将共享证据当成两个独立训练样本。

全部 CPU，保持原 actor seed109、256/8/4、原 PPO 参数。clock256→1280，仍处成功原 schedule 的 speed_weight=0 区间；不强行设置200k或2M offset。普通 mixed reset 保持 E1 原成功 scene，不在此加入 IQN recovery snapshot 采样；完整训练分布适配仍未验证。

pilot 记录每个 rollout 的 entropy/KL/value loss/ValueNorm/真实参数变化/原 reward 分量、已结束 episode 与末尾删失；只报告末点 checkpoint，不按 best score 选结果。新工具仅收集与核验数据，冻结环境、learner、reward、observation、actor/critic 源码不修改。

E1 未 PASS 前不运行 pilot。若已有五项复验失败，不修源来追认当前 SHA；保存最小复现、影响范围及可继续工作的账本/静态合同。非阻断范围记 warning。pilot 不增加任务成功率或未定义性能阈值作为新门禁。

是否进入 long run 在 pilot 结果中明示；本批没有长期预算，不能因短检查通过自动扩展。若后续要求 N2/R2 单变量因果训练，先定义同一冻结 E1 上的 control 与有效 delta，而不是复跑两条完全相同的长训练。
