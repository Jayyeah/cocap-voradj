# Batch02-A N2 / R2 pilot 结果

**E1 已独立 PASS 并冻结；N2 observation 与 R2 reward 消费资格 pilot 完成。本批不进入 long run。** 用户确认本批采用消费核验与短 pilot，完整 CoCap observation/reward 已在 E1 接通，二者相对 E1 的任务合同 delta 为零，因此本批不估计两项迁移各自的因果增益。

## 固定合同与 code diff

公共 E1 固定 `8827c11ef3534e5e9894bfb6b223405f24544be3`，成功 IQN-CoCap Z05 Stage1 resolved hash `6955ecf28383f7e146a0ee485ef3e33cb1d1782b197a51ceadd2ef93968d3b5a`。独立 QA Agent 未参与原实现；其 SHA/config/replay/update/resume 全部 PASS，原 checkpoint 按原期望续跑16步精确一致。绑定与签字 hash 见 [frozen_E1_binding.json](../../artifacts/2026-10-11_batch02_a/frozen_E1_binding.json)。本轮没有更改 E1 源码，也没有新增门禁。

新增 pilot 工具与配置已在 `experiment/terl-mappo-batch02-a-20261011@1a5475c` 提交；[code delta](../../artifacts/2026-10-11_batch02_a/code_delta.json) 和 [实际 patch](../../artifacts/2026-10-11_batch02_a/pilot_source_delta.patch)。`src/`、`vendor/` 相对冻结 E1 的 diff 为零。新工具复用 `build/collect/load_checkpoint/save_checkpoint`，通过单独原 provider 重放核验数据、记录账本，不修改训练路径。没有历史失败 MAPPO-CoCap 合同继承。

| 线 | 相对 TERL baseline 的迁移 delta | 相对成功 IQN-CoCap 的 task delta | 本轮对 E1 的 delta 与结果 |
|---|---|---|---|
| N2 | self4/19 slots → 完整 self9/22 slots；friend5→8；原 native 信息 → 成功 VorAdj、NormSense、CE centroid、boundary/clearance、Z05 与原 mask/type/order | 零；Z λ=η=.5、floor=.10、physical_only、原 global-friendly、直接探测 E/O 合同全部保留 | 零 feature transform，仅观察消费；1024步完整 actor/rollout 输入与 provider 精确一致，重复 query 不推进 Z |
| R2 | 原 time/distance/global/emergency/collision/goal → 完整成功 CR-MS/cartesian_disk/support/CE/PBRS/safety/event reward | 零；原系数、pre-action role、事件分配、phase/terminal correction 保留 | 零 reward hook，仅记录组件；1024步 raw reward/infos/APF/done 与 provider 精确一致，进入既有 GAE/PPO |

TERL→CoCap 的环境、4P容量和完整生命周期变化已属于公共 E1，不能归因给 N2 或 R2。IQN expected-Q/replay → 既有 TERLActor categorical/on-policy MAPPO 是 **MAPPO adaptation**；self 输入尺寸绑定、256/8/4、critic、AW9 head、PPO 默认均沿用 E1。N2/R2 相同合同的数据只计作一份证据，不伪造单变量 treatment 或重复长训。

## 短 pilot

CPU，共用一条1024个新增 decisions / 4096 active-agent transitions 的真实 on-policy 轨迹。共同起点是经独立复验的 E1 第256步 checkpoint，完整恢复 actor/V/Adam/ValueNorm/env/APF/RNG；末点1280 decisions。原 rollout256，共4次 PPO 调用 / 24次 minibatch 更新，另1024个 provider 重放 decisions，不计为新增训练样本。没有 V1/T1/C0、额外训练预算或自动扩展。

| 新增 decisions | raw reward mean | entropy | value loss | mean KL | actor update L2 |
|---:|---:|---:|---:|---:|---:|
| 256 | -0.256602 | 2.196778 | 0.029283 | 0.000042344 | 0.046569 |
| 512 | -0.698455 | 2.196206 | 0.535310 | 0.000033662 | 0.038210 |
| 768 | -0.830268 | 2.195500 | 0.614968 | 0.000035620 | 0.036964 |
| 1024 | -0.509116 | 2.194713 | 0.497666 | 0.000030887 | 0.038054 |

每次 PPO update 的参数、指标有限，actor 更新非零，没有 KL early-stop；所有 rollout 的 behavior log-prob 复验通过。value loss/ValueNorm随状态和死亡奖励波动，此短曲线不能证明稳定收敛。原始 [training curve](../../artifacts/2026-10-11_batch02_a/training_curve.json) 与 [CSV](../../artifacts/2026-10-11_batch02_a/training_curve.csv) 保留每次 update 的完整指标。

[N2消费报告](../../artifacts/2026-10-11_batch02_a/n2_consumption.json)：self9、P8×7、E8×7、O5×5、22 masks/types 全部原样到 rollout/PPO；没有新增 actor 信息。gradient 账本保留9个 self 输入列，不用单列零梯度判定 feature 被裁剪。

[R2消费报告](../../artifacts/2026-10-11_batch02_a/r2_consumption.json)：raw total=-2349.508267；capture=-143.757269、coverage=-565.750998、safety=-1640、terminal=0。四个顶层分量重构残差为0；CE center/PBRS/support 是子分量，不能再次累加到 total。clock256→1280，原 CE speed_weight=0，未覆盖200k schedule。

## 结果边界与 long run 决定

本次训练中4个已结束 episode 长度379/195/181/306，均因 boundary 终止，capture event=0、post-capture CE success=0；所有4096个 agent transitions 为 pre_capture。末尾 episode 走到3步，在 pilot 预算处删失。第一段从 checkpoint 中的 episode 第256步开始，episode 汇总含该已保存前缀；不是独立 held-out evaluation。

因此确认的是完整合同消费、真实 PPO 更新和数据一致性；没有证明 capture/recovery/CE 学习成功，也没有 N2/R2 因果增益。Boundary failure 是原成功合同的实际终止输出，未据此修改 reward、boundary 或任务定义。goal terminal/recovery 本次未触发；原 E1 相应 fixture 的通过仍是 transition 证据，不能代替学习能力。

**entered_long_run=false。** 当前未授权长期预算，且两条消费线对 E1 没有有效 treatment。下一次训练设计应明确共同 CoCap baseline、预算、reset/recovery 训练分布与评估方式；如要分离 observation/reward 因果贡献，另定义同一 E1 环境下的 control。它们是未来科学选择，不是本次新增门禁。

末点 checkpoint 留在本地 `runs/batch02_a_pilot_20261011/last.pt`，SHA `d335e2250e9a08469be5cd58faba884d321ad1f342f3e8f23b7ca8fdfd310dc9`；只记录 last，不选 best，不上传权重。原 E1 checkpoint 保持原 SHA。证据分类 `CONSUMPTION_QUALIFICATION_PILOT_PASS`，本 pilot 由 Commander 执行，`pilot_independent_validation=false`，不提升 Batch01 PROVISIONAL 或 BASE QA 状态。
