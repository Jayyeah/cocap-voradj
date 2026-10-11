# Batch02 E1 environment bridge

本轮完成 TERL-MAPPO learner → 成功 IQN-CoCap Forward-Final V2 的最小闭环。环境、adapter parity、真实 PPO update、fresh-process checkpoint continuation 均通过。分类为 **E1_ENGINEERING_CLOSED_LOOP_PASS**；按上一轮计划中已明确的独立验证定义，**正式 E1_PASS=false，independent_validation=false**。这是证据分类，当前没有未修复的阻断 bug，不新增性能门禁。

用户本轮授权 E1 接入及有限 PPO 检查，禁止长期训练、reward/observation 设计修改和 actor/critic 结构修改。尺寸冲突经用户明确回复：“后续一致，二者冲突时，以cocap的为准，因为我们的目的是成功迁移为原生cocap，而不是terl修改版”。因此保留 CoCap 完整输入，只绑定网络输入尺寸与实体容量；以下明确标为 **MAPPO adaptation**。

源码起点为 Batch01 V3r2 delivery `d6e201052f0eac319342c35cd4efd9171df3fd54`，独立分支 `experiment/terl-mappo-batch02-e1-20261011`。锁定 Batch01 工作树和六条 ARM 均未修改。

## CoCap 合同来源

使用成功 IQN 工作树 `evaluation/iqn-z05-independent-20260923@1a9da054ecb5652726005d00d08ab05a2188dd8e` 的 Z05 Stage1 配置：`configs/experiments/iqn_z_unified_decay_curriculum_20260919/z05_stage1_4p1e1obs_2m.yaml`。`load_config → enable_v2 → scene_config → VorAdjEnv` 的 resolved hash 为 `6955ecf28383f7e146a0ee485ef3e33cb1d1782b197a51ceadd2ef93968d3b5a`。

成功参考是 Stage1 selected1300k IQN checkpoint（SHA `d35b7984d8ca03fa7b536811794578ff402e9b8ad48589e3318f2cb1077d702b`）的既有独立验证，不是本轮 MAPPO 的初始化权重。Stage1 成功不代表全课程成功。本轮没有使用历史失败 MAPPO-CoCap 的配置、reward 简化、观测裁剪或生命周期。

[源码来源证据](../../artifacts/2026-10-11_batch02_e1/source_provenance.json)：9 个实际 CoCap task/provider 文件与成功 IQN 逐字节相同；7 个既有 TERL learner/model/PPO 文件与 delivery 起点逐字节相同。**相对成功 CoCap 的 env/reward/observation 源码 diff 为零**。

## Env diff：TERL 原始合同 → 成功 CoCap 合同

| 轴 | 原 TERL MarineEnv / Batch01 | 本轮 CoCap provider |
|---|---|---|
| physics | AW9，dt=.05×10，P vmax3/E3.5 | 保留成功 AW9 与 .5s 决策周期，直接调用已有积分器 |
| current | 原生 vortex current 每子步参与运动 | 原成功 CoCap discrete 路径零 current；不注入 TERL vortex |
| APF | vendored legacy APF | CoCap `v2_fixed`、`theta_speed` 解释；每个 E 独立 APF 状态 |
| collision | decision 末端 perception collision | `synchronized_swept_v1`；原碰撞类型、死亡、事件输出 |
| boundary | 越界处罚，原生没有成功 CoCap 的硬死亡语义 | 成功 hard boundary / collision death / proximity，系数不变 |
| spawn | 原 Stage1 3P1E1O / 原生 reset 分布 | 成功 Stage1 mixed 4P1E1O map_random；coverage 使用其原 scene reset |
| terminal | 全 E 捕获即结束；active<3；原 wrapper 判限3001 | Mixed 捕获后继续 recovery/CE；active<4；horizon3000；CE hold30 / window500 |

本轮 env diff 是 provider 切换，不是改写 CoCap task。原成功 CR-MS/support/CE/PBRS/safety reward、self9/22-slot observation、Z05、K10、capture event 和 recovery/reset 都从 provider 原样透传。Capture scene 仅使用成功 IQN evaluator 本已有的 capture terminal flags；Mixed 不因 capture 自动 reset。

## Adapter diff 与 MAPPO adaptation

| 旧调用 | 新文件/调用 | 实际改变 |
|---|---|---|
| `native.NativeStage1 → MarineEnv` | [batch02/cocap_env.py](/home/yjq/rl/CoCap1/terl-mappo-batch02-e1-20261011/src/terl_mappo/batch02/cocap_env.py) `CoCapAdapter → VorAdjEnv` | AW9 validation、原 CoCap APF、task-native step；保存原 infos、reward、terminated/truncated；不 auto-reset |
| `native.pack_local/empty_observation` | `pack_observations` | 活跃行各字段值、顺序、mask、type 不变，只 stack；死亡 None 行使用实际 CoCap template 的零行/false mask，active loss 排除 |
| 固定 self4 / 19-slot 输入 | [batch02/runtime.py](/home/yjq/rl/CoCap1/terl-mappo-batch02-e1-20261011/src/terl_mappo/batch02/runtime.py) `build` | **MAPPO adaptation**：actor self encoder 输入4→9；friend 容量5→8，既有 forward 已动态读 slot 数；AW9 head 不变 |
| 固定3P / self25 central V | `central_state/build` | **MAPPO adaptation**：复用既有 action-free geometry encoder；实际4P1E1O，零 vortex 容量；去除不存在的16列 current，self25→9；只作训练 V 输入，不进入 actor |
| Batch01 runner 的 native 采样 | `collect` | 原 task reward；pre-action active masks；先取物理 next V，再处理 reset；既有 PPO/GAE/ValueNorm 算法不变 |
| native runtime checkpoint | `save_checkpoint/load_checkpoint` | 完整 trainer、Adam、ValueNorm、CoCap/APF/cache/phase/Z/K10、RNG 和 counters 原子保存；fresh constructor 地图不覆盖 saved map |

[网络尺寸证据](../../artifacts/2026-10-11_batch02_e1/network_binding.json)：actor/critic 所有 module 路径与类、参数 key 均不变。只有 `entity_encoders.self.0.weight` 的形状 `[256,4]→[256,9]`，以及 critic `self_encoder.0.weight` 的 `[256,25]→[256,9]` 改变；4P 容量绑定没有新增参数层。保持 `TERLActor`、`CentralValueNetwork`、256/8/4、attention、target-selection、AW9 head。采用 fresh 模型，不承诺旧 TERL checkpoint 的跨尺寸权重恢复。

以上 central V schema 是已有 TERL 的训练输入适配，尚未证明是 CoCap 的充分 value 表示；它不是 CoCap observation 设计。完整 reward 归因、训练 clock/reset distribution、N2 表示适配、V1 学习正控与 M1/M2 性能结论不由本次 smoke 替代。

## Rollout parity

[原始 provider parity](../../artifacts/2026-10-11_batch02_e1/provider_parity.json) 直接调用成功 IQN 原工具的 `resolved/make_env` 构建独立实例，和新 adapter 使用相同 state/action。Mixed32、Capture16、Coverage16 共64个 scripted transitions，另11个 edge fixtures；raw observation、reward/components/infos、APF actions、done/terminated/truncated、物理 next state、env RNG 均一致。多次 query/packing 不额外推进 Z，每 decision 更新一次。

覆盖普通运动、零 current、远敌 sensor、硬边界、swept obstacle contact、capture→recovery、horizon、active loss、death+timeout terminal 优先、recovery window499→500。CE hold29→30 独立补充 fixture 保持成功原 threshold 和 reward；[边界证据](../../artifacts/2026-10-11_batch02_e1/boundary_wiring.json) 通过。

真实 actor 采样的256步轨迹，再把相同 P actions 送入成功 IQN 原始 provider 重放：所有 actor 输入、raw reward、task infos、APF actions、episode end 逐步一致。PPO update 前 log-prob 最大误差 `2.384185791015625e-07`。

最初 fixture 将 `post_capture_started` 错误预期为最后 capture 同一 decision 开始。成功源实际为捕获 event/E deactivation 发生当步，recovery 于下一 decision 开始。只修正一次 fixture 预期和下一步断言，未改 provider；[最小复现及修复记录](../../artifacts/2026-10-11_batch02_e1/fixture_expectation_correction.json) 保留原失败。没有两轮失败的未解决问题。

## PPO update 与 checkpoint

全部在 CPU 执行，无长期训练、无 GPU job、无正式 selection/final eval、无训练预算扩张。

| 项目 | 结果 |
|---|---|
| 首次 PPO batch | 256 decisions / 1024 active-agent transitions |
| Optimizer | 原配置3 epochs、6 minibatch updates；参数和指标全部有限 |
| Actor / value loss | -0.023109 / 0.669968 |
| Entropy / mean KL | 2.197118 / 0.000086486 |
| Actor 实际更新 L2 | 0.0596113，非零 |
| 全部 smoke 规模 | 272 unique on-policy decisions；额外16步恢复重放；共3次 PPO 调用（含重放） |
| 额外边界诊断 | 2步 timeout rollout，无 PPO update；不属于上述272步训练轨迹 |
| Timeout / GAE | 第3000步使用物理 pre-reset next V，truncation 保留 bootstrap，advantage 不跨 reset |

[PPO 原始指标/配置/shape](../../artifacts/2026-10-11_batch02_e1/ppo_update.json)。本轮仅证明 learner 能消费真实 CoCap transitions 并执行 PPO 更新，不能据此宣布 capture/coverage 已学会或训练稳定。

Checkpoint 存在本地 `runs/e1_cpu_closed_loop_20261011/resume.pt`，SHA `e69025598651c38c01eb2600ca6d476ee472d3c550b7d2a9efcfeac5e0775190`，保存第256步/1024 agent transitions/6 optimizer steps 的状态；权重不放 Git。

[fresh-process restore](../../artifacts/2026-10-11_batch02_e1/checkpoint_restore.json)：新 Python 进程先用不同 env seed 实例化，再恢复 checkpoint，执行16步 continuation 与下一次 PPO update；batch、episode/trace、metrics、actor/critic 参数、Adam、ValueNorm、完整 runtime、Python/NumPy/Torch RNG 的 canonical hashes 全部与未中断 continuation 相同。验证的是本轮 CoCap checkpoint 的精确恢复，不是旧 TERL 权重迁移。

## E1 状态与 warnings

**工程闭环通过，当前无阻断 bug。正式 E1_PASS 尚未取得：false。** 上一轮已写明独立 bridge 复验后才记正式 PASS；本轮所有执行均由 Commander 完成，因此不能将这些自测改名为独立验证。保留该事实为 warning；不追加训练、性能或形式化门禁。

另外两个未验证范围：本轮 reward clock 从成功 provider 默认0开始，没有覆盖200k CE-control weight 调度；smoke 没有完整 on-policy 学习分布/recovery snapshot pool，也没有正式任务成功率评估。原成功 provider 的 reward/observation/lifecycle 不因这些范围限制而被改写。

R2/N2/V1 可继续无依赖的源码/组件账本准备，共用本轮 E1 source；本轮未实施这些独立 ARM，也未开启任何后续训练。Batch01 的 PROVISIONAL 与 BASE 独立 QA pending 状态保留。

可重放命令和依赖路径见 [artifact README](../../artifacts/2026-10-11_batch02_e1/README.md)。本轮交付不要求再增加工程测试或训练来替代缺少的独立复验。
