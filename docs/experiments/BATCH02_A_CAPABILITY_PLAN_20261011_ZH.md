# Batch02-A N2/R2 long-run capability validation

本阶段只验证 TERL-MAPPO learner 在 CoCap 合同下的学习能力，保持 **PROVISIONAL**。用户明确授权 N2/R2 各 **1,000,000 joint decisions**、约每 **25,000 decisions** 保存并 evaluate，不自动延长。不是最终 CoCap-MAPPO，不启动 V1/T1/C0、backbone 重构或 lifecycle 迁移。

## 固定版本与任务来源

- 代码 parent：`53ed5b2eabe3b48c280741e29629038b450e7d7b`；公共 E1：`8827c11ef3534e5e9894bfb6b223405f24544be3`。E1 已独立 PASS 并冻结；新增文件在 `batch02_a_long`，冻结19个源码 hash 不变。
- CoCap 主参考：成功 IQN-CoCap `1a9da054ecb5652726005d00d08ab05a2188dd8e` 的 Z05 Stage1 / Forward-Final V2；resolved hash `6955ecf28383f7e146a0ee485ef3e33cb1d1782b197a51ceadd2ef93968d3b5a`。
- TERL 主参考：原 vendored `143359b2722d49c29b4fecc0ad1fd8d46326e45a` 的 `robots/pursuer.py::perception_output`、`robots/robot.py`、`robots/perception.py`。不使用历史失败 MAPPO-CoCap 的 observation/reward/terminal。
- experiment SHA 在提交后写入中央登记及每条线 `manifest.json`；source/config/evaluator hash、checkpoint 目录在首次动作前固定。

| 线 | 命名与 observation | 相对冻结 E1 delta | 相对成功 IQN-CoCap delta |
|---|---|---|---|
| N2 | TERL-MAPPO + CoCap observation + CoCap reward；完整 self9/22 slots、VorAdj/NormSense/Z05/mask/type/order | 无任务/input 变换；新长训/评估 orchestration | task env/reward/observation 为零；learner 替换为 TERL categorical PPO |
| R2 | TERL-MAPPO + TERL observation + CoCap reward；原 self4/19 slots、friend5/E8/O5 | actor 输入绑定 self9→4；直接调用原 TERL sensor 在隔离副本上构造输入 | intentional TERL consumer view：原 raw metre、robot frame、20m friend/obstacle sensor、global enemy、pursuing bit；这是实验定位用 **MAPPO adaptation**，不是 CoCap 原合同 |

两线共用 E1 的 central V self9 / 4P 容量与 action-free geometry。原 TERL critic 的25维中16个 vortex column已在E1按实际无 vortex 环境绑定为9维；本阶段不改 critic。TERLActor/CentralValueNetwork 类、参数键、层拓扑、256/8/4、AW9、masked_pool=false 和 PPO/Adam 保持原样，仅 R2 恢复所需 actor self 输入宽度。两条线不是严格单变量因果实验。

**reward 调度是成功原合同的执行恢复。** `CoCapTrainer._set_coverage_ce_control_weights` 原来位于 IQN trainer，E1 256步 smoke 的 clock 范围没有跨过切换点。新 wrapper 调用同一原函数：pre-action global clock `<200000` 时 `coverage_ce_speed_weight=0`，`>=200000` 时 `0.0005`。不改系数/时点、scale/shaping/辅助 reward、APF、collision、spawn、terminal 或课程。冻结 E1 文件不改写。

两条线均 scratch，fresh actor/V/Adam/ValueNorm/env/RNG，无 teacher/IQN/Batch01 权重 transfer。train env seed `2056101110`、actor seed `109` 配对。沿用 E1 ordinary mixed reset 的4P1E1O scene，不增加 recovery snapshot 采样或 scale/stage curriculum；相对成功 IQN 的 off-policy mixed/recovery replay训练分布，这仍是已声明的 **MAPPO adaptation / 未验证的训练分布**，不能声称完整 lifecycle 已迁移。

PPO：gamma=.99、GAE=.95、clip=.2、3epochs×2minibatches、actor_lr=3e-5、critic_lr=1e-4、entropy=.01、value=1、grad=.5、KL=.02；原 ValueNorm beta=.99999/epsilon=1e-5 与 Adam epsilon=1e-5。rollout256；精确25k边界的末段使用原 TERL runner 的剩余长度处理，不调整优化器或超参。

## 固定 evaluator

版本 `batch02-a-cocap-native-mixed-v1`。任务终止、capture/collision/episode record/time 来源是成功 CoCap provider；IQN expected-Q action selection 改为 MAPPO argmax/sample，标 **MAPPO adaptation**。评估只有原4P1E1O mixed task与普通reset；不做 pure coverage、scale 或 lifecycle 实验。

| 域 | physical seed base | 每模式 episodes | 用途 |
|---|---:|---:|---|
| screen | 2106101100 | 10 | initial0 + 每25k checkpoint 的完整曲线 |
| selection | 2116101100 | 20 | screen top3 + initial + last 的独立选择 |
| final | 2126101100 | 50 | 选定后 evaluate initial、selected 和 last，不参与选择 |

两个模式为 argmax/sample；sample torch seed=physical+100000。两线共享配对 seed，所有域与训练及旧 Batch01 域分离。工程 evaluator smoke 只执行首个 screen seed，明确标 engineering，不参与 best 选择或学习结论；selection/final 不用于工程 smoke 或 tuning。

固定 evaluator reward clock 与成功 IQN formal `make_env/run_episode` 一致：每个 episode 从新 env clock0 开始，不调用训练调度 callback，保留配置初始 CE 权重。训练 clock0→1M 按原调度推进。训练 raw reward 与固定评估 reward 不混合排序；主要能力指标为 capture completion，辅以 collision/boundary、episode length、条件 capture time 与 failure type。

最佳选择顺序在启动前固定：两个模式平均 capture completion 最大、collision 最小、boundary 最小、条件 capture time 最短、较早 step。final 不参与选择；即使 selected=initial 或 selected=last 也照实保留。保留全部 `step_*.pt`，约每2048步写 `resume_latest.pt`；不删除弱 checkpoint。

记录每个 rollout 的 raw reward、capture/collision/boundary episode、entropy、actor/policy loss、value loss、KL、梯度与参数 finite；逐 checkpoint 保留各模式性能与 failure type，报告 best、last、late-window 波动。capture completion 与 full CoCap episode_success 分开记录，非capture时间删失，不给失败 episode 虚构 capture time。

阶段末由 bounded runner 自动生成每线曲线/逐checkpoint表/selection+final依据/学习判断，以及两线对比与 A/B/C/D 下一阶段建议；保持 PROVISIONAL，不自动启动下一批或继续训练。简单率差仅为描述性信号，不新增成功门槛或统计门禁。

## 资源、阻断与范围

GPU0=N2，GPU1=R2；各进程只看自己的 physical GPU，process-local `cuda:0`，CPU线程1。现有 GPU 作业保持不动；资源不足仅排队/降低并发，不变合同。训练错峰启动，周期评估在本线 GPU 串行执行，避免跨线 evaluator 污染 RNG。

新输入/wiring 的64步provider parity、256步PPO update、16步fresh-process exact resume与2episode evaluator smoke只是工程检查，**不代替 held-out capability evaluation，也不提升既有独立验证范围**。无性能 early-stop、reward/architecture/curriculum tuning、自动 retry、预算扩展。若阻断，保存 failure/traceback/最近checkpoint并停受影响线；同一问题两轮修复仍失败或属于环境设计问题就停止，继续不依赖它的工作。非阻断问题记 warning。
