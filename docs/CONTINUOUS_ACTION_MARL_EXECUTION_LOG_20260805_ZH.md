# 连续动作 MARL 实际训练实验专用记录

> 建立日期：2026-08-05  
> 维护对象：PSA-MASAC 连续动作重构线  
> 当前状态：**formal matched 5k 任务 gate 未通过；2026-08-05 已启动 ax/ay 与 AW 各三条 500k scratch 长训，当前只等待 milestone/final report，结果必须按核心台账的深度复盘协议判定**
> 本文用途：给后续 Codex/Luna 等模型使用的实际训练台账、范围冻结和故障处理合同。  
> 上游详细台账：`docs/CONTINUOUS_ACTION_MARL_EXPERIMENT_TRACKER_20260804_ZH.md`

> **动作合同修订：** 用户已确认采用显式 `a_x,a_y`。旧 velocity feasible-layer 的 P6 结果全部保留为历史故障证据，不可与新线合并或续训；当前规范见 `docs/CONTINUOUS_ACTION_AXAY_REFACTOR_20260805_ZH.md`。

本文是执行层记录，不替代完整架构文档。每次训练、修复或改变判定前，先更新本文；旧结果不删除、不覆盖、不用新结果改写历史结论。

## 1. 最终检查结论

截至 2026-08-05，原 velocity 入口已关闭；ax/ay 需要从下面的唯一新入口开始：

```text
4v1 / scratch / local SAC / `acceleration_2d_body` / a_max=0.8 / 新 tag
```

旧 P0--P5 工程记录已通过各自历史门禁；P6 的 velocity 25k 因动作合同错误作废。ax/ay 的 25k report、replay、严格 resume 和六个 screening 点均通过工程审计，但 capture/CE strict 全部为 0，不能进入趋势扩展、central 或课程晋级。

本轮 ax/ay 验证证据：

- 聚焦回归：31 passed，2 warnings。
- local SAC smoke：12 joint transitions、1 update、`all_finite=true`，report `artifacts/2026-08-04_continuous_marl_refactor/p4_local_sac_smoke_report_axay_contract.json`。
- central SAC smoke：12 joint transitions、1 update、`all_finite=true`，report `artifacts/2026-08-04_continuous_marl_refactor/p5_central_sac_smoke_report_axay_contract.json`。
- P6 local 16-step screening：`replay_size=16`、13 updates、`all_finite=true`、`action_validation_rate_mean=0.0`，新 tag `axay_p6_smoke`；随机 scratch 的三类 screening 均未成功且 collision=1.0，仅作工程 smoke，不作性能结论。
- speed audit smoke：4 steps、`all_finite=true`、`action_validation_rate_mean=0.0`；三类 screening 的 `speed_limited_rate` 已写入 report（约 0.27--0.64），证明 env `v_max` 触发可被单独统计。
- ax/ay oracle 修正后：三候选 `a_max=0.4/0.8/1.6` 均 `contract_ok=true`、`no_collision_ok=true`、`nan_free=true`、`action_validation_failure_rate_mean=0`，candidate gate 全部通过。

当前 ax/ay 25k run（已完成）：

```text
PYTHONPATH=src:. python3 tools/run_continuous_p6_screening.py \
  --critic-mode local --a-max 0.8 --init scratch \
  --seed 2026080508 --total-steps 25000 --batch-size 64 \
  --update-every 1 --screen-interval 5000 --screen-episodes 2 \
  --device cuda --tag axay_local_amax08_scratch_25k_20260805_r2
```

PID `391852` 已正常退出。已生成 step1/5000/10000/15000/20000/25000 checkpoint 与 replay；最终 report 时间约 14:15，无 OOM/NaN/Inf/动作合同报错。

最终 report：`artifacts/2026-08-04_continuous_marl_refactor/p6_screening/axay_local_amax08_scratch_25k_20260805_r2_report.json`；`replay_size=25000`、`updates=24937`、`all_finite=true`、`action_validation_rate_mean=0.0`，manifest 为 `acceleration_2d_body`、`v_max=3.0`、`a_max=0.8`。六个 screening 点完整覆盖三场景、每点 2 episodes；capture 与 CE strict 全部 0，后五个点 collision 全 0，但 episode 多为 128 步未成功，属于低速/停滞而非任务学会。step25k 的 CV<=0.15 偶有单 episode 命中，但 strict 均为 0，不能替代 strict。

严格 resume：`axay_local_amax08_scratch_25k_20260805_r2_resume4_report.json`，从 step25000 checkpoint/replay 恢复 4 steps，`replay_size=25004`、`updates=4`、`all_finite=true`、manifest/action contract 一致。

关键产物 SHA256：final report `67c67e6523c08e38bebac54dbed0093a54ed1700b30a36acacd3715934341fb2`；final replay `973588494466485904c82c74c690d28aa61d61ccbcdd897b8731a26d0955c15b`；step25000 checkpoint `55baa3a03aa18da613dac97647d90d930cf3f4732e1676e8bf385ceb41833572`；resume report `677307c310454dc3e59d075ac1a07f8fc748064fbf7927929b07a216032891e7`。

前一条同 seed 进程处置记录：

```text
axay_local_amax08_scratch_25k_20260805
```

该 PID（320048）约 17 分钟时仅有 step1 checkpoint，随后已安全终止；事后基准显示 100 transitions/37 updates 约 16.9 秒，25k 约需 1 小时，因此它处于训练阶段而非 screening 死锁。该 run 未产生最终 report，不计入 ax/ay 实验结果；step1 checkpoint/replay 保留作审计。独立三场景 screening 基准为 CPU 约 24.5 秒、CUDA0 约 5.7 秒。

## 2. 冻结的实验实现范围

### 2.1 首版必须实现

1. 仅连续化 pursuer：每个 pursuer 输出 body-frame 二维加速度 `a_cmd=(a_x,a_y)`，且 `||a_cmd||<=a_max`。
2. `v_max=3.0` 由环境动力学限制；`a_max` 为策略侧可调参数，首选 `0.8`，`1.6` 只作压力候选，`0.4` 仅作安全对照。
3. Actor/adapter 只保证 `a_max`；env 负责速度上限、碰撞、边界和子步积分，速度上限触发必须记录，不将裁剪后的速度写回 action。
4. yaw 首版 hold，连续线 `yaw_init=0`；不加入 yaw action、速度定 yaw 或扇形 FOV。
5. 共享局部 Transformer Actor；训练期使用两套完全独立的 centralized entity-attention twin critics（PSA-MASAC）；Actor 只看局部观测，Critic 才看全局实体集合、joint final command 和 mask。
6. 一个 joint replay ring，采样比例固定为 70% uniform、20% regime、10% event，并保留 coverage-only floor；replay 保存真实发送的 `actions`（body-frame acceleration）。
7. 保持新 capture、CE、VCT-LS、support/CR-MS reward 语义；保持现有有限半径全向感知和 boundary self token。
8. 从 step 0 开 deterministic screening；每个 screening checkpoint 同步保存 checkpoint、replay 和 report。stage-best 才自动正式 20-rollout/10-GIF，尾迹默认关闭。

### 2.2 首版明确不做

- 不删除、替换或重写旧 `CoCapIQN/DQN`，旧 `unicycle_discrete` 默认行为必须不变。
- 不连续化 evader，不改 reward、termination、capture/CE 判定和 boundary 表示。
- 不把 phase、regime、event、global state 或 agent ID 输入 Actor。
- 不同时引入 PER、HER、TQC、GRU、flow、FACMAC、Set-Attention MATD3 等消融结构。
- 不做 IQN transfer，除非先完成确定性的 encoder key/mapping 合同和独立回归。
- 不在前置 gate 未通过时启动大规模课程或 2M 正式训练。

## 3. TODO 总表与完成情况

状态含义：`[x]` 有命令、结果和产物证据；`[~]` 正在进行或部分完成；`[ ]` 未开始；`[!]` 失败/阻塞；`[S]` 首版暂缓。

| 阶段 | 内容 | 状态 | 完成/进入条件 |
|---|---|---:|---|
| P0 | 旧 CR-MS 基线、manifest、三规模固定 seed regression、namespace 隔离 | [x] | 三规模 report passed；旧轨迹 hash 匹配；`draw_trails=false` |
| P1 | `acceleration_2d_body`、body/world transform、yaw hold、`AccelerationActionAdapter`、env speed cap | [x] | ax/ay 合同测试通过；旧离散线回归未变 |
| P2 | `a_max=0.4/0.8/1.6` scripted oracle 筛选 | [x] | 0.8 冻结主候选，1.6 压力候选 |
| P3 | `LocalEntityTokenEncoder`、radial acceleration Actor、mask/log-prob、`a_max` contract | [x] | Actor/local/central 窄测试通过；旧 IQN 未触碰 |
| P4 | joint replay、70/20/10 sampler、local SAC、strict resume | [x] | replay/采样/数值/恢复 smoke 通过 |
| P5 | centralized twin critics、PSA-MASAC、central smoke | [x] | central contract smoke 通过；无性能结论 |
| P6-a | ax/ay 4v1 local smoke/preflight | [x]/[!] | 工程 report/replay/resume/screening 已齐且通过；capture/CE strict 学习 gate 失败，转入简化任务诊断 |
| P6-b | ax/ay 100k--200k 趋势筛选 | [ ] | 仅在 ax/ay 25k 工程与学习 gate 通过后开启 |
| P6-c | 200k--300k local/central、a_max 对比 | [ ] | P6-b 保留至少一个候选 |
| P6-d | stage-best 正式 20-rollout/10-GIF | [ ] | 非劣门槛达标；尾迹关闭 |
| P6-e | 晋级组合 3 seeds | [ ] | stage-best 非偶然；报告 mean/std |
| P7 | 4v1 晋级后 8v2/12v3 课程和 screening | [ ] | 仅从 P6 solid/stage-best 启动；各阶段保留可恢复产物 |
| P8 | 离散/连续、local/central、critic 等单变量消融 | [S] | P6/P7 solid 后再做 |
| P9 | 三规模正式对照、展示和 handoff | [ ] | 所有 stage-best、manifest、GIF、3-seed 结果齐全 |

## 4. 已完成证据与当前 run 台账

### 4.1 已完成的关键证据

- 全量 P0--P5 回归：`45 passed, 2 warnings`。
- 修正前 local/central P6 短 smoke：均 16 transitions、7 updates、finite。
- ax/ay local/central smoke：`a_max` 合同通过；速度上限改由 env 统计 `speed_limited`，不再使用策略 `projection_rate_mean` 作为当前 gate。
- strict resume：从最终 checkpoint+replay 继续训练，`resumed_from`、replay_size 增长、updates finite。
- 修正前 25k 诊断 run：`replay_size=25000`、`updates=24937`、`all_finite=true`，但 raw desired action 与 replay/env 的 final feasible action 不一致，性能结论作废。
- ax/ay r2 25k：`replay_size=25000`、`updates=24937`、`all_finite=true`、`action_validation_rate_mean=0.0`；六个 screening 点齐全，capture/CE strict 全 0，后五个点 collision=0 但 episode 以 128 步结束，判定“工程 gate 通过、学习 gate 失败”。runtime joint replay schema_version=3，共 25,000 records/100,000 agent actions，动作范数 mean `0.6782`、p95 `0.7948`、max `0.7999`，reward mean `-0.2639`，terminal fraction `0.00644`。
- replay 元数据诊断：`coverage_only/pure_coverage=8353`、`active_target/pre_capture=16647`、`discovery event=169`，terminal-any fraction `0.00644`；数据不是单一场景或完全没有 discovery，但当前 runner 的事件样本仍需在后续 sampler 诊断中确认实际抽样占比。
- SAC 数值诊断：final update tail 全部 finite，alpha 已从初始约 `0.2` 降至约 `0.05448`，Q 均值约 `-28--35`；不是 NaN/Inf，但探索温度明显收缩，与后期低速停滞现象一致，下一步优先做 alpha/log-prob/奖励尺度的最小单变量诊断。
- ax/ay r2 strict resume：从最终 checkpoint/replay 继续 4 steps，`replay_size=25004`、`updates=4`、`all_finite=true`，manifest/action mode 一致。

### 4.2 核心实现哈希（当前工作树）

| 文件 | SHA256 |
|---|---|
| `tools/run_continuous_p6_screening.py` | `ff13d8be9022071e5955a75fa0cfe187e43e070f99478016983a1b17157b02a4` |
| `src/cocap_voradj/models/continuous/radial_actor.py` | `1d42ebb4c3668bd635a2fb7b2b12b197ebbef966cc0131c2fe61441e82552e52` |
| `src/cocap_voradj/training/continuous/local_sac.py` | `07d3433d0d8c67f0b97788e5cdf57fd5bdf76ea62f826ba73c343c6a9b3436a5` |
| `src/cocap_voradj/training/continuous/central_sac.py` | `819dffb93e16c400ae70785f27336a9dab94a9fac55936b231521169c07bcbbb` |

### 4.3 运行记录

| run | 状态 | 结论/产物 |
|---|---|---|
| `p6_local_smoke4` | passed | 16-step local screening smoke |
| `p6_central_smoke` | passed | 16-step central screening smoke |
| `p6_local_amax08_scratch_25k` | invalidated | raw/final action mismatch；保留 report/replay/checkpoint 作为 BUG-20260805-02 证据 |
| `p6_local_amax08_resume_after25k` | passed as recovery test | strict resume 机制通过，不挽回原始性能结论 |
| `p6_local_amax08_feasible_25k` | historical velocity; invalid | report manifest 为 velocity_2d_body，不作为当前结果 |
| `axay_local_amax08_scratch_25k_20260805_r2` | engineering preflight passed; learning failed | 25k report/replay/checkpoint/resume 齐全；六点 capture/CE strict 全 0，不晋级 |

## 5. P6-a 完成判据和下一步顺序

修正后 25k 的工程 preflight 已满足下列条件并标 `[x]`；其中任务成功率趋势另作为学习 gate：

1. report 显示 `replay_size=25000`、updates 与实际步数一致、`all_finite=true`。
2. step 0/5k/10k/15k/20k/25k 均有 screening、checkpoint、replay snapshot；capture/pure-CE/mix 三类字段齐全。
3. Actor 动作满足 `a_max`；环境速度满足 `v_max`，并单独记录 `speed_limited`、actual acceleration 和 jerk。
4. 从最终 checkpoint+replay 严格 resume 至少 4 steps，manifest、RNG、replay size 和 updates 正确。
5. 记录 wall time、updates/s、CPU/GPU、显存、replay 大小；无 OOM、NaN、Inf、死锁或 silent clip。
6. 将成功率、collision、CE strict、steps、CV、速度/加速度/jerk、speed-limited rate 一起审计。短 smoke 或 loss 下降不得单独判定任务学会。

通过后严格按此顺序推进：

```text
P6-a 25k report/audit
  -> P6-b local 100k--200k trend
  -> central 与 a_max=1.6 单变量比较
  -> 保留组合进入 200k--300k
  -> stage-best 20-rollout/10-GIF（默认无尾迹）
  -> 3 seeds
  -> P7 8v2/12v3 课程
```

若 P6-a 数值或合同失败，停在 P1--P5 诊断；本次合同正确但任务完全无趋势，必须先查数据、reward、bootstrap、log-prob、sampler 和初始策略/碰撞机制，再决定单变量实验，不能直接扩大规模。

## 6. 遇错修正逻辑

每次故障都新建 `BUG-YYYYMMDD-NN` 记录，保留首个坏日志、batch、seed、checkpoint 和原始产物。

| 等级 | 典型问题 | 允许动作 |
|---|---|---|
| L0 | 日志、拼写、测试夹具 | 直接修；重跑相关测试 |
| L1 | shape、mask、frame、索引、key、数值容差 | 最小修复；新增回归测试；受影响 run 明确标记 |
| L2 | lr、tau、alpha、batch、UTD、capacity | 一次只改一个变量；保留对照 run |
| L3 | action、observation、reward、termination、算法结构 | 停止长训；先更新本文并征得用户确认 |

固定决策树：

- 旧线回归失败：立即停，不更新 baseline hash 来“接受”失败。
- 动作合同失败：只查 adapter/frame/dynamics；env 不加入 silent clip。
- NaN/Inf：保存首个坏 batch、RNG 和 checkpoint，先做最小复现。
- replay/manifest 不一致：strict reject；没有确定性 converter 不跨 run。
- local 与 central 都不学：优先查 action/log-prob/reward/bootstrap/replay。
- 合同通过但 25k 所有 strict/capture 均为 0：先做最小可学习性诊断（随机/脚本策略、奖励分解、初始碰撞率、观测尺度和 episode 终止），不把长训当作修复。
- local 学而 central 不学：锁定 Actor/action/replay，只查 global schema、mask、target 和 critic 梯度。
- speed-limited 长期高：查 `a_max`、`v_max`、制动距离和环境积分；不能将速度裁剪改写成 action，也不能放宽 env 物理上限。
- capture 好而 CE 差：先查 deterministic eval、a_max、alpha、抖动；不能用 CV 代替 CE strict。
- 4v1 好而大规模差：查 masked mean、critic capacity、课程和 failure/mask，不先添加固定 agent ID。
- OOM/吞吐问题：先降低 batch、replay capacity、评测并发或日志频率，不改任务语义。

BUG-20260805-02/03 已按用户确认通过显式 `a_x,a_y` 关闭：加速度是 Actor、replay、critic 和 env 的同一 action，取消 velocity feasible layer，旧 velocity replay/schema 不可续训。当前若出现速度上限触发，只诊断 env 动力学与 `v_max`，不得重新引入策略速度投影。

## 7. 预期结果与后续调整

这些是预注册假设，不是通过证明：

- 0.4 约束最安全但可能过慢；0.8 预期是主平衡点；1.6 只用于压力比较，不能因速度高自动晋级。
- 连续动作应减少离散切换、速度抖动和无效动作，同时保持 CR-MS 的 capture/CE strict；只变平滑而成功率下降算失败。
- local SAC 若在 pure CE、capture 出现趋势，说明动作、replay 和基础 credit 链可用；central 才有资格继续比较。
- PSA-MASAC 若在 mixed/有限视域协同上不劣于 local 且多规模稳定，才证明 CTDE 有价值；否则记录为“连续动作可行、central 未带来收益”。
- 3-seed PSA-MASAC 达到当前 CR-MS 非劣门槛且至少一项路径/平滑/制动指标改善，判成功；只 local 达标判部分成功；合同或数值失败不得形成算法性能结论。

当前 CR-MS 对照门槛：capture、mix capture、pure/mix CE strict 均不得比对应基线低超过 5 个百分点；collision 不得高超过 2 个百分点。排序固定为 capture → CE strict → collision → steps → CV，CV 不能替代 strict。

调整原则：

1. 达到趋势但 speed-limited 高：先比较 `a_max=0.8/1.6` 和环境速度饱和率。
2. 达到 capture 但 CE 不足：只做 CE/coverage 数据和探索单变量调整。
3. 达到 local 而 central 失败：只改 central critic 的 schema/mask/容量，不重写 Actor。
4. 4v1 稳定后再启动 8v2/12v3；每个规模都必须 screening、checkpoint、resume 和 stage-best 展示。
5. 只有正式 stage-best 才生成 20-rollout/10-GIF；尾迹默认关闭，显式开启必须写入命令和 metadata。

### 7.1 25k 无学习趋势后的诊断与当前对照

- 零速度脚本：pure-CE 2 episodes、capture/mixed 各 2 episodes 均无碰撞，但也未成功；说明初始布局本身不是必然碰撞。
- 未训练 Actor：首步动作均落在 `a_max*DeltaT` 边界（0.8 时速度增量 0.4）；同一初始 seed 的 pure/capture/mixed 均在随机方向持续运动后碰撞。
- 修正后 25k replay：reward mean `-0.9134`、terminal fraction `0.00072`、实际 action speed mean `2.139`、max `2.9997`；最终 deterministic screening 仍 capture/CE strict 全 0、collision 50--100%。
- reward_terms：`coverage_ce_speed_weight=0`、motion/early-speed penalty disabled；因此先把问题归为“高速探索/缺少 settle 信号”的可学习性风险，不擅自修改 reward。
- 旧命名长程与对照复核：`p6_local_amax08_feasible_25k`、`p6_local_amax04_diag5k` 的 report manifest 均为 `action_mode=velocity_2d_body`，不是当前 ax/ay 合同；这些结果只作 velocity 历史/故障证据，不能替代 ax/ay 的学习 gate。
- Oracle 诊断：`tools/run_continuous_oracle_diagnostic.py` 使用同一连续环境、adapter 和 Voronoi centroid 脚本目标；seed `2026080410/2026080411` 均在 55/60 步内无碰撞达到 CE strict，center RMS `0.00351/0.00300`。JSON 产物：`artifacts/2026-08-04_continuous_marl_refactor/diagnostics/continuous_oracle_pure_ce.json`。因此环境可达性和动作合同成立，当前瓶颈锁定为 Actor/SAC 学习信号，不改环境合同。

### 7.2 简化任务并行诊断（当前新增）

25k 全任务的工程合同通过但 capture/CE strict 全部为 0，因此把复杂任务拆成两个相互独立的可学习性诊断是必要的：

| 诊断线 | 场景 | 唯一变化 | 保持不变 | 要回答的问题 |
|---|---|---|---|---|
| pure coverage | `pure_ce`，4 pursuer、无 evader | 将 coverage 的 speed/acceleration/angular-velocity 正则和 cell-center speed penalty 置 0 | `a_x,a_y`、`a_max=0.8`、env `v_max=3.0`、CE 判定、边界/碰撞 | 去掉控制正则后，基本 coverage 几何是否能被 SAC 学会？ |
| broadcast capture | `capture`，4v1 | `perception.global_evader_visibility=true`，所有我方 agent 直接收到敌人实体 | 动作、动力学、capture reward、termination、critic mode | 若不受 discovery/FOV 限制，capture 是否出现趋势？ |

这里的 pure coverage “控制消融”不是把速度/加速度动力学删掉：Actor 仍输出 body-frame `a_x,a_y`，策略仍硬满足 `||a||<=a_max`，环境仍积分并施加 `v_max`。只消融 reward 中用于鼓励慢速/低加速度/低角速度的辅助项，以隔离 coverage 几何 credit。

广播线复用已有全局可见开关，不改变 observation schema 的其余部分；它是“旧 VorAdj 感知围捕、只解除敌人有限可见性”的最小 ablation，不代表最终部署设定。

代码入口：`tools/run_continuous_p6_screening.py` 的 `--scene-set {pure_coverage,broadcast_capture,broadcast_mixed}`、`--coverage-control-ablation`、`--global-evader-visibility`。简化因素写入 report/manifest，不写入 joint replay 的受限 metadata。

已通过 16-step CPU smoke：

- `axay_pure_coverage_ablation_smoke16`：`replay_size=16`、7 updates、`all_finite=true`、action validation rate `0`。
- `axay_broadcast_capture_smoke16`：`replay_size=16`、7 updates、`all_finite=true`、action validation rate `0`。

两条互不续训的 5k local/SAC screening 已完成（各自新 seed、独立 tag；不是正式晋级）：

```bash
# GPU0
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. python3 tools/run_continuous_p6_screening.py \
  --critic-mode local --a-max 0.8 --init scratch --seed 2026080511 \
  --scene-set pure_coverage --coverage-control-ablation \
  --total-steps 5000 --batch-size 64 --update-every 1 \
  --screen-interval 1000 --screen-episodes 2 --device cuda \
  --tag axay_pure_coverage_ablation_5k_20260805

# GPU1
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. python3 tools/run_continuous_p6_screening.py \
  --critic-mode local --a-max 0.8 --init scratch --seed 2026080512 \
  --scene-set broadcast_capture --global-evader-visibility \
  --total-steps 5000 --batch-size 64 --update-every 1 \
  --screen-interval 1000 --screen-episodes 2 --device cuda \
  --tag axay_broadcast_capture_global_5k_20260805
```

判定顺序固定为：先看 replay/finite/action contract，再看 screening 的 collision、capture/CE strict 和 episode length 趋势。两条线都不学时，优先继续查 alpha/log-prob、reward scale、bootstrap 和初始碰撞；只有 pure coverage 学而 broadcast capture 不学，才把 discovery/捕获 credit 作为主要瓶颈；只有 broadcast capture 学，才回到有限视域线做 discovery 专项修复。5k 只能决定下一步诊断方向，不能替代 25k/正式 non-inferiority gate。

最终结果：

| run | replay / updates | finite / action 校验 | 末点 screening | 解释 |
|---|---:|---|---|---|
| `axay_pure_coverage_ablation_5k_20260805` | 5000 / 4937 | `true` / 0 | 2 episodes：CE strict 0/2、collision 0/2、末速度约 0.10--0.14 | 后期基本停住，未形成严格覆盖；消融控制正则未解决 coverage credit |
| `axay_broadcast_capture_global_5k_20260805` | 5000 / 4937 | `true` / 0 | 2 episodes：capture 0/2、collision 2/2、末速度约 0.72--2.98 | 全局可见仍高速碰撞；瓶颈不只是有限视域 discovery |

因此不启动这两条线的 25k 晋级，也不启动 central、`a_max=1.6` 或 8v2/12v3。下一步只做一次一个变量的 SAC 可学习性诊断：先固定环境/动作/场景，比较初始策略 action scale/alpha 与 reward/bootstrap 统计；任何 reward 或算法结构改动都需单独 tag、短 smoke 和回归记录。

只读 Actor 诊断发现默认 `log_std_max=1.0` 时初始 stochastic action 半径中位数约为 `0.56--0.76`（`a_max=0.8`），容易把速度推到环境上限。为验证这一共性风险，runner 新增 `--actor-log-std-max`，默认仍为 `1.0`；当前仅降低到 `-1.0`，不改 action contract、reward、env 或 SAC target entropy。

已通过低噪声 16-step smoke：`axay_pure_coverage_logstdm1_smoke16`，`replay_size=16`、7 updates、`all_finite=true`、manifest `actor_log_std_max=-1.0`。该单变量 5k 诊断已用默认 pure coverage 的同一 seed `2026080511` 完成：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. python3 tools/run_continuous_p6_screening.py \
  --critic-mode local --a-max 0.8 --init scratch --seed 2026080511 \
  --scene-set pure_coverage --coverage-control-ablation \
  --actor-log-std-max=-1.0 --total-steps 5000 --batch-size 64 \
  --update-every 1 --screen-interval 1000 --screen-episodes 2 \
  --device cuda --tag axay_pure_coverage_logstdm1_5k_20260805_same_seed
```

同 seed 对照结果：默认 `log_std_max=1.0` 在 4k/5k screening 均 collision `0/2`、episode `128/128`；低噪声 `-1.0` 在 4k/5k 均 collision `1/2`，两者 CE strict 均 `0/2`，finite/action contract 均通过。结论是低噪声上限没有修复学习问题，且该 seed 稳定性更差；该因素不晋级，runner 开关保留为可复现实验参数但默认不改变。

## 8. Luna/后续维护模型的最小操作规程

1. 先读本文、上游 tracker 和 compact design，检查 `git status --short`；不清理用户 dirty/untracked 资产。
2. 每轮只处理一个最小 TODO；修改前先给出命令和预期产物，修改后补测试、哈希和记录。
3. 六条 500k 长程线已登记并正在运行；不得复制启动未登记的正式长程 run。若 worker 仍存活，只做只读资源/产物检查，不杀进程、不换 seed。
4. 任何新 run 使用新 tag/namespace；失败 run 原样保留，不能换 seed 或覆盖报告。
5. 只在本表的 gate 通过后推进下一阶段；遇 L3 变更先停并请求用户确认。
6. 状态汇报必须包含：当前进程、阶段/step、report/replay/checkpoint、all_finite、成功率/CE/collision、speed-limited、资源、下一动作。

## 9. 复验入口

```bash
cd /home/yjq/rl/CoCap1/cocap-voradj
PYTHONPATH=src:. pytest -q \
  tests/test_continuous_action_contract.py \
  tests/test_p1_velocity_env_contract.py \
  tests/test_continuous_actor_contract.py \
  tests/test_local_sac_smoke.py \
  tests/test_central_sac_smoke.py \
  tests/test_joint_replay_contract.py
```

当前完整 P0--P5 结果：`45 passed, 2 warnings`；AW bridge 后 focused regression 另为 `20 passed`，16-step AW smoke `all_finite=true`。P6 工程门禁已通过但 matched 5k 学习门禁未通过；六条 500k 结果尚待实际 report/replay/checkpoint 和动作轨迹复盘，不得只用单一 success 指标下结论。

## 10. 20260805 并行 500k 长训登记（启动即记录，结果待复盘）

用户要求的三条任务线已并行启动；每条均为独立 scratch、local SAC、formal P6 profile、500000 transitions、100000-capacity hot replay、每 100000 steps 保存 checkpoint/replay 并做 4-episode deterministic screening。stdout 另存为同 tag 的 `_stdout.log`，因此后续可以在不依赖 tmux 的情况下复盘 late-stage metrics。

| 线 | scene | 动作 | seed | device | tag | 初始状态 |
|---|---|---|---:|---|---|---|
| ax/ay mix | `full`，非 global | radial `a_x,a_y`，`a_max=0.8` | 2026081501 | GPU0 | `p6_axay_500k_mix_non_global_20260805` | 进程存活，已过初始化 |
| ax/ay pure | `pure_coverage` | radial `a_x,a_y`，`a_max=0.8` | 2026081502 | GPU1 | `p6_axay_500k_pure_coverage_20260805` | 进程存活，已过初始化 |
| ax/ay capture | `broadcast_capture` + global visibility | radial `a_x,a_y`，`a_max=0.8` | 2026081503 | GPU0 | `p6_axay_500k_global_capture_20260805` | 进程存活，已过初始化 |
| (a,w) mix | `full`，非 global | factorized box SAC，`a_max=0.4,w_max=pi/6` | 2026081601 | GPU0 | `p6_aw_500k_mix_non_global_20260805` | 进程存活，已过初始化 |
| (a,w) pure | `pure_coverage` | factorized box SAC，`a_max=0.4,w_max=pi/6` | 2026081602 | GPU1 | `p6_aw_500k_pure_coverage_20260805` | 进程存活，已过初始化 |
| (a,w) capture | `broadcast_capture` + global visibility | factorized box SAC，`a_max=0.4,w_max=pi/6` | 2026081603 | GPU0 | `p6_aw_500k_global_capture_20260805` | 进程存活，已过初始化 |

产物根目录统一为 `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/`。当前不能把“进程存活”写成性能结论；后续复盘必须读取 report 的 `all_finite`、`update_metrics_tail`、reward 分解、`collision_transition_count/collision_reward_sum`、screening 中 coverage 的 `CV<0.15`/末态速度/episode length/collision，以及 capture 的真实 capture rate/collision。

## 11. IQN 老 replay 为连续 SAC 初始样本的可行性与实现边界

### 11.1 结论

可行，但不是“把旧 pickle 直接喂给新 trainer”。旧 IQN replay 是逐 pursuer 的 tuple：`(obs, action_index, reward, next_obs, done, metadata)`；新 `JointReplayBuffer` 要求一个 4-agent 原子 transition、固定 slot、`actions.shape=(4,2)`、分开的 `terminated/truncated`、global state 和受限 metadata。必须经过带 manifest/hash 的离线转换或在新环境中用旧 IQN policy 重新采样。

### 11.2 两种动作合同的可行性不同

1. **(a,w) bridge：可做高保真 teacher replay。** 旧 `Pursuer.action_list[index]` 解码为 `(a,w)`，其端点正是 `a∈{-0.4,0,0.4}`、`w∈{-pi/6,0,pi/6}`；这些离散点是连续 box 中的合法精确采样点。当前 bridge 动力学保留旧水阻尼和 yaw 积分，因此动作转换不需要近似。
2. **ax/ay：不应直接转换。** 旧 `(a,w)` 是非完整车体的前向加速度/yaw-rate，而 ax/ay 是 body-frame 二维加速度；同一个旧动作没有唯一的 `(a_x,a_y)`，把它硬映射到某个向量会制造错误的 transition/reward/critic target。旧 replay 只能用于 observation/encoder 预训练、行为克隆，或由旧 policy 在新 ax/ay 环境里重新 rollout 后再入新 replay。

### 11.3 推荐实现（后续 agent 可直接按此拆任务）

1. 新工具 `tools/convert_iqn_replay_to_continuous.py`：输入旧 replay、旧 config、目标 action mode；先验证旧 observation/action-list/schema hash，再按 `index -> pursuer.action_list[index]` 解码，拒绝越界、缺 slot、NaN 和 config mismatch。
2. 将逐 agent 数据按时间戳/episode/agent slot 重组为 joint transition；缺失或已失活 slot 用与现 runner 相同的 zero padding，但必须保留动作前 `active_mask`，并从旧 `done + info.state` 重算 `terminated` 与 time-limit `truncated`。旧 replay 没有可靠原因字段时宁可丢弃该 transition，不要把 timeout 当 terminal。
3. 通过显式 `iqn_obs_to_entity_v1` 适配器把旧 obs 映射到新 token keys、shape、mask/type；不能只按数组 reshape。global state 必须由可复现 env snapshot 重建；如果旧 replay 没有 snapshot，就不要伪造 global state，改用新环境 teacher rollout。
4. 新 replay 的 metadata 只写允许的 `regime/event_ids/phase/scene/task_label/coverage_only/active_target`，来源、旧 buffer id、转换版本放在 replay manifest，不写入 transition metadata；manifest 记录 `source=iqn_teacher`、源/目标 config hash、转换脚本 hash 和丢弃计数。
5. 初始训练只混入 5--20% teacher transitions（按 event/coverage floor 采样），其余保持新环境 scratch rollout；在 10k--50k steps 内逐步退火到 0%，避免旧 policy 的偏差锁死 SAC。先做 1k conversion smoke、`all_finite`/action contract/reward 重算，再做 25k 单 seed 对照。

### 11.4 更稳妥的替代方案

若旧 replay 缺少完整 next state、终止原因或 global snapshot，首选“旧 IQN policy → 新 AW 环境 teacher rollout”：旧网络仍输出 action index，按 action list 解码后直接执行 bridge，环境实时产生新 obs/reward/done/global state，写入新 `JointReplayBuffer`。这样保留旧策略的行为先验，却不继承旧 replay 的语义污染。对 ax/ay 也只能采用这种“在新动作合同下重新生成 transition”的方式，不能做离线动作硬映射。

### 11.5 当前决策

本轮不把 IQN replay 注入正在运行的六条 500k scratch 线；它们保持纯 scratch，确保 late-stage 结果可归因。IQN transfer 作为后续独立 tag（建议 `aw_iqn_teacher20pct_25k`，先 AW，再考虑 ax/ay representation-only）进入 TODO，必须通过转换 smoke 和合同审计后才可开长训。

## 12. 2026-08-06：checkpoint/replay 持久化间隔变更（每 25k 一存）

- 背景：六条 500k 长训启动时，runner 把 checkpoint/replay 保存绑定在 `--screen-interval=100000` 上；若服务器在两次保存之间停机，最多丢失约 100k 步训练进度，恢复只能从最近 milestone 以 `seeded_episode_boundary` 续训。
- 变更：`tools/run_continuous_p6_screening.py` 新增 `--save-interval`（默认 25000），在 step1 与每 25k 保存 checkpoint+replay snapshot；deterministic screening 仍按 `--screen-interval`（长训为 100k）执行，两者解耦。最终 report 新增 `save_interval` 字段。
- 证据：`PYTHONPATH=src:. python3 -m pytest -q -p no:cacheprovider tests/test_p6_save_interval_contract.py` → 4 passed；`python3 -m py_compile tools/run_continuous_p6_screening.py` 通过。
- 影响：**正在运行的六条 500k 线不受影响**，仍按 step1 + 每 100k 保存（不重启、不换 tag、不覆盖）；新开长训默认每 25k 一存，停机最坏丢失 ≤25k 步。
- 续训合同不变：checkpoint schema 2 / replay schema 3、manifest strict match、`seeded_episode_boundary`、`exact_env_state=false`。
- 续训兼容：六条运行线启动时的 `implementation_hash` 基于当时的 runner 文件（SHA256 `4303cc326c82f8e261836bac07f7a6fa0f71f8f22c24bf180f8c2345668a78b4`）；本轮 runner 改动后，若某线中断需要恢复，必须先用启动时版本备份 `tools/run_continuous_p6_screening.py.pre_25k_save_20260806.bak` 覆盖运行文件再 resume，恢复完成后可继续用备份跑完该线；不得用新版 runner 直接 resume 运行中的旧线。

## 13. 2026-08-06 15:xx：前两条 100k 里程碑复盘（pure coverage 线）

- 触发：`p6_aw_500k_pure_coverage_20260805_step100000.pt/replay`（12:32 落盘）与 `p6_axay_500k_pure_coverage_20260805_step100000.pt/replay`（15:01 落盘）出现；尚无最终 report。
- 评估：加载 100k checkpoint 跑 2× deterministic pure_ce（seed+10000/+10001），并分析 100k replay snapshot。
- ax/ay pure：两局均碰撞（step 14/16），末速 3.0（v_max），area CV 0.500/0.421，strict 与 CV<0.15 均 0/2；replay 中 collision event=2403、action norm mean 0.70/p95 0.80（近饱和）、speed mean 2.03、后 10% speed（2.30）高于前 10%（1.60）。判定：学习失败，高速乱撞。
- AW pure：两局均跑满 128 步、无碰撞、末速 0；逐局探针显示 mean a≈-0.32（持续制动）而 speed 恒 0，w 前 10 步转向后归零，即原地制动/转圈 no-op；replay 中 collision=1、speed mean 0.019、后 10% speed≈0.0016、action norm max 0.658。判定：学习失败，退化到不动。
- 结论：两条 pure 线 100k 均无乐观信号，不晋级；继续跑完 500k 只作长程数据积累。下一步仍按核心台账 9.2/10 诊断 SAC 学习信号（alpha/log-std/奖励尺度/bootstrap），不可直接扩 central 或改 reward。
- 产物：`artifacts/2026-08-04_continuous_marl_refactor/p6_screening/{p6_axay_500k_pure_coverage_20260805,p6_aw_500k_pure_coverage_20260805}_step100000.pt` 与 `_replay_step100000.pkl`。

## 14. 2026-08-06 16:30：两条 pure 线提前关闭（用户决策）

- 用户指示：已有 100k checkpoint 的线提前关闭；后续要大改方案，本批基本不再作为晋级证据。
- 操作：`kill 2773811 2803603`（SIGTERM）终止 ax/ay pure、AW pure 的 python worker；对应 tmux 会话随命令退出自动消失。其余四条（ax/ay mix/global、AW mix/global）继续运行，未动。
- 保留产物：step100000 checkpoint/replay、100k 复盘与正式 coverage 20-rollout（5 GIF）产物均保留在 `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/` 与 `artifacts/2026-08-06_continuous_marl_formal_20rollout10gif/`；无最终 report。
- 状态：四条线进程存活（PID 2773802/2773896/2803595/2803626），GPU0 ~21 GiB；GPU1 训练占用已释放。
