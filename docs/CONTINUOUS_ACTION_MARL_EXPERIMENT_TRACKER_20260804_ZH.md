# 连续动作 PSA-MASAC 专用实验记录与执行清单

> 建立日期：2026-08-04  
> 状态：CTDE_CONTRACT_20260806（formal central MASAC 合同整改已完成；5k/25k 训练与 25k 20-episode eval 均已完成；GPU0 旧四条 500k local 线已按用户要求停止）
> 当前阶段：审计文档 `cocap_voradj_masac_ctde_task_contract_audit_20260806_zh.md` 的 P0/P1 合同项已落地并通过 126 项 pytest；5k/25k formal CTDE 训练报告已生成；25k eval 三场景 20 episodes 均无 capture/coverage 成功，collision 0.45--0.55，100k gate 未开。
> 用途：后续 Codex/Luna 等模型实施、测试、训练和维护时的唯一执行台账；新 agent 默认只需先读本文件，再按“接手入口”跳转必要资料。

> **2026-08-05 合同修订：** 当前连续线已从“策略输出速度 + feasible velocity layer”切换为显式机体系加速度 `a_x,a_y`。本文件中旧 velocity 文字和 P1--P6 记录保留为历史审计，不得作为当前配置、checkpoint 或 replay 的续训合同；当前实现与迁移规则以 `docs/CONTINUOUS_ACTION_AXAY_REFACTOR_20260805_ZH.md` 为准。

配套资料：

- 完整设计：`docs/CONTINUOUS_ACTION_MARL_ARCHITECTURE_REFACTOR_20260804_ZH.md`
- 浓缩设计：`docs/CONTINUOUS_ACTION_MARL_ARCHITECTURE_REFACTOR_20260804_ZH_COMPACT.md`
- ax/ay 修订：`docs/CONTINUOUS_ACTION_AXAY_REFACTOR_20260805_ZH.md`
- 项目交接：`docs/PROJECT_HANDOFF_20260802_ZH.md`
- CR-MS 课程：`docs/CRMS_SUPPORT_APPROACH_AUTO_CURRICULUM_20260803_ZH.md`

## 0. 给新 agent 的唯一接手入口

### 0.1 当前结论（先读这一段）

- 旧 IQN/DQN 主线是 baseline，保持不动；旧 velocity 连续实验、checkpoint 和 replay 不可直接续训。
- 当前 canonical 连续动作是 ax/ay：`action_mode=acceleration_2d_body`，radial SAC Actor 输出 body-frame `a_x,a_y`，`||a||<=a_max`；环境负责 `v_max`、边界和碰撞。
- 另有独立的 AW bridge 实验：`action_mode=aw`，box SAC Actor 输出连续 `(a,w)`，保留旧水阻尼和 yaw 积分；AW 结果不得与 ax/ay 直接混成同一性能结论。
- 2026-08-05 已启动六条 500k scratch local/formal 线：ax/ay 与 AW 各自包含 mix 非 global、pure coverage、global capture；启动/step-1 checkpoint 已确认，当前没有 100k 或最终 report，不能把进程存活当作训练成功。
- formal matched 5k 的任务 gate 已失败：pure coverage `CV<0.15=0/2` 且 collision=100%；global capture capture=0/2 且 collision=100%。六条 500k 只是用户要求的长程证据积累，不能跳过后续深度复盘直接晋级。
- 最近一次实际核验：2026-08-06 00:04 CST；6 个 Python worker 仍存活，step-1 checkpoint/replay 均存在，尚无 `step100000` 或最终 report；GPU0/1 显存约 21.8/12.8 GiB，磁盘剩余约 94 GiB。

### 0.2 推荐阅读顺序与地址

1. **本文件**：唯一状态、合同、run 台账、判断门槛和 TODO。
2. [执行日志](./CONTINUOUS_ACTION_MARL_EXECUTION_LOG_20260805_ZH.md)：逐轮命令、BUG 修复、六条 500k 登记、IQN replay 可行性分析。
3. [ax/ay 合同修订](./CONTINUOUS_ACTION_AXAY_REFACTOR_20260805_ZH.md)：当前 ax/ay 动作、动力学、迁移边界。
4. [架构浓缩版](./CONTINUOUS_ACTION_MARL_ARCHITECTURE_REFACTOR_20260804_ZH_COMPACT.md)：PSA-MASAC、观测/replay/critic 结构总览。
5. [完整架构设计](./CONTINUOUS_ACTION_MARL_ARCHITECTURE_REFACTOR_20260804_ZH.md)：需要改网络、SAC 或 CTDE 时再查。
6. [旧项目交接](./PROJECT_HANDOFF_20260802_ZH.md)：CR-MS、CE、VCT-LS 和旧 IQN baseline 背景。
7. 正式训练配置：[p6_formal_local_sac_4v1.yaml](../configs/experiments/continuous_marl_20260804/p6_formal_local_sac_4v1.yaml)；训练入口：[run_continuous_p6_screening.py](../tools/run_continuous_p6_screening.py)。
8. 实现入口：[`local_sac.py`](../src/cocap_voradj/training/continuous/local_sac.py)、[`joint_replay.py`](../src/cocap_voradj/training/continuous/joint_replay.py)、[`radial_actor.py`](../src/cocap_voradj/models/continuous/radial_actor.py)、[`box_actor.py`](../src/cocap_voradj/models/continuous/box_actor.py)、[`continuous_action.py`](../src/cocap_voradj/dynamics/continuous_action.py)、[`robot.py`](../src/cocap_voradj/dynamics/robot.py)。
9. 产物根目录：`artifacts/2026-08-04_continuous_marl_refactor/p6_screening/`；六条长训的 tag/seed/命令见第 8.1 节。

### 0.3 接手前只读检查（不要先改代码、不要先杀进程）

```bash
cd /home/yjq/rl/CoCap1/cocap-voradj
git status --short
pgrep -af 'run_continuous_p6_screening.py'
nvidia-smi
df -h /home/yjq
find artifacts/2026-08-04_continuous_marl_refactor/p6_screening -maxdepth 1 -type f \
  \( -name '*500k*report.json' -o -name '*500k*step*.pt' -o -name '*500k*replay_step*.pkl' \) -printf '%f\n' | sort
```

若 worker 仍在运行：只记录 PID、GPU、日志大小和最新 checkpoint，不重启、不换 seed、不覆盖 tag。若某条退出：先读取 stdout 尾部和最后 checkpoint/replay 的 manifest，判断是正常完成、合同错误、OOM、NaN 还是环境异常，再决定是否恢复；恢复只能遵守 `seeded_episode_boundary`，不能声称 exact env-state resume。

### 0.4 运行时合同速查

- 六条长训均 `critic_mode=local`、`trainer_profile=formal_p6`、`warmup_steps=5000`、hidden=256、lr=1e-4、grad clip=10、`screen_interval=100000`、`screen_episodes=4`、`replay_capacity=100000`。
- checkpoint/replay 持久化：runner 自 2026-08-06 起与 screening 解耦，新 run 默认 `--save-interval=25000`（step1 + 每 25k 保存 checkpoint/replay）；正在运行的六条 500k 线启动时未传该参数，仍按旧耦合逻辑在 step1 + 每 100k 保存，不得为改成 25k 而重启或换 tag。停机后只能从最近一次已保存 milestone 以 `seeded_episode_boundary` 续训，`exact_env_state=false`。六条运行线的续训必须使用启动时版本 runner 备份 `tools/run_continuous_p6_screening.py.pre_25k_save_20260806.bak`（SHA256 `4303cc326c82f8e261836bac07f7a6fa0f71f8f22c24bf180f8c2345668a78b4`，与台账登记的启动时 hash 一致），不能用含 `--save-interval` 的新版 runner 直接 resume（manifest strict 会拒绝）。
- 运行总步数是 500000；replay 是 100000-capacity hot ring，不是只训练 100000 步。runner 已修正为尊重 `--replay-capacity`，不再强制按总步数分配整个 replay。
- runtime `JointReplayBuffer.schema_version=3`；旧文档中“schema v2”只表示动作字段语义的历史说法，不是当前可加载版本。checkpoint schema 为 2。
- AW 参数是 `a_max=0.4,w_max=pi/6`；ax/ay 参数是 `a_max=0.8`。两种 action contract、dynamics 和 manifest hash 必须分别核对。

### 0.5 结果不能只看一个成功率：强制深度复盘协议

任何 100k milestone 或最终 report 都必须形成“工程合同 + 训练动力学 + 行为轨迹 + 任务结果”四层记录；单看 `success_rate`、单个 strict 指标或单个 episode 不得决定晋级。

**A. 工程/数值层（先查，不通过则停止性能解释）**

- report 是否真的达到该阶段 step；`replay_size`、`updates`、`transition_attempt_count`、`terminal_transition_drop_count`、`inactive_next_slot_transition_count` 是否自洽。
- `all_finite`、action validation、manifest action mode/a_max/w_max、config/implementation hash、checkpoint/replay schema 是否匹配；不得跨 ax/ay、AW 或旧 velocity resume。
- `terminated/truncated`、collision transition/reward、metadata event/source/fallback、sampling regime/event/coverage floor 是否有异常空洞。
- update tail 和各 milestone 的 `critic_loss/actor_loss/alpha_loss`、Q/target-Q、TD error、log-prob、log-std、entropy proxy、actor/critic/alpha grad norm 是否有限且有趋势；只要 NaN、无限梯度、严重 TD 爆炸或 terminal drop > 0，先做数据/训练诊断。

**B. pure coverage（不能只看 strict）**

- 主指标：`CV<0.15` 达成率；并列报告每个 episode 的 `coverage_strict_success`、area CV、center/inside/disconnected 几何量、episode length、末态 mean/max speed、collision、speed-limited、jerk/加速度统计。
- strict 是更严格的次级指标。出现 strict=0 但 CV<0.15、几何中心/连通性已明显改善，且 collision 低、末态速度可停稳、episode length 合理时，标为“乐观的 coverage partial trend”，不能标失败也不能标通过；必须检查是否只是高速穿越、短暂几何接近或末态未稳住。
- 必须对比 step1/100k/中后期 episode 记录，而非只看最终 4 个 screening episode；必要时从 replay 的 `global_state/local_obs/actions` 计算动作范数、速度、转向/加速度饱和和碰撞前窗口。

**C. global capture（主要看真实围捕事件）**

- 主指标是 `capture event` 数和 episode capture rate；coverage strict/CV 不能替代 capture。并列报告 capture 前后 episode length、collision、末态速度、speed-limited、evader 相对距离/邻接变化和 reward term（capture、safety、terminal）。
- 若长期 capture=0，必须深入检查：动作范数/方向是否近似常值或饱和；AW 的 `w` 是否接近 0、符号是否切换、是否有有效转向；ax/ay 的切向加速度是否不足；碰撞发生在 capture 前多少步；evader 是否被发现、是否进入 active target、capture event 是否进 replay/event pool；alpha/log-std 是否过早收缩；critic TD/grad 是否主导安全惩罚。
- 先区分“没有接近”“接近但未形成环”“已形成环但 capture 判定/事件写入错误”“靠近后高速碰撞”四类原因，再决定改动作、reward、采样还是 capture 逻辑。没有轨迹级证据不得直接归咎于网络或环境。

**D. SAC/探索层**

- 当前没有 IQN 式 epsilon；探索由 SAC 的 Gaussian log-std + 自动 alpha 组成，warmup 只延迟 update，不等于安全随机探索。需同时看 alpha、log_std、log_prob、latent/action norm 与碰撞率，不能用 alpha 单个数代替探索度。
- 若行为显示高速随机撞击，优先记录 action scale/alpha/log-std 与 reward/TD 的同时间趋势，再决定是否做混合安全探索、动作幅度、reward scale 或 critic 稳定性单变量；不可同时改 reward、网络和环境。

**E. 每条线的复盘产物**

至少保存：原始 report、stdout 尾部、对应 milestone checkpoint/replay、一个逐 episode 指标表、一个 action/速度/碰撞前窗口摘要、SAC diagnostics 摘要、结论（通过/partial/失败/合同异常）及下一步理由。所有结论必须注明 seed、step、tag 和产物路径。

---

## 1. 台账规则

状态：`[ ]` 未开始；`[~]` 部分完成；`[x]` 已完成且证据齐全；`[!]` 失败/阻塞；`[S]` 首版搁置。

一个 TODO 只有同时记录修改路径、精确命令、返回码、关键结果、产物路径、日期、seed/step/config hash（如适用），才能标 `[x]`。禁止凭“代码看起来正确”、进程存活或单次幸运 rollout 标完成。

小模型固定工作流：

1. 先读本文和浓缩设计，有歧义再查完整设计。
2. 一次只做一个最小 TODO，不跨阶段顺手改 reward、observation 或旧 IQN。
3. 修改前检查 `git status --short`；现有 dirty/untracked 文件均是用户资产，不清理、不覆盖。
4. 新功能必须由新配置或 namespace 隔离；旧配置默认行为不变。
5. 先窄测试，再 smoke，再长训练；前置 gate 不通过不得跳级。
6. 当轮结束前回填本文的状态、证据、问题和下一动作。
7. 失败 run 原样保留；不得删结果、换 seed 或降判据制造通过。

---

## 2. 冻结的首版合同

未经用户明确确认不得改变。

| 项目 | 冻结决定 |
|---|---|
| 算法 | PSA-MASAC：共享局部 Transformer Actor + 两套完全独立的集中式实体注意力 Critic + joint replay |
| 动作 | 每架 pursuer 的二维连续 body-frame 加速度 `a_cmd=(a_x,a_y)` |
| Actor | radial-squashed Gaussian 输出 `||a_cmd||<=a_max` 的加速度 |
| 约束 | `v_max=3.0`；`a_max` 可调，先筛 0.4/0.8/1.6 |
| 约束位置 | 策略只保证 `a_max`；env 负责 `v_max` 物理速度上限、碰撞和边界，不把速度裁剪写回 action |
| 周期 | 暂沿用 `DeltaT=0.5 s`，每周期最大速度变化为 `a_max*DeltaT` |
| 坐标 | Actor 观测和动作均在生成观测时的 body frame；进入动力学前只旋转一次到 world |
| yaw | 连续首版 `hold`，所有 pursuer `yaw_init=0`；旧线 reset 不变 |
| 感知 | 保持有限半径、2π 全向局部感知，不引入扇形 FOV |
| boundary | 保持 `nearest_vector_robot_oob` 和 out-of-bound flag，不改四墙 token |
| evader | 保持现有离散/APF，只连续化 pursuer |
| reward | 保持新 capture、CE、VCT-LS、support/CR-MS 语义 |
| Actor 信息 | 仅部署可得局部观测；禁止 global state、phase/event 标签 |
| Critic 信息 | 仅训练期全局实体集合、final joint command、active mask |
| twin critics | 两套 encoder/normalizer/attention/head 完全独立，不复用 Actor learned feature |
| replay | 一个 joint ring；regime/event index 只存 transition ID |
| replay action | 只存实际发送的 body-frame acceleration `actions`；runtime `JointReplayBuffer.schema_version=3`，不存 final feasible velocity |
| 采样 | 70% uniform + 20% regime + 10% event，另设 coverage-only 下限 |
| 任务结果排序 | capture/mix 的真实 capture event -> pure/mix 的 CV<0.15+几何/末态速度 -> strict CE 作为次级 -> collision/steps；不得只按 strict 或单个 success_rate 排序 |
| 展示 | 大阶段最优模型正式 20-rollout/10-GIF；faded trail 默认关闭 |

旧配置在未传 theta 时仍随机初始化 yaw；连续线已通过独立配置分支固定 `yaw_init=0`，旧路径不变。

### 2.1 最终审计修正

原 TODO 要先从 `CoCapIQN` 抽出共享 encoder，会直接触碰已 solid 的旧线。首版改为：

1. 新增独立 `LocalEntityTokenEncoder`，按旧前端功能合同复制；
2. 提供旧 IQN encoder key 到新 encoder key 的确定性映射；
3. 固定输入验证映射后表示一致；
4. 不改 `CoCapIQN` 的执行路径；
5. 连续线 solid 后再考虑共享抽象，并重新做旧线回归。

P1 当前实现为 `AccelerationActionAdapter`，只验证/记录 `a_max`；P3 Actor、local/central SAC 和 env 统一使用同一个 acceleration action。旧 `FeasibleVelocityAdapter` 仅为兼容别名，不恢复速度投影语义。

---

## 3. 实验实现范围

推荐新增 namespace（可按现有包布局微调，但要在决策记录说明）：

~~~text
src/cocap_voradj/models/continuous/
  local_entity_token_encoder.py
  radial_actor.py
  central_attention_critic.py
  feasible_velocity.py
  psa_masac.py
src/cocap_voradj/training/continuous/
  joint_replay.py
  samplers.py
  masac_trainer.py
  checkpointing.py
configs/experiments/continuous_marl_20260804/
tests/continuous/
runs/continuous_marl_<run_id>/
artifacts/2026-08-04_continuous_marl_refactor/
~~~

允许 gated 修改的旧模块：action-mode/config 分发、pursuer/base 连续动力学分支、连续线 reset yaw、rollout/evaluation/checkpoint loader 新算法分支、screening/GIF metadata。每处都必须证明旧 `unicycle_discrete` 配置数值行为不变。

首版明确不做：

- 不删除、替换、重写现有 IQN/DQN；
- 不连续化 evader，不改 reward/termination；
- 不改 boundary 表示，不加 yaw action、速度定 yaw 或扇形 FOV；
- 不把 phase/regime/event/origin 标签输入网络；
- 不先加 agent ID、PER、HER、TQC、GRU、flow、FACMAC；
- 不在一个 smoke 同时改动作、observation 和 reward；
- 不在 gate 未通过时启动 700k/2M 正式训练。

---

## 4. 当前状态与参考基线

### 4.1 代码事实（2026-08-04）

| 项目 | 状态 | 结论 |
|---|---|---|
| 完整/浓缩设计 | `[x]` | 已存在 |
| 本实验台账 | `[x]` | 本文件 |
| `acceleration_2d_body` | `[x]` | 当前 canonical 配置与 env gated 分支；旧 velocity 字符串仅兼容 |
| 连续线 `yaw_init=0` | `[x]` | 仅连续 pursuer 分支 hold；旧 reset 不变 |
| radial Actor / MASAC | `[x]` | P3 radial Actor 已通过合同门禁；P5 central SAC/PSA-MASAC 已通过 smoke |
| joint replay | `[x]` | P4 joint ring、70/20/10 sampler、strict resume 已通过 |
| 连续动作训练 | `[~]` | ax/ay P2/合同/25k engineering gate 已通过；capture/CE strict 全 0，learning gate 未通过 |

结论：旧 P0--P5 工程记录保留；ax/ay 代码合同 31 项聚焦测试、P2 三候选 scripted gate、25k report/replay/resume/screening engineering gate 均通过。`a_max=0.8` 的 local scratch 25k 六个 screening 点 capture 与 CE strict 全 0，未达到学习 gate；不得直接进入 100k、central、`a_max=1.6` 或 8v2/12v3。

### 4.2 CR-MS 正式对照

根目录：`artifacts/2026-08-02_three_line_stage_best_20rollout10gif/`

| 规模 | 正式 artifact | capture | pure CE strict | mix capture | mix CE strict | collision |
|---|---|---:|---:|---:|---:|---:|
| 4v1 | `crms_supportapproach_4v1_s1_step_2000000` | 100% | 100% | 100% | 90% | capture 0%、pure 0%、mix 5% |
| 8v2 | `crms_supportapproach_8v2_s2_step_300000_cefirst_comparison` | 100% | 100% | 100% | 100% | 均 0% |
| 12v3 | `crms_supportapproach_12v3_s3_step_700000` | 100% | 100% | 100% | 95% | 均 0% |

CE strict 对应正式结果的 `coverage_success_rate/coverage_settled_rate`；CV<=0.15 是后排指标，不等同于 strict。8v2 历史 500k 仍保留，但新线主对照采用 CE-first 补跑的 300k。12v3 的历史 `selection.json` 文本仍把 CV 放在 CE 前；新线不得复制，统一按本文 CE-first 排序。

---

## 5. 分阶段 TODO、Gate 与状态

### P0：基线冻结与隔离

状态：`[x]`，P0 基线、隔离和三规模固定 seed regression 已完成。

- [x] 定位 4v1/8v2/12v3 正式 20-rollout/10-GIF。
- [x] 确认 CE strict 与 CV<=0.15 是不同指标。
- [x] 建 baseline manifest：checkpoint、effective config、seed、SHA256。
- [x] 固定 capture/pure/mix 的 screening 与 formal seed。
- [x] 保存旧 IQN 固定 seed 轨迹摘要 hash 和核心指标。
- [x] 记录 commit、dirty 清单、Python/PyTorch/CUDA/GPU。
- [x] 建独立 config/run/artifact namespace，验证不覆盖旧资产（manifest namespace + `configs/experiments/continuous_marl_20260804/README_zh.md`）。
- [x] 用旧配置重跑固定 seed regression。

Gate：以上全绿，manifest 可脚本复验。  
证据：

- manifest：`artifacts/2026-08-04_continuous_marl_refactor/baseline_manifest.json`
- 构建：`python3 tools/build_continuous_baseline_manifest.py`
- 校验：`python3 tools/build_continuous_baseline_manifest.py --check`
- 测试：`PYTHONPATH=src:. pytest -q tests/test_continuous_baseline_manifest.py tests/test_crms_baseline_regression.py`，3 passed
- 4v1 report：`artifacts/2026-08-04_continuous_marl_refactor/old_iqn_regression/4v1_seed_2026081201/regression_report.json`
- 8v2 report：`artifacts/2026-08-04_continuous_marl_refactor/old_iqn_regression/8v2_seed_2026082201_evaders_2/regression_report.json`
- 12v3 report：`artifacts/2026-08-04_continuous_marl_refactor/old_iqn_regression/12v3_seed_2026082301_evaders_3/regression_report.json`
- 三份 report 均 `status=passed`、核心字段无差异、capture/coverage/mix 轨迹摘要 hash 全匹配、`draw_trails=false`。

下一动作：P0 gate 已结束；P1--P5 已完成，进入 P6 4v1 学习型筛选。不得把 P4/P5 smoke 的 loss trend 当作任务性能结论。

### P1：动作与动力学合同（无 RL）

状态：`[x]`（2026-08-04，合同实现与窄门禁完成）

- [x] gated `unicycle_discrete` / `acceleration_2d_body`。
- [x] 定义 shape、dtype、单位、body/world frame、inactive padding。
- [x] 实现 `AccelerationActionAdapter`，策略侧只验证 `a_max`；`v_max` 由 env 负责。
- [x] env 对越约束 command 报错，不静默 clip。
- [x] body->world 只旋转一次；10 substeps 用线性速度参考/梯形积分。
- [x] 连续线 `yaw=hold, yaw_init=0`；旧 reset 不变。
- [x] pursuer 连续 + evader 旧 APF 的 mixed interface。
- [x] substep callback 在每个子步执行，边界/碰撞检查无高速跳步。
- [x] acceleration/speed/theta/trajectory 字段与连续 step 保持一致。
- [x] `infos`/VorAdj replay metadata 含 speed、actual acceleration、jerk、projection rate/delta。
- [x] 测 shape/NaN、零速、90°、停车、speed/a_max、反向合同拒绝。
- [x] 测 yaw 不漂移、body/world 旋转和梯形轨迹一致。
- [x] 测 4/8/12、inactive mask、mixed interface、子步回调。
- [x] 旧 CR-MS 固定 seed regression 不变。

Gate：合同测试全通过，旧线 hash 不变，scripted rollout 无 NaN、隐式 clip、穿障。  
证据：

- 配置：`configs/experiments/continuous_marl_20260804/p1_acceleration_contract_4v1.yaml`（旧 `p1_velocity_contract_4v1.yaml` 仅保留兼容）
- 实现：`src/cocap_voradj/dynamics/continuous_action.py`、`src/cocap_voradj/dynamics/robot.py`、`src/cocap_voradj/envs/base.py`、`src/cocap_voradj/envs/voronoi_adjacency.py`
- 测试：`PYTHONPATH=src:. pytest -q tests/test_continuous_action_contract.py tests/test_p1_velocity_env_contract.py`，9 passed；联合旧配置/拓扑门禁为 19 passed
- 配置/核心测试 SHA256：`fffea0a444165cadb91b581fe324b375de9adf6e80108514814572efee34ed03`、`e5c72ce30c703f7c23b7c39fb8f5633f3410cae4fc8580b305cbe9227378a1bf`、`7d647b47ab86b079261c0954b6c2f6fe5cb39f7fcd3b4d3c914a399220b27b1e`、`f1c4046b6a91d4be7ce7d0e9e375c20151aa508d3f1d525bd9ec60798f530573`
- P1 后旧线报告：`artifacts/2026-08-04_continuous_marl_refactor/old_iqn_regression_after_p1/{4v1_seed_2026081201,8v2_seed_2026082201_evaders_2,12v3_seed_2026082301_evaders_3}/regression_report.json`
- 三份报告均 `status=passed`，capture/coverage/mix trajectory hash 全匹配，核心字段无差异。

### P2：`a_max` 脚本/oracle 筛选

状态：`[x]`（2026-08-05，ax/ay 三候选 scripted/oracle 重新通过；旧 velocity projection 仅作历史）。`a_max=0.8` 固定为首轮主候选，`1.6` 保留为压力候选。

- [x] CE step response：加速、停车、过冲、收敛。
- [x] 90°/180° 转向：速度响应与 jerk。
- [x] 围捕环径向/切向制动。
- [x] 障碍/边界：制动距离、碰撞、穿障子步检查。
- [x] 历史线统计 projection、acceleration、jerk、path；ax/ay 新线改统计 `speed_limited`、actual acceleration、jerk、path。
- [x] 保留 top 1--2 进入训练因子。

| a_max | CE | 转向 | 制动/碰撞 | projection | 结论 |
|---:|---|---|---|---|---|
| 0.4 | 通过但 step 慢 | 通过 | 无碰撞；clearance 5.28 | 56.9% | 淘汰：响应偏慢 |
| 0.8 | 通过 | 通过 | 无碰撞；clearance 3.68 | 47.0% | 主候选：速度/安全折中 |
| 1.6 | 通过 | 通过 | 无碰撞；clearance 0.88 | 29.0% | 压力候选：快但 jerk/安全裕度较差 |

Gate：至少一个候选完成全部场景，无合同违规和明显不可控过冲/碰撞。  
证据：

- 脚本：`tools/run_continuous_action_oracle_screen.py`，SHA256 `cc811d79eb7cce3ebe408f1d1aea88835a318d5412bbd20973ee7f5443129e24`
- 命令：`PYTHONPATH=src:. python3 tools/run_continuous_action_oracle_screen.py`，exit 0
- 产物：`artifacts/2026-08-04_continuous_marl_refactor/p2_a_max_screen.json`，SHA256 `a45acd71e5c8b2932dbf74e1ecc96f5d9a1f1b26c1664310b260aa3b38ecd1c4`
- 固定 `seed=2026080401`、`v_max=3.0`、`DeltaT=0.5`，6 场景 × 3 候选；三候选均 `contract_ok=true`、`nan_free=true`、无碰撞。
- 汇总：0.4 的最大速度 2.0、最大 jerk 0.8；0.8 为 3.0/1.6；1.6 为 3.0/3.2。0.8 作为主训练因子，1.6 只作为压力/消融因子，0.4 不进入首轮训练。

#### 2026-08-05 ax/ay 重筛结果

| a_max | contract | no-collision | max speed | max actual accel | max jerk | speed-limited | min clearance | 结论 |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 0.4 | true | true | 0.905 | 0.4 | 0.8 | 0 | 7.282 | 安全但过慢 |
| 0.8 | true | true | 1.810 | 0.8 | 1.6 | 0 | 5.682 | 首轮主候选 |
| 1.6 | true | true | 3.000 | 1.6 | 3.2 | 有触发 | 2.895 | 压力候选 |

证据：

- 命令：`PYTHONPATH=src:. python3 tools/run_continuous_action_oracle_screen.py --output artifacts/2026-08-04_continuous_marl_refactor/p2_a_max_screen_axay_20260805.json`
- 产物 SHA256：`4a5a0056e64016d99d66d51b75d7ae3701d8d79476b89bb47d8ae8f697e064e0`
- 脚本 SHA256：`0bcd5cf2fd97de5fade0702e24623ad66d2ce5ce19cae907fc2c8e51ef90f9ba`
- 六场景 × 三候选均 `contract_ok=true`、`no_collision_ok=true`、`nan_free=true`、动作验证失败率为 0；旧场景“零速度即停车”已按 ax/ay 语义改为反向加速度制动。

### P3：兼容 Local Encoder 与 radial Actor

状态：`[x]`（2026-08-04，独立网络 namespace 与数值门禁完成；未接训练器）

- [x] 新增 `LocalEntityTokenEncoder`，不改 `CoCapIQN` 路径。
- [x] encoder key mapping 与覆盖率报告。
- [x] 映射后固定输入表示一致，旧 IQN regression 保持通过。
- [x] radial Gaussian、稳定 log-prob、deterministic radial squash。
- [x] Actor deployment 调用 P1 `AccelerationActionAdapter`，只验证 `a_max`。
- [x] dropout 首版显式 0。
- [x] 测 4/8/12 shape、padding、mask、entity permutation。
- [x] body-frame 输入不接 global critic state；圆盘边界 log-prob/gradient 通过。
- [x] 准备 scratch 与 IQN-encoder-transfer 两种初始化开关。

Gate：接口/数值测试全过；10k 随机前后向无 NaN/Inf；旧线无回归。  
证据：

- 配置：`configs/experiments/continuous_marl_20260804/p3_actor_contract_4v1.yaml`，SHA256 `fe582a11022d763de5f512daefa91ba8628288e6990d3280904a5ea3f8cc3006`
- 实现：`src/cocap_voradj/models/continuous/local_entity_token_encoder.py`（`5f6ecf0f...`）、`src/cocap_voradj/models/continuous/radial_actor.py`（`7d66e033...`）
- 测试：`PYTHONPATH=src:. pytest -q tests/test_continuous_actor_contract.py`，5 passed；与 P0/P1 联合门禁为 17 passed
- 固定输入映射：旧 `CoCapIQN.features` 与新 encoder 的 `self_token/mean_context/max_context/evader_features/evader_mask` 全部 `allclose`；mapping key 集合完整覆盖新 encoder。
- 10k gate：`PYTHONPATH=src:. python3 tools/run_continuous_actor_random_gate.py`，exit 0；产物 `artifacts/2026-08-04_continuous_marl_refactor/p3_random_gate.json`，`samples=10000`、`finite_outputs=true`、`finite_grads=true`、`max_action_norm=2.9998679`。
- 代码/测试/脚本/产物 SHA256：`dc3ad53973e74207f8e29812eaec86595798cd4bfd9cc1444cfdad63f9b315dc`、`4066113be18452747a946e44cc8563992c04b04a76ff4fadbef354990dad1e31`、`51590e338a02836d4480f6efb0b785058f87a45dc765b23ab487cf7f8b56ccae`。

### P4：Joint replay 与 local shared SAC smoke

状态：`[~]`（底层 joint ring、SAC 数值 smoke 与 GPU 资源测试通过；2026-08-05 审计发现 P6 在线接入未满足 terminal/event/resume 合同，集成 gate 重新打开）

- [x] 一个 joint ring；regime/event index 只存 ID。
- [x] ring 覆盖时原子清除所有旧索引引用。
- [x] 最小 schema 与 P6 terminal/replay 写入已接通；失活 slot、动作前 active mask、terminated/truncated 均有 runner 级覆盖。
- [x] 不重复存 `v_raw/v_candidate/v_world`。
- [x] `critic_mode=local` shared SAC，使用同一 joint schema。
- [x] masked mean 单元测试通过；P6 写入动作选择时 active mask，终止 transition 保留并按 slot padding。
- [x] 70/20/10 sampler 与 coverage floor 单元测试通过；P6 已从动作前 active evader 与真实 outcome 派生 regime/event，并记录 source/fallback、事件池 fallback 和实际 source slots；真实 capture/CE-success 集成样例已通过。纯 coverage smoke 事件池为空时如实记录 fallback，不据此伪造事件占比。
- [x] 以 10k 真实 transition 测 bytes，hot ring 初步定 200k（300k 作为高存储预算选项）。
- [~] checkpoint 保存网络/优化器与 CPU/CUDA torch、Python/NumPy RNG 及 scene runtime；exact env state 未保存，当前仅承诺 seeded episode-boundary continuation。
- [~] replay/checkpoint 已按当前 manifest 拒绝显式不兼容项，并纳入网络/优化/SAC、effective config hash、implementation hash 与 RNG/runtime state；当前 resume 明确是 seeded episode-boundary，未保存 exact env state，不能称完全确定性 strict resume。
- [x] batch=32 完成 CPU smoke；64 显存/吞吐待测。
- [x] smoke：pure CE -> independent capture -> mixed CR-MS，各 3334 条。
- [x] 10k 查数值；[x] 25k trend；[x] 100k--200k 只看趋势的入口条件已定义，长训留到 P6。

采样含义：

- uniform：ring 内所有有效 ID 等概率；
- active_target：动作选择时至少一个 active evader；
- coverage_only：没有 active evader，post-capture 与 pure coverage 合并；
- discovery：敌人首次进入团队可见集合，或 role coverage->capture；
- capture：loose/stationary capture，可带前后 K 步窗口；
- collision：任意实体碰撞或 boundary breach；
- CE-success：CE strict/settle latch 上升沿。

batch=64 名义约 45/13/6；coverage-only 总占比下限建议 25%，不足时替换 uniform 槽。event 空池需回退并记日志。所有标签只用于采样，不能输入网络。

Gate：三个 local smoke 均可学习；索引无悬挂；resume 后采样/更新可复现；无 NaN/OOM。  
证据：

- 配置：`configs/experiments/continuous_marl_20260804/p4_local_sac_smoke_4v1.yaml`，SHA256 `644369772ab3046e5dac84d2398a0902fcd4bb94fbc9f46d152556a1c3e0e886`
- 实现：`src/cocap_voradj/training/continuous/joint_replay.py`、`src/cocap_voradj/training/continuous/local_sac.py`
- 合同：`PYTHONPATH=src:. pytest -q tests/test_joint_replay_contract.py tests/test_local_sac_smoke.py`，7 passed
- 真实 smoke：`PYTHONPATH=src:. python3 tools/run_continuous_local_sac_smoke.py --steps-per-scene 3334`，exit 0；pure-CE/capture/mixed 各 3334，共 `replay_size=10002`；3 次更新 `finite=1.0`。
- 产物：`artifacts/2026-08-04_continuous_marl_refactor/p4_local_sac_smoke_report.json`（SHA256 `4a1bd82b...`）与 `p4_local_sac_smoke_joint_replay.pkl`（70M，SHA256 `97809023...`）。约 `7.16 KiB/transition`，线性估计 200k≈1.4GiB、300k≈2.1GiB（未含压缩/优化）。
- strict resume：同 manifest load 成功；错误 manifest 被 `ValueError` 拒绝。
- 25k 混合 trend：`p4_local_sac_smoke_report_25k.json`，25002 条、batch=64、20 updates，critic loss 约 `1.73→0.50`，全 finite。
- 分场景 trend：`p4_local_sac_per_scene_trend.json`，pure-CE `0.93→0.44`、capture `2.00→0.55`、mixed `0.55→0.37`，三者均 `critic_loss_decreased=true`、`all_finite=true`。
- GPU：`p4_local_sac_gpu_batch_report.json`，RTX A6000；batch32 `10.28 updates/s/34.1MB`，batch64 `15.44 updates/s/51.6MB`，均 finite、无 OOM。
- 以上证据只满足 P4 底层数据结构与数值 smoke；P6 在线 terminal/event/bootstrap/resume 集成修复并新增回归前，P4 不得重新标为完整 `[x]`。35 项连续动作/SAC 聚焦回归于 2026-08-05 复验通过，但现有测试未覆盖上述在线语义。

### P5：Centralized twin critics / PSA-MASAC

状态：`[x]`

- [x] 新建 `central_schema.py` 的 world-frame global entity schema，不含 phase/origin/event oracle。
- [x] 两个 Critic 从 raw feature 到 head 完全独立；target critic 冻结、eval、Polyak 更新。
- [x] final joint command 绑定对应 pursuer token，inactive/padding 正确 mask。
- [x] 一次前向输出全部 active focal pursuer 的 `Q_i`。
- [x] twin soft-Q、target、Polyak、actor loss、自动温度。
- [x] loss 按 active 数 masked mean，4/8/12 尺度合同通过。
- [x] 梯度边界、permutation、padding、teammate-action sensitivity、failure、全 inactive 非法输入测试。
- [x] local/central 共享 action/replay 合同，只切 critic mode；同一 joint batch 两模式更新均 finite。

Gate：相同 batch 上两模式均稳定；central 对 teammate command 有可解释 Q 响应，规模变化不触发 mask/shape 错误。  
证据：

- 配置：`configs/experiments/continuous_marl_20260804/p5_central_critic_contract_4v1.yaml`，SHA256 `215d7877c50cd88673afba6ad5bef1c2cf6707d82af84d2d3b79cf55c7bd88ff`。
- 实现：`src/cocap_voradj/models/continuous/central_attention_critic.py`（`09764a09...`）、`src/cocap_voradj/training/continuous/central_schema.py`（`d9aab3e6...`）、`src/cocap_voradj/training/continuous/central_sac.py`（`d5e8ad15...`）。
- 合同：`PYTHONPATH=src:. pytest -q tests/test_central_attention_critic_contract.py tests/test_central_sac_smoke.py`，8 passed。
- 联合回归：P0–P5 目标测试集共 44 passed，2 warnings，exit 0。
- 真实环境 smoke：`PYTHONPATH=src:. python3 tools/run_continuous_central_sac_smoke.py --steps-per-scene 16 --batch-size 8 --updates 1 --tag quick`，pure-CE/capture/mixed 各 16 条，共 48 条；central SAC update finite，`critic_loss=0.9963`、`twin_q_gap=0.8889`、`alpha=0.19994`。
- 产物：`artifacts/2026-08-04_continuous_marl_refactor/p5_central_sac_smoke_report_quick.json`（`da2d2330...`）与 joint replay（`65ff8821...`）。

### P6：4v1 筛选与正式训练（ax/ay/AW 结果待长训复盘）

状态：`[~]`（ax/ay 数值/action 与 BUG-04--07 工程 gate 通过；formal matched 5k 双线已完成但任务 gate 失败；六条 500k scratch 正在运行，尚无 milestone report，不能作性能结论）

因子：ax/ay radial SAC 与 AW box SAC 的独立桥接比较；当前均为 local/scratch/formal，central、IQN transfer、a_max=1.6 和多 seed 暂冻结。

- [x] runner 从 step 0 开 deterministic screening，并保存每个 screening checkpoint。
- [x] formal matched 5k 工程 preflight：两条均 finite、无 terminal drop，但 pure coverage 与 global capture 任务 gate 失败，不晋级。
- [~] 100k--200k 趋势筛选：六条 500k 线已启动；需等真实 milestone report/replay/checkpoint，再按第 0.5 节深度复盘，不能只看最终 success。
- [ ] 200k--500k 中期/末期比较：只有完成逐 episode、动作轨迹、SAC 诊断和碰撞前窗口分析后，才能淘汰或保留组合。
- [ ] 正式最多 2M，由 screening 决定早停/继续，不自动跑满。
- [ ] stage-best 自动正式 20-rollout/10-GIF，尾迹关闭。
- [ ] 晋级组合至少 3 seeds，报告 mean/std 和失败 seed。

非劣门槛（相对当前 CR-MS）：

- capture、mix capture 各不低于基线超过 5 个百分点；
- pure/mix CE strict 各不低于基线超过 5 个百分点；
- 各场景 collision 不高于基线超过 2 个百分点；
- 同时报 steps、settle、CV、path、acceleration、jerk、projection。

Gate：至少一个 3-seed PSA-MASAC 组合达标；若仅 local SAC 达标，只能判“连续动作可行、CTDE 未成功”。  
证据：

- 实现：`tools/run_continuous_p6_screening.py`（当前工作树 SHA256 `4303cc326c82f8e261836bac07f7a6fa0f71f8f22c24bf180f8c2345668a78b4`；每条运行的 manifest 仍以启动时 `implementation_hash` 为准），支持 `local/central × a_max{0.4,0.8,1.6} × scratch`、`pure_coverage/broadcast_capture` 简化诊断和可选 `actor_log_std_max` 单变量；transfer 默认拒绝，待 IQN 映射合同单独完成；每个 screening checkpoint 同步保存 replay snapshot。默认 ax/ay 仍为 0.8 / `log_std_max=1.0`。
- 当前 AW bridge 扩展：runner 支持 `--action-mode aw --w-max`；新增 `AccelerationAngularVelocityActionAdapter`、旧阻尼/yaw 连续积分和 `SquashedGaussianAccelerationAngularVelocityActor`。16-step CPU smoke `p6_aw_smoke_20260805` 与 focused regression `20 passed`；六条长训中 AW 三条使用 `a_max=0.4,w_max=pi/6`，与 ax/ay 分开审计。
- local 短程：`--critic-mode local --a-max 0.8 --total-steps 16 --batch-size 4 --update-every 2 --screen-interval 8 --screen-episodes 1 --device cpu --tag p6_local_smoke4`，16 transitions、7 updates、all finite、step1/8/16 screening、checkpoint/replay/report 均生成。
- central 短程：同参数 `--critic-mode central --tag p6_central_smoke`，16 transitions、7 updates、all finite、screening/checkpoint/replay/report 均生成。
- 原始长程：`p6_local_amax08_scratch_25k` 已完成 25k，`replay_size=25000`、`updates=24937`、`all_finite=true`、最终 report/replay/checkpoint 齐全；但它在 policy-side feasible action 补丁前运行，SAC 用 raw action 而 replay 保存 final feasible command，故性能结果作废，仅保留为 BUG-20260805-02 诊断证据。其 `projection_rate_mean=0.87759`，三类 screening 均无成功且 collision 偏高。
- 历史 velocity strict resume：使用旧 velocity checkpoint+replay 继续 4 steps，机制通过但不代表当前 ax/ay 合同；报告 `p6_local_amax08_resume_after25k_report.json`。
- ax/ay 短程：local/central 各 16 transitions、7 updates，均 finite，`action_validation_rate_mean=0.0`，step1/8/16 screening 与 checkpoint/replay/report 均生成；产物前缀分别为 `p6_local_feasible_smoke16`、`p6_central_feasible_smoke16`。
- 旧命名长程复核：`p6_local_amax08_feasible_25k` 与 `p6_local_amax04_diag5k` 的 report manifest 均为 `action_mode=velocity_2d_body`，不是当前 ax/ay 合同；其 checkpoint/replay/report 只保留为 velocity 历史/故障证据，不计入当前学习 gate，也不直接续训。
- 当前 ax/ay 长程：`axay_local_amax08_scratch_25k_20260805_r2`，seed `2026080508`、local/scratch、`a_max=0.8`、25k 已完成；PID `391852` 正常退出，step1/5k/10k/15k/20k/25k checkpoint/replay、最终 report 和严格 resume 均齐全。report 的 `replay_size=25000`、`updates=24937`、`all_finite=true`、manifest `action_mode=acceleration_2d_body`；六点 capture/CE strict 全 0，后五点 collision=0 但多为 128 步结束。
- 前一条同 seed ax/ay 进程 `axay_local_amax08_scratch_25k_20260805` 在 step1 后被安全终止；100 transitions/37 updates 基准约 16.9 秒，说明应按约 1 小时长训估计，不把该次处置误记为训练错误。
- 2026-08-05 集成审计：P6 在 `outcome.observations` 含 `None`（pursuer 因碰撞/边界永久失活）时于 `replay.add` 前退出，导致碰撞 action、`-160` safety reward 和 terminal transition 不入池；现有 broadcast capture 的碰撞学习结论因此不完整。
- 2026-08-05 集成审计：P6 把全部 `outcome.dones` 写为 `terminated`、把 `truncated` 固定为 false，而 local/central SAC 又以 `terminated|truncated` 停止 bootstrap；128-step time limit 没有按冻结设计作为可 bootstrap truncation 处理。
- 2026-08-05 集成审计（BUG-06 修复前）：P6 曾用 scene 硬编码 `active_target/coverage_only`，并把所有非 pure 场景第 1 步当作 `discovery`；该旧行为已由 `_derive_transition_metadata` 替换，但真实 capture/CE-success 集成样例和最终在线采样占比仍未闭环。
- 2026-08-05 配置审计：P6 仍固定 hidden=16、lr=3e-4、replay>=64 即更新、无 5k warmup/grad clip；这是 smoke 参数，不是设计预注册的 hidden=256、lr=1e-4、warmup=5k、grad clip=10 正式入口。修复数据语义后必须显式冻结有效 P6 config，再形成新的性能结论。
- 2026-08-05 BUG-04 实施：`tools/run_continuous_p6_screening.py` 增加固定 agent slot 的 `_stack_obs_with_padding`；失活 next obs 零 padding，transition 写入前 active mask 取动作选择时状态；新增 2 个 padding/shape 回归测试。
- BUG-04 窄验证：`PYTHONPATH=src:. pytest -q -p no:cacheprovider tests/test_continuous_p6_simplified_task_contract.py tests/test_joint_replay_contract.py tests/test_local_sac_smoke.py`，`13 passed`；新 tag `bug04_deactivation_smoke_counts2_20260805` 运行 64 steps/49 updates，`all_finite=true`、`replay_size=64`、`terminal_transition_count=1`、`inactive_next_slot_transition_count=1`、`collision_transition_count=1`、`collision_reward_sum=-160.0`，report/checkpoint/replay 齐全。BUG-04 的 replay-level 证据已闭环。
- BUG-05 实施：runner 新增 `_split_termination_flags`，按 env info 将 `too long episode` 分为 `truncated`；local/central SAC 共用 `sac_bootstrap_mask`，只由 `terminated` 关闭 bootstrap。
- BUG-05 窄验证：同一组 local/central SAC、replay、CE 测试扩展为 `PYTHONPATH=src:. pytest -q -p no:cacheprovider tests/test_continuous_p6_simplified_task_contract.py tests/test_joint_replay_contract.py tests/test_local_sac_smoke.py tests/test_central_sac_smoke.py tests/test_ce_coverage.py`，`28 passed`；其中真实连续环境跑满 128 steps 并断言 `too long episode -> truncated=true, terminated=false`，真实 capture→post 两步均断言不 terminal。另有 `bug05_termination_semantics_smoke_20260805` 64-step/49-update report：`all_finite=true`、`terminal=1`、`terminated=1`、`truncated=0`、`inactive=1`、`collision=1`、`collision_reward_sum=-160.0`。BUG-05 集成证据闭环。
- BUG-06 实施：`_derive_transition_metadata` 用动作前 active evader、真实 `last_capture_events`、`info.state` collision/boundary、replay task transition 和 coverage success latch 派生 regime/event；runner 不再按 scene/首步伪造；sampler report 增加 `event_pool_available/event_fallback_count`。
- BUG-06 窄验证：真实 capture/CE-success 样例加入后，`PYTHONPATH=src:. pytest -q -p no:cacheprovider tests/test_continuous_p6_simplified_task_contract.py tests/test_joint_replay_contract.py tests/test_local_sac_smoke.py tests/test_central_sac_smoke.py tests/test_ce_coverage.py` 为 `32 passed`；`bug06_regime_event_smoke_20260805` 64-step/49-update 曾真实写出 `discovery=1/collision=2`，`bug06_regime_event_smoke_counts2_20260805` 进一步写出 source slots `uniform=539/regime=147/fallback_uniform=98`、`sampling_event_fallback_count=49`、`sampling_coverage_floor_min_ratio=1.0`、`all_finite=true`。后一个 seed 64 步均停留 pure coverage，事件池为空的 fallback 已如实保留；capture/CE-success 由真实环境 runner-level 测试覆盖。BUG-06 数据语义与审计统计闭环。
- BUG-07 实施/验证：local/central checkpoint schema 升为 2，joint replay schema 升为 3；manifest 纳入 `trainer_contract`（含 actor a_max/dt、critic hidden/layers/heads、lr/gamma/tau/entropy/warmup）、每 scene effective config hash、implementation hash、RNG 列表与 resume mode；checkpoint/replay 保存 Python/NumPy/CUDA/torch RNG 及 `next_scene_index` runtime state。`bug07_runtime_contract_smoke_20260805` 16 steps/7 updates、`all_finite=true`，随后 resume 到 step20 为 2 updates finite，`resumed_from.next_scene_index=1`、`continuation_mode=seeded_episode_boundary`；同 checkpoint 以 `actor_log_std_max=-1.0` resume 被 `ValueError ... contract mismatch` 拒绝。最新 `bug07_manifest_contract_smoke_20260805` 4 steps/2 updates 复核新增 critic 字段。当前只证明严格合同与 seeded episode-boundary 恢复，exact env state 仍明确为 false。
- formal profile 首轮资源 smoke：新增 `configs/experiments/continuous_marl_20260804/p6_formal_local_sac_4v1.yaml`，冻结 hidden=256、encoder 4 layers/8 heads、lr=1e-4、warmup=5000、grad clip=10；1-step CUDA smoke 返回 `replay_size=1`、`updates=0`、`all_finite=true`，manifest trainer/config hash 一致。随后 `p6_formal_profile_1k_resource_smoke_20260805` 返回 `replay_size=1000`、`updates=0`、`all_finite=true`；`p6_formal_profile_5k_warmup_preflight_20260805` 跑满 5064 steps、跨过 warmup，`updates=65`、`all_finite=true`、`sampling_event_fraction=0.1269`、`sampling_coverage_floor_min_ratio=1.0`。两者均为资源/数值 preflight：纯 coverage 随机初始策略 collision 偏高，不构成学习趋势或晋级依据。
- 报告诊断字段实施：local/central SAC update 新增 `target_q/td_error_abs/log_prob/entropy_proxy/log_std/latent_norm/actor-critic-alpha_grad_norm`；runner 新增 `reward_term_sums/means`、`transition_attempt_count`、`terminal_transition_drop_count`。命令 `PYTHONPATH=src:. python tools/run_continuous_p6_screening.py --critic-mode local --a-max 0.8 --scene-set pure_coverage --total-steps 16 --batch-size 4 --update-every 2 --screen-interval 8 --screen-episodes 1 --device cpu --tag p6_diag_metrics_smoke2_20260805` 返回 exit 0、`replay=16`、`updates=7`、`all_finite=true`、`terminal_drop=0`，report 中 `entropy_proxy_mean` 已写入；窄回归 `33 passed`。
- formal matched 5k 双线最终：pure 与 broadcast 两份 report 均 `replay_size=10064`、`updates=5065`、`all_finite=true`、`terminal_transition_drop_count=0`。pure 在 step1/5000/10000 均 `CV<0.15=0/2`、collision=2/2，末态 mean speed 约 2.20--3.00；broadcast 在 step1/5000/10000 均 capture=0/2、collision=2/2。工程 gate 通过，学习/任务 gate 不通过，不开 25k。
- 交接注意：现有 25k 仅保留 action/数值/资源与故障诊断证据；修复上述集成问题、新增回归并重跑短 gate 前，不得进入 100k--300k、central、`a_max=1.6` 或课程晋级。
- resume 验证：从 `p6_resume_seed_step4.pt` + `p6_resume_seed_replay_step4.pkl` 以 `--resume-step 4` 继续到 step8，`resumed_from`、strict manifest、2 updates 均 finite；报告 `p6_resume_seed_cont_report.json`（`c64d584e...`）。
- r2 final strict resume：`axay_local_amax08_scratch_25k_20260805_r2_resume4_report.json`，从 step25000 恢复至 replay_size=25004，4 updates，`all_finite=true`，manifest/action mode 一致。
- 最小可学习性诊断：零速度脚本在 pure-CE/capture/mixed 小样本无碰撞但未成功；r2 replay 动作范数 mean `0.6782`、p95 `0.7948`、max `0.7999`，reward mean `-0.2639`，terminal fraction `0.00644`。r2 后五个 screening 点 collision=0 但大多 128 步结束、capture/CE strict 全 0，优先检查 reward/bootstrap/log-prob/sampler/观测尺度，不直接扩容长训。
- replay 元数据诊断：`coverage_only/pure_coverage=8353`、`active_target/pre_capture=16647`、`discovery event=169`；不是单一场景数据，但仍需确认 70/20/10 sampler 的实际抽样占比。
- SAC 数值诊断：final update tail 全部 finite，alpha 从约 `0.2` 降至约 `0.05448`，Q 均值约 `-28--35`；探索温度收缩与后期低速停滞一致，下一步先做 alpha/log-prob/奖励尺度单变量诊断。
- Oracle 诊断：`tools/run_continuous_oracle_diagnostic.py` 与产物 `artifacts/2026-08-04_continuous_marl_refactor/diagnostics/continuous_oracle_pure_ce.json`；seed `2026080410/2026080411` 分别 55/60 步无碰撞、CE strict=1、center RMS 0.00351/0.00300。环境与动作合同可达，当前问题是 Actor/SAC 学习信号，不得用“环境不可解”解释失败。

### P7：8v2/12v3 课程

状态：`[ ]`

- [ ] 只从 P6 solid/stage-best 晋级。
- [ ] 8v2、12v3 各建议上限 700k，100k screening，不保证跑满。
- [ ] 各阶段 stage-best 正式 20-rollout/10-GIF，尾迹关闭。
- [ ] 4/8/12 交叉规模矩阵。
- [ ] mixed-size replay、mask/failure 测试。
- [ ] 比较 loss、entropy、Q、updates/s、GPU memory。

Gate：8v2/12v3 达各自非劣门槛，无 mask、loss 缩放或规模稳定性错误。  
证据：`待填`

### P8：主线 solid 后消融

状态：`[S]`，现在不实施：离散 vs 连续、local vs PSA、PSA vs Set-Attention MATD3、scratch vs transfer、nearest boundary vs four-wall、TQC/GRU/flow/FACMAC 等。P6/P7 solid 后才逐个单变量做。

### P9：正式评测、展示与交接

状态：`[ ]`

- [ ] deterministic screening 覆盖 capture/pure/mix。
- [ ] 排序为 capture -> CE strict -> collision -> steps -> CV。
- [ ] metadata 含 algorithm/action_mode/a_max/yaw_mode/deterministic/config hash。
- [ ] stage-best 固定 seed 20-rollout/10-GIF，faded trail 默认关闭。
- [ ] 3-seed 与当前 CR-MS 三规模正式结果并排。
- [ ] checkpoint/config/manifest/screening/formal artifacts 齐全。
- [ ] 更新 handoff、设计文档和本台账。

---

## 6. 遇错修正逻辑

修正等级：

| 级别 | 含义 | 处理 |
|---|---|---|
| L0 | 拼写、日志、测试夹具等不改语义 | 可修，重跑相关 gate |
| L1 | 明确实现 bug：mask、shape、frame、索引、key | 可修，必须新增复现测试 |
| L2 | lr、tau、alpha、batch、UTD、capacity | 只做单变量对照 |
| L3 | 动作、observation、reward、termination、算法结构 | 停止并请用户确认 |

故障优先级：

1. 旧线回归失败：立即停；不能更新 baseline hash 来接受回归。
2. 动作合同失败：只修 adapter/frame/dynamics；env 不加 silent clip。
3. NaN/Inf：停长训，保存首个坏 batch、RNG、checkpoint，做最小复现。
4. replay/manifest 不同：strict reject；无确定性 converter 不跨 run。
5. local SAC 不学：不进 central；先查 reward、bootstrap、log-prob、sampler、数据。
6. local 学而 central 不学：锁定 Actor/action/replay，只查 global schema、mask、target、梯度、capacity。
7. OOM/吞吐：先调 batch、capacity、评测并发、日志频率，不改语义。
8. 性能不足：按下表单变量分支，失败 run 原样保留。

| 结果 | 首查 | 下一调整 | 禁止 |
|---|---|---|---|
| local/central 都不学 | action/log-prob/reward/bootstrap/replay | 回 P1--P4 最小复现 | 直接改 reward/加复杂 critic |
| local 学、central 不学 | global schema/mask/Q target/gradient | 单独修 critic | 同时改 Actor/sampler |
| capture 好、CE 差 | deterministic eval/a_max/alpha/抖动 | 比 P2 a_max；单调 entropy/UTD | 把 CV 当 strict 或改 CE reward |
| CE 好、capture 差 | discovery/active/event/support 数据 | 单调采样或探索 | 破坏 CE 配置 |
| 随机训练好、确定评测差 | mean action/entropy/log_std | 调 alpha target 或 bounds | 挑幸运随机 rollout |
| projection 长期高 | raw action/a_max/target entropy | 比 P2 候选、查增量参数化 | env 放宽约束 |
| collision/穿障高 | substep/制动/a_max | 修检测或换已筛候选 | 只看终局碰撞 |
| 4v1 好、大规模差 | masked mean/critic capacity/课程 | 单变量规模诊断 | 加固定 agent ID |
| transfer 差于 scratch | mapping/normalization/表示偏置 | scratch 主线，transfer 留消融 | 强行冻结旧 encoder |
| pure 好、post/mix 差 | termination/timer 隐状态 | 统一合同；语义变化升 L3 | 只把 phase 给 Critic |

每个错误按此闭环：

~~~text
BUG-YYYYMMDD-NN
run/step/seed：
症状与最小复现：
首个错误日志或 batch：
级别与根因：
修正文件：
新增回归测试：
修前/修后结果：
受影响 run 是否作废：
下一动作：
~~~

---

## 7. 预期结果与判定

预期是假设，不是通过证明：

- P1：command 精确满足 `v_max/a_max`，yaw 不漂移，无 clip、NaN、穿障。
- P2：0.4 可能过慢，0.8/1.6 更可能晋级，但完全由数据决定。
- P3：圆盘中心/边缘 log-prob 和梯度有限，映射 encoder 表示一致。
- P4：local SAC 在 pure CE、独立 capture 至少出现上升趋势；否则 CTDE 不能救基础链。
- P5：central critic 在 mixed/有限感知协同不差于 local，并保持多规模 loss 稳定。
- P6：连续控制降低切换与抖动，同时保持 CR-MS 成功率；不能拿平滑换性能退化。
- P7：变规模能力必须由交叉规模和 mask 失败测试证明，不能只看 shape。

判定：

- **成功：** 3-seed PSA-MASAC 达非劣门槛，且平滑/路径/制动至少一项稳定改善。
- **部分成功：** local SAC 达标，central 无收益；连续动作可行但 CTDE 未成功。
- **结构性失败：** 合同正确且合理单变量调参后 local/central 均不学；再评估 SAC 参数化或 Set-Attention MATD3。
- **实现失败：** gate、数值或旧线回归未过；不得形成算法性能结论。

---

## 8. 运行、问题与决策记录

### 8.1 Run 台账

| run_id | P | 日期 | config/hash | critic | init | a_max | seed | steps | GPU | 状态 | artifact | 下一动作 |
|---|---|---|---|---|---|---:|---:|---:|---|---|---|---|
| 待填 | P0 | - | - | none | none | - | - | 0 | - | 未开始 | - | baseline manifest |
| old-iqn-reg-4v1-2026081201 | P0 | 2026-08-04 | stage1 / frozen manifest | old IQN | baseline | - | 2026081201 | 1x3 episodes | CPU | passed | `old_iqn_regression/4v1_seed_2026081201` | P0 complete |
| old-iqn-reg-8v2-2026082201 | P0 | 2026-08-04 | stage2 / CE-first 300k | old IQN | baseline | - | 2026082201 | 1x3 episodes | CPU | passed | `old_iqn_regression/8v2_seed_2026082201_evaders_2` | P0 complete |
| old-iqn-reg-12v3-2026082301 | P0 | 2026-08-04 | stage3 / 700k | old IQN | baseline | - | 2026082301 | 1x3 episodes | CPU | passed | `old_iqn_regression/12v3_seed_2026082301_evaders_3` | P0 complete |
| p1-contract-4v1-2026080401 | P1 | 2026-08-04 | p1_velocity_contract_4v1 / `fffea0a4...` | none | scripted | 0.8 | 2026080401 | 1 step + contract tests | CPU | passed | `configs/experiments/continuous_marl_20260804/p1_velocity_contract_4v1.yaml` | P2 complete |
| p2-amax-screen-2026080401 | P2 | 2026-08-04 | p1 config / `a45acd71...` | none | oracle scripted | 0.4/0.8/1.6 | 2026080401 | 6×3 scripted sequences | CPU | passed | `artifacts/2026-08-04_continuous_marl_refactor/p2_a_max_screen.json` | P3 encoder/Actor |
| p3-actor-contract-2026080403 | P3 | 2026-08-04 | p3_actor_contract_4v1 / `fe582a11...` | none | scratch + mapping | 0.8 | 2026080403 | 5 tests + 10k fwd/bwd | CPU | passed | `artifacts/2026-08-04_continuous_marl_refactor/p3_random_gate.json` | P4 replay |
| p4-local-sac-smoke-2026080404 | P4 | 2026-08-04 | p4_local_sac_smoke_4v1 / `64436977...` | local | scratch | 0.8 | 2026080404 | 10002 real joint + 3 updates | CPU | partial | `artifacts/2026-08-04_continuous_marl_refactor/p4_local_sac_smoke_report.json` | 25k trend + batch64 |
| p4-local-sac-trend-2026080404 | P4 | 2026-08-04 | p4 smoke / 25k | local | scratch | 0.8 | 2026080404 | 25002 real joint + 20 updates | RTX A6000/CPU | passed | `artifacts/2026-08-04_continuous_marl_refactor/p4_local_sac_smoke_report_25k.json` | P5 twin critics |
| p5-central-smoke-2026080409 | P5 | 2026-08-05 | p5_central_critic_contract_4v1 / `215d7877...` | central | scratch | 0.8 | 2026080409 | 48 real joint + 1 update | CPU | passed | `artifacts/2026-08-04_continuous_marl_refactor/p5_central_sac_smoke_report_quick.json` | P6 screening |
| p6-local-smoke-2026080410 | P6 | 2026-08-05 | p6 runner / `p6_local_smoke4` | local | scratch | 0.8 | 2026080410 | 16 + step1/8/16 screen | CPU | passed | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/p6_local_smoke4_report.json` | 25k preflight |
| p6-central-smoke-2026080410 | P6 | 2026-08-05 | p6 runner / `p6_central_smoke` | central | scratch | 0.8 | 2026080410 | 16 + step1/8/16 screen | CPU | passed | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/p6_central_smoke_report.json` | 25k preflight |
| p6-local-amax08-scratch-25k-2026080410 | P6 | 2026-08-05 | pre-feasible runner / `fb87ea69...` | local | scratch | 0.8 | 2026080410 | 25k | RTX A6000 + CPU env | invalidated (action mismatch) | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/p6_local_amax08_scratch_25k_*` | retain diagnostic; rerun corrected contract |
| p6-local-amax08-feasible-25k-2026080410 | P6 | 2026-08-05 | legacy runner | local | scratch | 0.8 | 2026080410 | 25k complete; manifest `velocity_2d_body` | RTX A6000 + CPU env | historical velocity / invalid for ax/ay | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/p6_local_amax08_feasible_25k_*` | do not resume |
| p6-local-amax04-diag-5k-2026080411 | P6 | 2026-08-05 | legacy runner | local | scratch | 0.4 | 2026080411 | 5k complete; manifest `velocity_2d_body` | RTX A6000 + CPU env | historical velocity / invalid for ax/ay | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/p6_local_amax04_diag5k_*` | do not promote |
| p6-local-axay-amax08-25k-r2-2026080508 | P6 | 2026-08-05 | current runner / `354de314...` | local | scratch | 0.8 | 2026080508 | 25k complete; 6 screens; resume +4 | RTX A6000 + CPU env | numeric/action passed; replay-semantics invalid for promotion | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/axay_local_amax08_scratch_25k_20260805_r2_*` | fix integration; rerun new tag |
| p6-pure-coverage-ablation-5k-2026080511 | P6-diagnostic | 2026-08-05 | `scene_set=pure_coverage`, control ablation | local | scratch | 0.8 | 2026080511 | 5k; 6 screens | GPU0 + CPU env | engineering passed / learning failed | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/axay_pure_coverage_ablation_5k_20260805_*` | no 25k; inspect SAC/reward/settling signal |
| p6-broadcast-capture-5k-2026080512 | P6-diagnostic | 2026-08-05 | `scene_set=broadcast_capture`, global visibility | local | scratch | 0.8 | 2026080512 | 5k; 6 screens | GPU1 + CPU env | finite; performance inconclusive after terminal-drop audit | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/axay_broadcast_capture_global_5k_20260805_*` | fix terminal replay; matched rerun |
| p6-pure-coverage-logstdm1-5k-2026080513 | P6-diagnostic | 2026-08-05 | `pure_coverage` + `actor_log_std_max=-1.0` | local | scratch | 0.8 | 2026080513 | 5k; 6 screens | GPU0 + CPU env | engineering passed / learning failed; different seed | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/axay_pure_coverage_logstdm1_5k_20260805_*` | no promotion; retain exploratory evidence |
| p6-pure-coverage-logstdm1-5k-same-seed-2026080511 | P6-diagnostic | 2026-08-05 | `pure_coverage` + `actor_log_std_max=-1.0`, same seed control | local | scratch | 0.8 | 2026080511 | 5k; 6 screens | GPU1 + CPU env | engineering passed / learning failed | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/axay_pure_coverage_logstdm1_5k_20260805_same_seed_*` | low-noise factor closed; no 25k |
| p6-bug04-deactivation-smoke-2026080513 | P6-fix | 2026-08-05 | runner padding fix / `pure_coverage` | local | scratch | 0.8 | 2026080513 | 64 requested; 49 updates; screen 1/64 | CPU | finite; BUG-04 implementation smoke passed | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/bug04_deactivation_smoke_20260805_*` | add replay-level terminal assertion; then BUG-05 |
| p6-bug04-deactivation-smoke-counts2-2026080513 | P6-fix | 2026-08-05 | runner padding + counters / `pure_coverage` | local | scratch | 0.8 | 2026080513 | 64 requested; 49 updates; screen 1/64 | CPU | finite; terminal/inactive/collision/reward counts verified | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/bug04_deactivation_smoke_counts2_20260805_*` | proceed BUG-05; no long train |
| p6-bug05-termination-semantics-smoke-2026080513 | P6-fix | 2026-08-05 | terminated/truncated split + bootstrap mask / `pure_coverage` | local | scratch | 0.8 | 2026080513 | 64 requested; 49 updates; screen 1/64 | CPU | finite; collision terminal preserved; real 128-step timeout + capture→post runner assertions passed | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/bug05_termination_semantics_smoke_20260805_*` | proceed BUG-06; no long train |
| p6-bug06-regime-event-smoke-counts2-2026080519 | P6-fix | 2026-08-05 | real regime/event + source slots/fallback / full | local | scratch | 0.8 | 2026080519 | 64 requested; 49 updates; screen 1/64/3 scenes | CPU | finite; source slots uniform=539/regime=147/fallback=98; event fallback=49; floor min=1.0; pure-only seed had no event pool | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/bug06_regime_event_smoke_counts2_20260805_*` | BUG-06 closed; proceed BUG-07; no long train |
| p6-bug07-runtime-contract-smoke-2026080520 | P6-fix | 2026-08-05 | manifest/RNG/runtime state / `pure_coverage` | local | scratch | 0.8 | 2026080520 | 16 requested; 7 updates; screens 1/8/16 | CPU | finite; manifest schema2, trainer/config/code hashes, RNG/runtime state saved | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/bug07_runtime_contract_smoke_20260805_*` | resume smoke + mismatch reject passed; exact env state remains false |
| p6-bug07-runtime-resume-smoke-2026080520 | P6-fix | 2026-08-05 | strict manifest resume / `pure_coverage` | local | scratch | 0.8 | 2026080520 | 16→20; 2 updates | CPU | `resumed_from.next_scene_index=1`, seeded episode-boundary, finite; wrong log_std hard-rejected | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/bug07_runtime_resume_smoke_20260805_*` | BUG-07 partial; no long train |
| p6-bug07-manifest-contract-smoke-2026080521 | P6-fix | 2026-08-05 | extended trainer contract / `pure_coverage` | local | scratch | 0.8 | 2026080521 | 4 requested; 2 updates; screen 1 | CPU | finite; critic hidden/layers/heads + actor a_max/dt present in manifest | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/bug07_manifest_contract_smoke_20260805_*` | BUG-07 partial; formal 1k pending |
| p6-formal-profile-resource-smoke-2026080522 | P6-formal | 2026-08-05 | `p6_formal_local_sac_4v1` / `pure_coverage` | local | scratch | 0.8 | 2026080522 | 1 requested; 0 updates; screen step1/0 episodes | RTX A6000 GPU0 | finite; hidden=256, encoder 4×8, lr=1e-4, warmup=5k, clip=10; manifest consistent | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/p6_formal_profile_resource_smoke_20260805_*` | run formal 1k smoke only after BUG-07 final decision |
| p6-formal-profile-1k-resource-smoke-2026080524 | P6-formal | 2026-08-05 | `p6_formal_local_sac_4v1` / `pure_coverage` | local | scratch | 0.8 | 2026080524 | 1000 requested; 0 updates; screens step1/1000 | RTX A6000 GPU0 | finite; terminal/inactive/collision=24; warmup correctly suppresses updates | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/p6_formal_profile_1k_resource_smoke_20260805_*` | preflight only; no performance conclusion |
| p6-formal-profile-5k-warmup-preflight-2026080525 | P6-formal | 2026-08-05 | `p6_formal_local_sac_4v1` / `pure_coverage` | local | scratch | 0.8 | 2026080525 | 5064 requested; 65 updates; screens step1/5064 | RTX A6000 GPU0 | finite across warmup; source slots uniform=2925/regime=845/event=390; event fraction=.1269; floor=1.0; collision=178 | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/p6_formal_profile_5k_warmup_preflight_20260805_*` | preflight only; matched 5k pending |
| p6-formal-matched5k-pure-2026080514 | P6-formal-matched | 2026-08-05 21:11 start | `p6_formal_local_sac_4v1`, `scene_set=pure_coverage` | local | scratch | 0.8 | 2026080514 | 10064 requested (=5000 warmup+约5065 updates，inclusive boundary) | RTX A6000 GPU0 | completed 21:49; finite/replay=10064/updates=5065; collision=275/275; CV<0.15=0/2 at steps 1/5000/10000; end speed remained high; no promotion | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/p6_formal_matched5k_pure_coverage_2026080514_report.json` | no 25k; diagnose collision/reward/control |
| p6-formal-matched5k-global-2026080514 | P6-formal-matched | 2026-08-05 21:11 start | `p6_formal_local_sac_4v1`, `scene_set=broadcast_capture`, `global_evader_visibility=true` | local | scratch | 0.8 | 2026080514 | 10064 requested (=5000 warmup+约5065 updates，inclusive boundary) | RTX A6000 GPU1 | completed 21:53; finite/replay=10064/updates=5065; capture=0/2 and collision=2/2 at steps 1/5000/10000; no promotion | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/p6_formal_matched5k_global_broadcast_2026080514_report.json` | no 25k; diagnose collision/capture credit |
| p6-axay-500k-mix-2026080515 | P6-long | 2026-08-05 23:26 start | formal P6, `scene_set=full`, non-global | local | scratch | 0.8 | 2026081501 | 500000 requested; replay cap 100000; screens every 100000 | RTX A6000 GPU0 | running; initialized; no result yet | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/p6_axay_500k_mix_non_global_20260805_*` | late report must include CV<0.15/speed/collision and SAC diagnostics |
| p6-axay-500k-pure-2026080515 | P6-long | 2026-08-05 23:26 start | formal P6, `scene_set=pure_coverage` | local | scratch | 0.8 | 2026081502 | 500000 requested; replay cap 100000; screens every 100000 | RTX A6000 GPU1 | closed by user 2026-08-06 16:30 at step100000; learning gate failed: rollout collision at step 14/16, end speed 3.0, CV 0/2; no final report | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/p6_axay_500k_pure_coverage_20260805_*` | 本批不作晋级证据；产物保留待后续大改参考 |
| p6-axay-500k-global-2026080515 | P6-long | 2026-08-05 23:26 start | formal P6, `scene_set=broadcast_capture`, global visibility | local | scratch | 0.8 | 2026081503 | 500000 requested; replay cap 100000; screens every 100000 | RTX A6000 GPU0 | running; initialized; no result yet | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/p6_axay_500k_global_capture_20260805_*` | capture event rate primary; collision/length/speed secondary |
| p6-aw-500k-mix-2026080516 | P6-long-bridge | 2026-08-05 23:35 start | formal P6, `action_mode=aw`, `scene_set=full`, non-global | local | scratch | 0.4, pi/6 | 2026081601 | 500000 requested; replay cap 100000; screens every 100000 | RTX A6000 GPU0 | running; initialized; 16-step CPU smoke passed before launch | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/p6_aw_500k_mix_non_global_20260805_*` | bridge dynamics and box SAC; do not compare directly to ax/ay without action contract |
| p6-aw-500k-pure-2026080516 | P6-long-bridge | 2026-08-05 23:35 start | formal P6, `action_mode=aw`, `scene_set=pure_coverage` | local | scratch | 0.4, pi/6 | 2026081602 | 500000 requested; replay cap 100000; screens every 100000 | RTX A6000 GPU1 | closed by user 2026-08-06 16:30 at step100000; learning gate failed: no-op/brake-in-place (mean a≈-0.32, speed 0), CV 0/2, collision 0; no final report | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/p6_aw_500k_pure_coverage_20260805_*` | 本批不作晋级证据；产物保留待后续大改参考 |
| p6-aw-500k-global-2026080516 | P6-long-bridge | 2026-08-05 23:35 start | formal P6, `action_mode=aw`, `scene_set=broadcast_capture`, global visibility | local | scratch | 0.4, pi/6 | 2026081603 | 500000 requested; replay cap 100000; screens every 100000 | RTX A6000 GPU0 | running; initialized; 16-step CPU smoke passed before launch | `artifacts/2026-08-04_continuous_marl_refactor/p6_aw_500k_global_capture_20260805_*` | capture event rate primary; collision/length/speed secondary |
| p6-diagnostic-metrics-smoke2-2026080510 | P6-fix | 2026-08-05 | runner + local SAC diagnostics / `pure_coverage` | local | scratch | 0.8 | 2026080410 | 16 requested; 7 updates; screens 1/8/16 | CPU | finite; replay=16; terminal_drop=0; reward term sums + TD/log-prob/log-std/entropy proxy/grad norms/alpha/Q/source stats written | `artifacts/2026-08-04_continuous_marl_refactor/p6_screening/p6_diag_metrics_smoke2_20260805_report.json` | report-field gate closed; matched 5k pending |

### 8.2 指标

| run/step | 场景 | capture | CE strict | collision | steps | CV<=0.15 | final CV | speed | accel/jerk | projection | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| axay r2 step25000 | pure_ce | 0/2 | 0/2 | 0/2 | 128/128 | 1/2 | 0.2716/0.1077 | 0.262/0.325 | replay action norm p95 0.795 | 0 | 工程通过/strict 失败 |
| axay r2 step25000 | capture | 0/2 | 0/2 | 0/2 | 128/128 | 1/2 | 0.1653/0.1454 | 0.227/0.262 | replay action norm p95 0.795 | 0 | 工程通过/strict 失败 |
| axay r2 step25000 | mixed_crms | 0/2 | 0/2 | 0/2 | 128/128 | 0/2 | 0.1910/0.2315 | 0.305/0.317 | replay action norm p95 0.795 | 0 | 工程通过/strict 失败 |
| pure coverage ablation 5k | pure_ce | 0/2 | 0/2 | 0/2 | 128/128 | 0/2 | 0.215/0.265 | 0.135/0.100 | finite | 0 | learning failed; late stop |
| broadcast capture global 5k | capture | 0/2 | 0/2 | 2/2 | 23/59 | 0/2 | 0.515/0.302 | 2.981/0.715 | finite | 0 | learning failed; collision |
| pure coverage logstd=-1 same seed 5k | pure_ce | 0/2 | 0/2 | 1/2 | 69/128 | 0/2 | 0.5401/0.1687 | 1.981/0.531 | finite | 0 | no improvement vs default |
| formal matched pure coverage 10064 | pure_ce | 0/2 | 0/2 strict | 2/2 at steps 1/5000/10000 | 21--33 | 0/2 at all screens | 0.2606/0.3634 → 0.5208/0.6772 | 2.20--3.00 | finite; no drop | 0 | no trend; collision-dominated |
| formal matched global broadcast 10064 | capture | 0/2 capture event | n/a | 2/2 at steps 1/5000/10000 | 22--52 | n/a | 0.3158/0.4548 → 0.4365/0.7095 | 2.69--3.00 | finite; no drop | 0 | no capture; collision-dominated |
| axay pure 500k step100000 | pure_ce | 0/2 | 0/2 | 2/2（step 14/16） | 14/16 | 0/2 | area CV 0.500/0.421 | 3.0/3.0 | action norm mean 0.585/0.628, p95≈0.79 | 0 | 学习失败：满油门撞墙；replay collision=2403 |
| aw pure 500k step100000 | pure_ce | 0/2 | 0/2 | 0/2 | 128/128 | 0/2 | area CV 0.392/0.548 | 0.0/0.0 | mean a≈-0.32（制动）、w 先转后归零 | 0 | 学习失败：原地制动/转圈 no-op；replay collision=1 |

### 8.3 Bug

| bug_id | run/step | 症状 | 级别 | 修正 | 回归测试 | 结果 |
|---|---|---|---|---|---|---|
| 待填 | - | - | - | - | - | - |
| BUG-20260804-01 | 8v2/12v3 first attempt | regression wrapper hardcoded `capture_evaders=1`，导致 steps mismatch | L1 | 按规模改为 1/2/3；新 namespace 重跑 | 三规模 report + core/hash 对比 | 已修复；错误输出保留 |
| BUG-20260804-02 | P1 Robot substep | 初版连续积分在每个子步递归引用已更新速度，轨迹 x 偏离梯形积分合同 | L1 | 使用 `start_world` 固定起点速度，逐子步线性插值；保留 10 子步回调 | `test_robot_velocity_update_uses_all_ten_substeps_and_trapezoids` + P1 9 passed | 已修复；旧线回归仍通过 |
| BUG-20260805-01 | P6 online projection（历史 velocity 线） | adapter 投影极限值转 float32 后为 `0.40000001`，被 1e-8 容差误拒 | L1 | 历史兼容代码的 `atol` 调为 `1e-6`；当前 ax/ay adapter 不做速度投影 | 历史 P1 通过；当前 action 合同 30 passed | 已修复；不影响当前 `a_max` 合同 |
| BUG-20260805-02 | 原始 P6 25k | Actor/critic 训练采 raw desired velocity，但 replay/env 执行 final feasible command；`projection_rate_mean=0.87759`，25k screening 无成功 | L1 | 最终改为显式 `a_x,a_y`；移除 velocity feasible layer，env 只负责 `v_max` | 当前 ax/ay action/actor/replay/env 30 passed | 原始 velocity 25k 性能结论作废；保留 report/replay/checkpoint 作诊断 |
| BUG-20260805-03 | 修正后 P6 25k / feasible layer 初始状态 | `previous_body=0` 时 latent 半径全部映射为同一速度增量，径向 Jacobian 为 0；SAC 仍用 raw latent `log_prob` | L3 | 已按用户确认采用显式 `a_x,a_y`；加速度即 replay/env action，runtime joint replay schema 为 3，取消不可逆 velocity mapping | action、Actor、env、local/central/replay 共 30 passed | 旧 velocity run 作废；ax/ay 从 4v1 scratch 重新 screening |
| BUG-20260805-04 | P6 所有 collision/deactivation step | `next_observations` 过滤 `None` 后数量不足即在 `replay.add` 前 break；碰撞 action、safety reward、terminal transition 全部丢失 | L1 | 已实施：固定 4-agent slot、失活 next obs 零 padding、active mask 改为动作前状态；report 增加 terminal/inactive/collision/reward counters | 窄测试 `13 passed`；64-step smoke `terminal=1`、`inactive=1`、`collision=1`、`collision_reward_sum=-160.0`、`replay_size=64` | `[x]` 修复与 replay-level 证据完成；失败的旧 run 保留，不得续训 |
| BUG-20260805-05 | P6 timeout/bootstrap | runner 把 `outcome.dones` 全写为 terminated、truncated 恒 false；SAC 又对 terminated/truncated 一律停止 bootstrap | L1 | 已实施：`_split_termination_flags` + shared `sac_bootstrap_mask`；local/central 仅 terminated 关闭 bootstrap | `28 passed`；真实连续环境 128-step timeout、capture→post runner 边界、collision smoke 均通过；修前 25k Bellman 语义不合格 | `[x]` 修复与集成证据闭环；旧 25k 不得续训 |
| BUG-20260805-06 | P6 regime/event sampler | regime 按初始 scene 固定，非 pure 首步伪标 discovery；缺 post-capture/capture/collision/CE-success 与 source/fallback 统计 | L1 | 已实施：从动作前 active evader 和 outcome env/replay metadata 派生 regime/event；event source、source slots、coverage floor 与空事件池 fallback 写入 report | `32 passed`；真实 capture/CE-success runner-level 样例通过；两次 64-step smoke 分别覆盖 discovery/collision 与 source-slot/fallback 实际统计 | `[x]` 数据语义、事件索引与审计统计闭环；纯 coverage seed 的空事件池 fallback 保留，不作性能结论 |
| BUG-20260805-07 | P6 strict resume/config | manifest 缺网络/优化/entropy/warmup/effective config/code hash；仅恢复 CPU torch RNG，scene index 也不能精确续接 | L1 | 已实施：checkpoint/replay schema、trainer/config/implementation hash、Python/NumPy/CUDA/torch/replay RNG 与 scene runtime state；最终合同为 `seeded_episode_boundary`，`exact_env_state=false`；formal profile 显式冻结 | 24 项窄回归；16→20 resume finite；错误 log_std manifest hard reject；formal CUDA 1k/5064 preflight finite，warmup 后 65 updates | `[x]` 本周采用并文档化 seeded episode-boundary resume；exact env-state 续训明确不作为当前 gate，不能将此模式称为 exact strict resume |

### 8.4 决策

| 日期 | 决策 | 依据 | 范围 |
|---|---|---|---|
| 2026-08-04 | 不改 `CoCapIQN`，新增兼容 Local Encoder | 降低 solid 旧线风险 | P3 |
| 2026-08-04 | yaw hold + 连续线 init=0 | 全向有限半径感知，先减少变量 | P1--P7 |
| 2026-08-04 | boundary 保持当前 self token | 不同时改动作/Critic/observation | P1--P7 |
| 2026-08-04 | 70/20/10 + coverage floor | 兼顾总体、regime 和稀有事件 | P4--P7 |
| 2026-08-04 | P1 env 对 final command 只验证不静默投影；投影/jerk 诊断通过 info/replay metadata 记录 | 保持端到端动作合同，便于后续 Actor/adapter 审计 | P1--P7 |
| 2026-08-04 | P2 冻结 `a_max=0.8` 主候选、`1.6` 压力候选；0.4 首轮淘汰 | 0.8 在六场景中保持无碰撞且安全裕度明显高于 1.6；1.6 仅用于压力/消融 | P3--P7 |
| 2026-08-04 | P3 首版 encoder 不抽取/修改旧 IQN；radial Actor 仅在连续 namespace 注册 | 防止网络重构反向污染已 solid 的旧线；transfer 为显式开关 | P3--P7 |
| 2026-08-04 | 10k joint replay 实测约 7.16KiB/transition，hot ring 先按 200k 设计 | 300k 估计约 2.1GiB，需在 GPU/磁盘预算明确后再开 | P4--P7 |
| 2026-08-04 | P4 gate 全绿；仅允许进入 P5 central critic 实现，尚不开 100k/2M 正式训练 | 三场景 loss trend、strict resume、GPU batch32/64 均通过，但无任务成功率结论 | P5 |
| 2026-08-05 | 连续动作改为显式 `a_x,a_y` | 解决 velocity feasible layer 的不可逆映射和 raw/final action mismatch；策略只约束 `a_max`，env 约束 `v_max` | P1--P7 |
| 2026-08-05 | 所有旧 velocity 连续 run 不直接续训 | action/replay schema 改变；旧报告只保留历史诊断，新线从 4v1 scratch 重新 screening | P3--P7 |
| 2026-08-05 | 暂停所有 P6 扩长训，先修 terminal/replay/bootstrap/event 集成 | 碰撞终止 transition 被漏存、timeout 语义错误、在线事件索引未按合同派生；35 项单元回归通过但未覆盖这些路径 | P4--P7 |
| 2026-08-05 | `log_std_max=-1` 因同 seed 5k 无改善而关闭 | 同 seed 末点 CE strict 0/2、collision 1/2，不优于默认纯 coverage；继续调探索上限不能替代数据链修复 | P6 diagnostic |
| 2026-08-06 | checkpoint/replay 持久化与 screening 解耦，默认每 25k 保存 | 服务器停机时最多丢一个保存间隔（25k）；恢复仍为 seeded episode-boundary；正在运行的六条 500k 线保持启动时的 100k 存档不变，不重启 | P6 长训 |
| 2026-08-06 | 用户决定：已有 100k checkpoint 的 ax/ay pure、AW pure 两条线提前关闭；本批 500k 长训基本不再作为晋级证据，后续将大改方案 | 100k 探局已确认两条线学习失败（满油门撞墙 / 原地 no-op），继续 500k 只消耗资源；四条未到 100k 的线暂保留运行 | P6 长训 |

---

## 9. 学习型训练启动 Gate 与当前判定

以下全部 `[x]` 后，才能启动第一个学习型 smoke：

- [x] P0 manifest、seed、hash、环境、namespace 已冻结。
- [x] 旧 CR-MS 固定 seed regression 通过。
- [x] P1 action/dynamics 全部通过。
- [x] P2 至少一个 a_max 通过 scripted screen。
- [x] P3 Actor/encoder/log-prob/mask 通过。
- [~] P4 replay/采样/resume 与 P6 在线 terminal/event/bootstrap/resume 集成已通过窄回归；matched 5k 工程 gate 通过但任务 gate 失败，仍未满足长训入口。
- [x] 首训明确从 4v1 local SAC、scratch、`a_max=0.8` 开始；不直接 PSA-MASAC 2M。
- [x] screening 从 step 0 开；正式 GIF 尾迹默认关闭。
- [x] 已记录 GPU、显存预算、checkpoint/screening 周期和停止条件；P6 runner 支持 checkpoint/replay/resume。

当前判定：**P0--P5 与 BUG-04--07 在线集成/恢复 gate 通过；matched 5k 工程 gate 通过，但 pure coverage 的 `CV<0.15=0/2` 且 collision=100%，global broadcast capture=0/2 且 collision=100%，无可复现任务趋势。旧 velocity P6 结果作废，历史 ax/ay 25k/5k 不续训；六条新开的 500k scratch 只作为用户明确要求的长程数据积累，不代表任务 gate 通过，也不自动晋级 central、`a_max=1.6` 或 P7。**

### 9.1 简化任务并行诊断决策

当前新增两条 5k local/SAC 诊断线，均从 scratch、`a_max=0.8` 开始，互不续训：

- `pure_coverage`：只保留 `pure_ce`；有效继承配置在 step 0 时 speed/acceleration/angular-velocity 权重本来已为 0，cell-center speed penalty 本来已关闭，因此 `--coverage-control-ablation` 对本轮实际 reward 是 no-op，真正有效变化是只训练 pure scene。动作仍为 `acceleration_2d_body`，环境 `v_max=3.0` 不变。
- `broadcast_capture`：只保留 `capture`，开启 `global_evader_visibility=true`；这是解除有限视域 discovery 的诊断，不是最终部署配置。

本轮及后续固定评测口径：`pure_coverage` 不再只看 CE strict；必须联合报告 `CV<0.15` 达成率、末态 mean/max speed、episode length、collision rate，并把 strict 作为更严格的次级指标。`capture` 主要看真实 capture event/episode capture rate，同时报告 collision rate、episode length 和末态速度；不能用 coverage strict 替代捕获事件。

拆分是必要的，因为它把当前全任务失败分为两个可检验假设：coverage 基础几何/控制是否可学，以及在敌人直接可见时 capture credit 是否可学。两条线都必须先通过 finite、replay/action contract 和 screening 趋势，才有资格扩到 25k；不能因简化场景偶发成功而跳过 CE-first、collision 和多 seed gate。

5k 结果已回填：pure coverage 后期 collision 降为 0 但速度约 0.1、CE strict 0/2；broadcast capture 仍 collision 2/2、capture 0/2；同 seed `log_std_max=-1` 为 CE strict 0/2、collision 1/2，未改善并关闭该因子。由于 P6 会漏掉碰撞终止 transition，broadcast 结果只能判“不晋级”，不能据此断言有限视域不是主要瓶颈。下一步停留在 4v1 local，先修数据语义；不得继续调 log-std、改 reward 或扩规模。

### 9.2 修复后重新开放 P6 的最小 Gate（最高优先级）

按顺序完成，任何一步失败都停止：

1. `[x]` 固定 4-agent slot、collision/boundary transition padding、动作前 active mask 和 replay-level terminal/collision reward 计数均已验证；BUG-04 关闭。
2. `[x]` runner 已显式区分 task terminal 与 128-step time-limit truncation，local/central 使用同一 bootstrap mask；真实 128-step timeout、capture→post、collision replay smoke 与 28 项回归均通过。
3. `[x]` 已从真实动作前后状态派生 `active_target/coverage_only` 与 discovery/capture/collision/CE-success，并报告 source slots、coverage floor、event empty fallback 和实际采样占比；真实 capture/CE-success 集成样例与 32 项回归通过。
4. `[x]` 已冻结 `p6_formal_local_sac_4v1.yaml`：hidden=256、lr=1e-4、warmup=5000、grad clip=10；CUDA 1k 与 5064-step warmup-boundary preflight 均 finite，未做降配。该项只证明正式入口可训，不代表性能通过。
5. `[x]` manifest 已纳入网络/优化/SAC/warmup、effective config hash、implementation hash；checkpoint/replay 已保存 CUDA/Python/NumPy/torch/replay RNG，resume 合同明确为 `seeded_episode_boundary` 且 `exact_env_state=false`。这不是 exact env-state resume；后者不纳入本周训练 gate，避免把“可恢复”误称为“精确续训”。
6. `[x]` 报告新增 reward 分解、terminated/truncated/terminal-drop 数、entropy/log-prob/log-std、TD error、actor/critic grad norm、alpha/Q、事件池与采样来源统计；16-step 实际 smoke 与 `33 passed` 已验证字段写入且 finite。
7. `[x]` formal 1k/5064 preflight、诊断字段 smoke、窄回归与 matched 5k 双线已完成；两条均 finite/无 terminal drop，但 pure coverage 的 CV<0.15=0/2、collision=100%，broadcast capture=0/2、collision=100%，未出现可复现趋势，不重开 full local 25k。

当前工作目标：让六条已启动的 500k scratch 线完整留下可复盘的 milestone/final artifacts；是否恢复 25k、扩到 100k--200k、切 central、换 a_max 或做 IQN teacher，全部必须等真实 report/replay/checkpoint 和第 0.5 节深度复盘后决定。按历史吞吐，单条 500k 预计约 30--45 小时；本周内不保证所有线都完成，不能因 ETA 或资源压力跳过合同和行为复盘。

---

## 10. 接下来 2--3 步（交接给下一位维护模型）

以下顺序是当前唯一推荐路径；每一步都要把命令、返回码、seed/step、config hash 和产物路径回填本文。

1. `[x]` **补齐报告诊断字段（最小范围）**：只改 P6 runner/SAC report，不改 reward、observation、termination 或旧 IQN；已加入 reward 分解、terminated/truncated/terminal-drop、entropy/log-prob/log-std、TD error、actor/critic grad norm、alpha/Q、event-pool/source-slot 汇总，并以 16-step 实际 smoke + `33 passed` 复验。
2. `[x]` **正式 matched 5k 双线**：两条均已完成；`--total-steps 10064` 实际 5065 updates。工程/数值 gate 通过，但 coverage 综合指标与 capture 事件均未通过，碰撞率 100%，不晋级。
3. `[~]` **六条 500k scratch 长训已启动**：ax/ay 与 `(a,w)` bridge 各有 mix 非 global、pure coverage、global capture 三线；初始进程和日志已确认，等待 100k/最终 report 后统一复盘 collision、CV<0.15/速度和 capture 事件。不得把运行中状态写成通过。
4. `[ ]` **IQN teacher replay 转换（独立 tag）**：先实现/审计 `index -> (a,w)`、slot/termination/obs schema 转换；AW 可做 5--20% teacher warm-start，ax/ay 仅允许新环境 teacher rollout 或 representation pretrain，禁止旧离散动作硬映射。
5. `[ ]` **碰撞/控制信号单变量诊断**：待长训有阶段结果后，再决定是否开新的 local 25k；central、`a_max=1.6` 与 P7 仍冻结。
6. `[x]` **checkpoint/replay 持久化间隔改为默认每 25k（2026-08-06）**：runner 新增 `--save-interval`（默认 25000），保存与 screening 解耦；正在运行的六条 500k 线仍按启动时的 100k 存档（不重启、不换 tag）。证据：`tools/run_continuous_p6_screening.py`、`tests/test_p6_save_interval_contract.py`（4 passed）。
7. `[x]` **前两条 100k 里程碑深度复盘（2026-08-06 15:xx）**：ax/ay pure 与 AW pure 的 step100000 均已出现，checkpoint/replay 可加载、rollout 无 NaN；但学习 gate 均失败——ax/ay 仍满油门撞墙（rollout 14--16 步碰撞、末速 3.0、CV 0/2），AW 收敛为原地制动 no-op（mean a≈-0.32、全程速度 0、CV 0/2、无碰撞）。证据见 8.2；其余四条线未到 100k，继续等待。

### 10.1 结果依赖型 TODO（禁止提前执行）

| 状态标签 | 触发条件 | 必做检查/动作 | 不得做的事 |
|---|---|---|---|
| `[WAIT-RUNNING]` | worker 存活且无 milestone report | 只读检查 PID、stdout、GPU、磁盘、最新 checkpoint；保留原 tag/seed | 不杀进程、不换 seed、不覆盖报告、不提前宣布趋势 |
| `[DECIDE-AFTER-100K]` | 某线出现 `step100000` report/replay/checkpoint | 先做工程层，再按 coverage/CV+速度或 capture event+碰撞做逐 episode 深查；把结论写回 8.2/10 | 不因单个 screening episode 成功直接扩规模 |
| `[DEEP-REVIEW-ON-EXIT]` | 某线正常完成/异常退出 | 汇总 milestones、update tail、动作/速度/碰撞前窗口、reward/event/sampler、manifest；给出“合同异常/partial trend/任务失败/可保留” | 不只看最终 success_rate，不把 finite 当性能通过 |
| `[KEEP-BRANCH]` | 同一 action contract 内出现稳定 partial/正向趋势且 collision 可控 | 保留该 branch，必要时延长或开 3-seed；coverage 允许 strict=0 但 CV/几何/速度显示乐观 partial，capture 必须有真实事件证据 | 不把 AW 正向结果直接当 ax/ay 晋级，不跳过 seed |
| `[DIAGNOSE-CONTROL]` | finite 但长期 collision 主导、capture=0 或 coverage 速度异常 | 从 replay/obs/actions 分析 action 饱和、转向、制动、alpha/log_std、TD/grad、碰撞时序；只选一个变量做短回归 | 不同时改 reward、网络、探索、环境动力学 |
| `[CHECK-EVENT-PIPELINE]` | 动作已接近/围住但 capture event=0 | 检查 detection→active_target→capture 判定→event_ids→replay sampler 全链路 | 不直接归咎于有限视域或网络 |
| `[IMPLEMENT-IQN-IF-APPROVED]` | 六条结果复盘后仍需要 teacher warm-start，且用户确认 | 先实现 converter/teacher rollout smoke、hash/schema/termination 审计；AW 先于 ax/ay | 不把旧 discrete index 硬映射成 ax/ay，不把旧 replay 直接塞进 SAC |

明确不做：本周不实现 exact env-state resume；当前 `seeded_episode_boundary` 只能用于可审计的边界恢复，不能用于声称逐步确定性复现。

这样后续能明确区分：旧线回归、动力学合同、SAC 数据链和 CTDE 协同问题。

---

## 11. 2026-08-06 CTDE 任务合同整改记录

### 11.1 对照审计文档的完成情况

| 审计合同项 | 状态 | 证据/产物 |
|---|---|---|
| 唯一配置源 `algorithm=masac_ctde` / `critic_mode=central` | ✅ | [p6_formal_central_masac_4v1.yaml](../configs/experiments/continuous_marl_20260804/p6_formal_central_masac_4v1.yaml)、[formal_config.py](../src/cocap_voradj/training/continuous/formal_config.py)、[run_continuous_ctde_training.py](../tools/run_continuous_ctde_training.py) |
| 三 scene 深合并与 hash 不同 | ✅ | capture/pure_ce/mixed_crms 已分别配置；5k/25k manifest 中三个 hash 不同 |
| 旧环境分布（120×120、1 obstacle、spawn/recovery） | ✅ | `env.width/height=120`、`num_obstacles=1`、map_random/cluster/recovery pool 均在 formal runner 生效 |
| CTDE（twin central critic、focal item、local actor） | ✅ | [central_sac.py](../src/cocap_voradj/training/continuous/central_sac.py)、[joint_replay.py](../src/cocap_voradj/training/continuous/joint_replay.py) |
| 动力学 `continuous_parity_v1`（world acceleration、drag、L2 0.4） | ✅ | [robot.py](../src/cocap_voradj/dynamics/robot.py) 新增 `update_state_acceleration_world`；formal 拒绝 body/velocity 别名 |
| joint replay 存储单位 + focal-item 采样单位 | ✅ | 五桶索引 64/8/8/32/16；replacement/fallback 统计；generation 失效 |
| 优化参数（128/1e-4/0.99/0.005/0.2/-2/5000/4/0.5/2M） | ✅ | formal YAML + 5k/25k report 均读取 `central_critic`/`masac`/`training` |
| `discount.gamma` 统一 IQN/MASAC | ✅ | trainer 与 CE PBRS 均优先读取 `discount.gamma` |
| 19 项新增验收测试 | ✅ | [test_ctde_task_contract_20260806.py](../tests/test_ctde_task_contract_20260806.py) 24 项；完整 pytest 126 passed |
| 5k data-chain | ✅ | [ctde_5k_report.json](../artifacts/2026-08-06_ctde_contract/ctde_5k/ctde_5k_report.json) |
| 25k learning-signal | ✅（训练完成，eval 进行中） | [ctde_25k_report.json](../artifacts/2026-08-06_ctde_contract/ctde_25k/ctde_25k_report.json) |
| 100k gate | ⏳ 未开 | 25k 内 capture=0，post-capture focal 池为空 |
| legacy IQN 测试 | ✅ | 完整 pytest 126 passed；旧 IQN/CR-MS/CE/VCT-LS 相关测试均通过 |

### 11.2 5k/25k 核心指标

**5k（`ctde_5k_report.json`）**

- replay=5000、updates=1、all_finite=true；action norm mean=0.287、speed max=1.87、collision=70。
- focal pools：pre-capture pursuing=1021、support=2015、pre-capture coverage=9328、post-capture=0、pure-recovery=7636。
- sampling：128 items，actual 配额经 fallback 后 pure-recovery=48（post-capture 配额 32 全部 fallback）；unique transitions=127。

**25k（`ctde_25k_report.json`）**

- replay=25000、updates=5001、all_finite=true；action norm mean=0.261、speed max=2.56、collision=140、truncated=2。
- focal pools：pre-capture pursuing=6329、support=13549、pre-capture coverage=42126、post-capture=0、pure-recovery=37996。
- Q≈-11～-13、TD error≈1、alpha≈0.126、actor grad≈1～2；已有非零动作和状态访问，但尚无成功捕获。

**25k 20-episode eval（`ctde_25k_eval20.json`，真实 1000/1500 horizon）**

- capture：capture_rate=0/20、collision_rate=0.45、coverage 0/20；多条 episode 撞墙/障碍提前终止，其余跑满 1000。
- pure_ce：success=0/20、collision_rate=0.55；长 episode 跑满 1500，但 CE strict/CV<0.15 均为 0/20。
- mixed_crms：capture_rate=0/20、collision_rate=0.45；1000 步 pre-capture 大多跑满，无捕获后 coverage。
- 结论：25k 学习信号不足以产生任务成功；当前主要风险是碰撞和无法形成捕获/覆盖趋势。

### 11.3 GPU 与旧线处理（2026-08-06 21:xx）

- 按用户要求停止 GPU0 四条旧 local 500k 线：p6_axay mix/global、p6_aw mix/global；四条均保留 step100000 checkpoint + replay，无最终 report。
- CUDA1 出现 `device=1, num_gpus=1` / utilization N/A 的不稳定现象；25k 20-episode eval 最终在 `CUDA_VISIBLE_DEVICES=0` 的 GPU0 上完成。

### 11.4 后续 TODO（等待用户确认后执行）

1. `[x]` 读取 `ctde_25k_eval20.json`：三场景 20 episodes 已汇总，capture/coverage 均 0，collision 0.45--0.55。
2. 定位 25k 无捕获问题：reward scale/Q target/state coverage/action distribution 先做单变量诊断，不同时改多个参数。
3. 100k gate：要求 capture 或 pure CE 至少一项出现稳定非零成功后再开正式 2M。
4. 消融顺序保持单变量：`a_max=0.8`、no-drag、update_every=1、grad clip 1/10、sampler、map 55。
5. 确认分支 `continuous/masac-ctde-contract-20260806` 并推送 GitHub。

---

## 12. 2026-08-06 Phase 1/2：周期存档、诊断与 teacher-transfer 进展

### 12.1 周期 checkpoint bundle（P0）

- formal runner 已支持 `checkpoint_interval_env_steps=25000`、`metrics_flush_interval_env_steps=1000`、`diagnostic_eval_interval_env_steps=25000`。
- 每次 25k 保存 `checkpoints/step_%09d/`：trainer.pt、replay.pkl、runtime_state.pkl、effective_config.yaml、manifest.json、metrics.jsonl、diagnostic_eval.json。
- 保存采用临时目录 + atomic rename；`_verify_resume_steps` 拒绝 trainer/replay transition step 不一致的 resume。
- 新增测试 `tests/test_ctde_periodic_checkpoint_contract.py`：bundle 完整文件、失败不破坏旧 bundle、resume step mismatch、400 cap。

### 12.2 现有 25k 离线诊断

报告：[ctde_25k_offline_diagnosis.md](../artifacts/2026-08-06_ctde_contract/ctde_25k_offline_diagnosis.md)

- collision 140 transitions，reward mean=-40.4，非 collision=-0.69；碰撞有强负奖励长尾。
- scene 级 Q/TD：pure_ce Q≈-29、TD≈2.28；capture Q≈-11、TD≈0.71；mixed Q≈-11.8、TD≈0.75。pure CE critic 明显更差。
- capture min-distance（transition 级）：capture median 25.97、mixed 26.02；replay 无 episode id，无法做 episode 级 initial/final/AUC，需下一轮补。
- discovery event 37 条；discovery 前后 reward 差异小（-0.54 vs -0.84）。
- Actor 近零动作率≈0，radial saturation≈0.04%；动作分布不是主要失败原因。

### 12.3 Legacy IQN compatibility / encoder transfer

报告：[legacy_iqn_compatibility_report.md](../artifacts/2026-08-06_ctde_contract/legacy_iqn_compatibility_report.md)

- 使用 `stage1_4v1_step_2000000.pt`，SHA256 `2f39ea...`。
- `encoders.*`/`type_embedding.*`/`transformer.*` 全部 exact-shape 可迁移，loaded ratio=1.0；quantile/action/gate head 未加载。
- 测试 `tests/test_legacy_iqn_encoder_transfer.py` 覆盖 exact tensor、shape mismatch、policy/critic 隔离。
- formal runner 已支持 `initialization.actor_encoder.mode=legacy_iqn` + strict load + freeze_env_steps。

### 12.4 Snapshot curriculum

- 工具：[generate_legacy_iqn_curriculum_snapshots.py](../tools/generate_legacy_iqn_curriculum_snapshots.py)、[validate_curriculum_snapshots.py](../tools/validate_curriculum_snapshots.py)。
- 小规模数据集：`artifacts/2026-08-06_ctde_contract/curriculum_snapshots/`，514 snapshots（capture pre 109、post 191、pure 214）。
- `geometry_reset` restore 100/100 通过；`full_state` 测试通过；PBRS 首步无陈旧 baseline。
- runner 支持 `--scenes` + `--snapshot-dataset` 的 snapshot curriculum 模式，5 步 smoke 已通过。

### 12.5 Action translation gate

报告：[action_translation_gate.json](../artifacts/2026-08-06_ctde_contract/action_translation_gate.json)

- 10k 随机状态：rejection=0，radial saturation=98.7%，terminal velocity error mean=2.44、p95=5.14。
- **Gate 未通过**：按审计合同停止 action translation 与 BC/prefill；保留 encoder transfer + snapshot curriculum 路线。

### 12.6 Warmup 5k 数据链

- `uniform_disk` 5k pure-only 已完成，tag `ctde_warmup_uniform_disk_pure5k`；report 在 `artifacts/2026-08-06_ctde_contract/ctde_warmup_uniform_disk_pure5k/ctde_warmup_uniform_disk_pure5k_report.json`。
- 5k：collision=10/5000，action norm mean=0.266、p95=0.390、near-zero=0.3%；pure-recovery pool=20000；Q≈-0.1、TD mean≈1.03；critic pre-clip grad 50.9 → post 0.5（clip ratio≈0.0098）。
- 对比点：uniform disk 5k 的 collision 低于原随机初始化 25k 早期比例，但仍无 coverage 成功；需要与 actor-prior 5k 同 seed 对比后再下结论。
- `actor_prior` 5k pure-only 已完成，tag `ctde_warmup_actor_prior_pure5k`：collision=66/5000、speed mean=0.844、action norm mean=0.286、TD mean=1.83、critic pre-clip 88.4→0.5。
- **Warmup 对比结论**：uniform disk 比 actor prior 碰撞低约 6.6 倍、速度低约 2.5 倍，但两者 5k 均无 coverage 成功；先保留 uniform disk 作为低风险 warmup 候选，仍不满足任务 gate。

---

## 13. Phase 3 单任务 25k 实验矩阵（进行中）

统一参数：`a_max=0.4`、parity drag、update_every=4、grad_clip=0.5、map 120、400-step diagnostic cap、每 25k bundle。

| 实验 | scene | initialization | snapshot | 状态 |
|---|---|---|---|---|
| pure random | pure_ce | random | no | running `ctde_pure_random_25k` |
| pure encoder | pure_ce | legacy_iqn | no | pending |
| pure snapshot | pure_ce | random | yes | pending |
| pure encoder+snapshot | pure_ce | legacy_iqn | yes | pending |
| capture random | capture | random | no | pending |
| capture encoder | capture | legacy_iqn | no | pending |
| capture snapshot | capture | random | yes | pending |
| capture encoder+snapshot | capture | legacy_iqn | yes | pending |

启动命令模板：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. python3 tools/run_continuous_ctde_training.py \
  --tag <tag> --scenes pure_ce --total-steps 25000 \
  --screen-episodes 0 --diagnostic-eval-episodes 4 --device cuda:0
```

snapshot 实验追加 `--snapshot-dataset artifacts/2026-08-06_ctde_contract/curriculum_snapshots/snapshots.jsonl`；
encoder 实验通过 config 中 `initialization.actor_encoder.mode=legacy_iqn` 或独立 config override 启用。
