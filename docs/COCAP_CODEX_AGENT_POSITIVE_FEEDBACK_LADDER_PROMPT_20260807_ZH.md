# Codex Agent 持续执行提示词：CoCap 正反馈阶梯

你正在仓库：

`Jayyeah/cocap-voradj`

工作分支：

`ladder/implementation-20260807`（并行参考 `continuous/masac-ctde-contract-20260806`）

基准分支：

`main`

你的任务是持续执行“连续 MARL 正反馈阶梯”，不是继续在当前最终版上盲目堆训练步数。

开始前必须完整阅读：

`docs/COCAP_CONTINUOUS_MARL_POSITIVE_FEEDBACK_LADDER_20260807_ZH.md`

并创建或更新唯一专用实验台账：

`docs/CONTINUOUS_MARL_POSITIVE_FEEDBACK_TRACKER_20260807_ZH.md`

---

## 一、目标模式

持续推进目标：

```text
旧 IQN 复现
→ 连续 (a,ω) bridge 等价
→ MASAC + 连续 (a,ω) 极简单任务
→ pure coverage
→ capture 难度阶梯
→ episode-level 双任务
→ mixed capture→coverage
→ body-frame [ax,ay]
→ world-frame [ax,ay]
```

最终期望：

1. 优先完成 world-frame `[a_x,a_y]` 的完整 CTDE MASAC；
2. 若暂时无法完成，至少获得 body-frame `[a_x,a_y]` 或 continuous `(a,\omega)` 的成功版本；
3. 最低要求是获得可重复的乐观信号，并准确定位最终版本的阻塞层。

不得把“进程存活”“loss finite”或单次幸运 rollout 当作阶段成功。

## 0. 双线证据修订（2026-08-07）

本提示词以 `docs/COCAP_CONTINUOUS_MARL_POSITIVE_FEEDBACK_LADDER_20260807_ZH.md` 的 0.1 节为准。关键修订：

- Stage6/7 不再严格串行：Stage5 `(a,ω)` anchor 达到 PASS/OPTIMISTIC_PARTIAL 后，Stage6A 与 Stage7A1 可并行。
- Stage7A1 首轮目标是复现 `ctde_pure_random_25k_v2` 的 world-axay pure positive signal（actor-prior warmup），不是新 yaw 设计。
- Teacher-assisted 改为失败触发式；snapshot 优先用于 post-capture/mixed；encoder 不再默认。
- Action-translation BC：`FAILED GATE / NOT ACTIVE TODO`。
- uniform-disk warmup 降为 Stage7A1b 二级诊断。
- yaw=0/world-axis 不再视为 world action 不可学习的必要解释，改为样本效率/泛化/capture/mixed 难度的潜在因素。

---

## 二、执行纪律

### 1. 一次只改一个核心变量

禁止同一实验同时修改：

- 算法；
- 动作空间；
- dynamics；
- reward；
- observation；
- map；
- obstacle；
- visibility；
- replay quota；
- update ratio；
- grad clip。

每次配置改动必须标注：

- `[IQN-ALIGN]`
- `[ALGO-NECESSARY]`
- `[DIAGNOSTIC-ADAPT]`

其中：

- `[IQN-ALIGN]` 必须与 `main` 成功配置精确对齐；
- `[ALGO-NECESSARY]` 必须写必要性论证；
- `[DIAGNOSTIC-ADAPT]` 必须写适用阶段与恢复条件。

### 2. 不破坏旧 IQN

- 不修改旧 IQN 默认执行路径；
- 不覆盖旧 checkpoint；
- 不删除旧产物；
- 新功能必须使用新 config、namespace 或 action mode；
- 每次改动后运行 legacy regression tests。

### 3. 不擅自改冻结参数

未经用户确认，不得修改：

- map 120×120；
- perception 20、2π；
-原 reward结构/主要权重；
- `a_max=0.4`；
- `ω_max=π/6`；
- `v_max=3.0`；
- drag=`0.4/3`；
- dt=0.05；
- substeps=10；
- batch=128；
- gamma=0.99；
- update every 4；
- grad clip=0.5；
- local Actor / central twin critic；
- focal replay合同；
- speed reward=0；
- diagnostic eval cap=400。

---

## 三、每 25k 保存是强制要求

所有超过 25k transition 的训练必须在：

```text
25k, 50k, 75k, 100k, ...
```

分别保存完整 bundle。

每个 bundle 必须包含：

- Actor；
- critic1/critic2；
- target critic1/critic2；
- alpha；
-全部 optimizers；
- replay；
- focal indexes；
- recovery/snapshot pool；
- RNG states；
- transition/update counts；
- effective config/hash；
- commit；
- action/dynamics/observation/replay hashes；
- cumulative metrics；
- 400-step diagnostic eval；
- manifest。

保存必须原子化。checkpoint、replay、runtime step 不一致时禁止 resume。

每个 25k milestone 立即回填专用台账。

---

## 四、训练监控频率

训练启动后使用 `tmux`、`screen` 或 `nohup`。

轮询频率：

```text
约每 10 分钟检查一次
```

不要高频 busy-loop。

每次检查只做：

- PID/进程状态；
- 当前 step；
- 最新 checkpoint；
- NaN/Inf/OOM；
- GPU/磁盘；
- stdout tail。

若训练提前完成或报错，立即处理，不必等待下一个 10 分钟点。

不得因为 10 分钟内指标没有变化就重启或改参。

---

## 五、阶段输出规则

每个阶段开始时，在对话中明确输出：

```text
【阶段 X 开始】
目标：
本阶段唯一核心改动：
IQN 对齐项：
必要适配及理由：
临时简化及恢复条件：
预计运行：
当前台账条目：
```

每个阶段结束时必须单独、特意输出：

```text
【阶段 X 完成说明】

状态：PASS / OPTIMISTIC_PARTIAL / FAIL / CONTRACT_ERROR
本阶段唯一核心改动：
与 IQN 对齐项：
必要适配项及理由：
运行 seed / steps：
25k checkpoints：
关键指标：
对照结果：
失败或风险：
是否允许进入下一阶段：
下一步唯一动作：
台账更新位置：
```

不得只说“完成”。

每个子阶段也必须单独报告，例如 Stage 4A、4B、4C 不得合并。

---

## 六、专用实验台账

创建：

`docs/CONTINUOUS_MARL_POSITIVE_FEEDBACK_TRACKER_20260807_ZH.md`

文件顶部必须写：

- 当前 active stage；
-当前唯一 formal config；
-当前 action contract；
-当前 observation contract；
-当前 dynamics contract；
-当前 reward contract；
-最近 milestone；
-当前结论；
-下一步唯一动作。

每条 run 记录：

```text
stage
run_id
status
date
branch
commit
git status
config path/hash
seed
initialization
teacher/snapshot hash
action contract
observation contract
dynamics contract
reward contract
exact command
PID
log path
step
checkpoint paths
key metrics
comparison
decision
next action
```

更新时机：

- stage 开始；
-代码修改完成；
-训练启动；
-每25k；
-异常；
-stage 完成。

旧实验台账保留，但将与当前路线冲突的 local/body/world/旧sampler条目标记为 `HISTORICAL / SUPERSEDED`。

---

## 七、统一 MASAC 参数

除非阶段文档明确允许，不得改变：

```yaml
masac:
  gamma: 0.99
  tau: 0.005
  actor_lr: 0.0001
  critic_lr: 0.0001
  alpha_lr: 0.0001
  alpha_init: 0.2
  auto_entropy: true
  target_entropy: -2.0

training:
  batch_size: 128
  warmup_joint_transitions: 5000
  update_every_env_steps: 4
  gradient_steps: 1
  grad_clip_norm: 0.5
  replay_capacity_joint: 250000
  checkpoint_interval_env_steps: 25000
  metrics_flush_interval_env_steps: 1000
  diagnostic_eval_interval_env_steps: 25000
```

网络：

```yaml
actor:
  hidden_dim: 256
  num_heads: 8
  num_layers: 4
  dropout: 0.0

central_critic:
  twin: true
  hidden_dim: 256
  num_heads: 8
  num_layers: 4
  max_agents: 12
  dropout: 0.0
```

---

## 八、Stage 0：旧 IQN 基线

执行：

1. 在 `main` 或只读 worktree 运行旧成功 checkpoint；
2. 使用原 config、动作、动力学、reward、spawn、episode和seed；
3. 输出20-episode评估；
4. 保存可视化和hash；
5. 不修改旧代码。

Gate：

- checkpoint加载；
-主要成功行为可复现；
-指标与历史同数量级。

失败则停止所有新训练，先修回归。

完成后输出：

`STAGE_0_IQN_BASELINE_COMPLETION.md`

---

## 九、Stage 1：连续 `(a,ω)` bridge

旧 IQN仍输出离散index，精确映射为：

```text
a ∈ {-0.4,0,0.4}
ω ∈ {-π/6,0,π/6}
```

通过新的连续 `(a,\omega)` API执行。

必须保持与原 IQN 对齐：

- observation；
-robot frame；
-yaw；
-drag；
-dt/substeps；
-reward；
-spawn；
-seed；
-episode；
-target behavior。

执行 fixed-seed trajectory parity：

- action；
-position；
-velocity；
-theta；
-reward terms；
-events；
-success。

此阶段不训练 MASAC。

bridge 不通过不得进入 Stage 2。

---

## 十、Stage 2：MASAC + `(a,ω)` 极简单任务

场景：

```text
1 pursuer
1 stationary target
global visibility
0 obstacle
无 teammate
无多角色
```

动作：

```yaml
mode: acceleration_angular_velocity_body
bound_type: independent_box
low: [-0.4, -0.5235987756]
high: [0.4, 0.5235987756]
```

必须使用两个独立 squashed Gaussian 输出。不能使用radial disk，因为a和ω单位不同。

`[IQN-ALIGN]`：

- map=120；
-dt/substeps；
-drag；
-vmax；
-a/ω bounds；
-yaw；
-robot-frame observation；
-boundary。

`[DIAGNOSTIC-ADAPT]`：

- stationary；
-global visible；
-no obstacle；
-N=1；
-episode可400～600；
-简化距离progress reward。

必要性：只验证连续Actor、MASAC、replay、critic target和动作控制。

运行：

- 3 seeds；
-25k checkpoint；
-需要时继续50k；
-每25k diagnostic eval。

Gate：

- 3 seeds至少2个明显优于random/no-op；
-推荐20 episodes success≥80%；
-distance/AUC/episode length改善；
-有明确转向和推进。

若三seed至50k均无趋势：

1. 做实现审计；
2. 不推进复杂场景；
3. 同合同跑 MATD3；
4. MASAC和MATD3都失败→实现问题；
5. MATD3成功、MASAC失败→才分析算法。

---

## 十一、Stage 3：Pure Coverage

### 3A 无障碍

```text
4 pursuers
0 evader
0 obstacle
inner-cluster
continuous (a,ω)
CTDE MASAC
CE reward
speed reward=0
episode=1500
```

除了 obstacle，其他均 `[IQN-ALIGN]`。

Gate，3 seeds至少2个：

- CE energy末态相对初态改善；
-建议≥15%工程信号；
-优于random/no-op；
-CV或center-distance改善；
-动作朝centroid；
-collision不主导。

无strict但有稳定连续改善，标 `OPTIMISTIC_PARTIAL`，允许进入3B。

### 3B 恢复1 obstacle

唯一改动：

```text
obstacle 0 → 1
```

必须保持3A其他config和seeds。

Gate：

-保留大部分几何收益；
-collision可控；
-出现避障行为。

---

## 十二、Stage 4：Capture 难度阶梯

每个子阶段单独配置、单独训练、单独报告。

### 4A

```text
stationary evader
global visibility
0 obstacle
全部pursuers统一pursuing
```

Gate：

- capture rate推荐≥30%，或
- min-distance/ring geometry显著优于random。

### 4B

唯一改动：

```text
stationary → 原 moving evader
```

### 4C

用户确认（2026-08-07）：4C 就是动态敌人（moving evader），原本在 4B 之后；现为加速与 4B 并行开启。相对 4B 唯一改动：

```text
0 obstacle → 1 obstacle
```

即 4C 完整合同 = moving evader + 1 obstacle。历史表述“4C=stationary+obstacle 单变量”已被用户澄清替代，不再作为当前 TODO。

### 4D

唯一改动：

```text
global visibility → local VCT-LS
```

必须记录 discovery rate、discovery step和discovery后行为。

### 4E

唯一改动：

```text
恢复 support / pre-capture coverage角色
```

focal quota保持：

-64 pursuing；
-8 support；
-8 pre-capture coverage；
-32 post-capture；
-16 pure/recovery。

没有捕获时post-capture 32 fallback到pure/recovery，形成48 pure/recovery，这是正确逻辑，不得修改。

---

## 十三、Stage 5：双任务与 Mixed

### 5A Episode-level alternating

先按：

```yaml
capture: 0.5
pure_ce: 0.5
```

在不同episode交替，不立即要求同episode phase transition。

优先从Stage 4最佳Actor初始化。

默认先做：

```text
actor_only_init
critics/targets/optimizers reset
```

必要性：保留运动技能，避免旧单任务critic偏置新return分布。

Gate：

- capture和coverage均保留正向指标；
-一种任务不得完全覆盖另一种。

### 5B Mixed capture→coverage

恢复：

- pre-capture 1000；
-post-capture 500；
-total 1500；
-local sensing；
-obstacle=1；
-原reward；
-真实capture event。

保留post-capture fallback。

乐观gate：

- capture非零；
-post-capture pool增长；
-capture后CE energy下降；
-或capture/coverage两类连续指标同时正向。

---

## 十四、Stage 6：body-frame `[ax,ay]`

前置：Stage 5至少 `OPTIMISTIC_PARTIAL`（2026-08-07 修订：不再要求先 body 后 world；Stage6A 可与 Stage7A1 并行）。

动作：

```text
a_body=[ax,ay]
||a_body||<=0.4
a_world=R(theta)a_body
```

yaw不能恒0。

默认：

```text
speed>epsilon: yaw=velocity heading
speed<=epsilon: hold last valid yaw
```

必须测试：

- observation/action同为body frame；
-整体旋转等价；
-低速yaw稳定；
-只旋转一次。

按顺序重复：

1. pure coverage；
2. capture；
3. mixed。

不得直接从 `(a,\omega)` mixed跳到body ax/ay mixed。

---

## 十五、Stage 7：world-frame `[ax,ay]`

前置：Stage 5 `(a,ω)` anchor 达到 PASS/OPTIMISTIC_PARTIAL（旧表述“必须先 body 成功”已被跨线证据 SUPERSEDED）。

Stage7A1 首轮目标：复现 `ctde_pure_random_25k_v2` 的 world-axay pure positive signal（actor-prior warmup、1 obstacle、local VCT-LS、parity drag、25k），不作为新 yaw 设计实验；若无法复现再按合同 0.1 的检查清单定位。

先做坐标一致方案：

### 7A

```text
world-frame observation
world-frame action
```

允许全局旋转数据增强。

### 7B

若保留robot-frame observation，则必须：

-提供yaw方向信息，或
-使用等变表示；
-证明local→world变换可学习。

禁止只随机yaw却不向Actor提供方向信息。

依次：

1. pure coverage；
2. capture；
3. mixed。

---

## 十六、旧 IQN 教师辅助

> 2026-08-07 修订：教师辅助改为失败触发式，不再默认依次尝试。

触发条件（同时满足）：

- bridge成功；
- 当前新阶段 random-init 两个 seed 到 25k 均无连续正向信号；
- 工程链路和 reward 检查正常。

按任务优先级：

- Pure coverage：random-init → 若失败再 snapshot → 再 encoder → encoder+snapshot。
- Capture：优先 snapshot/state curriculum（discovery/approach state coverage 问题）。
- Stage5B mixed：post-capture 数据稀缺时优先 legacy capture/post-capture geometry snapshots。

> Action-translation BC：`FAILED GATE / NOT ACTIVE TODO`。`action_translation_gate.json`：radial saturation 98.68%、terminal velocity error mean 2.436、position error mean 1.307；除非 action contract/teacher formulation/映射方法实质改变，否则不要重复。

禁止：

-直接导入旧离散replay；
-旧action写入新replay；
-旧reward/next-state训练新critic；
-伪造post-capture。

snapshot reset后，必须由新环境生成新action/reward/next-state。

---

## 十七、算法切换

不因完整任务25k为0直接切MADDPG。

只有同一极简合同下才比较：

```text
MASAC
MATD3
MADDPG
```

优先MATD3：

```yaml
gamma: 0.99
tau: 0.005
actor_lr: 0.0001
critic_lr: 0.0001
policy_delay: 2
target_policy_noise: 0.1
target_noise_clip: 0.2
exploration_noise_std: 0.1
```

算法切换属于用户确认边界。先生成对照设计和必要性报告，再请求确认。

---

## 十八、失败处理

同一合同最多：

- 3 seeds；
-先观察25k；
-有可能时继续50k；
-三条均无连续指标改善则停止重复。

按阶段定位：

- Stage2失败：实现/算法链；
- Stage3失败：coverage reward/credit；
- Stage4失败：按stationary/global/moving/local/support定位；
- Stage5失败：多任务/遗忘/phase；
- `(a,ω)`成功而ax/ay失败：动作、yaw、旋转等变；
-world失败而body成功：保留body版本并继续研究world表征。

---

## 十九、每阶段产物

每个阶段目录至少包含：

```text
effective_config.yaml
manifest.json
metrics.jsonl
diagnostic_eval.json
checkpoint bundles
stdout.log
trajectory summary
STAGE_<ID>_COMPLETION.md
```

每个stage完成后：

1. 更新专用台账；
2.提交阶段报告；
3.在对话中特别输出完成说明；
4.根据gate自动进入下一阶段，或明确停止原因。

---

## 二十、必须停止并请求用户确认的情况

-改reward结构/主要权重；
-改map；
-改a/ω/vmax；
-改drag；
-改observation字段；
-给Actor全局/oracle信息；
-改focal quota；
-改grad clip/update ratio；
-正式切MATD3/MADDPG；
-跳过失败gate；
-同时改两个核心变量；
-删除失败结果；
-将临时诊断配置升为formal最终配置。

除此之外，在本文阶梯内持续推进，不要每个小步骤都重复请求确认。

最终交付应明确项目达到：

-完整版；
-强可用body ax/ay；
-可靠continuous (a,ω)；
-或可重复的乐观partial signal。

即使最终world `[ax,ay]`尚未成功，也必须保留所有成功锚点、失败证据和下一步可执行路径。
