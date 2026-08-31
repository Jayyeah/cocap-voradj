# 小步 Actor-Critic 迁移实验台账（2026-08-28）

## 1. 当前结论与状态

本轮不是继续调 MASAC，而是用四条最小差异线拆开两个问题：Actor-Critic 本身是否可行，以及连续 `(a,w)` 是否是主要退化源。代码、12 份正式配置、双 GPU 串行 supervisor、双评估、完整 resume 和三种子汇总已经实现。截至 2026-08-30，MAPPO-9、MAPPO-AW、IQN-VXY9、TD3-AW 四线三种子均已自然完成 300k；TD3 seed3 final capture 0%、collision 65%，所以 TD3-AW“三种子原配方失败”的结论已封版。后续 fixed-τ复评、reward-tail审计、IQN-VXY Full 与 MAPPO-9-v2 的最新状态见第18节。

GitHub 于 2026-08-28 重新 `fetch --prune` 核验。Golden 基线是远端 `origin/ablation/all-agent-oldmix-20260810` 的最新提交 `3ee4d94909a163b5f54689b6a4b1e62ac4baf286`；后续 `repro/open-ctde-20260823` 是另一任务合同的复现分支，不替换 CoCap Golden 环境，仅提供已经验证过的 PPO/GAE/MAPPO 语义参考。

历史 IQN 锚点文件：

- `/home/yjq/rl/CoCap1/cocap-voradj/runs/iqn_scratch_200k_20260808/checkpoints/step_125000.pt`
- SHA256：`5a0ad1c1400d0004334669908c85db6f7b8496fb2987f36f992040c26bd0344d`（已重新实测一致）
- 网络：local entity Transformer `hidden=256 / heads=8 / layers=4`、`voradj_single_head`、9 actions、distributional IQN。

## 2. Golden 冻结合同

四条正式线共同继承：

- corrected Pure-Capture、K3 normal capture、stationary fallback（speed `<0.2`、hold 10、至少2艇）；
- corrected `min_active_pursuers=2`、K10 `is_pursuing_release_delay_steps=10`；
- synchronized swept collision v1，每个物理子步检查；
- 4 pursuers / 1 moving APF evader / 1 obstacle，120m×120m，episode/pre-capture horizon 1000；
- local Legacy-VorAdj，range 20m，不广播全局敌人，actor 仅用本地 observation；
- moving APF enemy `v2_fixed`，capture terminal；
- `dt=0.05`、10 substeps、decision interval 0.5s、`v_max=3m/s`、线性阻力 `0.4/3`；
- `(a,w)` 端点 `a=±0.4m/s²`、`w=±π/6rad/s`；
- shared decentralized actor，Transformer 容量统一为 256/8/4；
- 训练从 scratch/new replay 开始，真实 environment transition 计步；
- 同三组 seed `2026082801/02/03`，同 formal eval seed base `2026082900`；
- 每25k保存轻量 model+optimizer milestone、原子覆盖单个完整 `resume_latest.pt`，并各做 deterministic20 + stochastic20；最终不按单 seed 或单次 capture 下结论。启动时原计划保留 MAPPO 两线与每条 off-policy 线 seed1 的 final full resume；2026-08-30 在完整性核验和用户再次授权后，已完成 run 改为只保留全部轻量 milestone、formal eval 和 metrics，滚动 full resume 统一清理，详见第17节。

## 3. Parity table

差异分类只允许 `SAME`、`ALGORITHM_REQUIRED`、`ACTION_REQUIRED`、`IMPLEMENTATION_ONLY`。本表审计结果：`UNEXPLAINED=0`。

| 合同字段 | Golden / IQN-AW@125k | MAPPO-9 | MAPPO-AW | IQN-VXY9 | TD3-AW |
|---|---|---|---|---|---|
| 环境、spawn、APF、horizon | SAME | SAME | SAME | SAME | SAME |
| Pure-Capture/K3/stationary/K10 | SAME | SAME | SAME | SAME | SAME |
| collision / boundary | SAME | SAME | SAME | SAME | SAME |
| local observation / no broadcast | SAME | SAME | SAME | SAME | SAME |
| actor Transformer 256/8/4 | SAME | SAME | SAME | SAME | SAME |
| local pursuer padding slots | 8 | SAME=8 | SAME=8 | SAME=8 | SAME=8 |
| 物理动作 | 9离散 `(a,w)` | SAME | continuous `(a,w)` `ACTION_REQUIRED` | 9离散 body `(vx,vy)` `ACTION_REQUIRED` | continuous `(a,w)` `ACTION_REQUIRED` |
| 学习算法 | distributional IQN | PPO+GAE `ALGORITHM_REQUIRED` | PPO+GAE `ALGORITHM_REQUIRED` | SAME | TD3 `ALGORITHM_REQUIRED` |
| 训练期 critic | target IQN | action-free centralized V `ALGORITHM_REQUIRED` | action-free centralized V `ALGORITHM_REQUIRED` | target IQN | centralized twin Q `ALGORITHM_REQUIRED` |
| replay / rollout | item replay | on-policy rollout `ALGORITHM_REQUIRED` | on-policy rollout `ALGORITHM_REQUIRED` | SAME item replay | uniform joint replay `ALGORITHM_REQUIRED` |
| entropy | epsilon-greedy | categorical entropy `ALGORITHM_REQUIRED` | Gaussian base entropy `ALGORITHM_REQUIRED` | epsilon-greedy | none `ALGORITHM_REQUIRED` |
| optimizer runtime/checkpoint wrapper | legacy trainer | unified runner `IMPLEMENTATION_ONLY` | unified runner `IMPLEMENTATION_ONLY` | unified runner `IMPLEMENTATION_ONLY` | unified runner `IMPLEMENTATION_ONLY` |

### VXY9 物理能力审计

VXY9 的 3×3 网格是 heading-aligned/body-frame：

```text
vx, vy ∈ {-3/√2, 0, +3/√2} m/s
```

这样对角目标速度范数恰好是 Golden `v_max=3m/s`，轴向目标速度不超过同一上限。动作是**目标速度**而不是瞬时改写状态；每个0.05s子步由速度 servo 在 `0.4m/s²` 上限内跟踪，并继续应用相同线性阻力、速度 clip、boundary 与 synchronized swept collision。因允许非零横向机体系目标，它比原 unicycle `(a,w)` 更接近 holonomic；这是本消融唯一允许的动作归纳偏置变化。

## 4. 算法实现

### MAPPO-9 / MAPPO-AW

- shared local entity Transformer actor；MAPPO-9 输出 categorical logits[9]，训练 sample、deterministic argmax；
- MAPPO-AW 使用标准 diagonal Gaussian + tanh/物理 scaling，训练 sample、deterministic mean；
- training-only centralized action-free entity-attention V；actor 不接触 global state；
- clipped PPO、GAE λ=0.95、γ=0.99、10 epochs、4 minibatches、clip 0.2、actor/critic lr 1e-4；
- time-limit truncation bootstrap，但 GAE 不跨 episode 边界；
- 记录 entropy、value loss、clip fraction、approx KL、explained variance、grad norm、throughput。

PPO/GAE 语义参考仓库中已经通过 smoke/resume 的 Open-CTDE MAPPO 线，但 observation、reward、environment 和网络均为当前 CoCap-native 实现，没有沿用 Roundup 任务。

### IQN-VXY9

- 保留 `CoCapIQN`、256/8/4、8个pursuer padding slots、1,000,000 item replay、3,000-item warmup、10k env-step等价 target update、distributional quantile-Huber loss、target IQN、epsilon schedule、item replay；
- 只把 action index 对应的物理表从 AW9 换成 rate-limited body VXY9；
- deterministic 用 mean-Q argmax，stochastic formal eval 保留原 IQN `epsilon_final=0.05` 的 epsilon-greedy 语义。

### TD3-AW

- shared deterministic local actor，tanh scaling 到 `[±0.4, ±π/6]`；
- 复用 centralized entity/action twin critics 和 uniform joint replay；
- target actor + target twin critics、target policy smoothing、delayed actor update=2、Polyak τ=0.005；
- 无 Gaussian actor、entropy objective、alpha/log-alpha。
- 训练探索噪声为固定 `0.1×action scale`，仅训练期使用；TD3是确定性策略，deterministic/stochastic formal 报告均直接使用无噪声 actor（保留双报告只为 matched bookkeeping）。

## 5. 已知性能证据（必须区分历史、smoke、formal）

### 5.1 历史 matched corrected-env 锚点

| 线 | 评估 | normal | stationary | total capture | collision | visited 3+ |
|---|---:|---:|---:|---:|---:|---:|
| IQN-AW@125k | 100 episodes | 61/100 | 24/100 | 85/100 | 10/100 | 65/100 |
| MASAC B0-K10@200k | deterministic20 | 0/20 | 0/20 | 0/20 | 19/20 | 0/20 |
| MASAC B1-K0@200k | deterministic20 | 0/20 | 0/20 | 0/20 | 20/20 | 0/20 |

B0 训练内200k仅出现3次 normal、B1仅1次；它们不能当作稳定策略。上述 IQN 85% total 中含24% stationary，因此主要 matched 目标仍是 normal 61%，并同时要求 stationary 占比下降。

### 5.2 1000-step CUDA smoke（不是性能结论）

每条线完成 `0→500 checkpoint→resume→1000`，网络仍是正式 256/8/4；smoke 中 MAPPO缩短rollout/epochs、TD3缩短warmup/batch以覆盖更新路径；当前IQN复测保留正式3k warmup/128 batch。双 formal screen 各1 episode×10 step，预期均无 capture，仅用于执行路径验证。

| 线 | GPU | updates@1000 | 末次核心学习指标 | smoke throughput* | resume |
|---|---:|---:|---|---:|---|
| MAPPO-9 | 0 | 8 | entropy 1.7761；value loss 563.7862；clip 0.0102；KL -0.00084；EV -0.00014 | 62.33 step/s | PASS |
| MAPPO-AW | 1 | 8 | entropy 3.6915；value loss 6.9631；clip 0.0805；KL 0.01855；EV -0.00107 | 53.50 step/s | PASS |
| IQN-VXY9 | 0 | 24 | IQN loss 16.8895；grad norm 3.1809；replay 3379 | 74.83 step/s | PASS |
| TD3-AW | 1 | 219 | critic loss 9.3047；Q1/Q2 -1.2975/-1.1279；gap 0.2217；replay 1000 | 43.01 step/s | PASS |

`*` resume 后吞吐以累计 step 除本进程墙时，数值偏乐观，只证明运行余量，不用于算法公平速度排名。所有学习指标 finite；TD3 末次是奇数 critic-only update，故该行 actor loss/grad=0，前一偶数 update 已实际执行 delayed actor update。

### 5.3 正式性能表（自动更新源）

| 线 | 3×300k | deterministic20×3 | stochastic20×3 | 当前结论 |
|---|---|---|---|---|
| MAPPO-9 | DONE | DONE | DONE | 三种子一致失败；无稳定 capture，末期 deterministic collision 100% |
| MAPPO-AW | DONE | DONE | DONE | 三种子一致失败；PPO KL/clip 失控并出现动作饱和/静止退化 |
| IQN-VXY9 | DONE | DONE | DONE | 明确学会 normal capture；225k 最强，300k 明显退化 |
| TD3-AW | DONE | DONE | DONE | 三种子 final capture 均为0；原配方强失败 |

最终数据源：`artifacts/2026-08-28_small_step_ac/<line>_seed{1,2,3}/evaluations/step_000300000.json`；无挑选汇总：`artifacts/2026-08-28_small_step_ac/summaries/<line>_three_seed.json`。

最终模型+optimizer milestone 三种子全部保留；完成 run 的滚动 resume/full replay 已按第17节清理，不随 Git 同步。

GPU0：`MAPPO-9 seed1→2→3`，自动生成三种子汇总，再 `IQN-VXY9 seed1→2→3`。

GPU1：`MAPPO-AW seed1→2→3`，自动生成三种子汇总，再 `TD3-AW seed1→2→3`。

每个 supervisor：单实例 `flock`、独立 PID/status/log、非零退出最多自动恢复3次、从最新原子滚动完整 `resume_latest.pt` 恢复、完成条目不重跑。实时状态源为：

- `artifacts/2026-08-28_small_step_ac/supervisor/gpu0_status.json`
- `artifacts/2026-08-28_small_step_ac/supervisor/gpu1_status.json`
- `artifacts/2026-08-28_small_step_ac/<run>/learning_metrics.jsonl`（每1k真实步）
首批进程由 `b2926ce` 启动，其 supervisor status 在首个25k checkpoint前仍显示0，当前真实步数以 learning metrics 为准；后续 runner 已补每1k写入 status 的监控增强，不中断当前训练。

验收门槛不是“有一次 capture”，而是三种子 deterministic/stochastic 的 normal、stationary、collision/type、2+/3+、3+ hold、angular/ring quality、return 与学习曲线共同判断。只有到 300k×3 完整结束才形成每条线的正式结论。

## 7. 启动时 Per-seed 状态、ETA、Gate 与异常

本节表格是 2026-08-28 15:08 CST 的启动 health 快照，保留用于复现实验启动过程，不代表当前状态；2026-08-30 的完成度、正式曲线和阶段性结论见第 9–17 节。

| GPU | 当前/后续条目 | 状态 | step | ETA | 下一自动任务 |
|---:|---|---|---:|---|---|
| 0 | MAPPO-9 seed1 | **RUNNING**（supervisor 2365773 / child 2365790 / tmux `cocap_small_ac_gpu0`） | 4k/300k | 当前18.33 step/s，约4h29m到本seed 300k（不含formal eval） | MAPPO-9 seed2 |
| 0 | MAPPO-9 seed2/seed3 | QUEUED | 0/300k | 依赖前序 | IQN-VXY9 seed1 |
| 0 | IQN-VXY9 seed1/2/3 | QUEUED | 0/300k | 依赖前序 | lane complete |
| 1 | MAPPO-AW seed1 | **RUNNING**（supervisor 2365779 / child 2365798 / tmux `cocap_small_ac_gpu1`） | 4k/300k | 当前16.02 step/s，约5h08m到本seed 300k（不含formal eval） | MAPPO-AW seed2 |
| 1 | MAPPO-AW seed2/seed3 | QUEUED | 0/300k | 依赖前序 | TD3-AW seed1 |
| 1 | TD3-AW seed1/2/3 | QUEUED | 0/300k | 依赖前序 | lane complete |

4k health snapshot（2026-08-28 15:08 CST）：MAPPO-9 update15，entropy 1.9400、value loss 55.3378、clip 0.3510、KL 0.03950、EV 0.23508；MAPPO-AW update15，entropy 2.6704、value loss 11.1041、clip 0.2898、KL 0.04898、EV 0.18214，Gaussian mean `(a,w)=(-0.6851,0.1142)`、std `(1.6241,0.6554)`。指标均 finite；这是早期 health，不是性能结论。首个正式 checkpoint 路径将在25k生成，当前为 `PENDING`。

Gate：25k仅 health；50k检查数值、approach、2+；100k看趋势；150/200/250/300k看稳定 capture 增长。当前 formal Gate 全部 `PENDING`，smoke Gate 为 `DONE/PASS`。除 NaN/Inf、明确 simulator/config/action mapping 错误或数值发散外，不因早期0 capture重启。

已在 smoke 中发现并修复：CUDA resume 时 CPU RNG ByteTensor 被 `map_location` 搬到GPU；IQN eval 错继承 AW adapter；APF controller 曾每步重建；PPO final partial rollout 未消费。所有修复后复测通过。存储审计将训练中重复 full replay 改为单个原子滚动 `resume_latest.pt`，25k milestone 不含 replay；按用户明确授权，IQN-VXY9/TD3-AW seed2/3 仅在自然完成、最终 milestone 与双 formal eval 均落盘后删除 final full replay，seed1完整保留。当前异常状态：`NONE`。


## 8. 验证记录

- compile：新增/修改 Python 全部通过；12 YAML 全部 resolve，Golden fingerprint 相同；
- regression：`49 passed`（动作、Pure-Capture、K10/corrected min-active、collision、IQN corrected eval、checkpoint）；
- CUDA：四条线均完成1000步、实际 optimizer update、500→1000完整 resume、deterministic+stochastic screen；统一为最终8-slot local actor合同后，四线另完成128步双评估 smoke；IQN seed2终态路径实测仅删除 full replay，milestone/eval/status 完整保留；
- action bounds/mask/determinism/action-free V/VXY acceleration/speed/checkpoint roundtrip 均有独立测试；
- 当前未发现未解释 parity 差异：`UNEXPLAINED=0`。

## 9. 四线阶段性完整结论（2026-08-30）

### 9.1 数据完整度与结论边界

截至本节最终回填时，四条线的 seed1/2/3 均自然训练到 300k，并且每个 seed 都有 25k、50k、……、300k 共 12 个 checkpoint 的 deterministic20 与 stochastic20 formal eval。每个 checkpoint 的跨种子均值来自 60 个 episode；单 seed 的成功率最小变化单位为 5%，三种子合并后的最小变化单位为 1.67%。TD3-AW seed3 也已完成，故该线现在是完整三种子结论。

TD3-AW seed3 已由原 supervisor 自然跑至300k并完成 final 双评估；terminal milestone保留，配置按既定 `retain_final_full_resume:false` 删除滚动 full replay。随后 GPU1 自动转入 MAPPO-9-v2 seed1；GPU0 则运行 IQN-VXY Full stage1，详见第18节。

结论摘要：

- **唯一明确、可跨种子复现地学会 capture 的新线是 IQN-VXY9。**
- **MAPPO-9 失败。** 离散 AW9 动作并没有挽救当前 PPO/GAE Actor-Critic，所以这批结果支持“退化不只是 continuous action 导致，当前 Actor-Critic 训练本身已有主要问题”。
- **MAPPO-AW 失败，而且失败机制比 MAPPO-9 更明显。** 除 capture 消失外，还出现 KL/clip 持续失控、tanh 前 Gaussian 均值漂到大幅饱和区，以及不同 seed 分裂成静止、满速或撞击策略。
- **TD3-AW 三种子最终失败。** 三个 final checkpoint 均为0 capture；seed3 final collision 65%、Q1/Q2约 -51.77/-52.44、twin gap 4.20、裁剪前 critic grad 903.4，与前两 seed 一致支持“当前配方不可用”。
- **AW 归纳偏置明显提升样本效率和稳定性，但不是学会任务的必要条件。** 历史 IQN-AW@125k 很强；IQN-VXY9 在 125k 明显更慢，却在 225k 达到三种子 deterministic normal capture 70%，证明 VXY9 也能学会，而且不依赖 stationary fallback。
- **当前 continuous Actor-Critic 没有成功，但 MAPPO-AW 与 TD3-AW 的故障形态不同。** 前者主要表现为 PPO 更新过猛与策略饱和，后者主要表现为 critic 数值尺度恶化；不能简单归结为同一个“连续动作 bug”。
- **最值得作为后续实验基础的是 IQN-VXY9 的 225k–250k 窗口，而不是 300k final。** 300k 时 capture 大幅回落，同时部分 reward/ring 代理指标仍上升，是过训练或代理目标错配的强信号。
- 不建议把四条线统一盲目延长到 500k。IQN-VXY9 值得先严格复评最佳 checkpoint，再做带学习率衰减和 early stop 的受控延长；两条 MAPPO 和 TD3 应先修训练稳定性，再重新从 scratch 运行。

### 9.2 指标分别反映什么，以及如何识别趋势或 bug

| 指标 | 主要反映的性能 | 本批应如何解读 | 可能暴露的问题 |
|---|---|---|---|
| normal capture | 移动敌人条件下真正完成围捕 | 最重要的主任务指标；需要跨种子、跨 checkpoint 稳定增长 | 单点成功可能只是 20 episode 小样本波动 |
| stationary capture | 敌人几乎静止时由 fallback 判定的捕获 | 可计入 total capture，但不能替代 normal capture | 占比高说明策略依赖敌人停滞或几何偶然 |
| total capture | normal + stationary | 用于总体完成率；必须同时拆分 normal/stationary | 只看 total 会掩盖 stationary 依赖 |
| collision / collision type | 安全性和控制质量 | capture 上升而 collision 下降才是可信进步 | agent-agent 高常指编队失败；boundary 高常指动作参数化、速度控制或避界失败 |
| visited 2+ / visited 3+ | 形成部分包围的前驱能力 | 可观察策略是否先学会接近，再学会完成 | 3+ 很高但 capture 很低说明 shaping/代理目标与终止目标错配 |
| 3+ hold steps | 环形结构的持续性 | 与 visited 3+ 一起判断是否真正保持包围 | hold 高而 capture 下降，说明“站成某种几何”未转换成正确 capture |
| largest angular gap / min separation | 环形覆盖质量与个体分散程度 | 最大角隙通常越低越好；最小间距在不过度分散前提下越高越好 | 单独优化这些量可能形成漂亮但无效的几何 |
| distance progress | 靠近敌人的能力 | 是必要前驱，但不是围捕完成 | 高 progress + 零 capture 说明只会追近，不会闭环包围 |
| episodic return | 当前 reward 下的总体代理分数 | 只能与 capture、collision、ring 指标联合看 | return 上升、capture 下降是 reward hacking/代理错配信号 |
| action norm / speed | 动作幅值、静止或饱和 | 极低 speed 可识别消极静止；满速和高 action norm 可识别饱和 | 连续策略可能在 tanh 两端饱和，表面 entropy 仍不低 |
| entropy / action std | 探索强度 | 离散 entropy 可观察策略塌缩；Gaussian base entropy 只反映 tanh 前分布 | base entropy 不能代表经过 tanh 和物理 scaling 后的有效动作多样性 |
| PPO clip fraction / approx KL | PPO 每轮策略改变幅度 | clip 长期大于约 0.4、KL 长期大于约 0.1 已很激进；KL 达 2–8 属明显 runaway | epochs 太多、actor LR 太高、无 target-KL early stop |
| explained variance | critic 对当前 return target 的拟合 | 高 EV 只证明 value 能拟合当前代理目标 | EV 高但 capture 为零表示 critic 可能只拟合了错误代理 |
| TD3 Q1/Q2 gap / critic grad | 双 critic 一致性和数值稳定性 | gap、Q 绝对值、裁剪前梯度持续升高是恶化 | reward/Q 尺度不当、critic LR 或 loss 不稳；梯度裁剪只能压输出，不能消除根因 |
| IQN loss / grad | 分布式 Q 的训练尺度 | replay 变大时 loss 上升本身不等于 bug，要和行为曲线联合判断 | loss 与行为同时恶化时才更像过训练或目标漂移 |

### 9.3 曲线阅读约定

下列 formal eval 曲线均按 25k 间隔排列：

25k / 50k / 75k / 100k / 125k / 150k / 175k / 200k / 225k / 250k / 275k / 300k。

除特别注明外，“跨种子”是三个 seed、每 seed 20 episode 的 deterministic 合并均值。训练内 50k 分箱曲线按 0–50k / 50–100k / 100–150k / 150–200k / 200–250k / 250–300k 排列。formal eval 是判断泛化表现的主数据；训练内 capture 只用于观察学习发生和消退的时间位置。

## 10. MAPPO-9：离散动作 Actor-Critic 线

### 10.1 跨种子 formal 曲线

| 指标 | 25k→300k 的跨种子 deterministic 曲线 |
|---|---|
| capture | 1.7 / 0 / 1.7 / 0 / 0 / 5.0 / 1.7 / 1.7 / 1.7 / 0 / 0 / 0 % |
| collision | 96.7 / 98.3 / 96.7 / 96.7 / 78.3 / 91.7 / 96.7 / 83.3 / 93.3 / 86.7 / 100 / 100 % |
| visited 2+ | 43.3 / 41.7 / 63.3 / 50.0 / 23.3 / 55.0 / 33.3 / 40.0 / 48.3 / 46.7 / 33.3 / 50.0 % |
| visited 3+ | 1.7 / 3.3 / 6.7 / 1.7 / 1.7 / 1.7 / 1.7 / 5.0 / 10.0 / 11.7 / 1.7 / 3.3 % |
| return | -366.8 / -221.3 / -244.5 / -235.8 / -188.5 / -222.3 / -234.6 / -151.1 / -237.8 / -329.1 / -220.4 / -216.1 |

曲线没有形成“接近→3+→capture”的稳定递进。150k 的 5% capture 是全线最高跨种子点，但下一 checkpoint 即回落；250k 的 visited 3+ 达 11.7% 时 capture 仍为 0；275k 和 300k collision 都达到 100%。这不是训练步数不足所呈现的缓慢单调改善，而是始终没有建立有效策略。

300k stochastic 跨种子均值仍只有 capture 5%、collision 91.7%。随机采样偶尔改变轨迹，却没有产生稳定围捕。

### 10.2 每个 seed 的最好点与最终点

| seed | 最好 deterministic checkpoint | 300k deterministic | 300k stochastic |
|---|---|---|---|
| seed1 | 150k：capture 10%（normal 5% + stationary 5%），collision 90% | capture 0%，collision 100% | capture 5%，collision 95% |
| seed2 | 225k：capture 5%，collision 90% | capture 0%，collision 100% | capture 5%，collision 95% |
| seed3 | 150k：capture 5%，collision 95% | capture 0%，collision 100% | capture 5%，collision 85% |

三个 seed 的“最好点”都只有 1–2 个成功 episode，且 checkpoint 不稳定；三个 final deterministic 全部 0 capture / 100% collision。失败具有跨种子一致性。

### 10.3 训练内 capture 曲线

| seed | 每 50k 训练分箱 capture |
|---|---|
| seed1 | 0 / 1.5 / 0 / 3.8 / 2.7 / 1.2 % |
| seed2 | 0 / 0 / 0 / 8.8 / 0 / 4.5 % |
| seed3 | 0 / 0 / 7.6 / 0 / 1.6 / 7.7 % |

训练过程中确实偶尔遇到 capture，但不能转化为稳定 formal 行为，且各 seed 的短暂峰值出现在不同时间。这支持“稀疏偶发成功被 PPO 更新冲掉或未被可靠固化”，不支持“只要再训一段就会自然收敛”。

### 10.4 PPO 学习指标

按 25k→100k→200k→300k：

- seed1：entropy 1.56→0.83→0.95→0.90；clip fraction 0.32→0.50→0.46→0.48；KL 0.009→0.360→0.309→0.175。
- seed2：entropy 1.56→1.05→0.88→0.64；clip fraction 0.32→0.43→0.41→0.42；KL 0.032→0.099→0.080→0.395。
- seed3：entropy 1.67→0.96→0.96→0.75；clip fraction 0.40→0.45→0.41→0.44；KL 0.015→0.201→0.061→0.207。

clip fraction 长期约 0.4–0.5，多个点 KL 达 0.17–0.40，说明当前 10 PPO epochs、actor LR 1e-4、无 target-KL early stop 的组合对该任务过于激进。seed2 final explained variance 达 0.932，但行为仍是 0 capture / 100% collision；这是“critic 很会拟合当前回报代理，并不等于策略学会任务”的直接例子。

**结论：MAPPO-9 失败是 solid 的。** 离散 AW9 动作保持后仍失败，说明 IQN→当前 Actor-Critic 的迁移本身造成了主要退化。原配置不值得直接延长训练。

## 11. MAPPO-AW：连续动作 PPO 线

### 11.1 跨种子 formal 曲线

| 指标 | 25k→300k 的跨种子 deterministic 曲线 |
|---|---|
| capture | 3.3 / 3.3 / 1.7 / 1.7 / 0 / 1.7 / 1.7 / 0 / 0 / 0 / 0 / 0 % |
| collision | 95.0 / 95.0 / 98.3 / 98.3 / 100 / 78.3 / 65.0 / 66.7 / 66.7 / 65.0 / 66.7 / 60.0 % |
| visited 2+ | 45.0 / 48.3 / 21.7 / 23.3 / 11.7 / 16.7 / 20.0 / 13.3 / 11.7 / 11.7 / 8.3 / 18.3 % |
| visited 3+ | 0 / 10.0 / 1.7 / 1.7 / 1.7 / 3.3 / 0 / 0 / 1.7 / 1.7 / 0 / 0 % |
| return | -212 / -273 / -128 / -54 / -59 / -63 / -86 / -69 / -65 / -101 / -40 / -78 |

150k 后 collision 从约 100% 降到 60%–78%，但 visited 2+/3+ 和 capture 没有同步上升；这不是更好的围捕，而是部分 seed 学会了静止或绕开交互。return 大幅变好同样没有转换成任务成功，属于明显代理错配。

### 11.2 每个 seed 的最好点、最终行为和动作状态

| seed | 最好点 | 300k deterministic 行为 |
|---|---|---|
| seed1 | 50k：capture 5%，collision 90% | capture 0%，collision 100%，speed 1.979，action norm 0.551；最后 50k 训练 collision 100% |
| seed2 | 25k：stationary capture 5%，collision 95% | capture 0%，collision 0%，visited2+/3+=0，return +2.552，speed 0，action norm 0.658 |
| seed3 | deterministic 150k：capture 5%，collision 95%；stochastic 50k：capture 15%，collision 85% | deterministic capture 0%、collision 80%；stochastic capture 0%、collision 100%；speed 2.807，action norm 0.637 |

连续二维物理动作的最大组合 norm 约为 0.659。seed2 final action norm 0.658 但 speed 为 0，说明 actor 并非输出“小动作”，而是落在强制动/强转向等饱和端点，最终形成消极静止；其 stochastic final 也仅 return -1.395、speed 0.004。seed3 则是相反的满速饱和，seed1 回到高碰撞。三个 seed 分裂成不同坏吸引子。

### 11.3 训练内 capture 曲线

| seed | 每 50k 训练分箱 capture |
|---|---|
| seed1 | 7.3 / 3.6 / 0.5 / 0.5 / 0 / 0 % |
| seed2 | 1.0 / 0 / 1.8 / 0 / 0 / 0 % |
| seed3 | 3.9 / 3.1 / 2.1 / 0 / 0 / 0 % |

三条训练曲线都在前半段出现少量成功，后半段统一归零，说明训练继续进行反而把偶发有效行为推走。

### 11.4 PPO 更新失控与 tanh 饱和

按 25k→100k→200k→300k：

- seed1：KL 0.01→1.34→3.72→6.88；clip fraction 0.31→0.63→0.62→0.78。
- seed2：KL 0.13→1.34→8.06→4.67；clip fraction 0.40→0.66→0.79→0.75。
- seed3：KL 0.48→1.35→2.81→4.94；clip fraction 0.46→0.72→0.69→0.75。

final tanh 前 Gaussian mean 分别约为：

- seed1：(+7.26, +1.78)
- seed2：(-5.22, +7.72)
- seed3：(+7.36, -4.58)

这些均远超 tanh 的近线性区，动作实际被压在物理边界附近。记录的 Gaussian base entropy 不能反映 tanh 后动作已经几乎失去有效多样性，所以“entropy 尚可”不能排除此处策略塌缩。

**结论：MAPPO-AW 失败非常 solid。** 原配置继续加步数没有意义。它也不能单独证明 continuous action 一定不可行，因为 PPO 更新机制本身已经明显不稳定；应先修 MAPPO-9 的 PPO 稳定性，再讨论连续动作增量。

## 12. IQN-VXY9：本批唯一明确成功、但后期退化的线

### 12.1 跨种子 formal 曲线

| 指标 | 25k→300k 的跨种子 deterministic 曲线 |
|---|---|
| capture | 0 / 0 / 6.7 / 6.7 / 21.7 / 36.7 / 56.7 / 60.0 / 70.0 / 45.0 / 43.3 / 8.3 % |
| collision | 100 / 100 / 93.3 / 93.3 / 78.3 / 63.3 / 43.3 / 40.0 / 30.0 / 51.7 / 53.3 / 50.0 % |
| visited 2+ | 43.3 / 41.7 / 43.3 / 56.7 / 56.7 / 63.3 / 73.3 / 68.3 / 93.3 / 95.0 / 88.3 / 91.7 % |
| visited 3+ | 1.7 / 0 / 3.3 / 6.7 / 6.7 / 5.0 / 10.0 / 15.0 / 45.0 / 58.3 / 60.0 / 81.7 % |
| 3+ hold | 1.7 / 0 / 0.7 / 2.0 / 3.0 / 3.3 / 6.7 / 7.3 / 27.3 / 30.0 / 37.7 / 75.7 |
| return | -86 / -101 / -72 / -88 / -113 / -95 / -34 / -26 / +33 / +5 / +67 / +51 |

25k–125k 是慢启动；150k–225k capture 持续上升、collision 持续下降；225k 达到统一最佳平衡：deterministic normal capture 70%、stochastic capture 68.3%、collision 30%、visited2+ 93.3%、visited3+ 45%，stationary capture 为 0。这里是可信的任务学习，而不是 stationary fallback。

225k 以后发生明确解耦：visited3+ 和 hold 继续大涨，return 也保持为正，但 capture 从 70% 降到 45%、43.3%、8.3%。策略越来越会满足局部几何/奖励代理，却越来越不能触发正确的 capture terminal；这是本批最清晰的过训练或 reward-proxy mismatch 信号。

### 12.2 每个 seed 的峰值与 300k 回落

| seed | 最好 deterministic | 最好 stochastic | 300k |
|---|---|---|---|
| seed1 | 250k：capture 85%，collision 15% | 250k：capture 90%，collision 10% | deterministic 15%，stochastic 40% |
| seed2 | 225k：capture 60%，collision 40% | 175k：capture 50% | deterministic 0%，stochastic 10% |
| seed3 | 225k：capture 75%，collision 25% | 225k：capture 90%，collision 10%，其中仅 1 次 stationary | deterministic 10%，stochastic 15% |

峰值 checkpoint 在 seed2/3 上一致落于 225k，seed1 晚一个 checkpoint 到 250k；三个 final 都明显低于各自峰值。因此“VXY9 能学会”是跨种子可复现结论，“统一用 300k final 部署”则明确不成立。

### 12.3 训练内 capture 曲线

| seed | 每 50k 训练分箱 capture |
|---|---|
| seed1 | 2.7 / 8.1 / 31.7 / 50.0 / 42.6 / 40.9 % |
| seed2 | 2.6 / 7.2 / 34.3 / 33.9 / 33.9 / 0 % |
| seed3 | 0 / 5.9 / 23.0 / 40.9 / 51.4 / 16.4 % |

三条线都经历慢启动后在 100k–250k 明显增强；seed2 最后 50k 训练 capture 归零，seed3 也从 51.4% 降至 16.4%，与 formal 后期退化一致。seed1 训练分箱仍有较多 capture，但 final formal 同样回落，说明只凭训练内平均不能选模型。

### 12.4 与历史 IQN-AW@125k 的 matched 对照

| 线与 checkpoint | normal | stationary | total capture | collision | visited3+ |
|---|---:|---:|---:|---:|---:|
| 历史 IQN-AW@125k，100 episodes | 61% | 24% | 85% | 10% | 65% |
| 本批 IQN-VXY9@125k，三种子 deterministic20 合并 | 21.7% | 0% | 21.7% | 78.3% | 6.7% |
| 本批 IQN-VXY9@225k，三种子 deterministic20 合并 | 70% | 0% | 70% | 30% | 45% |

VXY9 在 125k 远慢于 AW9，说明 AW 动作归纳偏置对样本效率和稳定性非常有帮助；但 VXY9 到 225k 达到 70% normal 且完全不依赖 stationary，证明 AW 不是完成任务的必要条件。因为历史 AW 是 100 episode 单 checkpoint，而当前曲线点是三种子各 20 episode，尚不能据此声称 VXY9 最终优于 AW；要回答“谁更强”，必须做 matched 的三种子 IQN-AW 全曲线。

### 12.5 IQN 数值曲线与 final 代理错配

按 25k→100k→200k→300k 的 IQN loss：

- seed1：2.71→9.43→27.03→50.09。
- seed2：3.72→11.72→25.83→38.33。
- seed3：3.24→14.56→22.45→36.16。

replay 最终达到 1,000,000 item。loss 随 replay 分布和 Q 尺度变大而上升，不能单独判为数值 bug；真正值得警惕的是它与 225k 后行为退化同时出现。典型例子是 seed2 final：deterministic capture 0%，visited3+ 75%，hold 41，return 82.8，明确展示高代理分数并未转化为任务成功。

300k 的 deterministic collision type 跨种子均值约为 boundary 45%、obstacle 5%、agent-agent 0%。虽然 final capture 差，VXY9 已把 agent-agent collision 降为零，这仍是有价值的控制结果；剩余主要故障转为边界处理。

### 12.6 IQN “deterministic” 评估语义缺口

当前 IQN formal eval 虽然固定 RNG、可重复，但 mean-Q 计算仍会在每次 forward 随机采样 quantile tau。相关实现位于：

- src/cocap_voradj/models/small_step_ac.py:81
- src/cocap_voradj/models/iqn.py:83

因此原报告中的 deterministic 是“无 epsilon 随机动作、固定评估 RNG”，而不是数学意义上使用固定 tau 网格的完全确定性评估。该缺口已在第18节补齐：固定 midpoint τ、冻结新 seeds 的100回合复评全部完成。

**结论：IQN-VXY9 的能力证明是 solid 的。** 独立300回合聚合后，统一225k capture为58.33%、per-seed-best为64%；历史20回合70%含选择偏差/方差，300k仍不应作为默认部署 checkpoint。

## 13. TD3-AW：三个完整种子的最终失败

### 13.1 三 seed final

| seed | 最好 deterministic checkpoint | 300k deterministic |
|---|---|---|
| seed1 | 150k：capture 5%，collision 80% | capture 0%，collision 55%，visited2+ 30%，visited3+ 0%，return -282，progress 2.78，action norm 0.450，speed 0.266 |
| seed2 | 100k：capture 10%，collision 90% | capture 0%，collision 85%，visited2+ 70%，visited3+ 10%，hold 14，return -235，progress 15.77，action norm 0.323，speed 0.777 |
| seed3 | 中期无稳定capture | capture 0%，collision 65%，visited2+ 40%，visited3+ 0%，return -569.55，mean length 579.65 |

TD3 是确定性 actor，当前 bookkeeping 中的 deterministic/stochastic formal eval 都直接使用无噪声 actor，因此两份报告相同是设计语义，不是评估 bug。三个完整 seed 均只有极少数中期成功，300k 全部回到 0 capture。

训练内每 50k capture：

- seed1：0 / 0 / 3.3 / 3.5 / 0 / 0 %。
- seed2：0.9 / 2.3 / 0 / 1.8 / 0 / 0 %。

两条曲线后 100k 都为零，不支持盲目延长。

### 13.2 seed3 完整证据

seed3 的 25k formal deterministic/stochastic 都是 capture 0%、collision 0%、visited2+/3+=0、return 2.5525、progress 6.1549、action norm 0.5173、speed 0。该点表现为早期静止/被动策略，不是成功；自然训练到300k后仍是capture 0%，并转为collision 65%。

50k→300k 的 critic 从 loss 1.12、Q约 -1.88、gap .49、raw grad 26.18 走到 loss 49.10、Q1/Q2 -51.77/-52.44、gap 4.20、raw grad 903.4；全程 finite，但任务性能与数值尺度同时恶化。terminal checkpoint、final eval 与状态完整落盘。

### 13.3 critic 恶化曲线

按 25k→100k→200k→300k：

- seed1：Q1 -0.29→-24.7→-51.1→-70.3；Q gap 0.53→3.89→5.06→4.54；裁剪前 critic grad 31→1294→1185→1157。
- seed2：Q1 -1.47→-22.1→-40.1→-48.0；Q gap 1.43→4.06→4.52→4.60；裁剪前 critic grad 446→745→837→1166。
- seed3：50k时Q约 -1.88、gap .49、raw grad 26.18；300k时Q1/Q2 -51.77/-52.44、gap 4.20、raw grad 903.4。

没有 NaN，实际 gradient clip 为 0.5，但 Q 绝对值、双 Q 分歧和裁剪前梯度均持续变坏。梯度裁剪防止了直接爆炸，却没有修复 reward/Q 尺度或 critic 学习动力学。继续增加 env step 很可能只是继续在坏尺度上训练。

### 13.4 TD3 actor 指标的可观测性 bug

原 runner 每 1k step 记录一次，而 TD3 delayed actor update=2；记录点总落在奇数 critic-only update，所以旧日志里的 actor_loss 和 actor_grad 始终为 0，即使偶数 update 上 actor 实际已经更新。这是日志采样相位 bug，不是“actor 没训练”。本轮已增加最近一次 actor update 的 loss/grad/update_count 缓存并纳入 resume；为不改变轨迹，已运行的 seed3 进程没有热加载新逻辑。

**最终结论：TD3-AW 在三个完整种子上是强失败。** 原配置不值得在 300k 后追加训练；后续若做TD3-v2，centralized joint gradient与focal-gradient必须作为显式单变量消融。

## 14. 已确认的实现健康项、分析限制与可能 bug

### 14.1 没有发现的错误

- 未发现 simulator、collision、action bounds、checkpoint roundtrip 或 resume 语义错误。
- 全部新增/相关回归测试为 49 passed。
- 各线训练指标保持 finite，没有 NaN/Inf。
- MAPPO centralized V 是 action-free training-only critic，actor 未泄漏 global state。
- TD3 deterministic/stochastic 相同符合确定性策略的设定。

### 14.2 会影响结论精度或训练质量的问题

1. **formal eval 样本量。** 每 seed 每 checkpoint 只有 20 episode；单 seed 5% 就是一个 episode。跨种子曲线适合看大趋势，但所有 post-hoc 最佳 checkpoint 都必须用预先固定的 100 episode protocol 重评，避免选择偏差。
2. **MAPPO 更新过猛。** 两条 MAPPO 都长期高 clip fraction；MAPPO-AW KL 达 2–8。当前没有 target-KL early stop，是最优先的训练稳定性问题。
3. **MAPPO-AW 动作饱和。** tanh 前 Gaussian mean 漂到绝对值 5–8，base Gaussian entropy 不能代表物理动作空间中的实际 entropy。
4. **reward proxy mismatch。** MAPPO-AW seed2、IQN-VXY9 300k、TD3 中都能看到 return 或局部几何改善而 capture 不改善；后续必须把 capture terminal、ring shaping、boundary/collision 惩罚的贡献拆开记录。
5. **IQN deterministic tau。** 固定随机种子让结果可复现，但随机 tau 意味着它并非严格 deterministic；应补 fixed-tau eval。
6. **TD3 actor 日志。** 1k logging 与 delayed update 奇偶相位重合，导致 actor 指标虚假为零；应先修可观测性。
7. **checkpoint 选择。** IQN-VXY9 225k/250k 是从同一批 20-episode 曲线事后选出的，不能直接当无偏最终数字；能力趋势可信，精确率需独立复评。

## 15. 对启动时核心问题的逐项回答

| 启动时问题 | 当前答案 | 证据强度 |
|---|---|---|
| 从 IQN 迁到 Actor-Critic 是否本身造成退化？ | **是。** MAPPO-9 保持离散 AW9 仍三种子失败，而 IQN-AW 历史锚点很强。 | solid |
| AW 是否是唯一或主要退化源？ | **不是唯一原因。** 离散 MAPPO-9 已失败；但 AW 对 IQN 的样本效率确有明显帮助。 | solid |
| 去掉 AW、改 VXY9 是否完全学不会？ | **否。** IQN-VXY9 在 225k 达 70% normal capture，三个 seed 都出现高成功窗口。 | solid |
| continuous MAPPO 与 TD3 谁更可行？ | 当前两者都不合格，不能宣称赢家；MAPPO-AW 是策略更新/饱和问题，TD3 是 critic 尺度问题。 | 三 seed均solid |
| 是否获得稳定 deterministic continuous capture？ | **没有。** MAPPO-AW 和已完成 TD3-AW final 都是 0 capture。 | strong |
| 这批“小跳”是否完成了诊断目的？ | **是。** 它成功拆分出算法迁移问题、动作归纳偏置作用和不同连续 AC 故障形态。 | solid |

### 15.1 哪些实验成功

- **实验设计和执行成功：** 四条最小差异线、公平环境合同、三种子、25k 网格双评估、resume/supervisor、指标留痕均按计划工作。
- **IQN-VXY9 能力验证成功：** 证明 body-frame、rate-limited VXY9 在相同 IQN 主体下可学会 normal capture，且最佳窗口不依赖 stationary fallback。
- **因果诊断成功：** MAPPO-9 失败证明问题不能只归咎于 continuous AW；MAPPO-AW 与 TD3 的不同数值症状给出了下一轮可操作的修复方向。
- **控制侧部分成功：** IQN-VXY9 final 的 agent-agent collision 为 0，说明 VXY9 对多艇互撞控制有正面信号，虽然 boundary 仍是主要故障。

### 15.2 哪些实验失败

- **MAPPO-9 任务性能失败，且结论 solid。**
- **MAPPO-AW 任务性能失败，且更新稳定性失败，结论非常 solid。**
- **TD3-AW 三个完整种子任务性能均失败；最终结论 solid。**
- **IQN-VXY9 的 300k final checkpoint 选择失败。** 算法/动作能力成功，但持续训练到 300k 造成明显退化。
- **“当前配方直接得到稳定 continuous Actor-Critic”这一目标失败。**

### 15.3 哪些结果已经较为 solid，可作为后续基础

1. IQN-VXY9 能学会任务：solid。
2. IQN-VXY9 的有效窗口在约 225k–250k、300k 过训练：跨种子证据强；精确 checkpoint 成功率待 100 episode 复评。
3. MAPPO-9 原配方不可用：solid。
4. MAPPO-AW 原配方不可用且有严重 KL/饱和问题：非常 solid。
5. 当前 Actor-Critic 问题不只是 continuous action：solid。
6. TD3 原配方不可用：三个完整种子上均失败，结论 solid。
7. “VXY9 最终优于 AW9”：**当前不支持。** 现有证据只支持 VXY9 可行、AW 更快；缺 matched 三种子 AW 曲线。

## 16. 后续实验与是否扩训练步数

### 16.1 值得做受控扩展：IQN-VXY9

不要从 300k final 盲目续训。优先顺序：

1. 固定 midpoint tau、冻结 eval seeds 的独立复评已完成：uniform225跨300回合capture 58.33%，per-seed-best跨300回合capture 64%；后续引用这些数字，不再引用事后20回合70%作为正式率。
2. 若严格复评保持高 capture，再从各自最佳 checkpoint 做短程 continuation：降低学习率、减慢 target/探索变化，并设置基于独立 validation 的 early stop。
3. 明确记录 normal/stationary、collision type、visited/hold、angular gap、separation、return 分项，避免只按 return 选 checkpoint。
4. 若要形成统一发布模型，先比较统一 225k 与 per-seed best；不能把 per-seed 事后最优直接当公平最终指标。

这是本批唯一值得在现有 checkpoint 基础上增加训练预算的线，但扩展必须以“修复后期退化”为目标，而非单纯增加总步数。

### 16.2 不值得原样延长：MAPPO-9

MAPPO-9-v2 已从 scratch 启动，并落实以下稳定性修复：

- PPO epochs 从 10 降到约 2–4；
- actor LR 从 1e-4 下调，例如先测 3e-5；
- 加 target-KL early stop，首轮候选范围 0.01–0.03；
- 对 return/value target 做规范化或尺度审计；
- 继续记录每 epoch KL、clip fraction、entropy、EV，并按 minibatch 检查更新幅度。

只有 v2 在离散 AW9 上先出现跨种子稳定 capture，才值得扩到更长步数或迁移到连续动作。

### 16.3 不值得原样延长：MAPPO-AW

在 MAPPO-9 v2 稳定前不要继续 MAPPO-AW。之后再做时应：

- 继承 KL-controlled PPO；
- 记录 tanh 前后动作分布、饱和比例和物理动作 entropy；
- 检查 log-prob 的 tanh Jacobian 与动作 scaling；
- 对静止、满速、边界撞击分别做行为统计；
- 先用短程多 seed health gate 排除 mean 漂到正负 5–8 的情况。

### 16.4 不值得原样延长：TD3-AW

seed3 已自然跑完并完成既定三种子证据；原配方不再追加预算。下一版先：

- 修正 actor update 指标日志；
- 拆解 reward 与 Q target 尺度；
- 尝试 Huber critic loss、较低 critic LR、target/reward normalization 或 clipping 的受控消融；
- 持续看 Q1/Q2 gap 与裁剪前 grad，而不只看裁剪后的 grad；
- 先做短程两至三种子数值稳定 gate，再决定是否跑 300k。

### 16.5 推荐的新实验顺序

1. **IQN-VXY9 最佳 checkpoint 严格复评：** fixed tau、三 seed、每点 100 episodes。
2. **matched IQN-AW 三种子曲线：** 与 VXY9 使用相同 25k 网格，回答归纳偏置的样本效率和最终上限。
3. **KL-controlled MAPPO-9 v2：** 先验证离散 Actor-Critic 能否稳定学习。
4. **MAPPO-9 v2 成功后**，再最小差异迁移到 MAPPO-VXY9 或修复后的 MAPPO-AW；一次只改动作头。
5. **TD3 critic 修复后**才重启 TD3-AW，不在当前 checkpoint 上直接续跑。

## 17. 证据位置与存储策略

本地原始曲线和汇总源位于 artifacts/2026-08-28_small_step_ac/，其中：

- summaries/mappo9_three_seed.json
- summaries/mappo_aw_three_seed.json
- summaries/iqn_vxy9_three_seed.json
- 各 run 的 evaluations/step_*.json
- 各 run 的 learning_metrics.jsonl
- 最佳 IQN 模型：seed1@250k、seed2@225k、seed3@225k
- 每个已完成 run 的 checkpoints/step_000300000.pt 与全部轻量 milestone

这些训练产物受 .gitignore 排除，不上传 GitHub；GitHub 只同步代码、配置、测试与台账。2026-08-30 已逐 run 确认 300k model+optimizer checkpoint、12 个 formal eval 文件、metrics 和完成状态，并删除 8 个仍残留的已完成 run 滚动 resume_latest.pt/full replay；此前已删除的 IQN-VXY9/TD3-AW seed2/3 resume 也复核为证据完整。本次额外释放 11,930,662,878 bytes（约 11.93 GB 十进制），artifacts 总占用由约 25 GB 降至约 14 GB。滚动 resume 只服务中断恢复，删除后不可从任意中间 environment/replay 状态无损续训，但不影响模型推理、formal 复评或本文结论。TD3-AW seed3 完成后按其既定配置移除 final full resume；当前新增的 IQN-VXY Full 与 MAPPO-9-v2 活跃 run 各自保留滚动 resume。

## 18. 2026-08-30 中断恢复：AC/CTDE audit 与 MAPPO-9-v2

本节是在上一次 Codex turn 因 usage limit 中断后，从实际 worktree、git diff、tmux、PID、checkpoint、日志和GPU状态恢复得到；没有重跑已完成的旧三种子实验，也没有回滚第1–17节结论。详细公式与证据见 `docs/AC_CTDE_GAP_AUDIT_20260830_ZH.md`；Final-AW/VXY Full 另见当日两份 IQN 文档。

### 18.1 恢复点

- branch：`experiment/small-step-ac-migration-20260828`；恢复时local/remote均为 `9ed9f61a462c213b93b304a6106fa5c5f083e97a`；
- GPU1 的 TD3-AW seed3 属于中断前既有正确训练，已保留并自然完成到300k，没有被抢占；
- GPU0 fixed-midpoint τ 的100回合复评已全部完成：uniform225为58.33% capture，per-seed-best为64%；
- 已完成的MAPPO-v2/full-resume CUDA smoke产物均保留，不重复执行。

### 18.2 MAPPO-9-v2 正式记录

```text
STATUS: THREE_SEED_COMPLETE / GATE_PASS_TO_MAPPO_AW_V2
HYPOTHESIS: exact IQN decision representation + healthy PPO recipe可分离AC算法gap与历史迁移误差
ONLY_CHANGED_VARIABLE: IQN value learning -> categorical MAPPO；centralized V/GAE/ValueNorm是算法或优化必需
CONFIG: configs/experiments/mappo9_v2_20260830/{common,seed1,seed2,seed3}.yaml
COMMIT: dda2a23f20c4b857bc958c856d76bf869b50df02
SEED: 2026083001 / 2026083002 / 2026083003
START_STEP: 0
CURRENT_STEP: seed1/seed2/seed3均400k COMPLETE（gate于2026-08-31 13:28 CST写盘）
RESULT: 三seed累计0 restart、optimization healthy；best依次为10%/90%、50%/50%、5%/95%（capture/collision），mean best capture21.67%、mean collision78.33%，all-seed nonzero
GATE: 3×400k；每25k det/stoch20；KL/clip/value健康并跨seed重复非零capture
CONCLUSION: `PASS_TO_MAPPO_AW_V2`；证明修复后的离散AC达到预声明最低桥接门槛，但seed1/3高碰撞说明策略仍弱，不能写成强成功
ETA: MAPPO-9-v2已完成；MAPPO-AW-v2尚未启动，当前无训练ETA
NEXT: 进入严格单变量的MAPPO-AW-v2；除categorical AW9→continuous (a,w)外保持合同固定
```

v2 的关键冻结值为 rollout256、3 PPO epochs、2 minibatches、actor LR `3e-5`、critic LR `1e-4`、clip `.2`、target-KL `.02`、ValueNorm beta `.99999`、categorical logits gain `.01`、400k/seed。Actor exact复用 IQN 的 self/mean/max/target attention/summary attention/pursuing embedding/fusion decision feature，critic仍是action-free centralized V。

第一次2-step smoke暴露了ValueNorm value clipping的尺度语义错误：raw old value在更新running stats后重新normalize，可能进入零梯度clip支路。已修为rollout保存normalized prediction、GAE前用旧stats反归一化、更新stats后normalize return target。全新smoke得到 value loss `.1106`、raw value grad `6.1316`、actor update L2 `.05059`、max KL `1.38e-4`，resume/milestone/deterministic+stochastic eval全部完成。

### 18.3 MAPPO-v2 自动线

`tools/supervise_mappo9_v2_20260830.py` 已完成其职责并正常退出。它没有杀死或抢占历史 TD3；seed1→seed2→seed3均自动晋级并完成：

- rolling resume、PID/tmux/GPU/VRAM/温度/RAM/disk/ETA/checkpoint；
- finite metrics；KL>.05或clip>.3 WARN；KL>.1连续3点暂停；
- 三seed最终gate JSON；
- PASS路由MAPPO-AW-v2，healthy FAIL路由discrete centralized-Q/counterfactual CTDE；在结果未知前不擅自启动下一算法。

2026-08-31 13:28 CST 状态为 `three_seed_gate_complete`：`PASS_TO_MAPPO_AW_V2`，mean best capture21.67%、mean collision78.33%、all-seed nonzero=true、optimization healthy=true。该PASS仅表示达到预声明迁移门槛；seed1/3 best collision分别90%/95%，仍需在连续动作线中严格验证。

### 18.4 TD3 centralized-Q/gradient审计与final seed3

```text
STATUS: AUDIT_PASS / THREE_SEED_COMPLETE
HYPOTHESIS: 旧TD3失败来自critic尺度/credit dynamics，而不是target公式或actor完全未更新
ONLY_CHANGED_VARIABLE: 只修logging可观测性；不改变当前算法/训练轨迹
CONFIG: configs/experiments/small_step_ac_migration_20260828/td3_aw_seed3.yaml
COMMIT: dda2a23f20c4b857bc958c856d76bf869b50df02
SEED: 2026082803
START_STEP: 0（中断恢复时约193k，保留原进程续跑）
CURRENT_STEP: 300k / 300k，COMPLETE
RESULT: final capture 0%、collision 65%、visited2+ 40%、visited3+ 0%、return -569.55；critic loss49.10、Q1/Q2 -51.77/-52.44、gap4.20、raw grad903.4
GATE: 自然跑完且final formal完整；FAIL，不原样延长，不自动启动TD3-v2
CONCLUSION: target/mask/noise/delay实现通过；actor是明确的centralized joint gradient，不是focal gradient
NEXT: 三种子结论已封版；若后续TD3-v2，joint与focal必须作为单变量消融
```

新增合同测试覆盖：terminal不bootstrap、truncation bootstrap、dead-agent mask、normalized target noise、twin-Q、delayed target update和跨agent joint gradient。历史每1k日志恰落在critic-only update，使actor loss/grad表面恒0；新代码缓存 `last_actor_loss/grad/update_count` 并保留当前update指标。已完成的旧进程加载的是修改前Python代码，故其历史日志不会倒推出这些新字段；这没有修改live trajectory。

### 18.5 Paired reward-tail 审计

统一225k与300k使用同一组独立新 seeds、fixed midpoint τ，各做 `3×20` 回合并记录尾100/200步。225k→300k 的 failed-3+ 组中：episode数 `9→36`、mean return `-231.28→+20.19`、3+ hold `11.22→34.64`、length `429.67→773.08`，但最后100步 approach `.05395→.00196`、capture shaping `.41159→.27977`，terminal始终为0。300k并非瞬时代理奖励更强，而是低质量 nonterminal dwell 更长；结论是 proxy/time-horizon mismatch。本轮只增强可观测性，不改 reward。

### 18.6 验证与当前边界

最终合并定向suite：`78 passed`（另有2条第三方 protobuf deprecation warning，无失败），覆盖：

- Legacy IQN decision feature bit-exact parity且旧checkpoint keys不变；
- GAE terminal/truncation/active mask；
- PPO ratio、target-KL、ValueNorm、actor update与nonzero critic gradient；
- TD3 target/noise/mask/delay/joint-gradient；
- VXY9 action、Final-AW strict diff、full resume与两条supervisor gate。

本轮不引入Flow/Beta/full-cov、MATD3/FACMAC、新pooling、global enemy、新reward/replay或continuous VXY Actor-Critic。下一阶段仅由正式gate结果决定。
