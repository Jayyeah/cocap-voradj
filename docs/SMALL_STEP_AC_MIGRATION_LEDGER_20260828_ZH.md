# 小步 Actor-Critic 迁移实验台账（2026-08-28）

## 1. 当前结论与状态

本轮不是继续调 MASAC，而是用四条最小差异线拆开两个问题：Actor-Critic 本身是否可行，以及连续 `(a,w)` 是否是主要退化源。代码、12 份正式配置、双 GPU 串行 supervisor、双评估、完整 resume 和三种子汇总已经实现；正式 300k×12 的性能结论在训练完成前保持 **PENDING**，绝不把 smoke 的偶发结果写成算法优劣。

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
- 每25k保存轻量 model+optimizer milestone、原子覆盖单个完整 `resume_latest.pt`，并各做 deterministic20 + stochastic20；最终不按单 seed 或单次 capture 下结论。MAPPO 两线与每条 off-policy 线 seed1 保留最终 full resume；IQN-VXY9/TD3-AW seed2/3 仅在300k milestone及双 formal eval 成功后删除 full replay，异常/未完成时绝不删除。

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
| MAPPO-9 | PENDING | PENDING | PENDING | 不提前判断 Actor-Critic 可行性 |
| MAPPO-AW | PENDING | PENDING | PENDING | 不提前判断 continuous action 影响 |
| IQN-VXY9 | PENDING | PENDING | PENDING | 不提前判断 `(a,w)` 归纳偏置 |
| TD3-AW | PENDING | PENDING | PENDING | 不提前比较 PPO vs TD3 |

最终数据源：`artifacts/2026-08-28_small_step_ac/<line>_seed{1,2,3}/evaluations/step_000300000.json`；无挑选汇总：`artifacts/2026-08-28_small_step_ac/summaries/<line>_three_seed.json`。

最终数据源：`artifacts/2026-08-28_small_step_ac/<line>_seed{1,2,3}/evaluations/step_000300000.json`；无挑选汇总：`artifacts/2026-08-28_small_step_ac/summaries/<line>_three_seed.json`。最终模型+optimizer milestone 三种子全部保留；大 replay 的完整 resume 至少保留每条 off-policy 线 seed1。

GPU0：`MAPPO-9 seed1→2→3`，自动生成三种子汇总，再 `IQN-VXY9 seed1→2→3`。

GPU1：`MAPPO-AW seed1→2→3`，自动生成三种子汇总，再 `TD3-AW seed1→2→3`。

每个 supervisor：单实例 `flock`、独立 PID/status/log、非零退出最多自动恢复3次、从最新原子滚动完整 `resume_latest.pt` 恢复、完成条目不重跑。正式启动后的实时 PID/step/ETA 以：

- `artifacts/2026-08-28_small_step_ac/supervisor/gpu0_status.json`
- `artifacts/2026-08-28_small_step_ac/supervisor/gpu1_status.json`

为准。

验收门槛不是“有一次 capture”，而是三种子 deterministic/stochastic 的 normal、stationary、collision/type、2+/3+、3+ hold、angular/ring quality、return 与学习曲线共同判断。只有到 300k×3 完整结束才形成每条线的正式结论。

## 7. Per-seed 状态、ETA、Gate 与异常

| GPU | 当前/后续条目 | 状态 | step | ETA | 下一自动任务 |
|---:|---|---|---:|---|---|
| 0 | MAPPO-9 seed1 | QUEUED（首个 pre-launch commit 后启动） | 0/300k | 首个1k吞吐后估算 | MAPPO-9 seed2 |
| 0 | MAPPO-9 seed2/seed3 | QUEUED | 0/300k | 依赖前序 | IQN-VXY9 seed1 |
| 0 | IQN-VXY9 seed1/2/3 | QUEUED | 0/300k | 依赖前序 | lane complete |
| 1 | MAPPO-AW seed1 | QUEUED（首个 pre-launch commit 后启动） | 0/300k | 首个1k吞吐后估算 | MAPPO-AW seed2 |
| 1 | MAPPO-AW seed2/seed3 | QUEUED | 0/300k | 依赖前序 | TD3-AW seed1 |
| 1 | TD3-AW seed1/2/3 | QUEUED | 0/300k | 依赖前序 | lane complete |

Gate：25k仅 health；50k检查数值、approach、2+；100k看趋势；150/200/250/300k看稳定 capture 增长。当前 formal Gate 全部 `PENDING`，smoke Gate 为 `DONE/PASS`。除 NaN/Inf、明确 simulator/config/action mapping 错误或数值发散外，不因早期0 capture重启。

已在 smoke 中发现并修复：CUDA resume 时 CPU RNG ByteTensor 被 `map_location` 搬到GPU；IQN eval 错继承 AW adapter；APF controller 曾每步重建；PPO final partial rollout 未消费。所有修复后复测通过。存储审计将训练中重复 full replay 改为单个原子滚动 `resume_latest.pt`，25k milestone 不含 replay；按用户明确授权，IQN-VXY9/TD3-AW seed2/3 仅在自然完成、最终 milestone 与双 formal eval 均落盘后删除 final full replay，seed1完整保留。当前异常状态：`NONE`。


## 8. 验证记录

- compile：新增/修改 Python 全部通过；12 YAML 全部 resolve，Golden fingerprint 相同；
- regression：`49 passed`（动作、Pure-Capture、K10/corrected min-active、collision、IQN corrected eval、checkpoint）；
- CUDA：四条线均完成1000步、实际 optimizer update、500→1000完整 resume、deterministic+stochastic screen；统一为最终8-slot local actor合同后，四线另完成128步双评估 smoke；IQN seed2终态路径实测仅删除 full replay，milestone/eval/status 完整保留；
- action bounds/mask/determinism/action-free V/VXY acceleration/speed/checkpoint roundtrip 均有独立测试；
- 当前未发现未解释 parity 差异：`UNEXPLAINED=0`。
