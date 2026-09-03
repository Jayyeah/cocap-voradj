# MAPPO-AW-v2 连续动作桥接启动审计（2026-08-31）

## 1. 状态

```text
STATUS: IMPLEMENTED / PARITY_PASS / CUDA_SMOKE_AND_RESUME_PASS / FORMAL_SEED1_RUNNING / FIRST_25K_DURABLE
PARENT: MAPPO-9-v2
BRANCH: experiment/small-step-ac-migration-20260828
IMPLEMENTATION_COMMIT: b37ced92e21819608e7a7fb3dc89a30a1ab95754
CONFIG: configs/experiments/mappo_aw_v2_20260831/{common,seed1,seed2,seed3}.yaml
ARTIFACT: artifacts/2026-08-31_mappo_aw_v2/
SUPERVISOR: tools/supervise_mappo_aw_v2_20260831.py
ONLY_CHANGED_VARIABLE: categorical AW9 -> factorized squashed Gaussian continuous (a,w)
```

MAPPO-9-v2 的三 seed gate 已为 `PASS_TO_MAPPO_AW_V2`：三 seed best deterministic capture 分别为 `10% / 50% / 5%`，mean best capture `21.67%`、mean collision `78.33%`，优化全程健康。MAPPO-AW-v2 不恢复 coverage/post-capture，不改任务语义，只检查离散到连续动作这一条增量。

## 2. 严格父合同

新 common 直接继承 `mappo9_v2_20260830/common.yaml`。resolved-config 单测逐叶比较，除 algorithm/run identity 和 continuous head 的初始化/监控字段外不允许差异。冻结项包括：

- corrected Pure-Capture、4P1E1obs、local Legacy-VorAdj、无 global evader；
- exact legacy IQN decision feature：self/mean/max、target attention、summary attention、pursuing embedding、fusion；
- K3、`min_active_pursuers=2`、K10、stationary fallback；
- synchronized swept collision、同 horizon/spawn/APF/obstacle/reward；
- training-only centralized action-free `V_i(s)`、active/dead mask；
- terminal 不 bootstrap；truncation bootstrap 但不跨 reset 递归；
- rollout `256`、PPO `3 epochs / 2 minibatches / clip .2`；
- actor LR `3e-5`、critic LR `1e-4`、target-KL `.02`；
- ValueNorm beta `.99999`、variance floor `.01`；
- seed `2026083001/02/03`、eval seed base `2026082900`；
- `400k/seed`、每25k checkpoint、deterministic20 + stochastic20。

物理范围仍为：

```text
a ∈ [-0.4, +0.4]
w ∈ [-π/6, +π/6]
```

没有引入 Beta、Flow、full covariance、VXY、new reward/pooling/slot、TD3/MATD3 或 post-capture。

## 3. Continuous policy 数学与修复

策略为 factorized diagonal Gaussian：

```text
legacy IQN decision feature -> policy MLP -> mean[2], log_std[2]
z ~ Normal(mean, exp(log_std))
u = tanh(z)
action = u * [0.4, π/6]
```

训练用 `rsample()`；deterministic evaluation 严格使用 `mean -> tanh -> physical scale`。物理动作 log-prob 为：

```text
log π(action) = Σ log Normal(z)
              - Σ {log(scale) + 2[log(2) - z - softplus(-2z)]}
```

Jacobian 同时包含 tanh 与物理 affine scale。bounded physical entropy 无解析式，PPO entropy bonus 使用同一次 policy forward 上的新 `rsample()` 做 `-log π(action)` Monte-Carlo 估计；base Gaussian entropy另行记录，不再把二者混为一谈。

本轮修复了旧 continuous PPO 的重要数值问题：rollout 已保存 pre-tanh latent，但旧 update 从 float32 物理动作反做 `atanh`。当原 latent 为 `8–10` 时，tanh 动作已量化到边界，反解只能得到约 `7.25`，会制造很大的伪 log-ratio/KL。v2 update 现在直接用 rollout latent 重算 log-prob，物理 action 只用于环境与幅度监控。回归测试覆盖 latent `8/10`，在参数不变时 KL `<1e-6`、clip fraction 为0。

初始化固定为：policy MLP orthogonal gain `sqrt(2)`、mean head gain `.01`/zero bias、log_std head zero weight/zero bias，初始 latent std 为1；log_std clamp `[-5,1]`，post-tanh saturation 定义为 `|tanh(z)| >= .99`。这不是 sweep。

## 4. 可观测性与暂停 gate

每次 PPO update 记录：

- signed/absolute/max pre-tanh mean；sampled latent；log_std/std；
- per-coordinate/any-coordinate post-tanh saturation；physical action magnitude；
- squashed physical entropy、base Gaussian entropy、action log-prob；
- mean/max approx KL、clip fraction、target-KL early stop；
- actor/value loss、explained variance、ValueNorm mean/std；
- actor/value grad norm、actor update absolute/relative L2。

supervisor 不按25k/50k无 capture 停止。异常 gate 为：非 finite 立即暂停；KL `>0.1` 连续3点；clip `>0.3` 连续5点；saturation `>=.95` 连续5点；pre-tanh mean abs max `>=5` 连续5点；ValueNorm 下 value loss `>1e3` 或 value grad `>1e2` 连续3点。warning 阈值为 KL `.05`、clip `.3`、saturation `.5`。

GPU1 同时存在外部用户只读服务进程时，supervisor记录 PID/owner/cmdline，绝不向非本实验进程发信号。共享启动必须显式使用 `--allow-shared-gpu`，启动前仍要求26,000 MiB free；本 seed 已分配后改用2,000 MiB active headroom guard，避免把自身显存误判为启动阻塞。

## 5. 验证

CPU 定向合并回归：实现阶段 `119 passed`；active VRAM guard 修正并新增一条 supervisor case 后最终重跑为 `118 passed, 2 skipped`，两条 skip 仅因本 worktree 不含可选历史 P1 frozen manifest，无 failure。覆盖新 config/policy/supervisor 及既有 IQN parity、GAE、ValueNorm、target-KL、action/mask/collision/Pure-Capture/central critic/TD3 合同。仅有2条第三方 protobuf deprecation warning；修改 Python 全量 `py_compile` 与 `git diff --check` 均 PASS。

GPU1 fresh `0->2`、full-resume `2->4` smoke 均完成 optimizer update、ValueNorm/RNG/rollout restore、milestone 与 deterministic/stochastic 短评估。4-step health：

```text
update_count=2
approx_kl_max=1.0301e-4
clip_fraction=0
pre_tanh_mean_abs_max=.02971
std(a,w)=(1.0109,.9834)
post_tanh_saturation=0
value_loss=.8848
value_grad_norm=11.3428
all metrics finite
```

## 6. 正式运行

正式 lane 于 `2026-08-31 17:25:39 CST` 从 seed1 step0 启动；supervisor在显存状态语义修正后于 `17:29:42` 无损接管既有 child，训练 PID 未重启。`2026-08-31 17:55:48 CST` 持续运行快照：

```text
seed/step: 1 / 26,000 of 400,000; update_count=101
training PID: 3830951; supervisor PID: 3833154
tmux: cocap_mappo_aw_v2_seed1_gpu1 / cocap_mappo_aw_v2_supervisor_gpu1
pre-eval training throughput: 29.33 env step/s
current wall throughput including 40-episode eval: 14.47 env step/s
checkpoint: checkpoints/step_000025000.pt (86,626,012 bytes)
rolling resume: resume_latest.pt (present)
formal eval: evaluations/step_000025000.json (present)
supervisor: seed_active; attempts=0; restarts=0; all health streaks=0
GPU1: this run 6,910 MiB; total free 24,596 MiB
RAM available: 82.07 GiB; disk free: 49.93 GiB
```

25k formal evaluation：

| metric | deterministic20 | stochastic20 |
|---|---:|---:|
| capture / normal / stationary | 0 / 0 / 0 | 0 / 0 / 0 |
| collision | 0 | 30% |
| collision type | none | boundary 5；agent-agent 1 |
| visited 2+ / 3+ | 0 / 0 | 0 / 0 |
| max ring / max 3+ hold | 0 / 0 | 0 / 0 |
| mean angular gap / min separation | 0 / 0 | 0 / 0 |
| mean return per-agent / team | 2.5525 / 10.2099 | -108.5387 / -434.1550 |
| mean action norm | .1413 | .3817 |
| mean episode length | 1000 | 869.5 |

25k无capture不触发提前停止。评估后26k最新 PPO health 为：max KL `1.7895e-4`、clip `0`、any-coordinate saturation `.02148`、pre-tanh mean abs max `.27866`、`log_std(a,w)=(-.07166,-.03837)`、`std(a,w)=(.93085,.96236)`、bounded/base entropy `-.25197 / 2.72784`、action magnitude `.38661`、value loss `.00924`、explained variance `.03754`、actor/critic grad norm `1.0522 / .09279`；全部 finite，无 warning/critical。

三 seed 最终自动 gate 为：

- repeated deterministic capture 且 mean best capture `>=10%`、mean collision `<90%`、明确优于旧 MAPPO-AW：`CONTINUOUS_AC_ROUTE_ESTABLISHED`；
- 优化健康但未过上述门槛：`HEALTHY_DISCRETE_TO_CONTINUOUS_GAP`；
- 优化不健康：`UNHEALTHY_CONTINUOUS_POLICY_PPO_IMPLEMENTATION`。

## 7. 后台进展与逐 seed 性能（2026-09-01 10:02 CST）

seed1、seed2 已自然完成400k和全部16个25k formal evaluation；seed3已到325k，325k checkpoint与第13个20+20 formal evaluation均已持久化。下表的“best deterministic”严格使用supervisor排序：先最大capture，再最小collision，再取较晚step；seed3统计已完整写盘的325k及之前评估；325k未刷新best，详见下文。

| seed | 状态 | best deterministic | 同点collision | 非零det checkpoint | best stochastic |
|---|---|---|---:|---|---|
| 1 | 400k complete | 10% @400k，normal 10%，stationary 0 | 90% | 175k/225k=5%；325k/375k/400k=10% | 15% @175k，collision 50% |
| 2 | 400k complete | 5% @375k，normal 5%，stationary 0 | 80% | 100k/125k/250k/325k/375k=5% | 10% @400k，collision 40% |
| 3 | 325k running；latest eval=325k | 10% @300k，normal 10%，stationary 0 | 85% | 75k/150k/225k/250k/275k=5%；300k=10% | 10% @200k，collision 70% |

三 seed 都已重复出现非零 deterministic capture，说明 continuous actor 已经学出可复现而非单次偶然的捕获行为；但轨迹不是单调的，seed2 final 400k deterministic 回落到0%，因此必须按best-checkpoint gate而不能只看terminal checkpoint。当前best deterministic宏观值为：

```text
mean best capture = (10% + 5% + 10%) / 3 = 8.33%
mean paired collision = (90% + 80% + 85%) / 3 = 85.00%
old MAPPO-AW = capture 5.00%, collision 93.33%
improvement over old = capture +3.33pp, collision -8.33pp
MAPPO-9-v2 parent = capture 21.67%, collision 78.33%
remaining parent gap = capture -13.34pp, collision +6.67pp
```

因此截至300k完整评估，continuous AC 已明确优于旧 MAPPO-AW 且优化健康，但 mean-best capture 尚低于正式路线成立门槛10%。若seed3剩余350/375/400k中出现至少15% deterministic capture，并保持paired mean collision `<90%`，即可把三seed mean推到10%；在seed3结束前只记为provisional `HEALTHY_DISCRETE_TO_CONTINUOUS_GAP`，不提前封版。

325k最新评估没有改变best：deterministic capture/collision=`0%/100%`，stochastic=`0%/90%`；deterministic 2+/3+=`75%/20%`、3+ hold=`12`、mean episode length仅`173.8`，stochastic 2+/3+=`60%/15%`、3+ hold=`5`。策略仍能进入ring，但更早发生接触碰撞，说明checkpoint间性能方差和collision gap仍大；同期PPO health健康，不能把该回落归因于优化崩溃。

### 7.1 Best deterministic 的几何、碰撞与收益

| seed@step | capture/collision | collision types（20 episodes） | 2+ / 3+ | max ring；2+/3+ hold | angular gap / min separation | return agent/team | action norm |
|---|---:|---|---:|---|---:|---:|---:|
| 1@400k | 10% / 90% | agent-agent 12，boundary 3，evader 2，obstacle 1 | 70% / 15% | 3；13/3 | 3.195/.950 | -537.81/-2151.25 | .1771 |
| 2@375k | 5% / 80% | agent-agent 15，boundary 1，obstacle 1 | 65% / 15% | 3；39/4 | 3.425/.390 | -1068.39/-4273.56 | .1582 |
| 3@300k | 10% / 85% | agent-agent 4，boundary 1，evader 11，obstacle 2 | 70% / 20% | 4；38/22 | 3.357/.639 | -602.48/-2409.91 | .1876 |

几何指标显示三 seed best checkpoint 均能让至少两个追捕者进入capture ring（65–70%），但3+仅15–20%，且捕获仍伴随大量接触；seed3的3+ hold `22`步和max ring `4`最好，seed1/2则主要受agent-agent collision限制。负return主要由碰撞惩罚主导，不能把return与capture率混用为同一Gate。

### 7.2 Best stochastic 的补充信号

| seed@step | capture/collision | 2+ / 3+ | max 3+ hold | return agent/team | action norm |
|---|---:|---:|---:|---:|---:|
| 1@175k | 15% / 50% | 90% / 25% | 24 | -259.85/-1039.41 | .3648 |
| 2@400k | 10% / 40% | 75% / 15% | 4 | -90.43/-361.70 | .3005 |
| 3@200k | 10% / 70% | 85% / 5% | 1 | -340.04/-1360.14 | .3531 |

best stochastic mean capture为11.67%、mean collision为53.33%，说明Gaussian分布中存在明显优于deterministic mean-action的有效动作质量；但正式Gate仍只使用deterministic结果，以避免靠采样偶然性宣告路线成立。

### 7.3 全程 PPO/continuous-policy health

| seed | max KL | max clip | max saturation | max `|mean|` | max value loss / critic grad | std(a) range | std(w) range | 任何暂停阈值命中 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 complete | .0291 | .1372 | .0547 | .698 | 4.646 / 21.005 | .806–1.254 | .350–1.062 | 0 |
| 2 complete | .0382 | .1533 | .0469 | .749 | 5.518 / 24.231 | .580–1.081 | .335–1.141 | 0 |
| 3 through325k | .0235 | .1094 | .0390 | .691 | 2.142 / 12.887 | .732–1.176 | .348–1.088 | 0 |

三 seed 从未出现 KL `>0.1`、clip `>0.3`、saturation `>=.95`、pre-tanh mean `>=5`、NaN/Inf或value divergence。seed3在325k更新点为：max KL `.00528`、clip `.02962`、saturation `.00488`、pre-tanh mean abs max `.6699`、std(a,w)=`(.807,.502)`、value loss `.5285`、explained variance `.1158`、actor/critic grad `1.467/2.400`；仍为健康PPO。

### 7.4 剩余 ETA 与运行资源

```text
seed1: 400k complete, ETA 0
seed2: 400k complete, ETA 0
seed3: 325k/400k, 75k remaining
wall throughput including formal evals: 19.65 env step/s
runner ETA: 3,817 s = 63.6 min
projected three-seed/final-gate completion: 2026-09-01 11:06–11:15 CST
training PID / tmux: 4107314 / cocap_mappo_aw_v2_seed3_gpu1
supervisor PID / tmux: 3833154 / cocap_mappo_aw_v2_supervisor_gpu1
GPU1: 6,841 MiB used, 41,699 MiB free, no foreign compute process
RAM available: 116.56 GiB; disk free: 36.54 GiB
restarts=0; rolling resume present; all health streaks=0
```

剩余正式评估为350/375/400k。ETA按seed3迄今包含13轮formal eval的实际wall throughput估算，已包含评估开销；最终400k评估完成后supervisor会自动生成三seed Gate并退出。

## 8. 2026-09-03 最终 gate 复核与旧快照更正

重新读取 `artifacts/2026-08-31_mappo_aw_v2/gate_decision.json` 后，最终三 seed gate 已不是第 7 节中 seed3 尚未完成时的中间快照。最终 JSON 的决定为：

```text
decision: CONTINUOUS_AC_ROUTE_ESTABLISHED
total_step: 400000 / seed
seed1 best: 10% capture @400k, collision 90%
seed2 best:  5% capture @375k, collision 80%
seed3 best: 30% capture @350k, collision 65%
mean best deterministic: 15.00% capture, 78.33% collision
stable_nonzero_all_seeds: true
optimization_healthy: true
old MAPPO-AW: 5.00% capture, 93.33% collision
```

因此 continuous AC 路线达到本实验声明的“路线建立” gate，并且相对旧 MAPPO-AW 有改善；但 capture 仍低于 MAPPO-9-v2 parent 的 `21.67%`，不能写成已经追平 IQN 或离散 AC。第 7 节的 `8.33%/85.00%` 以及“seed3 仍剩余 75k”的内容保留为 `2026-09-01 10:02` 的历史中间状态，不再作为当前结果引用。
