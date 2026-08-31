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
