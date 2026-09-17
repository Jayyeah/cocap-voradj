# TD3 Local-AW Stage-1 Tracker（2026-09-16）

## 定位与边界

本线是 **continuous-control capability audit**，不是当前 CoCap / MAPPO 方法主线的既定替换。科学问题仅为：local TD3 是否能分别从 scratch 学会 pure coverage、pure capture，以及 IQN warm-start 对 sample efficiency / stability 的实际提升。

本轮明确禁止 FullMix、phase-aware/focal replay、MADDPG/MATD3、central critic、GNN/GAT，以及删除 `is_pursuing`。Stage 2/3 均未启动。

## 恢复的真实状态

- 上游实验分支：`experiment/density-normalized-sensing-v2-20260915`
- 审计时最新 HEAD：`72c59392436967162fd99cf85b256bae263f0884`
- 本线分支：`experiment/td3-local-aw-stage1-20260916`
- 隔离 worktree：`/home/yjq/rl/CoCap1/cocap-voradj-td3-local-stage1`
- 既有 NormSense Pure-Capture MAPPO 在另一 worktree / GPU0 独立运行；本线未发送信号、未改其文件。该进程随后自然完成 300k。
- 历史 IQN teacher：`artifacts/2026-08-04_crms_vctls_ce_final/checkpoints/stage2_8v2_step_300000.pt`
- teacher SHA256：`ef58ae9bdd018633afee0d16f2242d2f157a6cc18611e41474f115aed254b87e`

优先审阅了当前 root-cause ledger、NormSense V2 合同与 artifacts、Pure Coverage/Pure Capture/FullMix 合同、当前 MAPPO trainer/evaluator、IQN 模型与 checkpoint、历史 continuous SAC/MASAC tracker、positive-feedback ladder、theoretical feasibility audit，以及 replay/action adapter/dynamics/reward/capture/coverage 实现。

## 合同

除动作 API 外，TD3 config 逐字段继承当前 NormSense V2 pure-task contract；`assert_action_only_delta` 要求差异集合严格等于 14 个显式动作字段。reward、capture success、coverage、spawn、map、episode length、sensing、collision 均不得变化。

- 动作：continuous body-frame `(a,ω)`；`a∈[-0.4,0.4]`，`ω∈[-π/6,π/6]`
- dynamics：legacy-exact `acceleration_angular_velocity_body`，`dt=.05×10=.5s`，`vmax=3`，drag=`.4/3`
- yaw：`legacy_random`，避免 continuous 默认零朝向造成 representation/dynamics drift
- Actor observation：仅 self / pursuers / evaders / obstacles / masks / types；NormSense V2 local sensing；无 global enemy/full state/future
- Critic：共享参数的 per-active-pursuer `Q_i(o_i,a_i)`；Q1/Q2 独立；不读 dummy replay global field，不读 teammate action
- Actor/Q1/Q2 各自持有独立 `LegacyVorAdjFeatureBackbone`，参数和 optimizer 均不共享
- target actor/Q hard-copy 初始化；只在 delayed policy update 时 soft update
- Bellman：`r+γ(1-terminated)min(Q1',Q2')`；time-limit truncation 继续 bootstrap
- target smoothing：normalized noise std `.2`、clip `.5`，之后按物理分量 scale 并 clip
- `policy_delay=2`；无 entropy/alpha
- exploration：actor physical action + normalized Gaussian std `.1`，按 `[.4,π/6]` scale/clip
- replay：standard uniform joint-transition ring；每个环境决策恰好存一次；role/phase/event 仅诊断，不参与采样

AW9 index 按 `(a,w) for a in (-.4,0,.4) for w in (-π/6,0,π/6)` 确定性映射。映射表保留 float64 legacy 值；两个任务 × 9 动作的 fixed-seed 单步 dynamics parity 均在 `1e-9` 内通过。

## 网络与 IQN warm-start

Actor 保留完整 legacy decision path，包括 entity encoders、type embedding、4-layer Transformer、target Q/K/V、mean/max/target summaries、summary attention/fusion、`is_pursuing` late fusion 与 final decision feature；其后新增 MLP → tanh → physical `(a,w)`。

Q1/Q2 分别为独立 legacy-compatible decision backbone，随后 `concat(h_i,a_i,w_i)` → MLP → scalar Q。没有 IQN quantile head、global state 或 `a_-i`。

TD3-W 严格映射 IQN decision-backbone state dict。正式 4-layer teacher 应映射 86 keys；显式排除 `single_head`、`cos_embedding`、coverage/encirclement/gate heads。TD3 actor head做短期 physical-AW MSE behavior cloning；Q1/Q2 必须在 warm actor load 前按与 scratch 相同 seed 构造，且 hash 断言保持不变。

当前合同教师资格抽查（非最终数据集）：

| task | matched NormSense V2 seeds | 结果 |
|---|---:|---|
| coverage | 2026091700–01 | 2/2 CE success，196 / 345 steps，无碰撞 |
| capture | 2026091700–01 | 2/2 normal capture，130 / 86 steps，无碰撞 |

因此正式 warm 数据使用当前合同重新采集的 successful rollouts；旧 B0/K10 teacher dataset 不冒充 matched 数据。

## 实验矩阵与状态

| ID | task / algorithm | init | seed | 当前状态 |
|---|---|---|---:|---|
| D1 | pure coverage / IQN AW9 | scratch | 2026091501（待跑） | 历史 teacher 有当前合同冻结评估能力，但训练合同/预算不 matched，不能用于 sample-efficiency 排名 |
| D2 | pure capture / IQN AW9 | scratch | 2026091501（待跑） | 同上；当前合同资格抽查 normal capture 2/2 |
| D3 | pure coverage / categorical MAPPO AW9 | scratch | 2026091501（待跑） | 旧 sensing 结果不冒充 NormSense V2 matched baseline |
| D4 | pure capture / categorical MAPPO AW9 | scratch | 2026091501 | horizon=3000 baseline 已完成 250k；25/50/75k normal capture 均 0，ring3 visitation 为 0/0/15%，collision 为 0/100/100%；原 run 缺 100k eval，需用共同 seeds 补评。1000-step conservative 线是显式 ablation，不计入 D4 |
| C1 | pure coverage / local TD3 AW | scratch | 2026091501 | formal RUNNING；GPU0；首个 5k update finite |
| C2 | pure capture / local TD3 AW | scratch | 2026091501 | formal RUNNING；GPU0；6k 时 critic/policy=1001/500，delay=2 精确 |
| C3 | pure coverage / local TD3 AW | IQN-warm | 2026091501 | teacher dataset 完成；统一 eval seeds 的最终 BC 验证中 |
| C4 | pure capture / local TD3 AW | IQN-warm | 2026091501 | formal RUNNING；GPU1；BC 初始 policy 在训练前 rollout 为 8/8 normal capture |

四条 TD3 均使用与现有 MAPPO baseline 相同的训练 seed `2026091501`；两个任务的 deterministic eval 均使用共同 seed base `2026191501`。scratch/warm 同 task 的 critics 随机初始化 matched。正式 TD3 milestone=`25k/50k/75k/100k`，每点 20 个 deterministic eval episodes、atomic checkpoint、replay snapshot、diagnostics 与 manifests；100k 无趋势先审计，不自动延长。

## Hyperparameters

共同 config：`configs/experiments/td3_local_aw_stage1_20260916/common.yaml`。

- hidden 256，8 heads，4 layers，dropout module 保留 `.1` 但 TD3 deterministic forward 固定 eval mode
- batch 256；replay 200k；warmup 5k；UTD 1
- gamma `.99`；tau `.005`
- actor/Q learning rate 均 `1e-4`；grad clip `.5`
- BC：64 successful train episodes + 16 successful heldout episodes / task；30 epochs；batch 512；lr `3e-4`

## IQN teacher / BC 实际结果

- Capture：80/80 successful normal-capture episodes，0 collision，21,560 rows（train 16,936 / held-out 4,624），dataset SHA256=`00d2210293b260051a8e2e48261abdf1d9dd39c4134bf40ef27f3f1a1b3f76f9`。
- Capture BC epoch30：train/held-out physical-AW MSE=`.0161539/.0178840`，action-index agreement=`.854511/.827638`；共同 seeds `2026191501–08` 为 8/8 normal capture、0 stationary、0 collision。
- Coverage：80/80 successful CE episodes，0 collision，109,628 rows（train 89,820 / held-out 19,808），dataset SHA256=`b408c7a87e0132beb641f207e6103eb09473498677b1db4f888626af4f50415c`。原始 BC 的 held-out MSE/action-index agreement=`.0695674/.693457`，但其 rollout 使用旧 seed base；当前正复用同一冻结 dataset 以共同 seeds `2026191501–08` 重做确定性验证后再启动 C3。
- 两任务均严格映射 86 个 backbone keys；IQN heads 全部排除；critic transfer=`none`。

## 工程验证

- 聚焦与扩展单测：最近完整批次 `51 passed`（protobuf 仅 deprecation warnings）
- CPU real-env coverage smoke：12 env steps，replay 12，critic updates 10，actor updates 5，finite；atomic checkpoint/replay/eval 完成
- CUDA real-env capture smoke（GPU1）：24 env steps，replay 24，critic updates 19，actor updates 9，finite；atomic checkpoint/replay/eval 完成
- CUDA smoke resume：成功恢复 trainer、replay、环境 pickle 与 RNG；manifest 为 attempted=24、added=24、replay=24；`global_state_consumed=false`、`teammate_actions_consumed=false`

smoke 仅是工程证据，不是学习能力证据。

## Exact commands

```bash
python3 -m pytest -q \
  tests/test_local_td3_stage1.py \
  tests/test_td3_stage1_contract.py \
  tests/test_td3_warmstart.py \
  tests/test_td3_stage1_runner.py

python3 -u tools/prepare_td3_local_aw_warmstart_20260916.py \
  --task all --device cuda:1

python3 -u tools/train_td3_local_aw_stage1_20260916.py \
  --config configs/experiments/td3_local_aw_stage1_20260916/capture_scratch.yaml \
  --device cuda:0
python3 -u tools/train_td3_local_aw_stage1_20260916.py \
  --config configs/experiments/td3_local_aw_stage1_20260916/capture_warm.yaml \
  --device cuda:1
python3 -u tools/train_td3_local_aw_stage1_20260916.py \
  --config configs/experiments/td3_local_aw_stage1_20260916/coverage_scratch.yaml \
  --device cuda:0
python3 -u tools/train_td3_local_aw_stage1_20260916.py \
  --config configs/experiments/td3_local_aw_stage1_20260916/coverage_warm.yaml \
  --device cuda:1
```

资源冲突时启动优先级固定为 C2 → C4 → C1 → C3；不得停止任何已有正式进程。

## Artifact paths

- warm datasets / BC：`artifacts/2026-09-16_td3_local_aw_stage1/warmstart/{coverage,capture}/`
- formal runs：`runs/2026-09-16_td3_local_aw_stage1/{C1_*,C2_*,C3_*,C4_*}/`
- 每条 run：`launch.json`、`diagnostics/`、`evaluation/`、`checkpoints/`、`replay/`、`manifests/`、`status.json`、`final.json`

## Milestones / Gate

| milestone | C1 | C2 | C3 | C4 |
|---:|---|---|---|---|
| 25k | pending | pending | pending | pending |
| 50k | pending | pending | pending | pending |
| 75k | pending | pending | pending | pending |
| 100k | pending | pending | pending | pending |

正式启动时间（CST）：C2 `23:10`、C4 `23:15`、C1 `23:17`。三者均在独立 TD3 artifact namespace 下运行；原 MAPPO 已自然完成，本线从未对其发送停止信号。C1/C2/C4 首个真实 optimizer update 均 finite，target smoothing 约 `.155` normalized absolute mean，replay attempted/add 相等。C2 到 6k 的 critic/policy updates=`1001/500`，与 `policy_delay=2` 精确一致；尚未到 25k，因此这些只算工程健康，不算任务能力结果。

Coverage/capture gate 按任务书原定义执行；normal 与 stationary capture 永远分列。Q1/Q2、Q-gap、Bellman target、TD error、predicted Q vs empirical discounted MC proxy，以及 success/failure 条件分组均进入 milestone report。不能把 proxy 表述成 true Q。

当前 STOP / promotion 状态：**停在 Stage 1；Stage-1 task gate 尚无正式结果；Stage 2 不允许开启。**

---

## 2026-09-17 15:25 CST 会话交接快照

### 代码与验证

- 恢复基线：`experiment/density-normalized-sensing-v2-20260915@72c59392436967162fd99cf85b256bae263f0884`。
- 本快照对应最新实现 HEAD：`f451172d9db9a0e2b135fd9ca750eedc7dcd8b0e`；已推送到 `origin/experiment/td3-local-aw-stage1-20260916`。
- 新增严格隔离的 matched discrete runner：`tools/run_td3_stage1_discrete_baseline_20260917.py`。
- IQN 新增向后兼容的 `terminated_only` replay 模式：true terminal 不 bootstrap，time-limit truncation 继续 bootstrap；旧 `terminal_state` 默认语义未改变。
- 离散 runner 固定使用 single-task 的 2M reward clock。审计曾捕获通用 IQN trainer 会把 `coverage_ce_speed_weight` 按历史 step-0 schedule 静默改为 0；隔离 subclass 现保持 live env 精确等于 NormSense-V2 task contract。
- D1/D2 scratch 模型初始化 bit-exact；D3 初始 Actor/Value/ValueNorm hash 硬断言与 D4 相同。
- CPU IQN real-env/checkpoint smoke PASS；CPU MAPPO 256-step rollout、1 次 production PPO update、finite 与 resume hash smoke PASS。
- 最新聚焦回归：`51 passed`；仅 protobuf deprecation warnings。

### TD3 正式里程碑

下表均为同一组 20 个 deterministic seeds（`2026191501–20`）。`Qmin-MC` 只是 predicted Q 与经验折扣 return 的 proxy，不是真实 Q。

| run / step | task ability | geometry / ladder | collision | action | Qmin-MC |
|---|---|---|---:|---|---:|
| C1 scratch 25k | coverage 0/20 | CE RMS .2011 / CV .3667 | 0% | saturation 100%，速度 0 | +26.94 |
| C1 scratch 50k | coverage 0/20 | 与 no-op 相同 | 0% | saturation 100%，速度 0 | +12.04 |
| C1 scratch 75k | coverage 0/20 | 与 no-op 相同 | 0% | saturation 100%，速度 0 | +3.09 |
| C1 scratch 100k | coverage 0/20 | CE RMS .2011 / max .2954 / CV .3667 | 0% | 常量 `a=-.4,w=+π/6`，速度 0 | -4.88 |
| C2 scratch 25k | capture 0/20 | min distance 10.08；ring2/3=0 | 15% | saturation 96.5% | -16.47 |
| C2 scratch 50k | capture 0/20 | min distance 13.95；ring2/3=0 | 5% | saturation 28.4% | -41.84 |
| C2 scratch 75k | capture 0/20 | min distance 12.34；ring2/3=0 | 5% | saturation 20.7% | -53.57 |
| C3 warm 25k | coverage 0/20 | CE RMS .1395 / CV .2384 | 15% | saturation 76.0% | +18.61 |
| C3 warm 50k | coverage 0/20 | CE RMS .1768 / CV .3329 | 10% | saturation 54.4% | +12.82 |
| C3 warm 75k | coverage 0/20 | CE RMS .1281 / CV .2055 | 15% | saturation 59.0% | +6.92 |
| C4 warm 25k | capture 0/20 | ring2 20%；ring3 0；min 6.40 | 100% | saturation 67.1% | +25.76 |
| C4 warm 50k | **normal 1/20** | ring2 35%；ring3 10%；min 8.60 | 95% | saturation 86.8% | -42.15 |
| C4 warm 75k | capture 0/20 | ring2/3=0；min 9.31 | 65% | saturation 74.0% | -142.64 |

C1 已完成完整 100k checkpoint/replay/manifest/final，wall-clock `56,458.5s`，critic/policy updates=`95,001/47,500`。它不是高速穿越式假改善，而是“饱和负加速度指令 + 零速”的行为 collapse，精确复现 no-op control：

- no-op：CE RMS `.2011`、max `.2954`、CV `.3667`、success 0/20、collision 0；
- random：success 0/20、collision 20/20。

因此截至本快照，**Q1 / Coverage Gate 对 formal seed 已明确 FAIL**；不应自动续训超过 100k。

C2/C4 已完成 100k optimizer budget，正在跑 20-seed deterministic eval，尚不能把训练累计统计冒充最终结果。C3 为 93k：

| run | PID | 状态 |
|---|---:|---|
| C2 Capture Scratch | 1627516 | 100k deterministic eval / snapshot pending |
| C3 Coverage Warm | 1637678 | 93k；critic/policy=88,001/44,000 |
| C4 Capture Warm | 1630254 | 100k deterministic eval / snapshot pending |

### Warm-start 实际证据

| task | held-out physical-AW MSE | held-out AW9 agreement | RL 前 common-seed rollout |
|---|---:|---:|---|
| coverage | .06957 | 69.35% | 8/8 CE success，0 collision；CE RMS .0420，CV .0732 |
| capture | .01788 | 82.76% | 8/8 normal capture，0 collision；平均 37.1s |

warm-start 在 step 0 给出明确能力，但在 TD3 更新后不稳定：

- coverage：25/50/75k formal success 均为 0；相对 scratch/no-op 的 CE RMS 改善约 30.6% / 12.0% / 36.3%，但没有保持几何成功；
- capture：scratch 到 75k 没有 ring2/3；warm 在 25k 首现 ring2，50k 首现 ring3 与 1 次 normal capture，但 75k 消失；
- C4 50k 的唯一成功轨迹 `Qmin-MC=-111.57`，失败轨迹为 `-38.49`，是 rare success 被 twin-min 强烈低估的 proxy；不能表述为已知 true-Q bias。

阶段性 Q3 结论：IQN warm-start 带来巨大的 **初始能力与早期 geometry/sample-efficiency** 提升，但没有带来稳定的 TD3 retention；当前不能宣称稳定性提升。

### D4 与 matched baselines

D4 checkpoint 内 live environment 已逐对象验证与 `discrete_task_config("capture")` 完全相等：4v1、AW9、NormSense-V2 radius 52.2173、horizon 3000、reward clock 2.1M、seed/eval seeds 均 matched。100k 补评 SHA256：

- checkpoint：`b31e11fbcf10c710e7bae16cfe62ab78ac4848337a79b1a1679665ab19b0edbc`
- eval：`78703f5d9a8c8bb9a4792a72f0ad0dc46a1fa9a2fdb577ed3d54432d699c020e`

| D4 step | normal | ring2 | ring3 | collision |
|---:|---:|---:|---:|---:|
| 25k | 0% | 0% | 0% | 0% |
| 50k | 0% | 5% | 0% | 100% |
| 75k | 0% | 75% | 15% | 100% |
| 100k | 5% | 85% | 30% | 95% |

D4 到 100k 有 capture/ring 信号，但碰撞极高，不能描述为稳定安全能力。

D1/D2/D3 没有可冒充 matched 的历史 artifact，因此已发布新 runner 并建立非侵入式队列：

- `td3_d2_then_d1_baselines`：等待 C2 PID 1627516 退出后，在 GPU0 顺序运行 D2 Capture IQN → D1 Coverage IQN；
- `td3_d3_baseline`：等待 C3 PID 1637678 退出后，在 GPU1 运行 D3 Coverage MAPPO；
- 队列仅轮询 PID，不发送信号；本快照时均处于 sleep，未占用 GPU、未创建输出目录。

预定 artifacts：

- `runs/2026-09-17_td3_stage1_discrete_baselines/D2_IQN_Capture_Scratch/`
- `runs/2026-09-17_td3_stage1_discrete_baselines/D1_IQN_Coverage_Scratch/`
- `runs/2026-09-17_td3_stage1_discrete_baselines/D3_MAPPO_Coverage_Scratch/`

### 阶段性 Gate / STOP

- Coverage scratch：**FAIL at 100k**。
- Capture scratch：截至 75k **FAIL**；100k deterministic eval pending。
- Warm-start：step-0 capability 强，但 retention/stability **FAIL / pending final100**。
- teammate-conditioning：**没有直接证据**。当前 coverage scratch 本身失败，且 capture 失败可由 action collapse / severe Q under-estimation 解释；尚未建立“同一 `(o_i,a_i)` 因 `a_-i` 不同而 return 显著不同”的条件证据。
- Stage 2：**不允许开启**（coverage Gate 已失败，且完整 matched matrix 未收口）。
- Stage 3：**不允许开启**（没有 teammate-action conditioning 的归因证据）。
- 自动延训：禁止；100k 后先审计。

本线仍是 continuous-control capability audit，不是 CoCap/MAPPO 主线的既定替换。FullMix、phase-aware replay、MADDPG/MATD3、central critic、GNN/GAT 与 `is_pursuing` removal 均未启动。
