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
| D1 | pure coverage / IQN AW9 | scratch | 待 matched-run 确认 | 历史 teacher 有当前合同冻结评估能力，但训练合同/预算不 matched，不能用于 sample-efficiency 排名 |
| D2 | pure capture / IQN AW9 | scratch | 待 matched-run 确认 | 同上；当前合同资格抽查 normal capture 2/2 |
| D3 | pure coverage / categorical MAPPO AW9 | scratch | 待 matched-run | 旧 sensing 结果不冒充 NormSense V2 matched baseline |
| D4 | pure capture / categorical MAPPO AW9 | scratch | 现有正式 run | 独立 run 已自然完成 300k；100k argmax normal capture 15%、ring2 visitation 90%、ring3 visitation 70%、collision 75%，须按同 eval seeds 再核对后才进入 matched 表 |
| C1 | pure coverage / local TD3 AW | scratch | 2026091601 | runner ready；formal 未启动 |
| C2 | pure capture / local TD3 AW | scratch | 2026091602 | runner ready；formal 未启动 |
| C3 | pure coverage / local TD3 AW | IQN-warm | 2026091601 | 等 matched teacher dataset + BC |
| C4 | pure capture / local TD3 AW | IQN-warm | 2026091602 | 等 matched teacher dataset + BC |

scratch/warm 同 task 共用 seed；critics 的随机初始化 matched。正式 TD3 milestone=`25k/50k/75k/100k`，每点 20 个 deterministic eval episodes、atomic checkpoint、replay snapshot、diagnostics 与 manifests；100k 无趋势先审计，不自动延长。

## Hyperparameters

共同 config：`configs/experiments/td3_local_aw_stage1_20260916/common.yaml`。

- hidden 256，8 heads，4 layers，dropout module 保留 `.1` 但 TD3 deterministic forward 固定 eval mode
- batch 256；replay 200k；warmup 5k；UTD 1
- gamma `.99`；tau `.005`
- actor/Q learning rate 均 `1e-4`；grad clip `.5`
- BC：64 successful train episodes + 16 successful heldout episodes / task；30 epochs；batch 512；lr `3e-4`

## 工程验证

- 聚焦单测：`36 passed`（protobuf 仅 deprecation warnings）
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

Coverage/capture gate 按任务书原定义执行；normal 与 stationary capture 永远分列。Q1/Q2、Q-gap、Bellman target、TD error、predicted Q vs empirical discounted MC proxy，以及 success/failure 条件分组均进入 milestone report。不能把 proxy 表述成 true Q。

当前 STOP / promotion 状态：**停在 Stage 1；Stage-1 task gate 尚无正式结果；Stage 2 不允许开启。**
