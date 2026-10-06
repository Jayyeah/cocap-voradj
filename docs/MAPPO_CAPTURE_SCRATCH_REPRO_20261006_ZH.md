# MAPPO Pure Capture Scratch 复现

日期：2026-10-06（Asia/Shanghai）  
分支：`experiment/mappo-scratch-primitives-20260923`  
问题：标准 MAPPO 能否在 CoCap Pure Capture 上从随机初始化学出 capture？

## 1. 来源合同与已有证据

- Coverage 正控：`M-COV`，random-init，200k，legacy sensing R=20m。100k 的 argmax/sample strict coverage 为 100%/100%；200k 为 80%/100%，碰撞为 5%/0%。150k 曾达到 100%/100%，此处保留终点合同，不用最佳点替代终点评估。
- Capture 旧失败线：`M-CAP`，NormSense V2，500k，终点评估 capture 4/40、collision 36/40；此结果是新的 capture-only 运行背景，不会与本次 100k 窗口拼接。
- 合同依据：`docs/ops/AC_MASTER_DAG_20260921_ZH.md`、`docs/MAPPO_SCRATCH_PRIMITIVES_CONTRACT_AUDIT_20260923_ZH.md`、`docs/MAPPO_SCRATCH_PRIMITIVES_STATUS_20260923_ZH.md`、`docs/FORWARD_FINAL_SCRATCH_MAPPO_PREFLIGHT_20260914_ZH.md`。

## 2. 恢复的 Coverage MAPPO 合同

| 部分 | 配置 |
|---|---|
| Actor | LegacyVorAdjFeatureBackbone / Categorical AW9；Transformer hidden 256、8 heads、4 layers、dropout 0.1；self feature 9；friend 最多 8、evader 最多 8、obstacle 最多 5；pursuing embedding 8；policy head orthogonal，gain 0.01；全部参数从随机初始化并参与 PPO 更新 |
| Critic | centralized、action-free V；hidden 256、8 heads、4 layers；self feature 9；max agents 4、evaders 8、obstacles 5 |
| Optimizer | 独立 Adam；actor LR 3e-5，critic LR 1e-4，Adam eps 1e-5；fresh optimizer state |
| PPO | gamma .99；GAE lambda .95；clip .2；3 epochs；2 minibatches；entropy .01；value coef 1；max grad norm .5；target KL .02 |
| ValueNorm | beta .99999，epsilon 1e-5 |
| Rollout | 256 joint environment decision steps；checkpoint/evaluation cadence 25k |
| Reward/clock | 不缩放优化器输入或另加 shaping；Capture 使用原 Final `ring_importance_ms_v0` reward，capture support 权重归一化为 1、coverage 权重 0，CE reward 关闭；reward clock offset 2,000,000，保留 Coverage 正控代码合同 |
| Observation | actor 仍为 self-9、friend 8×7、enemy 8×7、obstacle 5×5 的 canonical Legacy VorAdj token schema；critic 仍用 geometry-only centralized observation；不加 Z 或新 token。Enemy token 仍受 onboard surface-distance detector 限制 |

不加载 BC、IQN、teacher、pretrained critic、warm start 或 imitation 数据。terminal/truncation、active-agent mask、collision、ValueNorm 与 PPO 更新代码沿用 Coverage 正控实现。允许的实验差异限定为 capture task environment 和相应 capture reward。

## 3. 最小 Pure Capture 合同

Pure Capture 在此指**场景与任务 reward 都是 Pure Capture**：4 pursuers、1 evader、1 obstacle，随机初始化与正常移动 evader；只使用 capture/support capture reward，不混入 coverage reward；成功 capture 后 episode terminal，不接 recovery。没有 Z、curriculum、initialization curriculum、SAC 或 AC。

采用正式 NormSense V2 resolver：有效 free area 14,360 m²，`k=0.8715`，surface-clearance 语义，resolved enemy sensing radius `R=52.2173245m`（floor 20m）。该范围属于 environment/detector 设置；Observation token schema 和 actor/critic 结构保持不变。记录为 task environment 差异。旧 M-CAP 也使用该 resolver，所以旧失败结果说明扩大感知范围本身不足以保证成功。

初始化：actor、critic、Adam、ValueNorm 均为 Coverage 正控 seed `2026091501` 的 fresh random initialization；Capture 场景按官方 map-random spawn/reset；没有预填成功状态池或课程初始化。

## 4. 运行与评估方案

fresh scratch，预算 100,000 joint decision steps；固定评估 checkpoint：0、25k、50k、75k、100k。每点 argmax 与 sample 各 20 个 episode，训练 seed `2026091501`，evaluation seed base `2026191501`。checkpoint latest-only，不按指标提前停止。

每个评估点报告：normal capture、stationary capture、2+ 与 3+ coalition visitation、collision/boundary、capture time（mean/median/p90 与 success n；删失与失败单列，不填造时间）。保留 argmax/sample 分列。完成后按 A learnable / B partial learning / C same failure as M-CAP 分类，并明确单 seed 与每模式20局的 screen 限制。

## 5. 运行结果

正式 run：`artifacts/2026-10-06_mappo_capture_scratch/run_seed2026091501/`；GPU0；fresh run，未 resume；100,000/100,000 joint decision steps；391 PPO updates；状态 `COMPLETE_BUDGET`。固定 seed、actor/value/ValueNorm 初始 hash 与 M-COV 正控一致。每个 checkpoint argmax/sample 各 20 episodes。0/25k/50k/75k/100k 均完成。PPO 数值 telemetry 保持 finite；没有因指标提前停止。

| steps | mode | normal capture | stationary capture | 2+ visited | 3+ visited | collision | capture time |
|---:|---|---:|---:|---:|---:|---:|---|
| 0 | argmax | 0/20 | 0/20 | 0/20 | 0/20 | 20/20 | 无成功，20 删失 |
| 0 | sample | 0/20 | 0/20 | 0/20 | 0/20 | 20/20 | 无成功，20 删失 |
| 25k | argmax | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 无成功；20 局到 3000-step horizon |
| 25k | sample | 0/20 | 0/20 | 0/20 | 0/20 | 14/20 | 无成功，20 删失 |
| 50k | argmax | 0/20 | 0/20 | 1/20 | 0/20 | 20/20 | 无成功，20 删失 |
| 50k | sample | 1/20 | 0/20 | 1/20 | 1/20 | 19/20 | 唯一成功 924s；19 删失 |
| 75k | argmax | 0/20 | 0/20 | 15/20 | 3/20 | 20/20 | 无成功，20 删失 |
| 75k | sample | 0/20 | 0/20 | 3/20 | 0/20 | 20/20 | 无成功，20 删失 |
| 100k | argmax | 0/20 | 0/20 | 14/20 | 4/20 | 20/20 | 无成功，20 删失 |
| 100k | sample | 0/20 | 0/20 | 8/20 | 2/20 | 20/20 | 无成功，20 删失 |

碰撞含边界/障碍/agent collision；其中 25k argmax 是唯一无碰撞的评估点，但 20 局均未进入 ring2 或完成 capture。100k 环形占位明显高于初始化，不过两种动作模式最终都 20/20 collision。时间只对成功 episode 计算；删失 episode 不填入虚构 capture time。评估报告逐 episode 保存在对应 `eval_step_*.json`。

## 6. 结论

**分类 B：partial learning。** Scratch MAPPO 在 capture 前驱几何上有变化：50k sample 出现 1 次 normal capture（924s），75k/100k 出现 ring2/3 coalition visitation。这个信号没有持续成稳定 capture：25k 与 75k 的 capture 均为 0/40，100k 为 0/40，而且 100k 两个模式 collision 都是 20/20。因此不能回答为“A：MAPPO capture learnable”。

相对于 2026-09-23 的 M-CAP 500k 失败合同（终点 4/40 capture、36/40 collision），本次 100k endpoint 更接近同一种失败形态，尽管中途出现一个 sample capture 与后续 ring 访问。最合适的总结是“有 partial capture-geometry 信号，但没有学会可靠 capture”；不是成功复现，也没有证据支持超出 100k 继续扩预算。单 seed、每模式20局是 screen，不构成多种子统计结论。

### 运行器/成本观察

- 正式评估没有训练数据泄漏；argmax/sample 各用固定 evaluation seed base，隔离于训练 stream。
- 25k 评估完成 40 episodes 用时约 49 分钟，是全程最慢的一点；其余点评估约 1–6 分钟。仿真 CPU wall time 是主要成本，GPU0 利用率大多低于 10%，显存余量充足。
- `progress.json` 的 `steps_per_second` / `eta_seconds` 把 checkpoint 后的评估等待时间计入统计，因此 25k 后 ETA 会高估纯训练时间；训练 step 仍连续推进，所有 checkpoint 有效。
- 本次没有发现需要改动环境、架构或 PPO 算法的代码问题。唯一 runner 改动是增加 `--budget` CLI 覆盖以准确执行 100k；actor、critic、reward 函数、PPO 更新和 observation 实现均保持原合同。
