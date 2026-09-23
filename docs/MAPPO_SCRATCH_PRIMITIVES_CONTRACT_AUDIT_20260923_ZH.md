# MAPPO Scratch Primitives 合同审计

日期：2026-09-23  
独立分支：`experiment/mappo-scratch-primitives-20260923`  
最新上游基线：`origin/ops/ac-master-dag-20260921`  
Canonical learner parent：`25dd0f8`（2026-09-15 corrected single-task MAPPO）  
Canonical coverage result：`d9ede24` / `e696dec`

## 审计结论

本轮只吸收 `CORRECTNESS_REQUIRED`，并为两个 scratch primitive 固定任务所需的 `TASK_REQUIRED` 差异。没有吸收 Z、IQN、A1/A2/A3/B 线、reward curriculum、initialization curriculum、local-friendly V3 或 post-capture recovery。

M-COV 是成功的 legacy-R20 Pure Coverage positive control；M-CAP 是明确标注为 `CAPABILITY_FIRST_CAPTURE_BASELINE` 的 Pure Capture baseline。两者不是严格 sensing-matched 的单变量比较：M-COV 使用 legacy 20m，M-CAP 使用正式 NormSense V2 resolver 的 runtime R。

## 逐项分类

| 合同/后续修改 | 分类 | 本轮处理 |
|---|---|---|
| collision semantics | `CORRECTNESS_REQUIRED` | 保留 `synchronized_swept_v1`，碰撞、边界和 safety 仍由环境合同处理。 |
| terminal / truncation / bootstrap | `CORRECTNESS_REQUIRED` | 保留 terminal-priority、truncation bootstrap、GAE recursion cut 和 reset-before-next-V 语义；不把 terminal row 误当成可 bootstrap 状态。 |
| active-agent handling | `CORRECTNESS_REQUIRED` | actor 与 centralized action-free V 使用 active-only mask；保持 `min_active=4`，不把 inactive agent 重新纳入 loss。 |
| fully-masked terminal row fix | `CORRECTNESS_REQUIRED` | 将 Astra fix 的最小 encoder fallback 移植到 MAPPO 使用的 LegacyVorAdjFeatureBackbone；仅在整行全 mask 时临时开放 self token，输出 mask 与 token schema 不变；加入 `test/test_mappo_terminal_rows_20260923.py`。 |
| MAPPO update | `CORRECTNESS_REQUIRED` | 使用 canonical rollout-256、PPO clip、3 epochs、2 minibatches、target KL 和 active-weighted actor/value update。 |
| ValueNorm | `CORRECTNESS_REQUIRED` | 保留 beta `.99999`、eps `1e-5` 及 canonical update/denormalization 路径。 |
| central V | `CORRECTNESS_REQUIRED` | 保留 centralized、action-free V，hidden 256 / 8 heads / 4 layers；不加入 action 或 Z。 |
| AW9 mapping | `CORRECTNESS_REQUIRED` | 保留 AW9 action mapping 与 action-mask 合同。 |
| observation token dimensions | `CORRECTNESS_REQUIRED` | 保留 self-9、max pursuers 8、max evaders 8、max obstacles 5、hidden 256、pursuing embedding 8；未增删 token。 |
| `is_pursuing` | `CORRECTNESS_REQUIRED` | 保留 canonical role/pursuit state 的 observation 语义；不以额外 heuristic 或初始化 curriculum 改写。 |
| Z | `EXPERIMENTAL` / out of scope | 不引入、不读取、不训练 Z。 |
| friend ordering | `CORRECTNESS_REQUIRED` | 保留 canonical friend ordering 与 historical VorAdj/free-mask-projected topology。 |
| legacy R20 sensing | `CORRECTNESS_REQUIRED` + M-COV task contract | M-COV 使用成功 positive-control 的 legacy local surface sensing 20m；不改为 V2。 |
| NormSense V2 | `TASK_REQUIRED`（仅 M-CAP） | M-CAP 调用正式 resolver：`R = 0.8715 * sqrt(A_eff / N)`，floor 20m；不手写 radius。 |
| spawn / reset | `CORRECTNESS_REQUIRED` | 保留 common map randomization、spawn/reset 和 no post-capture recovery 的相关状态边界；M-CAP 使用正式 Pure Capture stream。 |
| horizon | `CORRECTNESS_REQUIRED` + task contract | 保留 horizon 3000，不因前期 capture=0 提前终止实验。 |
| reward | `CORRECTNESS_REQUIRED` + task contract | M-COV 保留 Final CE centroid-energy + PBRS + speed/control/safety、无 success bonus；M-CAP 使用 Final `ring_importance_ms_v0` / CR-MS，support capture weight 归一化为 1、coverage weight 0，并保留 collision/boundary/safety。 |
| post-capture recovery / conservative recovery | `EXPERIMENTAL` | 不吸收；Pure Capture 成功即 terminal。 |
| reward / initialization curriculum | `EXPERIMENTAL` | 不吸收。 |
| IQN、Full-Mix、role/observation extensions | `EXPERIMENTAL` | 不吸收。 |

## 两条运行合同

### M-COV

4 pursuers、0 evaders、1 obstacle、120x120、legacy 20m、historical global-friendly VorAdj/free_mask_projected、AW9、random-init LegacyVorAdjFeatureBackbone、canonical MAPPO learner。预算 200k，formal checkpoints `0/25/50/75/100/125/150/175/200k`，每个 argmax20 + sample20。

### M-CAP

4 pursuers、1 evader、1 obstacle、120x120；任务 reward 和 direct detector 使用 Final Pure Capture primitive。保留 Final `neighbor_visible` 与 `approach_only`，无敌情/非-support agent 不给 global/oracle enemy information，不给 coverage task reward；capture terminal 后无 recovery/coverage episode。预算 500k，formal 每 25k，每点 argmax20 + sample20。实验标签固定为 `CAPABILITY_FIRST_CAPTURE_BASELINE`。

本次 runtime resolver 实测并保存为 `artifacts/2026-09-23_mappo_scratch_primitives/resolved_capture_sensing.json`：

```text
A_eff = 14360.0
k = 0.8715
raw R = 52.21732449580312 m
resolved R = 52.21732449580312 m
floor = 20.0 m
map scale = 1.0
surface-distance semantics = surface_clearance
N = 4
```

M-COV 与 M-CAP 的 sensing 不匹配已在 launch/report metadata 中显式标记；本轮不追加 R20-vs-NormSense Capture ablation。若 M-CAP 到 500k 仍无稳定 capture，`state visitation / exploration difficulty` 仅列为未来 A3 initialization-curriculum 候选原因，不在本轮实现。

