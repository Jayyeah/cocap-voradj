# AC-2B — Canonical Strong-BC Formal Critic Bank

日期：2026-09-20  
分支：`audit/ac-bc-critic-lineage-20260920`  
合同：`forward-final-aw9-4v1-swept-v1`

## Verdict

`FORMAL_BANK_PASS`

本轮只做 frozen-actor evaluation rollout、raw bank 写入和离线完整性审计；未进行 critic training、bootstrap、GAE、PPO 或任何参数更新。

## Generator identity

- canonical logical checkpoint：`artifacts/2026-09-08_forward_final/c2_distillation/actor_epoch_030.pt`
- 当前实际加载：`artifacts/2026-09-17_critic_identifiability_audit/frozen_policy/actor_epoch_030.pt`
- Actor SHA256：`7916a970226a0452a8f71a22235524f6ba4511aa4c9bda907a9c8368a60bf2cd`
- generator：categorical MAPPO Actor，frozen，actor updates = 0，actor tensor checksum bit-exact
- 未使用 IQN teacher、IQN recovery pool、random/noisy Actor、PPO-updated Actor 或 scratch MAPPO

实际加载副本与 canonical logical checkpoint 的 SHA 相同；因此 generator identity 通过。

## Bank composition

固定 Scheme A，共 80 episodes：

| scene | argmax | sampled | total |
|---|---:|---:|---:|
| mixed Full Task | 30 | 30 | 60 |
| pure coverage | 10 | 10 | 20 |
| total | 40 | 40 | 80 |

结果：mixed success `59/60`、failure `1/60`、collision episode `1/60`；pure coverage success `20/20`、failure `0/20`、collision `0/20`。

Bank 共 `16964` transitions、`67856` active-agent rows。实际 bank：
`artifacts/2026-09-20_ac2b/canonical_bc_critic_bank_v2.npz`，SHA256
`f77e0fe7c2b76dd143c78d407e16d5b780e10e0278af425f23e6e5726c132fac`，大小 `35279291` bytes。

## Environment and return contract

沿用 AC-1 的 `CONTRACT_MATCH`：

- environment：120x120；mixed 为 4 pursuers / 1 evader / 1 obstacle；coverage 为 4 pursuers / 0 evaders / 1 obstacle；horizon 3000；`synchronized_swept_v1` collision；capture 不终止，保留自然 post-capture recovery。
- sensing：`friendly_voronoi_comm_v0`、legacy-r20 Final VorAdj sensing；friend-only Voronoi adjacency；enemy/obstacle surface radius 20m；无 global enemy token、无 Z-token 环境。
- action：AW9 categorical index `0..8`，顺序为 `[(a,w) for a in (-0.4,0,0.4) for w in (-pi/6,0,pi/6)]`；runtime 映射审计通过。
- reward：canonical environment 的 individual per-agent reward；没有替换为 team reward。
- return：`G_t = sum_k gamma^k r_(t+k)`，`gamma=0.99`；完整 episode 反向计算；包含 terminal transition reward；无 bootstrap。
- coverage：canonical `inner_random_cluster`；mixed：canonical mixed initialization 和自然 post-capture continuation；没有注入 IQN recovery curriculum。

## Raw schema and critic readiness

每个 timestep 保留 current/next local observation、neighbor observation、central/global state、active masks、episode/timestep/scene/policy/split metadata、reward/done flags、phase/event flags、actor logits/probabilities、selected action、AW9 action，以及 MC return。

为 NQ 显式物化并审计了 `neighbor_action_index` 和 `neighbor_action_aw`。因此后续可直接构造：

- LQ：`local_* + own action`
- NQ：`local_* + neighbor_local_* + neighbor_ids/mask + neighbor actions`
- CQ：`global_* + masks + joint actions + agent identity/index`
- V：`global_* + masks + normal state/time information`，无 future label

所有自然访问 rows 保留；未做 class rebalance、downsample 或 oversample。

## Phase, event and semantic classes

active-agent rows：

| quantity | rows |
|---|---:|
| pure coverage phase / `recovery_pure` | 11564 |
| pre-capture phase | 20668 |
| `pursuing` | 6057 |
| `pre_capture_cover` | 14611 |
| post-capture phase / `post_capture_real` | 35624 |
| early recovery | 9460 |
| ring2 | 3156 |
| ring3 | 264 |
| capture transition rows | 60 transitions / 240 active-agent rows |

early recovery 原样沿用旧 A0/A1 定义：`post_capture` 且 `0 <= timestep-capture_step <= 40`；capture transition 本身保持 `pre_capture`。所有四个 semantic class 均达到 operational floor `5000` rows：pursuing `6057`、pre_capture_cover `14611`、post_capture_real `35624`、recovery_pure `11564`。

## Integrity and split audit

- required raw fields：通过
- NaN / Inf：`0 / 0`
- actor probability sum 最大绝对误差：`2.980232238769531e-07`
- AW9 action mapping 最大绝对误差：`0`
- current/next intra-episode continuity：`16884` links checked，0 violations
- independent MC recompute：100 rows，最大绝对误差 `7.5362939924161765e-06`，阈值 `1e-5` 内
- successful mixed capture alignment：`59` episodes，检查 `t_capture-1 / t_capture / t_capture+1`，0 violations
- split：episode-level only；train/validation/test = `64/8/8` episodes；无 transition random split、无 episode/seed leakage

## Artifacts

- bank：`artifacts/2026-09-20_ac2b/canonical_bc_critic_bank_v2.npz`（不提交 Git）
- seed/split：`artifacts/2026-09-20_ac2b/seed_split_manifest.json`
- phase/event/class counts：`artifacts/2026-09-20_ac2b/phase_counts.json`
- integrity：`artifacts/2026-09-20_ac2b/integrity_report.json`
- formal bank manifest：`artifacts/2026-09-20_ac2b/bank_manifest.json`
- compact report：`artifacts/2026-09-20_ac2b/report.json`
- collector：`tools/collect_ac2b_canonical_bc_bank_20260920.py`
- offline audit：`tools/audit_ac2b_canonical_bc_bank_20260920.py`

本轮到此停止；不进入 AC-3。
