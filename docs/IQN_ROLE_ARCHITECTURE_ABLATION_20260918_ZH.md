# IQN role architecture 正控消融（2026-09-18）

## 结论

最终判决：`CASE C — CURRENT_BC_PIPELINE_FAILS_POSITIVE_CONTROL`。

C0 在任何 BC update 前逐 Q 复现 Final IQN（最大绝对误差 `6.10e-5`，overall action agreement `0.9999665`，Q regret `0`），3-seed qualification 的 coverage/capture/mixed 均 `3/3 safe complete`。同一完整 architecture 经过与 B3 完全相同的 4-epoch BC/distillation 后：

- overall agreement：`0.99997 -> 0.39861`；
- pure-coverage agreement：`0.99991 -> 0.35072`；
- formal pure coverage：`0/20 CE`，collision `1/20`；
- formal pure capture：`20/20 normal capture`，collision `0/20`；
- formal mixed：`20/20 capture`，但 `0/20 post-capture CE/recovery`，collision `1/20`。

所以 B1/B3 的 architecture / z 因果结论必须降级。当前最强证据是 BC pipeline 本身会把完整 Final IQN 从可用 policy 拉坏；在修复该正控之前，不能把 C1 或 B3 的 failure 归因于 late fusion 或 z 定义。

## 实验合同与公平性

- 分支：`experiment/iqn-role-architecture-ablation-20260918`
- 基线：B3 `5f545de9233ce5e917919cb638462cf580ad4c56`
- teacher：Final IQN SHA256 `2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89`
- 数据 seed：`2026091801`；20 mixed + 20 pure coverage + 原 targeted ring3 seed `2026094101`
- 数据：29,864 rows；train/validation 仍按 `episode_id % 5`；优化集仍使用 B3 stratified subset（上限 65,536）
- 训练：4 epochs、batch 4096、Adam (`lr=3e-4`, `eps=1e-5`)、gradient clip 1.0
- loss：B3 原样的 categorical KL + `0.25 * normalized smooth-L1`
- rollout seed base：`2026092801`，scene offset 与 B3 完全相同；post-BC 每 arm 每 scene 20 episodes
- reward、sensing、topology、dynamics、termination 与 evaluator 均未修改

B3 dataset 没有保存可严格恢复 friend role/order 的原始字段，因此没有把 z 数值伪装成 `is_pursuing`。本轮按同一 teacher/seeds 重采 role-correct observations。41 个 shard 对 B3 的 teacher Q、greedy action、reward、episode/seed/timestep/agent、phase/role/direct/support/pursuing、ring2/ring3、capture transition、terminated/truncated 均逐项校验：离散字段完全相等，teacher-Q 与 reward 最大绝对差均为 `0`。各类计数也完全相同：ring2 64、ring3 16、direct 1,810、support 1,936、post-capture 12,384、pure coverage 10,812。

## Architecture 与迁移

### C0 — Full Original + Same Training

直接完整加载 Final IQN checkpoint；self/friend `is_pursuing`、role-dependent friend ordering、`pursuing_embed`、late fusion、Transformer/IQN/action head 全保留。所有参数参与与 B3 相同的 BC updates。

### C1 — Token-only `is_pursuing`

保留原 self/friend physical feature、`is_pursuing_i/j` 与 role-dependent friend ordering。只删除 `pursuing_embed` 和 late-fusion branch。所有 shape-compatible 权重逐位复制；`single_action_feature.0.weight` 只裁掉最后 8 个 role-embedding columns，backbone 未随机重置。独立 config 通过 `iqn.pursuing_late_fusion: false` 开启。

旧 Final checkpoint 默认仍启用 late fusion；B1/B3 旧 checkpoint 根据 `include_is_pursuing` 向后兼容加载，不覆盖 z architecture。

## Offline 结果

指标顺序均为 `action agreement ↑ / categorical KL ↓ / Q regret ↓ / Q-ranking agreement ↑`。

| arm | overall agreement | KL | regret | ranking |
|---|---:|---:|---:|---:|
| C0 step-0 | 0.99997 | 1.61808 | 0.00000 | 0.99999 |
| C0 post-BC | 0.39861 | 1.56899 | 0.03019 | 0.76457 |
| C1 step-0 | 0.71477 | 1.61066 | 0.00426 | 0.94984 |
| C1 post-BC | 0.36546 | 1.58158 | 0.03487 | 0.78551 |
| C2/B3 post-BC | 0.41441 | 1.57396 | 0.03938 | 0.76947 |

沿用 B3 的 categorical-KL 口径：teacher target 以 temperature `0.0031884` softmax，而 student side 使用原始 Q log-softmax。因此即使 C0 step-0 是同一个 teacher，KL 也不会为 0；C0 的实现有效性用逐 Q 误差、agreement 与 regret 判定，KL 仅按同一 B3 口径横向报告。

### C0 step-0

| slice | agreement | KL | regret | ranking |
|---|---:|---:|---:|---:|
| overall | 0.99997 | 1.61808 | 0.00000 | 0.99999 |
| pre-capture | 1.00000 | 1.41743 | 0.00000 | 1.00000 |
| post-capture | 1.00000 | 1.68272 | 0.00000 | 0.99999 |
| pure coverage | 0.99991 | 1.66777 | 0.00000 | 0.99999 |
| direct | 1.00000 | 1.23546 | 0.00000 | 1.00000 |
| support | 1.00000 | 1.24832 | 0.00000 | 1.00000 |
| coverage | 0.99996 | 1.67263 | 0.00000 | 0.99999 |
| ring2 | 1.00000 | 1.24683 | 0.00000 | 1.00000 |
| ring3 | 1.00000 | 1.30421 | 0.00000 | 1.00000 |

### C0 post-BC

| slice | agreement | KL | regret | ranking |
|---|---:|---:|---:|---:|
| overall | 0.39861 | 1.56899 | 0.03019 | 0.76457 |
| pre-capture | 0.57798 | 1.28535 | 0.05671 | 0.85548 |
| post-capture | 0.34383 | 1.66031 | 0.02363 | 0.73574 |
| pure coverage | 0.35072 | 1.63932 | 0.02136 | 0.74153 |
| direct | 0.67947 | 1.01213 | 0.11562 | 0.91958 |
| support | 0.64411 | 1.03243 | 0.06685 | 0.91604 |
| coverage | 0.36063 | 1.64747 | 0.02151 | 0.74265 |
| ring2 | 0.65625 | 0.94267 | 0.09790 | 0.90123 |
| ring3 | 0.68750 | 0.83965 | 0.03745 | 0.96914 |

### C1 step-0

| slice | agreement | KL | regret | ranking |
|---|---:|---:|---:|---:|
| overall | 0.71477 | 1.61066 | 0.00426 | 0.94984 |
| pre-capture | 0.77549 | 1.40212 | 0.01364 | 0.95312 |
| post-capture | 0.69598 | 1.67769 | 0.00158 | 0.94900 |
| pure coverage | 0.69885 | 1.66250 | 0.00154 | 0.94877 |
| direct | 0.71560 | 1.20807 | 0.04550 | 0.93002 |
| support | 0.93853 | 1.22899 | 0.00151 | 0.98547 |
| coverage | 0.69799 | 1.66730 | 0.00153 | 0.94863 |
| ring2 | 0.60938 | 1.21689 | 0.07790 | 0.92361 |
| ring3 | 0.43750 | 1.20849 | 0.09410 | 0.94136 |

### C1 post-BC

| slice | agreement | KL | regret | ranking |
|---|---:|---:|---:|---:|
| overall | 0.36546 | 1.58158 | 0.03487 | 0.78551 |
| pre-capture | 0.56419 | 1.29010 | 0.05855 | 0.86508 |
| post-capture | 0.30782 | 1.67433 | 0.02835 | 0.76079 |
| pure coverage | 0.30892 | 1.65510 | 0.02773 | 0.76475 |
| direct | 0.67317 | 1.00869 | 0.11991 | 0.91463 |
| support | 0.65806 | 1.03144 | 0.06196 | 0.91753 |
| coverage | 0.32246 | 1.66216 | 0.02682 | 0.76686 |
| ring2 | 0.71875 | 0.93547 | 0.06418 | 0.90432 |
| ring3 | 0.68750 | 0.85320 | 0.03745 | 0.95988 |

### C2/B3 reference

| slice | agreement | KL | regret | ranking |
|---|---:|---:|---:|---:|
| overall | 0.41441 | 1.57396 | 0.03938 | 0.76947 |
| pre-capture | 0.49760 | 1.33707 | 0.10864 | 0.83338 |
| post-capture | 0.38735 | 1.64735 | 0.02019 | 0.74885 |
| pure coverage | 0.39410 | 1.63599 | 0.01865 | 0.75366 |
| direct | 0.66858 | 0.97782 | 0.12312 | 0.90844 |
| support | 0.49535 | 1.24199 | 0.22973 | 0.86278 |
| coverage | 0.39084 | 1.63945 | 0.01912 | 0.75305 |
| ring2 | 0.59375 | 0.98036 | 0.15424 | 0.86420 |
| ring3 | 0.68750 | 0.82728 | 0.03745 | 0.94599 |

## Matched rollout

Step-0 qualification 为每场景 3 seeds；post-BC formal 为每场景 20 seeds。

| arm | coverage CE | coverage collision | time-to-CE mean | capture normal | capture stationary | capture collision | capture time mean | mixed capture | mixed collision | mixed CE / safe complete | recovery mean | mission mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 step-0 (3) | 3/3 | 0/3 | 71.17 s | 3/3 | 0/3 | 0/3 | 46.00 s | 3/3 | 0/3 | 3/3 | 49.50 s | 98.50 s |
| C1 step-0 (3) | 3/3 | 0/3 | 75.50 s | 3/3 | 0/3 | 0/3 | 48.83 s | 3/3 | 0/3 | 3/3 | 92.17 s | 141.67 s |
| C0 post-BC (20) | 0/20 | 1/20 | — | 20/20 | 0/20 | 0/20 | 40.75 s | 20/20 | 1/20 | 0/20 | — | — |
| C1 post-BC (20) | 0/20 | 1/20 | — | 18/20 | 0/20 | 2/20 | 40.42 s | 19/20 | 3/20 | 0/20 | — | — |
| C2/B3 post-BC (20) | 0/20 | 2/20 | — | 14/20 | 0/20 | 6/20 | 68.00 s | 15/20 | 7/20 | 0/20 | — | — |

Post-BC on-policy teacher agreement：C0 `0.11312`，C1 `0.21960`，既有 C2/B3 `0.18386`。C0 step-0 为 `1.0`；C1 step-0 为 `0.70379`。

## 因果分析

### C0 BC 前后漂移

- agreement：`-0.60136`
- categorical KL：`-0.04908`（受上述非对称 temperature 口径影响，不能解释为 policy 改善）
- Q regret：`+0.03019`
- Q-ranking agreement：`-0.23542`
- pure coverage：从 qualification `3/3 CE` 到 formal `0/20 CE`
- mixed recovery：从 qualification `3/3` 到 formal `0/20`

这不是 reconstruction 或 architecture surgery：C0 使用完整原 architecture 与完整 teacher weights，step-0 也复现 teacher。唯一新增因素是 matched BC updates。

### C1 相对 C0

C1 的单次 surgery 在 step-0 将 overall agreement 降至 `0.71477`，说明 late branch 删除确实会改变 logits/action；但 3-seed rollout 仍在三场景全部完成。post-BC 时，C1 相对 C0：overall agreement `-0.03315`、Q regret `+0.00468`；capture 从 `20/20` 降至 `18/20`，但两者 coverage/recovery 都是 `0/20`。

由于 C0 正控已经失败，不能把 post-BC 的 C1-C0 差异升级为 `LATE_FUSION_CAUSALLY_IMPORTANT`。本轮只允许得出 CASE C；C1 architecture 的正式因果判决要等 BC pipeline 先通过 C0。

### Pure coverage

Pure coverage 中 C0/C1 的 `is_pursuing=0`，B3 的 `z=0`。C0 step-0 正常、C0 post-BC 失败，直接说明 pure-coverage collapse 无需任何 z dynamics 或 late-fusion removal 就能发生。因此当前 B3 的 0/20 coverage 不能归因于 z；training pipeline 是已被正控证明的 major confound。

## 定位建议（本轮不做超参搜索）

下一步应先修 student training/reconstruction pipeline，并以 C0 post-BC 必须保持 Final 能力作为 gate。优先审计：极低 teacher temperature 与非对称 KL 定义、全参数 Adam 对已收敛 teacher 的 catastrophic forgetting、action/phase imbalance、rare action coverage、teacher Q scale，以及是否应冻结无需重建的 transferred layers。本轮没有启动任何 RL fine-tune 或下一阶段训练。

## z 后续 TODO（仅记录，未实施）

在 BC 正控修复后，才预注册一个小型 `(lambda, epsilon_z)` 矩阵：

1. 缩短当前 `lambda=0.95`、half-life≈`6.76 s`；
2. 计算 `z_tilde = max(d_i, lambda*z_i_prev, eta*max_neighbor_z_prev)`；
3. 施加 hard floor：`z_i = 0 if z_tilde < epsilon_z else z_tilde`。

本轮没有改 lambda/eta、没有 hard threshold、没有 local Voronoi、没有新 role、没有启动 scratch RL/PPO/TD3/Actor-Critic。

## 产物与验证

- `artifacts/2026-09-18_iqn_role_arch_ablation/contract.json`
- `C0_step0.json`、`C0_postbc.json`、`C1_step0.json`、`C1_postbc.json`
- `offline_comparison.json`、`rollout_comparison.json`、`FINAL_AUDIT.json`
- role-correct dataset shards/manifest 与四个可复查 checkpoint
- focused regression：15 passed（Final/C1/B3 三模式及既有 B1/B3 contracts）

详细 per-episode mission-event records 保存在 `rollout_comparison.json`；`FINAL_AUDIT.json` 仅保存 contract、摘要、因果判决与产物引用。
