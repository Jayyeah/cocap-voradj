# B0 — Final IQN Reference（2026-09-17）

状态：**冻结 / PASS**。本轮复用已有可信 matched artifact，不重复大规模 teacher rollout。

## Teacher

- checkpoint：`artifacts/2026-08-04_crms_vctls_ce_final/checkpoints/stage1_4v1_step_2000000.pt`
- SHA-256：`2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89`
- network：`voradj_single_head` IQN；hidden 256；Transformer 8 heads / 4 layers；32 quantiles；64 cosine features；AW9 action head。
- config：self 9；pursuer 8×7；evader 8×7；obstacle 5×5；`max_pursuers=8`；`max_evaders=8`；`max_obstacles=5`；`pursuing_embed_dim=8`。

## Contract

- observation：robot-frame / Final VorAdj/VCT-LS sensing；self 9；friendly token 7；enemy token 7；obstacle token 5；无 unconditional global enemy input。
- sensing：当前 Final local VorAdj/VCT-LS 与 friendly adjacency 语义保持不变；本 B0 不改拓扑或通信半径。
- AW9：`a∈{-0.4,0,0.4}`、`w∈{-π/6,0,π/6}`，按 `a` 外层、`w` 内层排列为 9 类。
- reward：沿用 Final CR-MS / support / coverage / CE-PBRS / capture / post-capture recovery contract；B0/B1 均不改 reward。
- coverage/capture/support inputs：coverage centroid / VorAdj features、local enemy/obstacle sensing、capture direct target、support candidate 与 K10 effective-role metadata；这些标签只在 policy bit 删除前后作为对照或 evaluator/replay metadata 保留。

## `is_pursuing` reference path

1. `_has_enemy_neighbor()` 从 capture/VCT-LS adjacency 计算 raw pursuit evidence。
2. `_effective_is_pursuing()` 结合 capture raw label、K10 release delay 与 `_pursuing_release_counters` 得到 effective state；reset 时清空，step 后由 `_update_effective_pursuing_flags()` 更新。
3. legacy policy observation 在 `_pack_agent_obs()` 中把 effective bit 放在 self 最后一列，并放在 friend token 最后一列；friend token 还按 effective role 优先排序。
4. `CoCapIQN.voradj_single_feature()` 读取 self 最后一列，经 `pursuing_embed(1→8)` 后与 256-d fused feature 拼接，再过 `single_action_feature(264→256)`。
5. reward / evaluator / replay 使用的是独立的 `task_label`、`reward_role`、`effective_pursuing`、`support_candidate` 等 metadata；B1 只删除 policy observation/late-fusion shortcut，不删除这些分析和奖励语义。

## Matched reference evidence

- `artifacts/2026-09-08_forward_final/c0_formal20/report.json`：mixed 20/20 capture，normal capture 1.0，collision 0，capture→CE mean 70.475 s；coverage CE/safe 1.0，collision 0。
- `artifacts/2026-09-08_forward_final/c3_formal100/iqn_greedy/report.json`：mixed capture 1.0，normal capture 1.0，CE 0.99，safe 0.99，collision 0.01，mean capture 43.37 s、capture→CE 71.11 s；coverage CE/safe 0.99，collision 0，mean mission 69.35 s。

这里的 pure capture evidence 是 Final matched evaluator 的 mixed pre-capture/capture prefix；该 Final contract 没有独立 standalone pure-capture scene。pure coverage 与 mixed lifecycle/recovery 均有正式 artifact 覆盖。

后续 B1 以及任何 cleanup 均以本文件 checkpoint hash/config/contract 为 reference。
