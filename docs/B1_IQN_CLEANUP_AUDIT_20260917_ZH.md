# B1 — No-is-pursuing Student Audit（2026-09-17）

状态：**`B1_FAIL_HIDDEN_STATE_REQUIRED`**。本结论只针对 policy shortcut；reward、replay/evaluator role metadata 未删除或改造。

## Intervention

- teacher：B0 frozen Final IQN，SHA-256 `2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89`。
- student config：self 8；friendly 8×6；enemy 8×7；obstacle 5×5；Transformer 256/8/4；IQN 32 quantiles / 64 cosine；`include_is_pursuing=false`、`pursuing_embed_dim=0`。
- 删除：self role column、friend role column、role-dependent friend ordering、`pursuing_embed` late fusion。
- transfer：除 3 个输入相关矩阵按列裁剪外，entity encoders、Transformer、target/summary path、action head 全部复用；无 `z_i`、无 global 信息、无新 role bit。
- optimization：teacher frozen；现有 C1 teacher data；episode-pair held-out split；2 epochs、16,384 条固定 phase/role-stratified optimization rows；KL(teacher Q || student Q)+centered-Q smooth-L1；无 reward、PPO、scratch RL。

## Offline distillation / BC

完整 C1 数据 146,612 rows，held-out validation 30,296 rows；student checkpoint SHA-256：`dbbe365c94a5d1fa94313e9f9d8b2238b36e19284c15cd134e4595c9f5df85b9`。

Full C1 rows 的结果：

| subset | rows | action agreement | categorical KL | Q regret | pairwise rank |
|---|---:|---:|---:|---:|---:|
| all | 146,612 | 0.3738 | 1.6593 | 0.0753 | 0.7341 |
| pre-capture | 32,132 | 0.3982 | 1.7131 | 0.2349 | 0.7744 |
| post-capture | 59,036 | 0.3652 | 1.6474 | 0.0361 | 0.7211 |
| pure coverage | 55,444 | 0.3689 | 1.6409 | 0.0245 | 0.7244 |
| direct role | 9,473 | 0.4944 | 1.4682 | 0.3434 | 0.8171 |
| support role | 9,999 | 0.2773 | 2.0242 | 0.4052 | 0.7680 |
| coverage role | 127,140 | 0.3725 | 1.6449 | 0.0293 | 0.7252 |

最明显的 phase-specific collapse 是 support（0.2773）以及 post-capture/recovery（0.3652）；不是只发生在 direct pursuit。

## Matched rollout

CPU-only、同一 Final scene contract，3 mixed + 3 coverage：

- mixed：capture 3/3，normal capture 3/3，collision 0/3；capture 后 CE/recovery 0/3，safe complete 0/3；capture time mean 528 s。
- pure coverage：capture 0/3（不适用），CE 0/3，safe complete 0/3，collision 0/3；3/3 到 1,500 s 截止。
- student 在自己轨迹上与 frozen teacher 的 action agreement：54,672 decisions，0.1722。

B0 的 mixed capture/reference 能力没有被 student 保持为完整 lifecycle；coverage/recovery 更早失败。

## Counterfactual / aliasing

在 146,612 rows 中，`pursuing=false/true` cross-role rows 为 137,139 / 9,473。对删除 role 列后的 159-d physical local vector 做最近邻：

- strict RMSE ≤ 0.05：0 pair，不能据此声称“几乎完全相同”的 aliasing。
- calibrated near-observation RMSE ≤ 0.10：118 pairs，其中 87 个 teacher action mismatch，mismatch rate 73.7%。RMSE≤0.15 时为 3,338 pairs / 75.4% mismatch。
- 87 个近邻 mismatch 全部落在 `target_recent_memory_or_local_target_evidence`：对应 pair 的 direct-target evidence mask 不同；未观察到由 neighbor-target mask 主导的样本。
- teacher own self-bit toggle（2,048 counterfactual rows）action flip 76.2%，mean absolute Q delta 23.71；friend role zeroing flip 8.5%，Q delta 6.57。

解释边界：counterfactual flip 证明 Final policy 对人工 bit 高度敏感，但单独不等于真实环境 aliasing；真正支持 hidden-state 判断的是 RMSE≤0.10 的 cross-role pair 及其 action mismatch。当前最像缺失的是 target recent memory / local target evidence（例如 target 是否刚刚被本地 evidence 解析、capture 后 release 的历史），不是可直接恢复的 role label。

## Gate

删除 `is_pursuing` 后整体和 phase agreement 均未达到基本复现，support/recovery 出现明显 collapse；同时 calibrated near-observation pair 存在高 mismatch。因此本轮分类为 **`B1_FAIL_HIDDEN_STATE_REQUIRED`**。不把 bit 加回 policy，也不因该结果进入 B2；下一阶段必须先由人工审阅 hidden-state 信息来源，再决定是否设计未来 `z_i`，本 worktree 不实现 B2/B3/B4。

完整机器可读结果见 `artifacts/2026-09-17_iqn_cleanup_b1/B1_FINAL_AUDIT.json`、`student_report.json`、`aliasing_audit.json`、`rollout_report.json`。
