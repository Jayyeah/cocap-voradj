# AC-2A.5 — Coverage / Recovery Bank Balance Audit

日期：2026-09-20

本轮只读取 AC-2A pilot、Final IQN trainer 合同和 canonical BC collector/evaluator；没有新 rollout、critic training、BC 修改或 AC-2B 正式采集。

## 1. Provenance

- branch：audit/ac-bc-critic-lineage-20260920
- HEAD：772ff649e5dae8699436690a0016e7673733ae43
- 通过显式 127.0.0.1:17892 mihomo 完成 git fetch --all --prune。
- local HEAD 与 origin/audit/ac-bc-critic-lineage-20260920 一致。
- pilot：artifacts/2026-09-20_ac2a/pilot_trajectory_bank_v2.npz
- pilot SHA256：348c17ea1fcfda02b99876afffa9345932d36c9a71b73382ac415eb458cc1bed

## 2. Episode length and active rows

Pilot 共 16 episodes、3,454 transitions、13,816 active-agent rows。每个 pilot transition 有 4 个 active pursuer rows；以下统计按 scene/policy mode 从 raw bank 直接计算。

| scene / mode | episodes | episode transitions | mean | median | min | max | active-agent rows |
|---|---:|---|---:|---:|---:|---:|---:|
| mixed argmax | 5 | 287, 148, 356, 144, 175 | 222.0 | 175 | 144 | 356 | 4,440 |
| mixed sampled | 5 | 334, 186, 213, 265, 230 | 245.6 | 230 | 186 | 334 | 4,912 |
| pure coverage argmax | 3 | 210, 190, 125 | 175.0 | 190 | 125 | 210 | 2,100 |
| pure coverage sampled | 3 | 304, 174, 113 | 197.0 | 174 | 113 | 304 | 2,364 |

合并后 mixed 平均 233.8 transitions / 935.2 active rows，pure coverage 平均 186.0 transitions / 744.0 active rows。pure coverage 确实较短，因此 episode-count 比例不能直接代表 row 比例。

## 3. Final IQN four-class replay mapping

Final trainer 的正式分类在 src/cocap_voradj/training/trainer.py::_voradj_replay_class：

- phase=post_capture → post_capture_real
- phase=pure_coverage → recovery_pure
- 其余 phase 中，task_label=capture → pursuing
- 其余 pre-capture → pre_capture_cover

当前 canonical observation contract 为 nearest_vector_robot_oob、9 维 local_self；src/cocap_voradj/envs/voronoi_adjacency.py::_pack_agent_obs 将 effective is_pursuing 放在 local_self[...,8]。因此 pilot raw bank 能严格重建上述分类，不需要从 global state 猜角色。active rows 上该字段只有 0/1，非二值为 0。

| class | Final IQN update ratio | AC-2A pilot raw rows | AC-2A raw-row ratio | difference |
|---|---:|---:|---:|---:|
| pursuing | 50.0% | 1,081 | 7.82% | −42.18 pp |
| pre_capture_cover | 12.5% | 2,479 | 17.94% | +5.44 pp |
| post_capture_real | 25.0% | 5,792 | 41.92% | +16.92 pp |
| recovery_pure | 12.5% | 4,464 | 32.31% | +19.81 pp |

结论：pilot 的 strong-BC natural visitation 没有低估 coverage/recovery；相反，它明显低估 pursuing，并相对过采样 post-capture / pure recovery。这里不能把两列当作同一分布：Final IQN 比例是 training replay minibatch/update sampling；AC-2A 比例是 frozen BC actor 的 raw behavior visitation。

## 4. Capture / recovery event rows

复用 AC-2A/A0/A1 的 phase/event 定义：

| event/phase | active-agent rows |
|---|---:|
| capture transition（normal_capture；stationary/capture-terminal 均 0） | 40 |
| early recovery | 1,600 |
| late recovery | 4,192 |
| pure coverage | 4,464 |

early recovery 仍是旧定义：post_capture 且 0 <= timestep - capture_step <= 40；没有重新定义窗口。

## 5. Canonical BC coverage/recovery initialization

### Final IQN trainer configuration

Final 配置 configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/common.yaml 和 resolved artifact artifacts/2026-09-08_contract_parity/canonical_main_runtime.json 明确包含：

    capture_state_pool_capacity = 1000
    captured_state_ratio = 0.75
    map_random_ratio_within_non_capture = 0.5

src/cocap_voradj/training/trainer.py 的 reset 逻辑是：当 capture pool 非空时，75% 取 capture snapshot；剩余 25% 中，50% map-random、50% synthetic inner cluster。因此配置意图为：

    75% capture snapshot
    12.5% ordinary map random
    12.5% synthetic inner cluster

但 pool 为空时会 fallback，故这是 conditional intended ratio，不是整个训练生命周期的无条件实际比例。

### Canonical strong BC collector/evaluator

canonical Full-Task BC 使用：

- collector：tools/collect_forward_final_dataset_20260908.py
- evaluator/runtime：tools/run_forward_final_bridge_20260908.py、src/cocap_voradj/training/forward_final.py
- bridge ledger：docs/FORWARD_FINAL_MAPPO_BRIDGE_LEDGER_20260908_ZH.md

collector 是 alternating mixed / pure-coverage 的完整 episode collection；voradj_coverage resolved task 使用 inner_random_cluster。它没有实例化 CoCapTrainer.recovery_init_pool，也没有执行 Final IQN trainer 的 capture-snapshot/map-random recovery reset。mixed 中的 recovery 是同一 episode 内真实 capture 后续状态。

因此本轮分类为：

    BC_COVERAGE_INIT_DIFFERS

更准确地说：Final IQN trainer 的 75/12.5/12.5 配置存在且代码可证，但 canonical strong BC collector 的 pure-coverage/recovery initialization 并未使用它。不能把这三个比例追认成 BC visitation 合同，也不应因此修改 canonical BC。

## 6. Formal-bank quota estimates

按 pilot 均值线性外推；不是新采集，也不是正式样本量承诺。

| scheme | mixed / pure episodes | transitions | active rows | pursuing | pre-capture cover | post-capture real | recovery pure | early recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 60 / 20 | 17,748 | 70,992 | 6,486 | 14,874 | 34,752 | 14,880 | 9,600 |
| B | 50 / 30 | 17,270 | 69,080 | 5,405 | 12,395 | 28,960 | 22,320 | 8,000 |
| C | 40 / 40 | 16,792 | 67,168 | 4,324 | 9,916 | 23,168 | 29,760 | 6,400 |

## 7. Scheme D — phase-quota adaptive proposal

不强迫 raw bank 复刻 64/16/32/16。建议先采一个 base：

    40 mixed + 20 pure coverage = 60 episodes

以每类至少 5,000 active rows 作为本轮规划用 operational floor，之后按 episode、每次 5 局补采：

- pursuing、pre_capture_cover 或 post_capture_real 不足：补 mixed；
- recovery_pure 不足：补 pure coverage；
- mixed 内优先补 pursuing，其次 post_capture_real，再其次 pre_capture_cover；
- 最多追加 20 mixed + 10 pure coverage，共 30 top-up episodes；
- 每次检查后若所有绝对 floor 已满足则停止，不为追求 replay 百分比继续采；
- train/validation/test 始终按 episode 隔离。

后续 critic training sampler 可以参考 IQN 的 64/16/32/16 做 phase-balanced minibatch sampling；raw bank 应保持 canonical BC natural visitation。test 应同时保留 natural-distribution metrics 与 phase-conditioned metrics，不能只报告 balanced sampler 上的分数。

## 8. Recommendation and stop

在 A/B/C 三个固定方案中，Scheme A 数据最健康：它保留最多 mixed exposure，提供更多 pursuing / post-capture / early-recovery rows，同时 20 个 pure-coverage episodes 已提供 14,880 个 recovery-pure rows；Scheme C 会进一步降低 rare pursuing rows，并使 pure recovery 占比过高。

本轮未执行 Scheme A/B/C/D，也未进入 AC-2B。

机器结果：[bank_balance_analysis.json](../../artifacts/2026-09-20_ac2a5/bank_balance_analysis.json)
