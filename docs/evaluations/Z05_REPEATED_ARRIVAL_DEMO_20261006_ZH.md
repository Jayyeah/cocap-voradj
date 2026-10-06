# EXP-PERSIST-01：Z05 Zero-shot Repeated Arrival Demo handoff

## 结果

`DEMO_CONTRACT_PASS`

只完成 3 regimes × 3 paired episodes 的 evaluation-only Demo。未训练、未做 optimizer/replay update，未改 checkpoint、reward、network、sensing 或 curriculum。成功率仅作为 Demo 逐局结果，不据此排序 policy。

## Branch / source / policy

| 项 | 事实 |
|---|---|
| evaluation branch | `evaluation/z05-repeated-arrival-demo-20261006` |
| implementation commit | `0472847` (`Add Z05 repeated-arrival demo evaluator`) |
| base branch / HEAD | `evaluation/iqn-z05-independent-20260923` @ `1a9da054ecb5652726005d00d08ab05a2188dd8e` |
| checkpoint | `/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05/stages/stage3/training/checkpoints/step_600000.pt` |
| selected step / SHA256 | `600000` / `8ee5c162c32883984f72aa4be4b86e82338ae1e8d8c912011d181987a476d095` |
| formal selection report | `/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05/stages/stage3/selection_report.json` |
| selection report SHA256 | `83a056abcdaf65e95529c927e55c5db361f19682ce29d29321fbc526fdcea134` |
| Stage3 config | `configs/experiments/iqn_z_unified_decay_curriculum_20260919/z05_stage3_12p3e3obs_700k.yaml` |
| Stage3 config SHA256 | `e7c180a5bdc61ae10a29c4b9ea571b80130cb0f49eb79a202ace8cc1403c0477` |
| fixed contract | 12 pursuers, 3 target slots/wave, 3 obstacles, 120×120, friend token cap 8, NormSense V2, synchronized_swept_v1, AW9, decision_dt 0.5 s, 3000 global steps, 700-step post-capture window |
| paired seeds | initial-world base `2026100601`; target-refresh RNG base `2026800601`; same episode indices across regimes |

Target refresh streams are paired. Delay sampling uses a separate per-episode stream. Validity rejection can consume different numbers of target-refresh draws, so Wave2/3 coordinates are not claimed to match across regimes.

## 实现

- 新增 `tools/run_iqn_repeated_arrival_demo_20261006.py`：沿用注册的 IQN midpoint-32 greedy action、NormSense V2 runtime、Stage3 map_random 目标采样规则和 canonical strict-CE recovery；只在 evaluator 中加入 repeated-arrival 调度与固定 target slot 刷新。
- Pursuer 的位置、速度、朝向和 active/deactivated 状态连续；obstacle layout、global step、Z 状态与更新计数连续。target slot 0–2 按 generation 复用。每波只清理捕获、coverage/recovery 与 capture timer 元数据。
- 新增 `tests/test_iqn_repeated_arrival_demo_20261006.py`，覆盖 A/B/C 决策边界、独立 delay stream、target slot 复用、初始间距、pursuer/obstacle/Z 连续性和 lifecycle timer 清理。
- 首次运行发现并修正两项 evaluator 问题：PERSIST-A 不得由 Z-clear 触发；轨迹文件必须在逐步追加前创建。另将 PERSIST-C delay stream 改为每局一次初始化、每波顺序抽样。

测试：`3 passed`（新合同测试）；`1 passed`（IQN deterministic evaluator）；`22 passed`（IQN Z curriculum 与 Z-v2 contracts）。`git diff --check` 通过。

## A/B/C Demo

时间统计为 mean / median / p90。每项均按名称、含义和方向报告。只有 3 episodes/regime，结果用于逐局说明，不作强统计结论。

| 指标 | PERSIST-A | PERSIST-B | PERSIST-C |
|---|---:|---:|---:|
| waves_captured `[0,3]` ↑，总数 / 平均 | 7/9 / 2.33 | 9/9 / 3.00 | 9/9 / 3.00 |
| waves_safe_completed `[0,3]` ↑，总数 / 平均 | 5/9 / 1.67 | 0/9 / 0.00 | 0/9 / 0.00 |
| all_3_captured ↑ | 2/3 | 3/3 | 3/3 |
| all_3_safe_complete ↑ | 1/3 | 0/3 | 0/3 |
| total mission time ↓，秒 | 581.83 / 639.50 / 714.30 | 112.83 / 113.50 / 116.70 | 135.17 / 136.00 / 144.80 |
| cumulative collision ↓，事件数 mean / median / p90 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| normal capture success ↑，逐波 | 7/7 | 9/9 | 9/9 |
| capture success ↑，逐波 | 7/7 | 9/9 | 9/9 |
| capture time ↓，秒 mean / median / p90 | 29.07 / 26.50 / 38.40 | 35.61 / 34.00 / 48.50 | 39.11 / 42.00 / 49.20 |
| first all-Z-zero time diagnostic，capture 后秒 mean / median / p90 | 1.50 / 1.50 / 1.50 | 1.50 / 1.50 / 1.50 | 1.50 / 1.50 / 1.50 |
| Z release time diagnostic，capture 后秒 mean / median / p90 | 1.50 / 1.50 / 1.50 | 1.50 / 1.50 / 1.50 | 1.50 / 1.50 / 1.50 |
| CE recovery success ↑，逐波 | 5/7 | 0/9 | 0/9 |
| recovery time ↓，秒 mean / median / p90 | 167.90 / 179.50 / 232.70（n=5） | 未完成（n=0） | 未完成（n=0） |
| safe-complete ↑，逐波 | 5/7 | 0/9 | 0/9 |
| active pursuer count ↑，波末 mean / median / p90 | 12 / 12 / 12 | 12 / 12 / 12 | 12 / 12 / 12 |
| coverage debt diagnostic，秒 mean / median / p90 | 248.64 / 231.00 / 374.40 | 37.61 / 36.00 / 50.50 | 45.06 / 48.50 / 54.50 |
| cumulative coverage debt diagnostic，episode 秒 mean / median / p90 | 580.17 / 637.50 / 711.50 | 112.83 / 113.50 / 116.70 | 135.17 / 136.00 / 144.80 |

PERSIST-A 的 E0 在 700-step window 到期且未 safe-complete，因此按合同不刷新 Wave2。PERSIST-B/C 的 Wave2/3 在 recovery 完成前到达，`interrupted_by_next_arrival=true`；没有将未完成恢复记成 safe-complete。

## Per-wave trace

`cap` 是 wave 开始至末目标捕获的 steps；`Z0` 是 capture 后至 `max_i z_i == 0` 的 steps；`D` 是 PERSIST-C 抽样 delay（PERSIST-B 为 0）；`rec` 是 capture 后 strict CE recovery steps；`I` 表示被下一 arrival 中断；`alive` 是波末 active pursuer 数。

25 个已执行 wave 的 collision count 均为 0；逐波计数也记录在每个 episode 的 events JSON 中。

| Regime / episode / wave | cap | Z0 | D | rec | CE / safe | I | alive |
|---|---:|---:|---:|---:|---|---:|---:|
| A / E0 / W1 | 46 | 3 | — | — | 否 / 否 | 否 | 12 |
| A / E1 / W1 | 84 | 3 | — | 379 | 是 / 是 | 否 | 12 |
| A / E1 / W2 | 72 | 3 | — | 523 | 是 / 是 | 否 | 12 |
| A / E1 / W3 | 46 | 3 | — | 359 | 是 / 是 | 否 | 12 |
| A / E2 / W1 | 42 | 3 | — | 163 | 是 / 是 | 否 | 12 |
| A / E2 / W2 | 64 | 3 | — | 255 | 是 / 是 | 否 | 12 |
| A / E2 / W3 | 53 | 3 | — | — | 否 / 否 | 否 | 12 |
| B / E0 / W1 | 46 | 3 | 0 | — | 否 / 否 | 是 | 12 |
| B / E0 / W2 | 101 | 3 | 0 | — | 否 / 否 | 是 | 12 |
| B / E0 / W3 | 68 | 3 | 0 | — | 否 / 否 | 否 | 12 |
| B / E1 / W1 | 84 | 3 | 0 | — | 否 / 否 | 是 | 12 |
| B / E1 / W2 | 85 | 3 | 0 | — | 否 / 否 | 是 | 12 |
| B / E1 / W3 | 54 | 3 | 0 | — | 否 / 否 | 否 | 12 |
| B / E2 / W1 | 42 | 3 | 0 | — | 否 / 否 | 是 | 12 |
| B / E2 / W2 | 96 | 3 | 0 | — | 否 / 否 | 是 | 12 |
| B / E2 / W3 | 65 | 3 | 0 | — | 否 / 否 | 否 | 12 |
| C / E0 / W1 | 46 | 3 | 6 | — | 否 / 否 | 是 | 12 |
| C / E0 / W2 | 86 | 3 | 9 | — | 否 / 否 | 是 | 12 |
| C / E0 / W3 | 104 | 3 | 9 | — | 否 / 否 | 否 | 12 |
| C / E1 / W1 | 84 | 3 | 5 | — | 否 / 否 | 是 | 12 |
| C / E1 / W2 | 83 | 3 | 10 | — | 否 / 否 | 是 | 12 |
| C / E1 / W3 | 91 | 3 | 9 | — | 否 / 否 | 否 | 12 |
| C / E2 / W1 | 42 | 3 | 7 | — | 否 / 否 | 是 | 12 |
| C / E2 / W2 | 97 | 3 | 6 | — | 否 / 否 | 是 | 12 |
| C / E2 / W3 | 71 | 3 | 10 | — | 否 / 否 | 否 | 12 |

## Agent behavior reuse / transition

仅称 behavior reuse / behavior transition。每个 source-wave 的 capture/support participant 在后续波次再次成为 capture/support participant 的总计如下：

| Regime | Wave1 → Wave2 reuse | Wave1 → Wave3 reuse | Wave2 → Wave3 reuse |
|---|---:|---:|---:|
| PERSIST-A | 23/36 | 24/36 | 23/23 |
| PERSIST-B | 36/36 | 35/36 | 35/36 |
| PERSIST-C | 36/36 | 36/36 | 36/36 |

| Transition（跨相邻波、按 primary behavior） | A | B | C |
|---|---:|---:|---:|
| coverage → support | 1 | 0 | 0 |
| support → capture | 8 | 16 | 20 |
| capture → coverage | 0 | 0 | 0 |
| coverage → capture | 0 | 0 | 0 |

各 regime 的 agent-wave 行为标签汇总（允许同一 agent 同一 wave 有多个 flags）：

| Flag | A | B | C |
|---|---:|---:|---:|
| direct detector | 79 | 101 | 98 |
| evidence-informed / support | 50 | 80 | 87 |
| capture participant | 61 | 73 | 76 |
| coverage-only | 84 | 108 | 108 |
| collision/deactivated | 0 | 0 | 0 |

## Artifacts / contract diagnostics

结果目录：`/home/yjq/rl/CoCap1/z05-repeated-arrival-demo-20261006-final`

- `summary.json`：9 episode 汇总、指标定义与方向、逐波事件、行为复用、contract diagnostics。
- `episode_{regime}_{index}.trajectory.jsonl`：9/9 完整逐决策轨迹，包含全 pursuer/target 状态、Z、wave/generation、capture/collision、strict CE。
- `episode_{regime}_{index}.events.json`：9/9 per-wave event、target generation、Z diagnostics、agent behavior flags。
- `gifs/persist-a_representative.gif`、`gifs/persist-b_representative.gif`、`gifs/persist-c_representative.gif`。

9/9 episode 的条件式 Wave2/3 arrival、slot reuse、无 ghost/stale observation、Z update count、Z reactivation、pursuer physical continuity、obstacle continuity、dead-agent persistence、timer reset、wave_id/generation、finite values 均通过。Evaluator parameter updates = 0。Checkpoint SHA256 前后相同；加载后模型 state SHA256 前后相同。全部 9 局 collision count 为 0。

发现的运行语义：PERSIST-A 未在 recovery window 内 safe-complete 时应停止刷新；PERSIST-B/C 可以在 safe-complete 前刷新，并记录 recovery interrupted。此行为已由最终轨迹覆盖。未发现阻止 evaluator 正式运行的 contract blocker。

## Formal evaluation handoff

分类：`READY_FOR_FORMAL_REPEATED_ARRIVAL_EVAL`。本任务未启动 20×3 formal run。

建议命令（正式运行前由调度方复核资源与输出目录）：

```bash
python tools/run_iqn_repeated_arrival_demo_20261006.py \
  --config configs/experiments/iqn_z_unified_decay_curriculum_20260919/z05_stage3_12p3e3obs_700k.yaml \
  --selection-report /home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05/stages/stage3/selection_report.json \
  --expected-step 600000 \
  --episodes-per-regime 20 \
  --output <FORMAL_OUTPUT_DIR> \
  --device cpu
```
