# EXP-PERSIST-01：Z05 Zero-shot Repeated Arrival Demo handoff

## 分类与结论

分类：`DEMO_CONTRACT_PASS`

就绪状态：`READY_FOR_FORMAL_REPEATED_ARRIVAL_EVAL`

本轮只修正 evaluator 的 Wave3 endpoint 与统一 persistent endpoint 指标，并使用原始配对 seeds 重跑 3 regimes × 3 episodes。没有启动 20×3 formal run。

Wave1/Wave2 的 PERSIST-A/B/C arrival scheduling、checkpoint、spawn、reward、Z、NormSense、horizon 和 recovery window 均保持原合同。B/C 的 Wave3 arrival trigger 现在只记事件；环境继续正常推进到 recovery success 或终止条件。9 局未生成 Wave4。

## Branch、policy 与固定合同

| 项 | 值 |
|---|---|
| Branch / evaluator fix commit | `evaluation/z05-repeated-arrival-demo-20261006` / `35a9aa44f1899cd2cc7e094517de65f8ba29e6ed` |
| 原始基线 HEAD | `b0bbad78b8b13dfb183bb5b464f132800d287d31` |
| Stage3 selected checkpoint | `/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05/stages/stage3/training/checkpoints/step_600000.pt` |
| selected step / SHA256 | `600000` / `8ee5c162c32883984f72aa4be4b86e82338ae1e8d8c912011d181987a476d095` |
| 正式 selection report | `/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05/stages/stage3/selection_report.json` |
| Stage3 config | `configs/experiments/iqn_z_unified_decay_curriculum_20260919/z05_stage3_12p3e3obs_700k.yaml` |
| 固定合同 | 12 pursuers、3 target slots/wave、3 obstacles、120×120、friend-token cap 8、NormSense V2、synchronized_swept_v1、AW9、dt 0.5 s、global horizon 3000、post-capture window 700 |
| 配对 seeds | initial-world base `2026100601`；target-refresh RNG base `2026800601`；每 regime 3 局 |

正式 selection report 仍选 step 600000；文件 checkpoint SHA256 与期望值完全相同。模型 state SHA256 前后同为 `baf15ba48d7286f5fa298a6295e9ae941566f48991236bb3aec5f0e8aa705cc1`。运行仅使用 CPU；9/9 evaluator parameter updates = 0。

## 本轮 evaluator 修正

- Wave1/Wave2 的 arrival scheduling 不变。
- Wave3 不再因 B/C 的 Z-clear + delay arrival boundary 结束 episode；该 boundary 记录为 `arrival_trigger_step`，之后继续正常 policy/environment stepping。
- Wave3 在 canonical strict-CE recovery success、capture 后 700-step window 到期、global horizon、collision/terminal failure 或 active-pursuer 不足时终止；不生成 Wave4。
- 新增 `final_recovery_success`、`final_recovery_time`（秒，capture 至 Wave3 strict CE）、`final_safe_complete`、`no_disqualifying_failure` 和 `persistent_service_complete`。
- `persistent_service_complete = all_3_captured AND final_recovery_success AND no_disqualifying_failure`。本次 no-disqualifying-failure 排除累计 collision、终端失败及 active-pursuer loss。
- `all_3_safe_complete` 保留为逐局诊断；summary 对 PERSIST-B/C 不将其作为主性能数字，因为 Wave1/2 recovery 可能被下一 arrival 中断。Coverage debt 保留为 secondary diagnostic。

## 3×3 Demo endpoint 结果

时间单位为秒。final recovery time 对 Wave3 从 capture 到 canonical strict-CE recovery；失败时为 null。Collision 为 episode 累计事件数。

| Regime | all_3_captured ↑ | final_recovery_success ↑ | final_recovery_time ↓（逐局） | final_safe_complete ↑ | persistent_service_complete ↑ | collision ↓ | all_3_safe_complete |
|---|---:|---:|---|---:|---:|---:|---|
| PERSIST-A | 2/3 | 1/3 | null, 179.5, null | 1/3 | 1/3 | 0 | 1/3（A 可直接解释） |
| PERSIST-B | 3/3 | 3/3 | 40.0, 152.5, 90.5 | 3/3 | 3/3 | 0 | 0/3（诊断；W1/2 被中断） |
| PERSIST-C | 3/3 | 3/3 | 74.5, 81.5, 111.5 | 3/3 | 3/3 | 0 | 0/3（诊断；W1/2 被中断） |

PERSIST-A 的两个未完成 episode 分别在 Wave1 与 Wave3 的 700-step recovery window 到期；因此依合同没有继续刷新下一 wave。B/C Wave1/2 的 `interrupted_by_next_arrival=true`，这些 wave 不记为 safe-complete。所有 regime 的 active pursuer 数均保持 12。

## Wave3 trigger 后 recovery 轨迹

| Regime / episode | Wave3 capture step | first all-Z-zero | delay steps | arrival trigger step | final recovery step | trigger 后继续 steps | recovery time |
|---|---:|---:|---:|---:|---:|---:|---:|
| A / E1 | 1106 | 1109 | — | — | 1465 | — | 359 steps / 179.5 s |
| B / E0 | 223 | 226 | 0 | 227 | 303 | 76 | 80 steps / 40.0 s |
| B / E1 | 231 | 234 | 0 | 235 | 536 | 301 | 305 steps / 152.5 s |
| B / E2 | 211 | 214 | 0 | 215 | 392 | 177 | 181 steps / 90.5 s |
| C / E0 | 259 | 262 | 9 | 272 | 408 | 136 | 149 steps / 74.5 s |
| C / E1 | 281 | 284 | 9 | 294 | 444 | 150 | 163 steps / 81.5 s |
| C / E2 | 231 | 234 | 10 | 245 | 454 | 209 | 223 steps / 111.5 s |

A/E0 没有 Wave3；A/E2 Wave3 capture 后 700 steps 内未 recovery。B/C 六局全部在 Wave3 arrival trigger 后持续运行，并以 canonical recovery success 结束。9 局的 target generation wave IDs 均为 `[1,2,3]`。

Wave1/2 简要 traces（capture steps；Z-clear 后 arrival delay；recovery 是否被新 arrival 中断）：

| Regime | E0 W1 / W2 | E1 W1 / W2 | E2 W1 / W2 |
|---|---|---|---|
| A | 46 / 未生成 | 84 / 72；均 recovery | 42 / 64；均 recovery |
| B | 46 / 101；D=0，均中断 | 84 / 85；D=0，均中断 | 42 / 96；D=0，均中断 |
| C | 46 / 86；D=6/9，均中断 | 84 / 83；D=5/10，均中断 | 42 / 97；D=7/6，均中断 |

## Per-wave diagnostics 与 agent reuse

逐波 capture、recovery 和 coverage-debt mean/median/p90 均记录在 `summary.json`。逐波 capture 成功数分别为 A 7/7、B 9/9、C 9/9；per-wave collision 总数均为 0。严格 recovery success 数为 A 5/7、B 3/9、C 3/9；B/C 的前六个波次因下一 arrival 中断，Wave3 recovery 均成功。Coverage debt 仅作 secondary diagnostic：episode cumulative mean 为 A 580.17 s、B 204.67 s、C 217.17 s。

Cross-wave behavior reuse（capture/support participants）：

| Regime | W1→W2 | W1/W2→W3 |
|---|---:|---:|
| PERSIST-A | 23/36 | 47/59 |
| PERSIST-B | 36/36 | 70/72 |
| PERSIST-C | 36/36 | 72/72 |

行为转换计数（仅称 behavior transition）：coverage→support 为 A/B/C = 1/0/0；support→capture = 8/16/20；capture→coverage 与 coverage→capture 均为 0。各逐局 agent flags 和 target participant 信息保存在 per-episode event JSON 中。

## Contract diagnostics / tests

9/9 contract diagnostics 全部通过：Wave2/3 conditional spawn、target-slot reuse、无 ghost target/stale observation、Z update count、Wave2/3 后 Z 可重新激活、pursuer physical state 连续、obstacle 不变、dead agent 不复活、capture/recovery timer 清理、wave_id/generation 正确、finite values、无 Wave4、evaluator parameter updates = 0。Checkpoint SHA256 与 model state SHA256 前后不变。

新增 focused tests：`tests/test_iqn_repeated_arrival_demo_20261006.py` 3 passed。连同 IQN deterministic、Z curriculum、Z-v2 regression tests，共 26 passed。最终逐局 endpoint 审计确认：global step ≤3000；Wave3 recovery success 均在 700 steps 内；所有 recovery-window-expired episode 均已达到 capture+700；B/C trigger 后 episode 至少继续 76 steps；persistent service 布尔公式逐局一致。

结果目录：`/home/yjq/rl/CoCap1/z05-repeated-arrival-demo-20261006-final-recovery`

- `summary.json`：9 局汇总、每项指标定义和方向、checkpoint/runtime hash、contract diagnostics。
- `episode_*.trajectory.jsonl` 与 `episode_*.events.json`：9/9 完整轨迹与逐波事件/Z/target generation。
- GIF：`gifs/persist-a_representative.gif`、`gifs/persist-b_representative.gif`、`gifs/persist-c_representative.gif`。

## Formal handoff

建议正式命令（本轮未执行）：

```bash
python tools/run_iqn_repeated_arrival_demo_20261006.py \
  --config configs/experiments/iqn_z_unified_decay_curriculum_20260919/z05_stage3_12p3e3obs_700k.yaml \
  --selection-report /home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05/stages/stage3/selection_report.json \
  --expected-step 600000 \
  --episodes-per-regime 20 \
  --output <FORMAL_OUTPUT_DIR> \
  --device cpu
```

按本轮 gate，Demo 已达到 `READY_FOR_FORMAL_REPEATED_ARRIVAL_EVAL`。formal 20×3 evaluation 未启动。
