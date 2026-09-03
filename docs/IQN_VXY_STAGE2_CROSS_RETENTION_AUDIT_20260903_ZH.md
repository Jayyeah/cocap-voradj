# IQN-VXY Stage2 cross-retention 与 support-latency 审计（2026-09-03）

## 1. 执行范围

本轮只做冻结权重评估，不训练、不续训、不改 reward。三个 checkpoint 全部以 SHA-256 fail-closed 校验后，放入同一个 Stage2 `8p2e2obs` 正式环境：

| 标签 | checkpoint | SHA-256 |
| --- | --- | --- |
| Stage2 parent | `step_600000.pt` | `1388ac6813fa6c6ee9f0cc6041446e04556a01ef611a07d0afe3071340a86940` |
| Stage3 selected | `step_150000.pt` | `b33dce2484f3394d1aadb13cd02ca5085929011a8bf7a2a9742ba75c0dba419b` |
| Pure-Coverage warm selected | `step_75000.pt` | `a7e2e930ee9c86fa2156d4d2518c9e060f2b69053d9e31235a2cc0ffb72138d0` |

Stage3 的 `pretrained_load.json` 明确指向 Stage2 selected 600k，`loaded_count=119`、`adapted_keys=[]`、`skipped_keys=[]`。这证明 Stage3 的模型 lineage 为 VXY Stage2 600k → VXY Stage3 150k；不能把它写成继承 Final-AW checkpoint。

## 2. 2026-09-03 合同结论复核

“当前 IQN-VXY 对齐 2026-08-04 Final IQN-AW”只在**课程/环境/reward/网络及训练合同**层面成立，严格基准是：

`configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/`

其 artifact release 为 `artifacts/2026-08-04_crms_vctls_ce_final/manifest.json`。VXY 首轮的唯一主要实验变量是 AW9 → body-frame rate-limited VXY9；checkpoint 血缘仍是独立的 VXY 三阶段链。Final-AW artifact 还记录：发布展示的 Stage2 selected 为300k，而历史 Stage3 实际从当时选中的 Stage2 500k warm-start，二者不能混写。

已由配置、实现和合同测试共同确认：

- direct detector 使用 `capture_reward_mode: ring_importance_ms_v0`；
- support reward 开启，capture/coverage 权重各 `.5`；
- capture 半项是 `neighbor_visible + approach_only`，只追逐一跳 pursuing 邻居直接可见的 enemy，不暴露全局 enemy 坐标；
- 因此 support 精确为 `0.5 × neighbor-visible approach-only + 0.5 × CE`；
- 旧 `omega_approach/omega_mean_shift/omega_front` 全部为 `0`；
- replay/recovery、mixed episode、CE、CR-MS、VCT-LS 与三阶段课程均属于 integrated contract。

对应定向合同回归为 `18 passed`；第三方 protobuf 仅有2条 deprecation warning。

## 3. Formal100 评估合同

评估器为 `tools/evaluate_vxy_stage2_cross_retention_20260903.py`，执行时 SHA-256 为 `3154663b57b1af1749c52acf669812ee4219f7ffca27c6518e4831456036f894`。它不会创建 optimizer、调用 backward/step 或写 checkpoint；manifest 显式记录 `training_or_weight_updates=false`。

共同冻结项：

- Stage2 `8 pursuers / 2 evaders / 2 obstacles`、legacy-end-step collision、VXY9；
- `model.eval()`、`torch.no_grad()`、`epsilon=0`、fixed-midpoint 32 quantiles；
- 三模型、三场景严格复用 seeds `2026083201..2026083300`；
- capture / standalone coverage / mixed 各100回合；
- horizon 分别为1000 / 1500 / 2500；
- pure coverage 按全局 episode index 精确交替 map-random / inner-random-cluster，各50回合；
- paired comparison 逐 seed 对齐，并对差值做10,000次 paired bootstrap 95% interval。

输出位于 `artifacts/2026-09-03_iqn_vxy_stage2_cross_retention_formal100/`。逐 episode records 保留 capture type、collision type、CV、centroid、长度、post-capture survival/settling，以及 support 时序诊断。

## 4. 指标解释边界

多敌机环境中，捕获一个 enemy 会产生部分 `loose` event，但只有全部非碰撞 deactivated 才算 episode-level capture。评估器同时保留原始 `capture_types`，且 normal/stationary 正式率仅在整局 capture 成立后计数，避免把“捕获1/2”误报为成功。

support 时序使用和 reward 相同的 VCT-LS 私有解析路径：先找一跳 pursuing friend，再取该 friend 的 `_vct_ls_direct_enemy_ids_for_pursuer`。现有环境 `support_enemy_distance_progress` 聚合诊断仍沿用 raw Voronoi enemy edge，在 VCT-LS 下可能漏掉真实 neighbor-visible target；本轮不改环境/reward，只在只读评估器中按实际 reward target 重建距离 progress。

`rate-limit pressure` 定义为期望速度与当前速度误差超过单 decision acceleration budget；`near-hazard` 定义为最小 surface clearance ≤2m。二者都是行为相关 proxy，不能单独证明因果。`post_capture_settled` 也是 episode-end 低速诊断；Final 合同的 settling terminal/reward 仍关闭。

## 5. Formal100 结果

### 5.1 Capture-only

| 模型 | all-enemy capture | normal | stationary | collision | mean length |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage2 600k | `.97` | `.97` | `0` | `.02` | `235.95` |
| Stage3 150k → Stage2 | `.98` | `.98` | `0` | `.02` | `269.27` |
| Pure-Coverage warm75 → Stage2 | `.80` | `.80` | `0` | `.13` | `428.95` |

Stage3 对 Stage2 的同 seed paired capture 差值为 `+1pp`，10,000次 paired bootstrap 95% interval 为 `[-3,+5]pp`；collision 差值为 `0pp [-4,+4]pp`。两者都不支持“Stage3 学出了更强 8v2 capture”。相反，Stage3 平均多用 `33.32` steps，95% interval `[4.55,61.92]`，说明回迁 capture 略慢。

warm75 对 Stage2 的 capture 为 `-17pp [-25,-9]pp`，collision 为 `+11pp [+5,+18]pp`，length 为 `+193.00 [142.79,245.82]` steps。绝对 capture 保留80%，相对 parent rate 保留 `80/97=82.47%`。

所有模型 stationary capture 均为0。warm75 的100个 capture-only 回合中95个至少捕获过一个 enemy，但只有80个完成2/2；另15个是部分捕获后失败。干扰因而不只是“完全不接敌”，还包含第二个 enemy 收尾能力下降。

### 5.2 Standalone coverage sanity

| 模型 | strict CE | CV≤.15 | collision | final / best CV | centroid RMS / max | mean length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage2 600k | `.05` | `.20` | `.01` | `.2351/.2329` | `.0610/.0849` | `1415.19` |
| Stage3 150k → Stage2 | `1.00` | `.49` | `0` | `.1555/.1462` | `.0386/.0580` | `81.12` |
| Pure-Coverage warm75 → Stage2 | `1.00` | `.79` | `0` | `.1220/.1144` | `.0312/.0489` | `76.30` |

Stage3 与 warm75 相对 Stage2 的 strict CE 均为 `+95pp [90,99]pp`。这里每个模型都严格包含50个 map-random 与50个 inner-random-cluster，避免旧多 worker 局部 index 造成初始化比例漂移。

### 5.3 Mixed capture→coverage

| 模型 | capture | strict CE | collision | survival（全体 / capture条件） | settled（全体 / capture条件） | mean length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage2 600k | `.98` | `.07` | `.06` | `.94/.9592` | `.94/.9592` | `766.84` |
| Stage3 150k → Stage2 | `.98` | `.97` | `.03` | `.97/.9898` | `0/0` | `344.38` |
| Pure-Coverage warm75 → Stage2 | `.82` | `.82` | `.15` | `.82/1.00` | `0/0` | `535.46` |

Stage3 对 Stage2：mixed capture `0pp [-4,+4]pp`，strict CE `+90pp [84,96]pp`，collision `-3pp [-9,+3]pp`。在98个 capture 回合内，Stage3 有97个完成CE（`.9898`），Stage2 只有7个（`.0714`）。因此核心问题的答案是：**Stage3 没有学出更强的8v2 capture，但明确学出了可回迁的 capture→coverage joint skill。**

warm75 对 Stage2：mixed capture `-16pp [-24,-9]pp`，strict CE `+75pp [66,83]pp`，collision `+9pp [+1,+18]pp`，全体 survival `-12pp [-21,-3]pp`。在它成功 capture 的82个回合中，82个全部 CE、全部 post-capture survival；但18个未捕获，其中13个已有部分 normal event。故 mixed CE 的提升是真实且巨大，却由明显的 capture/safety interference 换来。结论是**显著但非完全的 selective catastrophic interference**，不是“capture 完全遗忘”。mixed capture 相对 parent rate 保留 `82/98=83.67%`。

Stage2 的 settled 高而 Stage3/warm75 为0，不应解释成后两者 recovery 失败：settling reward/terminal 在 Final 合同中关闭，Stage3/warm75 常在CE达标后快速结束；Stage2 多数未达CE而跑满 post-capture window，低速结束。该列只作为用户要求的诊断保留，不参与模型优劣 gate。

## 6. Support-latency 行为审计

下表使用 capture-only 100回合；进度单位为单 decision step 的距离减少，正数表示靠近。

| 模型 | direct / support-target 首现 | support→direct / pursuing | direct / support speed | support→enemy / friend progress | direct / support rate pressure | reward异号竞争 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage2 600k | `3.95/3.95` | `84.19/84.19` | `1.207/1.032` | `.200/.084` | `.988/.948` | `.637` |
| Stage3 150k | `3.86/3.86` | `85.21/85.29` | `1.199/1.079` | `.167/.062` | `.989/.973` | `.606` |
| warm75 | `3.96/3.96` | `115.02/115.09` | `1.194/.993` | `.075/.015` | `.990/.985` | `.545` |

| 模型 | 2+ / 3+ episode rate | 2+ / 3+ time fraction | 首次2+ / 3+（出现条件下） | 2+ / 3+ mean max hold |
| --- | ---: | ---: | ---: | ---: |
| Stage2 600k | `.90/.25` | `.0761/.00636` | `72.66/126.64` | `8.26/.95` |
| Stage3 150k | `.98/.40` | `.1439/.01283` | `76.03/197.90` | `16.21/2.49` |
| warm75 | `.86/.15` | `.0492/.00236` | `107.49/299.20` | `7.94/.62` |

“support 补位慢”客观存在，但需要分层表述：

1. **不是首次局部信息到达慢。** 三模型中 first direct、first pursuing neighbor、first neighbor-visible target 的 episode-level earliest timing 都约为第4步且相同；没有证据支持初始 local information delay 是主因。
2. **support 角色推进确实慢于 direct。** support 平均速度比 direct 低约10%–17%，且转为 direct/pursuing 需约84–115步；warm75 的升级比 parent 再慢约31步，enemy/friend progress 也分别降到 parent 的约37%/18%。
3. **`.5 approach + .5 CE` 竞争是强相关候选。** 两分量符号相反的 support agent-step 比例为 `.545–.637`；这证明优化方向经常竞争，但本轮是观察性审计，尚不能单独证明它造成延迟。
4. **VXY rate-limit 是普遍约束，不是 support 独有故障。** direct/support pressure 都很高（`.948–.990`），能解释整体响应惯性；direct 同样接近饱和，所以不能把 direct-support gap 全归因于 rate-limit。
5. **碰撞规避保守不是主解释。** support 在 clearance≤2m 的 agent-step 仅约 `0–.14%`；该轻量 proxy 没有显示大多数慢补位发生在近障/近碰撞状态。

Stage3 的2+/3+出现率、时间占比和 hold 明显高于 Stage2，但首次3+并未更早、capture还略慢。这与“更持久地组织 coverage-like geometry，而非更快完成 capture”一致。warm75 则在升级、progress、ring entry 和最终2/2 capture 上同步退化，和 selective interference 一致。

## 7. 证据完整性与 paired GIF

新 evaluator 在 Stage2 parent 的前20个相同 seeds 上，与既有 Stage2 formal20 的 capture/coverage/mix 四个核心字段（capture、CE、collision、length）逐 episode 比较，三个场景均为 `0 mismatch`。formal100 manifest 状态为 `complete`，三模型×三场景均恰好100条，且 `FORMAL_DONE` 已写入。

行为 GIF 固定取 seeds `2026083201..2026083203`，三个模型都取完全相同的前三个 capture 回合，不按成功或失败事后筛选。输出位于：

`artifacts/2026-09-01_iqn_vxy_full_rollouts/paired_stage2_cross_retention_20260903/`

最终验收为三个模型各3张、共9张，seed 和 episode JSON 完全对应。Stage2 与 Stage3 的前三局都是2/2成功；warm75 的 seed `2026083201` 只捕获1/2并跑满1000步，后两局成功，因此样本实际包含失败例，不是成功 GIF selection。GIF 只用于解释；正式率和 paired 结论全部来自100回合 JSON。

## 8. 后续方向（仅 TODO，本轮未启动）

### 8.1 Matched Final IQN-AW / IQN-VXY

严格基准固定为 `configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/`：4v1 scratch2M → 8v2 warm700k → 12v3 warm700k，包含 CR-MS + VCT-LS + CE + support + mixed episode + replay/recovery。首轮只把 AW9 替换为 body-frame rate-limited VXY9，后续按 matched seeds/cadence 比较学习速度、best/final、capture、pure CE、mixed CE、collision、sample efficiency 和多 stage 稳定性。状态：`TODO / NOT_STARTED`。

### 8.2 IQN → MAPPO policy pretraining / distillation

从强 IQN-AW checkpoint 收集 local observation + greedy AW9 action dataset；以完全相同 Legacy feature backbone 监督预训练 MAPPO-9 Actor；先做 imitation policy formal eval，再做 MAPPO-v2 fine-tune；成对比较 fine-tune 前后 capture、collision 与策略退化速度，以区分“从零发现困难”和“PPO 更新破坏好策略”。状态：`TODO / NOT_STARTED`。

本轮没有为这两个方向创建训练进程、配置队列或 supervisor。
