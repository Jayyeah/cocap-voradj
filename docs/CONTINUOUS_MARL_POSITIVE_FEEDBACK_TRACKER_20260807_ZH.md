# CoCap-VorAdj 连续 MARL 正反馈阶梯专用实验台账

> 建立日期：2026-08-07  
> 最高执行合同：`docs/COCAP_CONTINUOUS_MARL_POSITIVE_FEEDBACK_LADDER_20260807_ZH.md`  
> Codex Agent 提示词：`docs/COCAP_CODEX_AGENT_POSITIVE_FEEDBACK_LADDER_PROMPT_20260807_ZH.md`  
> 分支：`continuous/masac-ctde-contract-20260806`  
> 本台账只记录本文定义的阶梯路线；旧台账 `docs/CONTINUOUS_ACTION_MARL_EXPERIMENT_TRACKER_20260804_ZH.md` 保留为历史，不再作为当前路线唯一依据。

---

## 0. 当前状态（每次更新必须保持最新）

- 当前 active stage：**reward 第一批三线并行（A1/A2/A3，基于 4B 场景，seed1 25k）**；4B/4C 原配置 FAIL 记录保留
- 当前唯一 formal config：`configs/experiments/positive_feedback_ladder_20260807/stage4b_capture_aw.yaml` / `stage4c_capture_aw.yaml`（Stage3A/3B/4A configs 已 PASS，保留为成功锚点）
- 当前 action contract：连续 `acceleration_angular_velocity_body`，独立 box 边界 `a∈[-0.4,0.4]`、`w∈[-π/6,π/6]`（Stage 1 已严格等价）
- 当前 observation contract：Stage 0 使用旧 IQN VCT-LS robot-frame observation；Stage 2+ 目标为同一 robot-frame local observation，Actor 不得读取全局/oracle
- 当前 dynamics contract：`continuous_aw_v1`（显式 Euler、10 substeps、dt=0.05、decision_dt=0.5、v_max=3.0、drag=0.4/3、yaw 积分、legacy_random 初始化、碰撞整步检查；与旧 IQN 完全一致）
- 当前 reward contract：Stage 3A 使用 CE centroid energy + PBRS，speed weight=0（`[IQN-ALIGN]`；Stage 2 简化 reward 已退出）
- 最近 milestone：Stage 0/1/2/3A/3B/4A 均 PASS；Stage4B/4C seed1 已并行启动
- 当前结论：连续 `(a,ω)` MASAC 已覆盖 capture 几何（stationary）与 pure coverage 无/有 obstacle 成功锚点
- 下一步唯一动作：**完成 Stage4B/4C 25k 训练与 gate 判定；各自 2-seed 验证**

---

## 0.1 跨线证据 TODO 修订（2026-08-07）

依据并行原线 `continuous/masac-ctde-contract-20260806` 最新结果：

- `ctde_pure_random_25k_v2`：world `[ax,ay]` pure-CE 25k 出现 4/4 positive signal → Stage6/7 不再严格串行；Stage5 anchor 后 Stage6A 与 Stage7A1 可并行。
- `ctde_pure_encoder_25k` / `ctde_pure_snapshot_25k`：teacher 不再默认；改为失败触发式；snapshot 优先用于 post-capture/mixed。
- `action_translation_gate.json`：FAILED GATE → action-translation BC 主线删除，标 NOT ACTIVE TODO。
- Stage7A1 首个目标：复现 `ctde_pure_random_25k_v2` 的 world-axay pure positive signal（actor-prior warmup）。
- uniform-disk warmup 降为二级诊断（Stage7A1b）。

## 0.2 [2026-08-08 PATCH] Stage4 诊断补丁（主线不变）

```text
[2026-08-08 PATCH]
1. Stage4 mainline unchanged.
2. E1 target_entropy=-4 plan paused pending entropy calibration.
3. Added action / geometry / reward / entropy / critic diagnostics.
4. Added legacy IQN scratch 25k/50k early-training reference.
```

- 现有 A1/A2/A3 训练与归因主线保持不变、不重排。
- E1 标记 `PAUSED — ENTROPY DIRECTION NOT JUSTIFIED`，由 E0（entropy calibration，只记录不调整）取代；方向待 E0 数据决定。
- 详细设计见 `docs/LADDER_STAGE4_REWARD_BATCH_ATTRIBUTION_20260808_ZH.md` §2/§3.5。

## 0.3 Stage4 Diagnostic Patch 2026-08-08（章节 A–H，逐步回填）

- A. Entropy calibration（E0）：alpha/log_alpha/alpha_loss、log_prob 分布、physical/normalized entropy、entropy residual、log_std_a/log_std_omega —— 接入训练 metrics。
- B. Action-component diagnostics：a/omega 拆分统计、speed 分位与占比；action_norm 降为辅助。
- C. Pursuit geometry diagnostics：d1–d4 排序距离、distance progress、radial closing velocity、fraction_closing。
- D. Reward density diagnostics：reward term 拆分 + nonzero/positive fraction。
- E. Replay state coverage：距离分桶 histogram + detected/undetected。
- F. Counterfactual critic Q-ranking：policy/random/seek 三动作 Q 排序。
- G. Paired deterministic/stochastic evaluation：同 seeds 双模式 20ep + paired delta。
- H. Legacy IQN scratch 25k/50k reference：原成功合同 scratch 早训时间尺度参照。

## 1. 执行纪律摘要

1. 一次只改一个核心变量；任何配置改动必须标注 `[IQN-ALIGN]` / `[ALGO-NECESSARY]` / `[DIAGNOSTIC-ADAPT]`。
2. 不修改旧 IQN 默认执行路径、不覆盖旧 checkpoint、不删除旧产物；新功能使用新 config/namespace/action mode。
3. 冻结参数不得未经用户确认修改：map、perception、reward 结构/主要权重、`a_max/ω_max/v_max/drag/dt/substeps/batch/gamma/update every/grad clip/local Actor/central twin critic/focal replay/speed reward=0/diagnostic cap=400`。
4. 所有 >25k 训练必须每 25k 原子保存完整 bundle，并立即回填本台账。
5. 每个 stage 结束必须单独输出完成说明并创建 `STAGE_<ID>_COMPLETION.md`。

---

## 2. 阶梯路线与 Gate 摘要

```text
Stage 0 旧 IQN 基线
  -> Stage 1 连续 (a,ω) bridge 等价
  -> Stage 2 MASAC+(a,ω) 极简单任务（1p1e stationary/global/no-obs）
  -> Stage 3A/3B pure coverage（0 obstacle -> 1 obstacle）
  -> Stage 4A-4E capture 难度阶梯
  -> Stage 5A/5B episode 双任务与 mixed capture->coverage
  -> Stage 6 body-frame [a_x,a_y]
  -> Stage 7 world-frame [a_x,a_y]
```

成功分级：Level A（完整 world-frame mixed）/ Level B（body-frame）/ Level C（可靠 continuous `(a,ω)`）/ Level D（明确乐观信号）。

---

## 3. Run 记录

### 3.1 Stage 0：旧 IQN 基线复现

```text
stage: 0
run_id: STAGE0_IQN_BASELINE_20260807
status: PASS
date: 2026-08-07
branch: continuous/masac-ctde-contract-20260806
commit: 9827770 + 本日 ladder 工具/配置改动
git status: dirty（新增阶梯工具/配置/台账；旧线未改）
config path/hash: configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/stage1_4p1e1obs_scratch2m.yaml
seed: 2026081201（沿用历史 20-rollout seed）
initialization: 旧 IQN checkpoint，无训练
teacher/snapshot hash: checkpoint SHA256 2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89
action contract: unicycle_discrete，9 动作 (a∈{-0.4,0,0.4}, w∈{-π/6,0,π/6})
observation contract: 旧 IQN VCT-LS robot-frame local observation
dynamics contract: map 120、4v1、1 obstacle、episode 1000/1200/1500、旧水阻
reward contract: 原 CR-MS + VCT-LS + CE reward
exact command: python3 tools/batch_rollouts.py --config configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/stage1_4p1e1obs_scratch2m.yaml --checkpoint artifacts/2026-08-02_three_line_stage_best_20rollout10gif/crms_supportapproach_4v1_s1_step_2000000/step_2000000.pt --output-root artifacts/2026-08-07_positive_feedback_ladder/stage0_iqn_baseline_20260807 --episodes 20 --gif-count 0 --scenarios capture coverage mix --seed 2026081201 --device cuda:1
PID/log: session 79028 / 完成
step: 旧 checkpoint step 2000000（只评估，不训练）
checkpoint paths: artifacts/2026-08-02_three_line_stage_best_20rollout10gif/crms_supportapproach_4v1_s1_step_2000000/step_2000000.pt
key metrics: capture 1.0/avg_steps 77.15；coverage 1.0/avg_final_CV 0.0493；mix episode 0.9/CV 0.95/collision 0.05
comparison: 与历史 20-rollout 完全同数量级（逐项一致）
decision: PASS，可进入 Stage 1
next action: Stage 1 完整 parity 报告
```

### 3.2 Stage 1：连续 (a,ω) bridge 等价验证

```text
stage: 1
run_id: STAGE1_AW_BRIDGE_PARITY_20260807
status: PASS（严格等价）
date: 2026-08-07
branch: continuous/masac-ctde-contract-20260806
commit: 9827770 + ladder 工具改动
config path/hash: 旧 stage1 YAML；桥接端使用 acceleration_angular_velocity_body + yaw.init=legacy_random
seed: 2026081201/2026081202/2026081203
initialization: 旧 IQN checkpoint，无训练
action contract: 旧离散 9 动作 -> 连续 (a,w) 独立 box
observation/dynamics/reward: 与 Stage 0 同一旧配置
exact command: python3 tools/run_stage1_aw_bridge_parity.py --seeds 2026081201 2026081202 2026081203 --max-steps 400 --device cuda:1 --out-root artifacts/2026-08-07_positive_feedback_ladder/stage1_aw_bridge_parity
PID/log: session 24963 / 完成
step: 每 run 最多 400 decision steps（实际 93–251）
key metrics: state_error_max mean 0.0、reward_abs_diff_max mean 0.0、event/capture/collision parity 9/9
decision: PASS；Bridge 采用 legacy-exact 显式 Euler + 整步碰撞语义，profile=continuous_aw_v1
next action: 启动 Stage 2
```

### 3.3 Stage 2：MASAC + 连续 (a,ω) 极简单任务

```text
stage: 2
run_id: STAGE2_SIMPLE_AW_20260807
status: PASS（seed1 30%、seed2 100%；seed3 于 14k 停止，保留为可选证据）
date: 2026-08-07
branch: continuous/masac-ctde-contract-20260806
commit: c84a5b4 / 24d5e26
config path/hash: configs/experiments/positive_feedback_ladder_20260807/stage2_simple_aw.yaml（hash 见 smoke report）
seed: 2026080701/02/03（正式 run）
initialization: scratch random-init
action contract: acceleration_angular_velocity_body, independent box, a_max=0.4, w_max=pi/6
observation contract: robot-frame local VCT-LS, global evader visibility=true
dynamics contract: continuous_aw_v1, legacy-exact
reward contract: [DIAGNOSTIC-ADAPT] legacy distance-progress + success + safety；退出条件为 Stage 2 gate PASS
exact command:
  seed1: PYTHONPATH=src:. python3 tools/run_continuous_ctde_training.py --config configs/experiments/positive_feedback_ladder_20260807/stage2_simple_aw.yaml --scenes capture --seed 2026080701 --total-steps 25000 --screen-episodes 0 --diagnostic-eval-episodes 4 --device cuda:1 --tag stage2_seed1_25k --artifact-root artifacts/2026-08-07_positive_feedback_ladder/stage2
  seed2: 同上 seed=2026080702 --device cuda:0 --tag stage2_seed2_25k
PID/log: seed1 PID 344158 / tmux ladder_s2_s1（完成）；seed2 PID 344451 / tmux ladder_s2_s2（完成）；seed3 PID 765177（已停止）
step: seed1=25000（完成）；seed2=25000（完成）；seed3=14000（停止，保留产物）
key metrics: baseline random capture=0/min-dist=40.6，noop capture=0/min-dist=50.5，oracle capture=1.0；seed1 eval20 capture=6/20、collision=0/20、min-dist≈15；seed2 eval20 capture=20/20、collision=0/20、avg length≈55、min-dist≈8.2
decision: Stage 2 PASS（3 seeds 中已有 2 个明显优于基线，且 seed2 达 100% capture）；进入 Stage 3A
next action: Stage 3A seed1/seed2 训练中；seed3 如需可作为第三条证据从 14k 续训
```

### 3.4 Stage 3A：Pure Coverage（无 obstacle）

```text
stage: 3A
run_id: STAGE3A_PURE_CE_AW_20260807
status: PASS（seed1 CE -96%、seed2 CE -15%；seed3 作为第三条证据运行中）
date: 2026-08-07
branch: continuous/masac-ctde-contract-20260806
commit: 3caefea + 后续
config path/hash: configs/experiments/positive_feedback_ladder_20260807/stage3a_pure_ce_aw.yaml
seed: 2026080701（后续 02/03）
initialization: scratch random-init
action contract: acceleration_angular_velocity_body, independent box
observation contract: robot-frame local VCT-LS（无 evader）
dynamics contract: continuous_aw_v1
reward contract: CE centroid energy + PBRS，speed weight=0
exact command: PYTHONPATH=src:. python3 tools/run_continuous_ctde_training.py --config configs/experiments/positive_feedback_ladder_20260807/stage3a_pure_ce_aw.yaml --scenes pure_ce --seed 2026080701 --total-steps 25000 --screen-episodes 0 --diagnostic-eval-episodes 4 --device cuda:0 --tag stage3a_seed1_25k --artifact-root artifacts/2026-08-07_positive_feedback_ladder/stage3a
PID/log: seed1 PID 823508 / tmux ladder_s3a_s1；seed2 PID 1019023 / tmux ladder_s3a_s2；监督 tmux ladder_s3a_sup
step: seed1=25000（完成）；seed2=25000（完成）；seed3 运行中
key metrics: seed1 eval20 CE progress=0.097（~96%）、collision=0/20；seed2 eval20 CE progress=0.0155（~15%）、collision=0/20；baseline random progress=0.0138/collision=1.0，noop progress=0/collision=0
decision: PASS（3 seeds 中已有 2 个满足 CE 改善 + collision 可控 + 优于 baseline）
next action: 进入 Stage3B；seed3 作为额外证据继续
```

### 3.5 Stage 3B：Pure Coverage（1 obstacle）

```text
stage: 3B
run_id: STAGE3B_PURE_CE_OBS_AW_20260807
status: PASS（r3 eval20 CE -51%、collision 15%）
date: 2026-08-07
branch: continuous/masac-ctde-contract-20260806
config path: configs/experiments/positive_feedback_ladder_20260807/stage3b_pure_ce_obs_aw.yaml
seed: 2026080701（后续 02/03）
initialization: scratch random-init
action contract: acceleration_angular_velocity_body
observation contract: robot-frame local VCT-LS（无 evader）
dynamics contract: continuous_aw_v1
reward contract: CE centroid energy + PBRS，speed weight=0
exact command: 同 Stage3A，config 换 stage3b，seed=2026080701
PID/log: tmux ladder_s3b_s1
step: r3=25000（完成）
key metrics: 待回填
decision: 待 25k
next action: 25k 后评估；PASS 后进入 Stage4A
```

### 3.6 Stage 4A：Capture（stationary/global/no-obs）

```text
stage: 4A
run_id: STAGE4A_CAPTURE_AW_20260807
status: PASS（seed1/seed2 25k：min-distance 22.1 vs random 26.6/noop 40.1；seed1 collision 5%、seed2 40%）
date: 2026-08-07
branch: continuous/masac-ctde-contract-20260806
config path: configs/experiments/positive_feedback_ladder_20260807/stage4a_capture_aw.yaml
seed: 2026080701 / 2026080702（用户确认：Stage4A 起 2 seeds 验证）
initialization: scratch random-init
action contract: acceleration_angular_velocity_body
observation contract: robot-frame local VCT-LS + global_evader_visibility=true
dynamics contract: continuous_aw_v1
reward contract: CR-MS ring_importance_ms_v0（stationary target）
exact command: 同 Stage3A，config 换 stage4a，seed 2026080701/02，device cuda:0/1
PID/log: tmux ladder_s4a_s1 / ladder_s4a_s2
step: seed1/seed2=25000（完成）
key metrics: baseline random capture=20%/collision=90%，noop capture=0%；正式结果待回填
decision: 待 25k（2 seeds 中至少 1 个明显优于 random/no-op 即可晋级）
next action: 25k 后分析+eval；按用户确认 4A 后 4B/4C 可并行
```

### 3.7 Stage 4B：Capture（moving evader）

```text
stage: 4B
run_id: STAGE4B_CAPTURE_AW_20260807
status: IN_PROGRESS（seed1 25k 完成，无信号；seed2 已启动）
date: 2026-08-07 19:31 启动
branch: ladder/implementation-20260807
config: configs/experiments/positive_feedback_ladder_20260807/stage4b_capture_aw.yaml
seed: 2026080701
唯一改动: stationary -> original APF moving evader
exact command: tools/run_continuous_ctde_training.py --config stage4b_capture_aw.yaml --scenes capture --seed 2026080701 --total-steps 25000 --screen-episodes 0 --diagnostic-eval-episodes 4 --device cuda:0 --tag stage4b_seed1_25k --artifact-root artifacts/2026-08-07_positive_feedback_ladder/stage4b
PID/log: 608229 / tmux ladder_s4b_s1；监督 ladder_s4b_sup1（tools/supervise_stage3a_milestones.py --analyzer tools/analyze_stage4a_25k.py --scenes capture）
step: seed1=25000（完成）；seed2 启动于 2026-08-07 23:17（PID 1434065 / tmux ladder_s4b_s2 + ladder_s4b_sup2）
key metrics: seed1 diagnostic 4ep：capture=0/4、collision=0、min-distance mean=15.29（random=13.99、noop=13.59）→ 未优于 baseline；training terminated=0、collision=0；eval20 进行中
baselines: random capture=0%/min-distance=13.99，noop min-distance=13.59（stage4b_baselines_moving.json）
decision: seed1 未过“明显优于 random/no-op”门槛；按 2-seed 规则继续 seed2，需至少 1 seed 达标
next action: 等 seed1 eval20 + seed2 25k；若两者均无信号，则 4B 判 FAIL 并做合同/行为诊断，不得直接晋级
```

### 3.8 Stage 4C：Capture（1 obstacle）

```text
stage: 4C
run_id: STAGE4C_CAPTURE_AW_20260807
status: IN_PROGRESS（seed1 25k 完成，无信号；seed2 已启动）
date: 2026-08-07 20:00 启动
branch: ladder/implementation-20260807
config: configs/experiments/positive_feedback_ladder_20260807/stage4c_capture_aw.yaml
seed: 2026080701
唯一改动（相对 4B）: num_obstacles 0 -> 1；4C = moving evader + 1 obstacle（用户确认 2026-08-07：4C 就是动态敌人分支，原为 4B 之后串行，现并行加速）
exact command: tools/run_continuous_ctde_training.py --config stage4c_capture_aw.yaml --scenes capture --seed 2026080701 --total-steps 25000 --screen-episodes 0 --diagnostic-eval-episodes 4 --device cuda:1 --tag stage4c_seed1_25k --artifact-root artifacts/2026-08-07_positive_feedback_ladder/stage4c
PID/log: seed1: 726327 / tmux ladder_s4c_s1；seed2: 重启于 23:59（tmux ladder_s4c_s2 + ladder_s4c_sup2）
step: seed1=25000（完成，有效证据）；seed2 训练中（seed 2026080702，cuda:1，tag stage4c_seed2_25k）
key metrics: seed1 diagnostic 4ep：capture=0/4、collision=0、min-distance mean=14.13（random=13.44、noop=13.68）；eval20：capture=0/20、collision=0、min-distance mean=14.96 → 未优于 baseline；training terminated=40、collision=39
baselines: random capture=10%/min-distance=13.44，noop min-distance=13.68（stage4c_baselines.json）
decision: seed1 未过“明显优于 random/no-op”门槛；按 2-seed 规则继续 seed2，需至少 1 seed 达标
next action: 等 seed2 25k；若两者均无信号，则 4C 判 FAIL 并做合同/行为诊断，不得再叠加其他未授权改动
```

### 3.8.1 跨线 TODO 修订与 Stage4 监控记录（2026-08-07 22:32）

- 文档修订（contract/tracker/implementation-line/agent-prompt）已提交推送：commit `cf45d65`，含 `SUPERSEDED BY CROSS-LINE EVIDENCE 2026-08-07` 标记、Stage6A/7A1 并行规则、teacher 失败触发式、action-translation FAILED GATE。
- 4B/4C seed1 训练中，监督脚本就绪（25k 后自动 analyze + eval20），本轮不做高频轮询，约 10 分钟一次检查。

### 3.8.2 Stage4B/4C 监控更新（2026-08-07 23:17）

- 4B seed1：25k 完成（5001 updates、all finite），诊断 4ep capture=0/4、min-distance=15.29 劣于 baselines（13.99/13.59）→ 无正向信号；eval20 由监督自动执行中。
- 4B seed2：23:17 启动（cuda:0，seed 2026080702，tag stage4b_seed2_25k），独立监督已就位。
- 4C seed1：25k 完成（2026-08-07 23:38）；diagnostic 4ep capture=0/4，eval20 capture=0/20、min-dist 14.96 劣于 baselines（13.44/13.68）→ 无正向信号；training terminated=40、collision=39。
- 4C seed2：首启 23:44，因中间误判配置为“单变量错误”而短暂停止；用户澄清（23:5x）4C 本就应为动态敌人+obstacle，配置已恢复 `autonomous: true` 并加防回归测试（`test_stage4c_moving_evader_with_obstacle`），seed2 于 23:59 重启（cuda:1，seed 2026080702，tag stage4c_seed2_25k），独立监督已就位。
- 用户澄清（2026-08-07）：4C 就是动态敌人，原本在 4B 之后，现为加速与 4B 并行；合同/台账/实现线已同步修正，旧“4C=stationary+obstacle 单变量”表述标记为历史。
- 观察：4B/4C seed1 的 collision 均远低于 random（0–5% vs 90%），说明策略学到规避但未学到接近；capture 终止事件稀疏（34–40/25k）。

### 3.8.3 4B/4C seed2 无信号备案（2026-08-07 制定）

- 触发：两 seed 均 25k 完成且 eval20 capture=0/20、min-dist ≥ min(random,noop)、capture 终止率无上升趋势。
- 顺序：①深层次分析（训练轨迹/400 vs 1000-1500 cap 重评估/oracle seek 上限/合同核对）→ ②有潜在信号则 50k 续训（唯一变量=steps）→ ③无信号且 oracle 可学则提交超参候选等用户确认（warmup/update_every/alpha，单变量单 seed 验证）→ ④oracle 也不可学则回合同层重新定标。
- 纪律：4B/4C 独立判定；不自动调参；不跳过 gate。完整版见 `docs/LADDER_IMPLEMENTATION_LINE_20260807_ZH.md` §5.2。
- Stage4D 准备（2026-08-08 01:00）：smoke20 通过（all finite，local visibility 生效，前 20 步角色为 coverage）；baselines 就绪：random capture=5%/min-dist=13.58、noop capture=0%/min-dist=13.65（`artifacts/2026-08-07_positive_feedback_ladder/stage4d/stage4d_baselines.json`）。待 4B seed2 积极信号即启动 4D 两 seed。
- Stage4D 提前启动（用户确认 2026-08-08）：4B seed2 先结束且积极 ⇒ 直接启动 4D 两 seed；4C 后结束积极 ⇒ 4D 正常维护；4C 无积极信号 ⇒ 4C 诊断与 4D 训练并行。

### 3.8.5 Stage4C 结论（2026-08-08 03:50）

- 4C seed2 25k：eval20 capture=0/20、collision=35%、avg min-dist=15.48（random 13.44 / noop 13.68）→ FAIL。训练期 terminated=56≈collision=56。
- 注意：seed2 的 4ep 诊断 min-dist=12.92 < baseline 使自动分析器判 PASS，但 20 集 eval20 无稳定改善（15.48），按 eval20 gate 判 FAIL；4ep 小样本 PASS 为假象，已在完成报告中记录。
- 2-seed 汇总：seed1 capture 0/20、min-dist 14.96、collision 0%；seed2 上述 → Stage4C FAIL（25k 合同下）。
- 完成报告：`artifacts/2026-08-07_positive_feedback_ladder/stage4c/STAGE_4C_CAPTURE_COMPLETION.md`。

### 3.8.4 Stage4B 结论与深层次分析（2026-08-08 03:10）

- 4B seed2 25k：eval20 capture=0/20、collision=100%、avg min-dist=17.15（random 13.99 / noop 13.59）→ FAIL。训练期 terminated=81、collision=79（终止≈碰撞，非真实 capture）。
- 2-seed 汇总：seed1 capture 0/20、min-dist 14.45、collision 5%；seed2 上述 → 4B 两 seed 均无“明显优于 random/no-op”信号 → Stage4B FAIL（25k 合同下）。
- 轨迹解剖：
  - seed1：6k 后 term/coll 归零，speed 0.09–0.16、action 0.35、Q 单调降至 -4.9 → 学成“低速安全游荡”。
  - seed2：11k–22k term≈coll 密集（最高 10/1k）、speed 0.4–0.76、Q 单调降至 -9.8 → 学成“高速冲撞”，eval 100% collision。
  - 4C seed1 同 seed1 模式（Q→-4.6）。
- 行为探测（1000 步上限重评估 25k checkpoint）：seed1 0/10 capture、0 collision，但 distance_progress 多为负（远离 evader）、speed_mean 0.02–0.09；seed2 0/10 capture、100% collision、60–351 步即撞停 → 排除“400 cap 掩盖慢接近”。
- 可行性上限（手写 seek 控制器，完美信息）：4B/4C 各 10/10 capture，平均 23–49 步完成，avg min-dist 7.3–8.0 → 任务合同可学，问题在 RL 训练信号。
- 结论：无潜在乐观信号（无 capture 事件趋势、无接近趋势、Q 单调下降、更长 horizon 也无改善）→ 按备案不走 50k 续训首选；进入机制/超参调整分支，候选需用户确认（warmup / update_every / alpha / reward 侧机制核对）。
- replay 定量证据（2026-08-08 03:53，加载 25k replay 逐条统计）：真实 capture 事件 4B s1=0、4B s2=2、4C s1=1、4C s2=0；碰撞事件 34/79/39/56；ce_success 全 0。active-agent 奖励：mean -0.29/-0.91/-0.27/-0.57、p50=0、p95 0.25–0.78、max（非 capture）≤1.88；≥50 的 capture 奖励仅 s2 各 1–2 条 → 训练 25k 内 capture 正反馈几乎为零，稀疏奖励假设成立。reward 配置核对：capture_reward_mode=ring_importance_ms_v0、capture_timestep_penalty=0、omega_ring_ms=2.0、clip=3.0、k_required=1、stationary capture enabled → 接线正确，问题在策略从未稳定接近（非机制 bug）。
- 跨线佐证（参考分支 2026-08-08 复核）：`ctde_capture_random_25k`（world [ax,ay] capture random init）同样 25k 内 Q1 降至 ≈-14.2、TD≈1.77（与我们 Q -10~-14 一致），training collision 仅 1/25000，无 final eval → “Q 单调坍缩”跨 body/world action 模式一致，进一步排除 action-mode 特因。
- 下一步：4C seed2 已完成（见 3.8.5），两条线均 FAIL；汇总调整方案供用户选择（warmup / update_every / alpha / reward 侧 / 50k 续训）。

### 3.8.6 reward 第一批三线并行（用户确认 2026-08-08 08:40）

- 用户决策：三线并跑 A1/A2/A3（reward-first 批）；若无明显积极，再做一轮以探索为改变的（entropy/alpha init/warmup 等，叠加在更积极一侧）；若奖励改动已积极，以更优且改动更小的继续进入下一阶段。
- A1：omega_ring_ms 2.0→4.0 + ring_ms_progress_clip 3.0→6.0。
- A2：A1 + omega_approach 0.0→3.0（旧 capture 距离进度吸引项叠加在 ring 之上；代码改动 `src/cocap_voradj/envs/voronoi_adjacency.py` capture_task_reward，默认 0 不影响旧配置；单步验证 A2>A1 奖励 +0.0727）。
- A3：A1 + warmup_joint_transitions 5000→10000。
- 配置：`stage4b_a1/a2/a3_reward_20260808.yaml`；合同测试新增 `test_stage4b_reward_first_variants_contract`（12/12 通过）；三线 smoke20 均通过（all finite）。
- 运行计划：三线各 25k seed1（seed 2026080801），A1/A3 共 cuda:0、A2 用 cuda:1；监督自动 analyze+eval20（baselines=stage4b_baselines_moving.json）；判定标准不变（eval20 capture≥1 或 min-dist 明显优于 random/noop）。
- 进度快照（2026-08-08 08:52）：A1/A2 均 ~6k（warmup 后训练开始，update=251）、A3 ~10k（warmup 10000 中）；GPU0 94%（A1+A3）、GPU1 100%（A2）；无异常；10k 初检待执行。
- 待办：若某线 25k 明显积极 → 补 seed2 并择优进入 4D/4C；若均不积极 → 第二轮探索批（target_entropy/alpha_init/warmup）在较积极侧增量调整。
- 归因与备案：完整消融归因矩阵、E 批归因、Case 1-8 决策树见 `docs/LADDER_STAGE4_REWARD_BATCH_ATTRIBUTION_20260808_ZH.md`（2026-08-08 固化为可执行规则）。

### 3.9 历史/已淘汰条目

```text
stage: OLD_LINE (HISTORICAL / SUPERSEDED)
run_id: ctde_25k_world_axay_20260806
status: FAIL（作为历史证据保留）
date: 2026-08-06
branch: continuous/masac-ctde-contract-20260806
commit: 9827770
config: configs/experiments/continuous_marl_20260804/p6_formal_central_masac_4v1.yaml
config_hash: fda900f01440214a421c581a5fc9c298a008fd71c105285df2bddfbdd03cdafc
seed: 2026080601
action contract: acceleration_2d_world (L2 disk, a_max=0.4)
dynamics: continuous_parity_v1
reward: 原 CR-MS+VCT-LS+CE
exact command: tools/run_continuous_ctde_training.py（2026-08-06）
step: 25000
key metrics: capture 0/20、pure_ce 0/20、mixed 0/20、collision 0.45–0.55、Q≈-11~-13、critic grad norm 140–400
comparison: 与 random/no-op 无行为差异
decision: 不晋级；因一次性改变过多核心变量，无法归因算法，转新阶梯重建
next action: 已由本台账 Stage 0 接管
```

```text
stage: OLD_LINE (HISTORICAL / SUPERSEDED)
run_id: aw_axay_500k_pure_20260805
status: FAIL（100k 提前关闭）
date: 2026-08-05/06
branch: continuous/masac-ctde-contract-20260806
config: p6 screening 5k/500k 诊断线
seed: 2026081501/1502
action contract: AW 与 ax/ay 混合实验
key metrics: AW pure 退化为原地制动/转圈；ax/ay pure 高速碰撞
decision: 保留 100k checkpoint 与复盘，不作为阶梯线晋级证据
```

---

## 4. TODO（本阶梯）

- [x] Stage 0：20-episode 旧 IQN 基线（PASS）
- [x] Stage 1：连续 `(a,ω)` bridge parity（PASS）
- [x] Stage 2：极简单任务 3 seeds × 25k（PASS）
- [x] Stage 3A：4p0e0obs pure coverage 3 seeds × 25k（PASS）
- [x] Stage 3B：1 obstacle pure coverage r3 25k（PASS）
- [ ] Stage 4A-4E：capture 难度阶梯，每子阶段单独报告（用户确认：4A 启动后 4B/4C 可并行；Stage4A 起 2 seeds 验证）
- [ ] Stage 5A/5B：双任务交替与 mixed capture→coverage
- [x] Stage 6 配置/实现预备：velocity-heading yaw + Stage6A/6B/6C body-frame configs
- [x] Stage 7 配置/实现预备：7A1-3、7B1-3 world/robot-obs configs + smoke
- [ ] Stage 6A/7A1 并行训练：Stage5 anchor 后启动（body pure / world pure 复现 `ctde_pure_random_25k_v2`）
- [ ] Teacher-assisted：失败触发式，snapshot 优先用于 post-capture/mixed；encoder 不再默认
- [x] Action-translation BC：FAILED GATE / NOT ACTIVE TODO（除非 formulation 实质改变）

---

## 5. 复验入口

```bash
cd /home/yjq/rl/CoCap1/cocap-voradj
PYTHONPATH=src:. pytest -q -p no:cacheprovider
```

每次阶段更新必须同步更新第 0 节、第 3 节对应 run 记录和本文件顶部结论。
