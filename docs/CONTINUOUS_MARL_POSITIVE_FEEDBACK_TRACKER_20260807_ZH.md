# CoCap-VorAdj 连续 MARL 正反馈阶梯专用实验台账

> 建立日期：2026-08-07  
> 最高执行合同：`docs/COCAP_CONTINUOUS_MARL_POSITIVE_FEEDBACK_LADDER_20260807_ZH.md`  
> Codex Agent 提示词：`docs/COCAP_CODEX_AGENT_POSITIVE_FEEDBACK_LADDER_PROMPT_20260807_ZH.md`  
> 分支：`continuous/masac-ctde-contract-20260806`  
> 本台账只记录本文定义的阶梯路线；旧台账 `docs/CONTINUOUS_ACTION_MARL_EXPERIMENT_TRACKER_20260804_ZH.md` 保留为历史，不再作为当前路线唯一依据。

---

## 0. 当前状态（每次更新必须保持最新）

- 当前 active stage：**Stage 4B / 4C 并行（moving evader / 1 obstacle）**
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
唯一改动: num_obstacles 0 -> 1
exact command: tools/run_continuous_ctde_training.py --config stage4c_capture_aw.yaml --scenes capture --seed 2026080701 --total-steps 25000 --screen-episodes 0 --diagnostic-eval-episodes 4 --device cuda:1 --tag stage4c_seed1_25k --artifact-root artifacts/2026-08-07_positive_feedback_ladder/stage4c
PID/log: 726327 / tmux ladder_s4c_s1；监督 ladder_s4c_sup1（--eval-device cuda:1）
step: seed1=25000（完成）；seed2 启动于 2026-08-07 23:44（tmux ladder_s4c_s2 + ladder_s4c_sup2）
key metrics: seed1 diagnostic 4ep：capture=0/4、collision=0、min-distance mean=14.13（random=13.44、noop=13.68）；eval20：capture=0/20、collision=0、min-distance mean=14.96 → 未优于 baseline；training terminated=40、collision=39
baselines: random capture=10%/min-distance=13.44，noop min-distance=13.68（stage4c_baselines.json）
decision: seed1 未过“明显优于 random/no-op”门槛；按 2-seed 规则继续 seed2，需至少 1 seed 达标
next action: 等 seed2 25k；若两者均无信号，则 4C 判 FAIL 并做合同/行为诊断，不得与 4B 叠加
```

### 3.8.1 跨线 TODO 修订与 Stage4 监控记录（2026-08-07 22:32）

- 文档修订（contract/tracker/implementation-line/agent-prompt）已提交推送：commit `cf45d65`，含 `SUPERSEDED BY CROSS-LINE EVIDENCE 2026-08-07` 标记、Stage6A/7A1 并行规则、teacher 失败触发式、action-translation FAILED GATE。
- 4B/4C seed1 训练中，监督脚本就绪（25k 后自动 analyze + eval20），本轮不做高频轮询，约 10 分钟一次检查。

### 3.8.2 Stage4B/4C 监控更新（2026-08-07 23:17）

- 4B seed1：25k 完成（5001 updates、all finite），诊断 4ep capture=0/4、min-distance=15.29 劣于 baselines（13.99/13.59）→ 无正向信号；eval20 由监督自动执行中。
- 4B seed2：23:17 启动（cuda:0，seed 2026080702，tag stage4b_seed2_25k），独立监督已就位。
- 4C seed1：25k 完成（2026-08-07 23:38）；diagnostic 4ep capture=0/4，eval20 capture=0/20、min-dist 14.96 劣于 baselines（13.44/13.68）→ 无正向信号；training terminated=40、collision=39。
- 4C seed2：23:44 启动（cuda:1，seed 2026080702，tag stage4c_seed2_25k），独立监督已就位。
- 观察：4B/4C seed1 的 collision 均远低于 random（0–5% vs 90%），说明策略学到规避但未学到接近；capture 终止事件稀疏（34–40/25k）。

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
