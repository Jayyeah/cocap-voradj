# CoCap-VorAdj 连续 MARL 正反馈阶梯专用实验台账

> 建立日期：2026-08-07  
> 最高执行合同：`docs/COCAP_CONTINUOUS_MARL_POSITIVE_FEEDBACK_LADDER_20260807_ZH.md`  
> Codex Agent 提示词：`docs/COCAP_CODEX_AGENT_POSITIVE_FEEDBACK_LADDER_PROMPT_20260807_ZH.md`  
> 分支：`continuous/masac-ctde-contract-20260806`  
> 本台账只记录本文定义的阶梯路线；旧台账 `docs/CONTINUOUS_ACTION_MARL_EXPERIMENT_TRACKER_20260804_ZH.md` 保留为历史，不再作为当前路线唯一依据。

---

## 0. 当前状态（每次更新必须保持最新）

- 当前 active stage：**Stage 2（MASAC + 连续 (a,ω) 极简单任务）**
- 当前唯一 formal config：**待 Stage 1 后锁定**（阶梯线尚未生成 Stage 2 formal config；旧 `configs/experiments/continuous_marl_20260804/p6_formal_central_masac_4v1.yaml` 属 world-frame `[a_x,a_y]` 旧线，标 `HISTORICAL / SUPERSEDED`，不得作为阶梯线唯一 formal config）
- 当前 action contract：连续 `acceleration_angular_velocity_body`，独立 box 边界 `a∈[-0.4,0.4]`、`w∈[-π/6,π/6]`（Stage 1 已严格等价）
- 当前 observation contract：Stage 0 使用旧 IQN VCT-LS robot-frame observation；Stage 2+ 目标为同一 robot-frame local observation，Actor 不得读取全局/oracle
- 当前 dynamics contract：`continuous_aw_v1`（显式 Euler、10 substeps、dt=0.05、decision_dt=0.5、v_max=3.0、drag=0.4/3、yaw 积分、legacy_random 初始化、碰撞整步检查；与旧 IQN 完全一致）
- 当前 reward contract：Stage 2 使用 `[DIAGNOSTIC-ADAPT]` 简化距离 progress/success/safety 项（legacy capture reward），退出后恢复原 reward
- 最近 milestone：Stage 0 PASS、Stage 1 PASS（严格 parity 9/9，state/reward diff 均为 0）
- 当前结论：旧 IQN -> 连续 `(a,ω)` bridge 严格等价；Stage 2 config 已创建并通过 20 步 smoke 与 6k 更新链冒烟（有限、SAC 更新正常），正式 3-seed 25k 待启动
- 下一步唯一动作：**启动 Stage 2 的 3 seeds × 25k MASAC 训练，并在每 25k 回填台账**

---

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
status: IN_PROGRESS（seed1/seed2 已启动，seed3 待前两个 25k 后启动）
date: 2026-08-07
branch: continuous/masac-ctde-contract-20260806
commit: 9827770 + ladder 改动
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
PID/log: seed1 PID 344158 / tmux ladder_s2_s1；seed2 PID 344451 / tmux ladder_s2_s2；stdout 见 stage2_seed{1,2}_25k_stdout.log
step: 启动中（0/25000）
key metrics: smoke20/smoke6k all finite、251 updates、focal fallback 正常；正式训练待回填
decision: 工程链通过；25k 后按 Stage 2 gate 判定
next action: 每 10 分钟轮询；25k 后诊断 eval + 20-episode 评估，并启动 seed3
```

### 3.4 历史/已淘汰条目

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

- [ ] Stage 0：运行 20-episode 旧 IQN 评估（capture/coverage/mix），输出 `STAGE_0_IQN_BASELINE_COMPLETION.md`
- [ ] Stage 1：实现/验证旧 IQN 离散 action -> 连续 `(a,ω)` bridge 的 fixed-seed trajectory parity
- [ ] Stage 2：生成 Stage 2 formal config（1p1e stationary/global/no-obstacle/`(a,ω)`），通过 smoke 后跑 3 seeds × 25k
- [ ] Stage 3A：4p0e0obs inner-cluster pure coverage，3 seeds × 25k（需要时 50k）
- [ ] Stage 3B：恢复 1 obstacle
- [ ] Stage 4A-4E：capture 难度阶梯，每子阶段单独报告
- [ ] Stage 5A/5B：双任务交替与 mixed capture→coverage
- [ ] Stage 6/7：body-frame / world-frame `[a_x,a_y]`

---

## 5. 复验入口

```bash
cd /home/yjq/rl/CoCap1/cocap-voradj
PYTHONPATH=src:. pytest -q -p no:cacheprovider
```

每次阶段更新必须同步更新第 0 节、第 3 节对应 run 记录和本文件顶部结论。
