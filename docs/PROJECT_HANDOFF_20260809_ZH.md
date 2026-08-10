# CoCap-VorAdj 阶梯实现线交接文档（2026-08-09）

> 本文档写给下一个接手 Agent / 后续窗口：快速建立项目全貌、找到相关文档、知道当前维护哪些训练线、当前最突出的问题是什么，以及哪些结论需要自行复核。
> 生成时间：2026-08-09 10:59 CST。生成时仓库 HEAD：`6db5105`，分支 `ladder/implementation-20260807`。
>
> **2026-08-09 11:55 增补：** 瓶颈已在隔离分支修复并通过 6k smoke/源码回归，25k 4A gate 正在运行；恢复合同、历史证据、磁盘风险与最终最短路线见 `docs/PROJECT_AUDIT_AND_FAST_FINAL_PLAN_20260809_ZH.md`。本增补优先于下文 10:59 快照中的 waiter/下一步描述。
>
> **2026-08-09 20:06 增补（当前权威状态）：** 原慢线已停止并保留 trainer；speedopt Stage4A/4C 分别为 PID 1326505/cuda:0 和 1326499/cuda:1，20:06 时约 77k/69k，均 `mean_finite=1`。A50 正式 eval20 为 capture 95%、collision 5%，A75 为 capture 25%、collision 75%，因此 A50 是已确认的 stationary capture 强 anchor，且证明 checkpoint 表现非单调；4C 到 50k diagnostic 仍 0/4，69k 训练窗口仍无 ring/capture 信号。当前 ETA 为 A 08-10 05:10–05:50、C 06:00–06:40，保守两线 07:00 前完成。可提前并行的正确候选是 Stage4B moving/no-obstacle 25k gate，不是 Stage4D/5；Stage4D 仍需 4B/4C moving 信号。硬件只建议在较冷的 GPU0 增加一条低优先级短线，GPU1 90–91°C 不叠加。现 runner 尚无 MASAC actor-only 跨配置初始化接口，启动 warm-start B 前须先补接口和合同测试；不得用 full resume 混入 A replay。
>
> **2026-08-09 20:49 增补（最新用户指定并行实验）：** 新增 Pure-CE 4P/0E/1obs 与 Legacy-VorAdj old-mix capture+CE coverage 两条 scratch 200k `(a,w)` 线；每 25k 模型+自动诊断，replay 仅保留滚动 latest，final 完整保留且自动 eval20。配置/指标/测试/ETA 见 `docs/PARALLEL_CE_LEGACY_VORADJ_200K_20260809_ZH.md`。20:49 旧 A/C 已到约 88k/79k、均 finite，GPU 78°C/90°C；为遵守“不影响现有线”并避开 GPU1 热上限，新双线由 tmux `parallel_ce_voradj_200k_queue` 在旧 PID 自然结束且温度/磁盘通过 gate 后同时启动，预计 08-10 06:20–07:10 起跑。该用户指定分支独立于原阶梯的 Stage4B 推荐，不将其误记为 Stage4D 解锁。
>
> **2026-08-10 17:17 增补（服务器停机恢复，当前最高优先级）：** 服务器约在 02:23 后停机，所有 tmux 丢失。停机前 A/C metrics 分别到 161k/145k，最后完整恢复点为 A150/C125；A150 diagnostic 为 capture 4/4、collision 0/4，C125 为 capture 0/4、collision 0/4。17:11 已直接启动用户指定 Pure-CE/Legacy-VorAdj 两条新 200k（PID 224391/cuda:0、224386/cuda:1）；17:16 使用 detached HEAD `54c7488` 的原 runner 且 manifest 零差异，从 A150/C125 严格恢复旧线（PID 240128/cuda:0、240122/cuda:1）。四个 tmux 均存活；旧线没有放宽 checkpoint 安全校验。GPU1 约 92°C，需后续观察，但按用户要求当前四线均保持运行。

## 0. 一句话现状

项目在跑 **Stage4A / Stage4C 原合同 200k 对照训练**（均约 61k / 200k，逐 25k 评估并与已完成的 IQN 200k 参照对照）。当前最突出的问题是**训练速度极慢且持续变慢**（22 小时仅 61k）；已做初步归因，但**归因仅供参考，接手者必须自行探查验证后再实施优化**（见第 4 节）。

## 1. 仓库与分支

- 仓库：`Jayyeah/cocap-voradj`（本地 `/home/yjq/rl/CoCap1/cocap-voradj`）
- 当前主要工作分支：`ladder/implementation-20260807`（基于 `main` 的阶梯实现线）
- 并行参考分支：`continuous/masac-ctde-contract-20260806`（原连续线，仅作跨线证据参考，不要在上面做阶梯实验）
- 基准成功分支：`main`（旧 IQN 成功线所在，用于对照配置/checkpoint）
- 算法/合同：MASAC CTDE；连续 `(a,ω)` body-frame 动作（a∈[-0.4,0.4]、ω∈[-π/6,π/6]）；central twin critic；focal joint replay；`continuous_aw_v1` 动力学（10 substeps、dt=0.05、decision_dt=0.5、v_max=3.0、drag=0.4/3）；VorAdjEnv 4v1 120×120、capture 距离 8、ring 8–10.5、VCT-LS 局部观测。

## 2. 相关文档与内容索引

| 文档（仓库相对路径） | 内容 | 何时必读 |
|---|---|---|
| `docs/COCAP_CODEX_AGENT_POSITIVE_FEEDBACK_LADDER_PROMPT_20260807_ZH.md` | 最高执行合同 / Agent 提示词：TODO 修订规则、Stage4/5/6/7 定义、4B/4C 并行规则、Stage5 后 6A/7A1 并行、teacher 失败触发式、action-translation FAILED GATE、每阶段输出格式、执行边界 | 每轮开始 |
| `docs/COCAP_CONTINUOUS_MARL_POSITIVE_FEEDBACK_LADDER_20260807_ZH.md` | 阶梯总合同：Stage0→7 的完整预期阶梯、gate 判据、冻结参数 | 判定 gate 时 |
| `docs/CONTINUOUS_MARL_POSITIVE_FEEDBACK_TRACKER_20260807_ZH.md` | 专用实验台账（0 节当前状态、0.3 诊断补丁、0.4 IQN 200k 并入、0.5 速度瓶颈归因） | 每次开始/修改/启动/25k/异常/完成都必须更新 |
| `docs/LADDER_IMPLEMENTATION_LINE_20260807_ZH.md` | 实现线当前位置、历史结论、实现边界、并行规则 | 快速定位当前位置 |
| `docs/LADDER_STAGE4_REWARD_BATCH_ATTRIBUTION_20260808_ZH.md` | Stage4 reward 批次归因计划（A 批 reward 优先、E 批探索优先、归因逻辑与备案） | 推进 reward/探索批时 |
| `docs/LADDER_STAGE4_BC_ADJUSTMENT_PROPOSALS_20260808_ZH.md` | Stage4 BC（行为克隆）调整提案（当前 NOT ACTIVE） | 评估 teacher/BC 时 |
| `artifacts/2026-08-08_200k_reference/LEGACY_200K_PROGRESS.md` | 200k 三线对照逐 25k 进度表 + IQN 分阶段汇总 + 运行速度与瓶颈分析 | 每次 25k 节点回填 |
| `artifacts/2026-08-08_reward_first_batch/LEGACY_IQN_SCRATCH_25K_DIAGNOSTIC.md` / `..._50K_DIAGNOSTIC.md` | IQN scratch 25k/50k 行为诊断 | 时间尺度对照结论引用 |
| `configs/experiments/positive_feedback_ladder_20260807/*.yaml` | Stage2–7B 全部正式/预备配置 | 启动训练前核验 |
| `tools/evaluate_ctde_formal.py` | MASAC eval20 正式评估（需 `PYTHONPATH=src:.`） | 每 25k 节点 |
| `tools/evaluate_stage4_paired.py` | paired seeded 行为评估（trained/random/noop/oracle） | Stage4 gate 判定 |
| `tools/evaluate_iqn_scratch_early.py` | IQN scratch 行为评估（离散动作统计） | IQN 节点 |
| `tools/diagnose_critic_q_ranking.py` | counterfactual critic Q-ranking | critic 学习诊断 |

## 3. 当前维护的训练线

### 3.1 正在跑（GPU 全占）

- **Stage4A 200k**（stationary capture，原合同，seed 2026080801）：PID 527834 / cuda:0，tag `stage4a_200k_20260808`，产物 `artifacts/2026-08-08_200k_reference/stage4a/`，当前约 61k。
- **Stage4C 200k**（moving evader + 1 obstacle，原合同，seed 2026080801）：PID 528005 / cuda:1，tag `stage4c_200k_20260808`，产物 `artifacts/2026-08-08_200k_reference/stage4c/`，当前约 61k。
- 规则：每 25k 保存完整 bundle（已验证 25k/50k 含 trainer/replay/runtime/effective_config/manifest/metrics/diagnostic）；每个 25k 节点做 eval20 → 回填 `LEGACY_200K_PROGRESS.md` 与台账 → commit/push。

### 3.2 已完成（历史结果，不要重跑）

- **IQN 200k 参照**：`runs/iqn_scratch_200k_20260808`，08-08 12:12→15:08 完成；每 25k 行为已评估并入对照表（25k–75k capture=0、100k 8/20、125k 峰值 15/20、200k 8/20；后期 collision 归零、2+ ring 36.9%）。
- **Stage0/1/2/3A/3B/4A 25k**：PASS（4A 2 seeds capture 5%/15%）。
- **Stage4B/4C 25k 双 seed**：FAIL（capture 0/20、min-dist ≥ baseline；oracle seek 10/10 证明合同可学）。
- **A 批 reward 变体**（4B 场景 25k seed1）：A1（ring 4/clip 6）collision 95%、A2（+approach 3）collision 100%、A3（+warmup 10k）安全但无信号 → E 批 Base=A3，待 GPU 释放。
- **4A/4C 200k @25k/50k 评估**：4A 25k no-op 坍缩、50k capture 5/20（25%）；4C 25k 弱接近、50k 0/20（slow approach）。

### 3.3 排队中（按 gate 推进）

- 4A/4C 200k 完成后输出与 IQN 200k 的完整三线对照结论。
- Stage4D（local visibility，需 4B 有积极信号才提前启动，当前未满足）/ Stage4E / Stage5A / Stage5B 按 gate。
- Stage6A / Stage7A1：Stage5 建立 continuous `(a,ω)` anchor（PASS 或明确 OPTIMISTIC_PARTIAL）后并行启动；Stage7A1 首轮复现原线 `ctde_pure_random_25k_v2`。

## 4. 当前问题：训练速度瓶颈（诊断仅供参考）

### 4.1 观测事实

- 4A/4C 200k 启动 08-08 12:31，到 08-09 10:59（约 22h27m）仅到 61k；速度从 25k 段 ~6.7k/h 降到 50k 段 ~2.5k/h，再到近期 ~1.2–1.5k/h。
- IQN 200k 同机、同外部负载、部分时间与 4C 共用 cuda:1，仍恒定 ~68–75k/h（每 25k 20–25 分钟）。
- 4A/4C 进程各只用 ~1.24 核（单线程为主），GPU 利用率 0%，内存 96G 可用。

### 4.2 已做初步归因（2026-08-09 基准）

- **外部线基本排除**：另一用户 `blenderproc` 占 ~55 核（5457% CPU，2d10h），loadavg≈95/128；但 IQN 在同一负载下仍快、4A/4C 不缺核、GPU 不忙 → 外部线不是主因。
- **主要嫌疑：`FocalReplaySampler` 采样开销**：
  - `sample_items` 实测 25k replay：3.0–5.3s/次；50k replay：5.5–9.0s/次；训练每 4 env steps 采样一次 → 摊销后每秒级成本，与观测 ~3s/step 同量级。
  - 根因疑似：`_role_pool('pre_capture_pursuing')` 每次全量重建（池大小=4×replay_size，50k 时约 20 万 item），且当前其余 focal bucket 全空，fallback 矩阵会重复重建同一池。
  - 修复方向（未实施、未验证）：role pool 增量缓存/索引、fallback 复用已构建池；不改变采样语义，预计单次采样降到 O(batch) 量级。
- 次要开销：env.step ≈32ms/step；完整单步循环（obs stack + central obs + step + geometry）≈55ms/step。

### 4.3 重要声明

**以上原因诊断仅供参考，实际原因需要接手者自行探查确认后再尝试改进。** 原因包括但不限于：

1. 单点基准可能受当时负载波动影响（基准进程与训练进程、blenderproc 同机并发），需要复测；
2. `blenderproc` 的 CPU 占用可能随时间变化，外部线是否曾持续 55 核无历史证据；
3. 还可能存在其他未排查因素：CPU 调度/频率、内存带宽争抢、`replay.add` 索引、`_metrics_record` 每 1k 的全量 `focal_index_sizes`、checkpoint 写盘（1.6GB replay）等；
4. 修复前必须做 A/B 验证：同 seed 复跑短段（如 5k–25k），确认速度提升且训练曲线/结果语义一致；
5. 若当前 200k 对照仍在跑，不要改动已加载的代码影响其可复现性；优化应先在独立副本/下一条线实施。

复测建议（快速验证方法，已在 2026-08-09 使用过）：
- 加载 `checkpoints/step_000025000` 与 `step_000050000` 的 `replay.pkl`（用 `JointReplayBuffer.load` + manifest），对 `FocalReplaySampler.sample_items` 计时并比较规模增长；
- 对 `VorAdjEnv.step`、`build_central_global_obs`、`_stack_with_batch`、`_pursuit_step_geometry` 单独计时（env 需先 `set_global_config(config)` + `env.reset()`）；
- 用 `ps -L -p <pid>` 看线程数、`nvidia-smi` 看 GPU、`sar`/`top` 看 CPU 争抢。

## 5. 运行状态与监控约定

- 训练进程：4A PID 527834（cuda:0）、4C PID 528005（cuda:1）；`Rl+`、~124% CPU、metrics 全 finite、无 OOM。
- tmux：`stage4a_200k` / `stage4c_200k`（原训练输出）及 `stage4a_speedopt_25k`（隔离低优先级验证）；当前未发现先前记录的等待器 session 64955，75k 评估需人工或重新建立监督器。
- 磁盘：`/` 用量 96%（约 42G 可用），注意 checkpoint/日志增长；`metrics.jsonl` 被 .gitignore 忽略，**不要提交**。
- 监控频率：约每 10 分钟一次；只查 PID/step/log tail/checkpoint/NaN/OOM/GPU/磁盘。
- 保存合同：每 25k 原子保存完整 bundle（trainer.pt / replay.pkl / runtime_state.pkl / effective_config.yaml / manifest.json / metrics.jsonl / diagnostic_eval.json）。

## 6. 下一步动作（接手即执行）

1. 检查 75k checkpoint 是否生成：`ls artifacts/2026-08-08_200k_reference/stage4{a,c}/stage4{a,c}_200k_20260808/checkpoints/`。
2. 并行评估 4A/4C eval20（命令模板见台账 0.4 / 摘要）：
   `PYTHONPATH=src:. python3 tools/evaluate_ctde_formal.py --config .../stage4a_capture_aw.yaml --checkpoint .../step_000075000/trainer.pt --tag ..._75k_eval20.json --episodes 20 --max-steps 400 --scenes capture --seed 2026080801 --device cuda:0`（4C 同理）。
3. 回填 `LEGACY_200K_PROGRESS.md`（4A/4C 75k 行）+ 台账，commit 并 push `ladder/implementation-20260807`。
4. 继续 100k/125k/150k/175k/200k 节点；200k 完成后输出三线完整对照结论。
5. 独立于训练线，复测第 4 节瓶颈归因；确认后提出采样器优化补丁（先 A/B 验证）。

## 7. 坑与边界（务必遵守）

- 不要 kill 训练进程；不要改动 4A/4C 正在使用的代码/配置（可复现性）；不要重跑已完成实验。
- 沙箱常见 `bwrap: loopback: Failed RTM_NEWADDR` → 命令需以已授权方式执行。
- `apply_patch` 工具可能被同一沙箱问题阻断；可用仓库内 `apply_patch` 可执行文件替代。
- 需用户确认才可改：reward 结构/权重、map、a_max/ω_max/v_max/drag、perception、Actor 全局信息、focal quota、grad clip、update ratio、正式切换算法、跳过 Stage gate、临时诊断配置转正式。
- 已批准无需再问：本提示词允许的 TODO 文档调整、Stage4/5 继续、Stage6A/7A1 并行、当前 200k 对照执行。
