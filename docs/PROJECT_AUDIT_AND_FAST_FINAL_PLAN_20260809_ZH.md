# CoCap-VorAdj 全面审查、训练提速与最终场景最短计划（2026-08-09）

## 2026-08-10 17:17 停机恢复与四线启动

- 服务器在 02:23 后停机，tmux 全部丢失。停机前 Stage4A/C metrics 到 161k/145k；可验证完整恢复点为 A150/C125，因此不可把未 checkpoint 的 11k/20k 当作可恢复进度。
- 停机前 A125/A150 diagnostic 均为 capture 4/4、collision 0/4；A150 平均 distance progress=20.04、平均最小敌距=8.30。C125 仍 capture 0/4、collision 0/4、平均最小敌距=15.02、平均速度=0.019。
- 为保持原 checkpoint 合同，旧线没有使用当前已改动 runner，也没有绕过 implementation hash。创建 detached worktree `/tmp/cocap_stage4_resume_exact` 指向 `54c7488`；其完整 manifest 与 A150 零差异（implementation `ca5e1cf9...`），随后从 A150/C125 恢复。
- 当前 PID/tmux：A 240128/`stage4a_speedopt_200k_resume`/cuda:0；C 240122/`stage4c_speedopt_200k_resume`/cuda:1；Pure-CE 224391/`parallel_pure_ce_200k`/cuda:0；Legacy-VorAdj 224386/`parallel_legacy_voradj_200k`/cuda:1。
- 两条新线 step-1 model-only milestone 与 `resume_latest` 完整 replay 已原子落盘。17:17 根盘余约 71 GiB；GPU0/1 约 80°C/92°C。当前按用户要求四线并行，首个新 1k/25k 后再按共存吞吐更新 ETA。

## 2026-08-09 20:35 用户指定 Pure-CE / Legacy-VorAdj 并行分支

- 已新增两条 scratch 200k 连续 `(a,w)` 线：Pure-CE 4P/0E/1obs，以及 legacy VorAdj capture reward/graph + CE coverage 的 4P/1E/1obs old-mix；详见 `docs/PARALLEL_CE_LEGACY_VORADJ_200K_20260809_ZH.md`。
- runner 新增 config 驱动 scene cycle、独立 diagnostic scenes、CV<0.20 与接敌正向信号汇总，以及 `rolling_latest` replay 模式。每 25k 的 trainer/评估永久保留，只有 replay 滚动替换；final 仍是完整恢复点。
- 核心新合同 13/13、相关整组回归 62/62 和双 32-step 端到端预检均通过。20:49 旧 A/C 约 88k/79k、GPU1 为 90°C，因此使用持久安全队列，预计旧线 08-10 06:20--07:10 完成冷却后再同时启动，不抢占当前线。
- 新双线 first-25k 是首个可信分流 gate：Pure-CE 看 CE strict、CV<0.20、CE progress；legacy capture 看 capture/collision，并同时看 detected、discovery step、最小敌距与 distance progress。200k 完成与自动评估前不做 checkpoint 清理。

## 2026-08-09 20:06 状态、选优与并行决策

- speedopt Stage4A/4C 进程健康：PID 1326505/cuda:0 与 1326499/cuda:1 仍存活；20:06 最新 metrics 为 77k/69k，`mean_finite=1`。
- 实测稳态吞吐已收敛到 A 约 13.6–13.7k steps/h、C 约 13.3k steps/h。更新 ETA：A 200k+final eval 为 08-10 05:10–05:50，C 为 06:00–06:40；保守两线 07:00 前完成。
- 新增正式 eval20：A50 capture=95%、collision=5%、平均 distance progress=24.86；A75 capture=25%、collision=75%、平均 distance progress=11.90。A50 明确 PASS 且必须永久保留；A75 是显著退化点，不得以 final 自动覆盖最佳点。
- 退化与 critic 尺度同时发生：A 的 critic loss 25k/50k/75k 为 5.0/775.1/1986.9，raw critic grad 为 41.8/582.8/1258.7，Q mean 从 -2.22/-2.86 翻到 +10.4；虽然 finite，75k 的高速度/高 closing 已转化为碰撞坍缩。这是“学会追捕后过度激进”，不是无学习。
- C 暂无晋级信号：25k/50k diagnostic 均 0/4 capture，50k speed mean=0.067、d1_min=14.83，69k 仍无 ring visitation/capture。C 保留到后续 checkpoint 观察，但当前不能据此开 Stage4D。
- 加速决策：可以提前并行 **Stage4B moving/no-obstacle**，它是 A50 后唯一合法的下一难度和 4D 的解锁路径；不能直接开 4D/4E/5。仅建议 GPU0 上一条低优先级 25k gate，先做 2k 共存吞吐门槛（现有 A/C 下降不超过 10%、GPU0 <88°C）再延长。GPU1 已到 90–91°C，不叠线。
- 实现前置：当前 runner 只支持同 manifest 的 full checkpoint+replay resume 和 legacy IQN encoder 初始化，没有 MASAC actor-only 跨配置初始化。full resume 会被 manifest 拒绝且不应通过绕过校验混入 stationary replay。优先补严格 actor-only 初始化、审计记录和合同测试，再从 A50 启动 Stage4B；若不补接口，只能做价值较低的 scratch B 重跑。

正式评估产物：`artifacts/2026-08-09_200k_speedopt/analysis/stage4a_50k_eval20_seed2026080801.json`、`stage4a_75k_eval20_seed2026080801.json`。

本节覆盖下文 14:45/11:55 的状态与 ETA；其余冻结合同继续有效。

## 2026-08-09 14:45 执行更新：speedopt 双线已替换原线

- 用户确认执行磁盘批次 R1/R2/R3，并授权立即关闭原 Stage4A/4C、保留旧 trainer checkpoint、不保留旧 replay。
- 原 PID 527834/528005 已于最后 metrics step=67,000 时停止；最后完整模型 bundle 为各自 step=50,000。旧根中的 6 个 replay 已删除，step1/25k/50k 的 `trainer.pt`、config、manifest、runtime state、metrics 与 diagnostic eval 均保留；两棵旧输出由各约 2.67 GiB 降至约 338 MiB。
- speedopt 同配置 200k 双线已于 14:45 启动：Stage4A PID 1326505 / cuda:0 / tmux `stage4a_speedopt_200k`，Stage4C PID 1326499 / cuda:1 / tmux `stage4c_speedopt_200k`。共同输出根为 `artifacts/2026-08-09_200k_speedopt/`，tag 分别为 `stage4a_speedopt_200k_20260809` 与 `stage4c_speedopt_200k_20260809`。
- 保存/评估合同不变：每 25k 写完整 checkpoint bundle 并执行内置 diagnostic eval，200k 终点写 final report/eval；在自动评估完成且用户再次确认前，不清理新线任何 checkpoint/replay。
- R3 已完成，当前使用的 VS Code Server `df53...`、OpenAI 26.803、Claude 2.1.226 均保留并继续运行。R1/R2 以 nice=19、idle I/O 独立执行，manifest 位于 `docs/audits/`；新训练为 nice=10。

本节覆盖下文 11:55 快照中“原线继续、等待追赶后再停”的旧状态；架构审查、最终场景路线与 gate 仍有效。

> 审查基线：`bd63549`；隔离优化分支：`perf/focal-replay-20260809`；本文快照时间：2026-08-09 11:55 CST。
> 用户最新边界：最终动作空间只要保持连续即可，`(a,ω)`、`[v_x,v_y]` 或 `[a_x,a_y]` 均可。因此 Stage6/7 动作迁移不再是最终场景训练的前置条件。

## 0. 执行结论

1. 原 Stage4A/4C 200k 进程未被修改、暂停或降优先级；仍分别运行在 PID 527834/cuda:0 与 PID 528005/cuda:1。
2. 训练变慢的主因已独立复现并修复：formal focal sampler 的成本随 replay 线性增长，50k replay 每次采样 5.5–9.0 秒；优化后同一 50k replay 的中位数为 0.004433 秒。
3. 优化保留 focal quota、fallback、唯一 focal pair、每 joint transition 上限及均匀抽样分布；大池的 RNG 映射发生变化，因此是“合同/分布等价”，不是与旧实现逐 transition 位级相同。
4. 6k 同配置 Stage4A 冒烟已完成，251 次更新全 finite；独立 25k Stage4A 正在 `nice=10` 验证，原线资源占用未见下降。
5. 暂不关闭原线。25k 通过后启动优化版同配置 4A/4C；优先在新线 checkpoint 进度超过原线且行为/数值 gate 通过后关闭原线。只有端到端吞吐达到至少 8 倍、25k 行为无新增退化且磁盘无法支持追赶方案时，才采用“直接关闭原线并换新线”。
6. 最短最终路线改为：稳定的连续 `(a,ω)` Stage4 capture anchor → Stage4D local visibility → Stage4E support/coverage → Stage5A 双任务 → Stage5B 真实 mixed capture→coverage → 8v2/12v3 warm curriculum。Stage6/7 仅作可选研究，不阻塞交付。

## 1. 提速验证

### 1.1 根因

旧实现的 `_role_pool()` 每次从索引集合重新构造完整 `FocalItem` 列表。Stage4A 当前只有 pursuing bucket 有数据，formal sampler 的主 quota 与四级 fallback 会在一次 128-item batch 中重复过滤同一大池：

| replay | pursuing focal items | 旧 `sample_items` | 旧 `_role_pool` 典型值 |
|---:|---:|---:|---:|
| 25k | 100k | 约 3.0–5.3 s/次；独立复测中位 3.1629 s | 约 0.15 s |
| 50k | 200k | 约 5.5–9.0 s/次 | 最高约 1.7 s |

训练每 4 env steps 更新一次，所以采样开销完全主导原线后半段吞吐。环境本身约 32 ms/step，包含 observation/central state/geometry 的循环约 55 ms/step，不是 replay 规模相关的减速来源。

外部 Blender PID 64227 当前约占 55 CPU cores，且在两张 GPU 上各保留约 6.3 GiB。它不能解释旧线随 replay 增大而线性恶化，故不是原主因；sampler 降到毫秒后，trainer/GPU 成为主路径，Blender 与原训练进程构成共享背景，可能限制优化线端到端上限。外部进程不属于本项目，不做停止或改优先级操作。

### 1.2 实现

- `4d88e15 perf(replay): make focal sampling independent of replay size`
  - role index 改为增量维护的 dense list + key→position map；ring overwrite 使用 O(1) swap-delete；records schema 不变，新 checkpoint 额外保存可选的派生 pool 顺序，旧 schema-4 replay 仍可从 records 重建。
  - 同一次 sample 调用复用 role pool。
  - 对大且稀疏排除的池使用均匀 rejection sampling without replacement；小池或排除密集时回落到穷举路径。
  - `focal_index_sizes` 改为 O(1) cardinality 查询。
- `0e7de8d fix(training): preserve exact replay sampling continuation`
  - 保存/恢复 runner generator 与 focal sampler generator 状态。
  - 恢复累计 update count、metrics history、scene/origin counts、sampling stats 与 finite 状态。
  - 修正 snapshot 分支 exact milestone checkpoint 的错误缩进，并去掉同一步完整 replay 的二次重复序列化。
- `16d7e19 fix(replay): preserve dense pool order across resume`
  - checkpoint 额外保存 dense focal pool 顺序；覆盖 ring 后 save/load 与 sampler RNG 一起可精确复现下一 batch。
- `2300725 perf(checkpoint): save final bundle only once`
  - 当 `total_steps` 正好命中 checkpoint interval 时，跳过循环内的重复 final bundle/eval；终点评估和 bundle 只执行一次。
  - 兼容用 standalone trainer/replay 名称改为 final bundle 文件的原子 hardlink（跨文件系统才 copy fallback），避免第二份完整 replay 占用。
- `43e7514` 为 final-save ownership 增加显式 gate 测试；`2f7e1fa` 仅在 replay 真正超过 capacity、发生 ring overwrite 后保存 pool 顺序，正常 200k/250k-capacity 训练不增加该元数据。

### 1.3 性能与正确性证据

同一真实 50k replay、20 次采样：

| 指标 | 优化后 |
|---|---:|
| `_role_pool` median | 1.48 µs |
| `sample_items` median | 0.004433 s |
| `sample_items` mean | 0.004639 s |
| `sample_items` max | 0.0114 s |
| 相对旧 5.5–9.0 s | 约 1,240–2,030× |
| batch / actual pursuing / fallback | 128 / 128 / 64 |
| unique transitions / max items per transition | 128 / 1 |

真实 25k replay 的完整 CPU batch 路径（10 次）进一步测得：`sample_items` median 0.001372 s、`batch_from_items` median 0.008702 s、完整 `replay.sample` median 0.009744 s（max 0.009930 s）。因此 residual 秒级 update 成本不在 replay/batch 物化，而在 central SAC/attention 计算及当前 GPU 共享；不为追求额外速度擅自引入 AMP/compile 等数值路径变化。

60 秒、5 秒间隔的设备级采样中，GPU0 SM 连续为 95–100%、功耗 249–293W；GPU1 多数为 0%，一次 burst 为 100%。这证明 speedopt 已把 central trainer 推到 GPU-compute 主路径；设备级计数不能拆分 trainer/Blender 的各自贡献，故不把 residual 唯一归因给外部进程。

最终 commit 上的可归档 50k/20-repeat 复测见 `artifacts/2026-08-09_focal_replay_speed_validation/focal_sampler_50k_benchmark.json`：`sample_items` median/mean/max=0.001052/0.003559/0.018813 s，role-pool median=0.607 µs；高负载下仍稳定为毫秒级。

回归结果：

- focal/replay/checkpoint 目标测试：16 passed；
- 完整 worktree 最终代码：154 passed；另 2 项只因隔离 worktree 不含原仓库未跟踪的历史 `runs/` 文件失败；使用原数据路径复核后 4 passed；
- 新增统计测试覆盖大池四分位均匀性、focal pair 唯一性、ring overwrite 后无 stale generation、sampler RNG round-trip。

6k Stage4A pipeline smoke：

- 11:35:16 启动，训练/最终模型/replay 于 11:43:32 落盘，最终 4×400 eval 于 11:45:04 完成；
- 6000 transitions、251 updates、replay 6000、`all_finite=true`；
- 最后一个 batch 为 128 items、128 unique transitions、fallback 64、无 replacement；
- 包含保存时间的训练段约 43.5k steps/h；该短跑前 5k 是无更新 warmup，只用于链路与资源验证，不据此宣称 200k 稳态吞吐。
- 6k smoke 与随后独立启动的 25k run 在相同 config/seed 下，前 6 条（1k–6k）metrics 字节级一致，SHA256 均为 `9925e0e5b8004f086ec84e920ccb14ed455b904ba02a52215f9f1020990006a7`；证明优化线自身可确定复现。
- 最新 `2f7e1fa` 另做 CPU 1-step final-save smoke：只生成一个 `step_000000001` bundle；standalone trainer/replay 与 bundle 内文件 inode 相同、link count=2，整个 run 实际占一份约 63 MiB；runtime schema=2、runner/focal RNG 均存在、未回绕 replay 的 `role_items=None`。

### 1.4 25k/迁移 gate

正在运行：tmux `stage4a_speedopt_25k`，PID 769401，cuda:0，`nice=10`，原 Stage4A config/seed 2026080801，25k 后 eval20。

12:26 快照：11k transitions / 1501 updates 全 finite，warmup 后约 14–15k steps/h；原 4A 同期从 63k 前进到 64k，原 4C 保持 63k 慢速运行，两个原进程均存活且 CPU 占用不降。该验证线继续在后台跑满 25k。

通过必须同时满足：

1. 25k/5001 次预期更新全部 finite，无 OOM/异常退出；
2. manifest、动作/dynamics/reward、quota、fallback 与原合同一致，只有 implementation hash 改变；
3. sampling stats 满足 batch=128、focal pair 唯一、每 transition 不超过 4；
4. 端到端 25k 吞吐至少为原 4A 首段的 5 倍，且原 4A/4C 在并行期间 CPU/GPU/step 速率无可见退化；
5. eval20 不用于要求逐轨迹相同，只检查没有新增数值错误或明显异常行为；学习结论仍需多 seed/里程碑判定。

## 2. 实现与恢复合同审查

### 2.1 已修复

- manifest 原声称保存 `replay_numpy`，但 formal focal sampler 实际使用自己的 generator 且未持久化；现已显式保存 `focal_sampler_numpy`。
- snapshot/reset 选择使用独立 runner generator，原先未持久化；现已保存 `runner_numpy`。
- resume 后 `update_count` 原用新进程 `len(updates)` 重新从 0 计数；现恢复累计计数。
- `metrics_history` 原虽在 runtime state 内但启动后被空列表覆盖；现恢复。
- scene/origin 累计计数和 sampling stats 原被重置；现恢复。
- snapshot-assisted 路径的 checkpoint 判断原在 episode 内循环之外，短 episode 可错过精确 25k；且同一 replay 会先保存空诊断再 overwrite 保存一次。现已在精确 step 保存一次完整 bundle。

恢复仍明确是 `seeded_episode_boundary`，`exact_env_state=false`；不能声称 mid-episode bit-exact env continuation。

### 2.2 仍需观察而非立即改参数

原线 63k 最新值：

| 线 | Q1 mean | critic pre-clip grad norm | clip ratio字段 | finite |
|---|---:|---:|---:|---:|
| 4A | -16.52 | 1111.27 | 0.000627 | 1 |
| 4C | -7.12 | 1916.91 | 0.02326 | 1 |

`grad_clip_norm=0.5` 下 raw critic gradient 长期达到数百至上千，Q 值持续负移；“finite”不足以证明 critic 健康。下一 checkpoint 必须做 Q scale/TD-error/target-Q/counterfactual ranking 诊断。grad clip、reward、update ratio 属冻结参数，当前不擅改；如需调整应先形成单变量提案并获得确认。

## 3. 历史证据与既有结论复核

| 项目 | 原记录倾向 | 审查结论 |
|---|---|---|
| Stage3A | seed3 仍在跑，Stage3A PASS | seed3 原始 eval 已存在：seed1/3 分别为明确正向，seed2 较弱；按 3-seed 至少 2 个方向性改善，PASS 有效，但完成文档过期，应补 seed3。 |
| Stage3B | PASS | 只有一个有效 fixed run 的明确正向证据；在原 3-seed/2-direction gate 下应标 `OPTIMISTIC_PARTIAL / provisional anchor`，而非充分复现 PASS。 |
| Stage4A 25k | PASS anchor | 两 seed capture 5%/15%、collision 5%/40%；random capture 20%/collision 90%。训练策略更安全且有几何信号，但 capture 未超过 random；应称“安全/几何 anchor”，不是稳健成功。 |
| Stage4B/4C 25k | FAIL | 双 seed 0 capture 且连续指标不足，FAIL 有证据支持；但 IQN 到 75k 仍 0 capture，所以只证明“25k 预算失败”，不能证明合同不可学。 |
| A3 | 安全但无信号 | A3 是安全、缓慢接近的正向连续信号，并非完全无信号；作为下一探索批 Base 合理。 |
| IQN vs MASAC | 4A 学习不劣于 IQN | 非同难度对比：IQN 200k 是 local VCT-LS/support/moving+obstacle/mixed 类完整合同；4A 是 stationary/global/no-obstacle/unified pursuing。IQN 只能作为时间尺度参考，不能用于算法优劣结论。 |

其他文档问题：

- `LEGACY_200K_PROGRESS.md` 上层表格仍有已在下文出现的 4A/4C 25k 数据空缺，需要在下个 checkpoint 回填时统一修正。
- 台账称 75k waiter/session 64955 存活，但当前 tmux 只有原 4A、原 4C 和 speedopt 验证；不存在该自动等待器。后续不能假设 75k 会自动评估/提交。
- 旧台账把“进程存活/loss finite”写得偏正面；正式 gate 必须看 deterministic/stochastic eval、continuous geometry、collision 和 critic 诊断。

## 4. 项目目标与实际路线是否匹配

项目真正任务是局部感知下多机器人搜索/围捕移动 evader，捕获后恢复覆盖，并从 4v1 扩展至 8v2/12v3。当前 Stage4A/4C 只覆盖 capture 子问题；尚未形成以下闭环：

```text
local discovery
  → moving-target capture
  → support/coverage role credit
  → same-episode post-capture CE
  → 8v2/12v3 multi-target scaling
```

因此“4A/4C 跑满 200k”是建立 anchor 和定位难点，不是项目完成。另一方面，用户已接受任何连续动作合同，所以继续把 body/world `[a_x,a_y]` 当关键路径会延迟真正场景目标；可靠 `(a,ω)` 已满足动作连续性，应优先完成场景和任务闭环。

## 5. 最快完成最终期望场景的更新计划

### P0：立即解决存储前置条件

根盘 `/` 仅余约 41 GiB（96% 使用）；项目约 89 GiB，其中 artifacts 约 77 GiB、runs 约 12 GiB。`/data/disk1` 余 2.6 TiB、`/data/disk2` 余 4.5 TiB，但当前用户没有写权限。

原两线若继续保存 75k–200k，保守估计还需超过 48 GiB；再加优化版双线会更高，根盘必然在完成前耗尽。必须在启动优化版双 200k 前二选一：

1. 首选：授予 `/data/disk1` 或 `/data/disk2` 一个专用目录写权限，把新 artifact root 指向该目录；
2. 备选：由用户批准一份精确的历史 artifact 归档/删除清单。未经批准不删除或迁移历史结果。

### P1：完成 speedopt 25k gate（当前进行中）

- 完成 Stage4A 25k + eval20；输出端到端吞吐、资源影响、数值和采样合同对比。
- 若低于 5×，profile trainer.update/env/序列化，不启动双 200k。
- 若通过，冻结 `0e7de8d` 代码，不再在迁移线叠加 reward/算法/配置变化。

### P2：优化版同配置 4A/4C 并行追赶

- 两张 GPU 各启动一条优化版 200k，seed/config 与原线一致，独立 tag/artifact root，`nice=10` 起步。
- 每 25k eval20；比较吞吐、finite、Q/grad、capture/distance/ring/collision。
- 正常策略：新线 checkpoint step 大于原线实际 step，且最新 checkpoint gate 通过后，优雅停止对应原线并保留其最后完整 checkpoint。
- 快速替换例外：25k 端到端 ≥8×、两线资源互不干扰、行为无新增退化，同时存储无法支撑追赶时，可在 25k 完整 bundle 后直接停原线。
- 4A/4C 到 200k 后选“历史最佳 checkpoint”，不机械选择 final；IQN 的 125k 峰值后回落已证明必须按 checkpoint 选优。

存储可写后使用以下同合同命令（`<writable-artifact-root>` 必须替换为数据盘专用目录）；追赶阶段保持 `nice=10`：

```bash
cd /home/yjq/rl/CoCap1/cocap-voradj-speedopt
nice -n 10 env PYTHONPATH=src:. python3 tools/run_continuous_ctde_training.py --config configs/experiments/positive_feedback_ladder_20260807/stage4a_capture_aw.yaml --scenes capture --seed 2026080801 --total-steps 200000 --screen-episodes 0 --diagnostic-eval-episodes 4 --device cuda:0 --tag stage4a_speedopt_200k_20260809 --artifact-root <writable-artifact-root>/stage4a
nice -n 10 env PYTHONPATH=src:. python3 tools/run_continuous_ctde_training.py --config configs/experiments/positive_feedback_ladder_20260807/stage4c_capture_aw.yaml --scenes capture --seed 2026080801 --total-steps 200000 --screen-episodes 0 --diagnostic-eval-episodes 4 --device cuda:1 --tag stage4c_speedopt_200k_20260809 --artifact-root <writable-artifact-root>/stage4c
```

实际启动时分别放入独立 tmux，stdout 指向各自 artifact root；不得复用 tag 或覆盖原线目录。

### P3：完成 capture 难度阶梯

1. 若 4C 得到明确 moving+obstacle capture/geometry anchor，直接以最佳 Actor 初始化 Stage4D（恢复 local visibility），两 seed 25k，积极则延至 50k。
2. Stage4D 通过后进入 Stage4E（恢复 support/coverage role 与 64/8/8 focal 配额），同样 25k→50k gate。
3. 若 4C 到 200k 仍无方向性，回到 4B moving/no-obstacle 做单变量定位；若 4B 成功而 4C 失败，瓶颈是避障；若二者失败，优先合法 capture snapshot curriculum，仍由新环境生成 transition。

### P4：完成 4v1 双任务与真实 mixed

1. Stage5A：从最佳 Stage4E Actor 初始化，先跑 `actor_only_init`；用一条短对照验证是否需要 full trainer init。capture/pure-CE 任何一项完全遗忘即停止扩步。
2. Stage5B：真实 capture→post-capture coverage，保留 local sensing、1 obstacle、原 reward、真实 phase transition 和 64/8/8/32/16 replay。
3. post-capture 数据不足时启用已授权的 legacy capture/post-capture geometry snapshot reset；禁止伪造标签或导入旧 transition。
4. 4v1 完整 gate：两 seed 中至少一条稳定非零 capture，capture 后 CE energy/CV 明确改善；最终选优点必须同时有 mixed capture 与 post-capture coverage，且 collision 可控。

4v1 配置已存在：`stage4d_capture_aw.yaml`、`stage4e_capture_aw.yaml`、`stage5a_alt_aw.yaml`、`stage5b_mixed_aw.yaml`；启动前仍需按上游最佳 checkpoint 生成独立初始化 override，不能直接把过程配置冒充最终配置。

### P5：扩展最终规模 8v2 → 12v3

- 用 4v1 Stage5B 最佳共享 Actor warm start；central critic/optimizer 是否复用先做小规模单变量对照。
- 先 8v2，再将最佳 8v2 注入 12v3；保留 local VCT-LS、support credit、mixed phase 和连续 `(a,ω)` 合同，只改变规模及对应 post-capture window。
- 每个规模按 capture、pure coverage、mixed 分别做 deterministic/stochastic eval20，并记录多目标分工、first discovery、support 命中率、capture 后 CE 恢复和 collision。
- 最终交付标准：12v3 mixed 中 capture 与 post-capture coverage 均可重复出现；若只到 4v1/8v2，必须明确规模阻塞层，不能以单次 rollout 代替完成。

当前 positive-feedback ladder 尚无 8v2/12v3 连续 MASAC 正式配置；4v1 Stage5B 通过后需先生成配置并验证 max_agents=12 下的 actor/critic/replay shape contract，再开 warm curriculum。

### 可选而非关键路径

- Stage6 body `[a_x,a_y]`、Stage7 world `[a_x,a_y]`：在 `(a,ω)` mixed 主线空闲 GPU 上再做，不阻塞上述最终场景训练。
- MATD3/MADDPG、grad clip、reward/update ratio 修改：只有同一简化合同多 seed 失败且实现/数据诊断排除后，提交单变量方案请求确认。

## 6. 时间估计与决策节奏

25k 稳态吞吐尚未完成测量，不给虚假的固定完成时间。若优化版稳定在 30–60k steps/h：

- 单条 200k 约 3.3–6.7 小时，两 GPU 并行不叠加墙钟；
- 每个 25k–50k、两 seed 子阶段约 0.5–1.7 小时训练，加评估与诊断；
- 4v1 Stage4D→5B 在所有 gate 首次通过的理想情况下约 1–2 天；任何失败归因、snapshot curriculum 或 8v2/12v3 扩展会增加时间。

以上是容量计划，不是成功保证。每 25k 只做三类决策：继续、选优并晋级、或停止并定位；不再让无方向性长训机械跑满。
