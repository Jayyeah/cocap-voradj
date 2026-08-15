# CoCap-VorAdj 全面审查、训练提速与最终场景最短计划（2026-08-09）

## 2026-08-11 17:45 All-Agent 主线规则切换

- Baseline B 已按用户决策在最新完整100k bundle定向停训，完成 Actor-Q exact VJP显存修复和严格零差异 gradient/Q/loss Gate；peak allocated约22.46 GiB→8.05 GiB，稳态update time未退化。commit `f19dac6`。
- B从100k原位恢复为no-clip；此前 `<100k` 仍是clip=.5历史。101--102k均finite，显存稳定，早期critic spike已回落，故不启用5.0 fallback。B现在是 Standardized All-Agent Development Line，A只作 Historical Focal Reference。
- P0显示旧clip对critic/actor触发率均100%，critic典型只保留约0.13% raw norm；critic counterfactual严格seek>policy>random仅21.9%，post-capture replay为0，entropy/Q约0.2%--0.4%。完整证据见 `docs/SAC_HEALTH_AUDIT_20260811_ZH.md`。
- 独立P1 no-clip-from-scratch CUDA32与2k/469-update optimizer preflight均finite，peak约8.05 GiB；正式200k已于18:03从step 0启动（PID 1171022 / tmux `p1_allagent_noclip_scratch_200k` / cuda:0），无任何resume或warm start。当前不启动Huber、UTD/LR sweep、MATD3、Formal Capture-Only或global visibility。

## 2026-08-11 10:08 状态、信号与ETA更新

### 完结线结论

- Stage4A结论不变：200k clean/finite，final 4/4 capture且0/4 collision；统计更强的当前最佳仍是50k eval20（19/20 capture、1/20 collision），50k与200k均保留。
- Stage4C结论不变：200k clean/finite，但所有里程碑capture均0/4，final collision 2/4；训练实现正确、任务学习失败。

### 在训线

- Baseline A：180k，最新175k checkpoint。capture从25k到175k始终0/4；50--150k多次出现正distance progress，但collision长期为2--4/4，175k进展回落到+3.47。pure-CE在50k strict 2/4、100k CV<0.20 4/4后，125--175k退化为strict 0/4、CV<0.20 2/4。结论：有coverage和接敌能力，但capture失败且后期非单调退化。
- Pure-CE：134k，最新125k checkpoint。最强75k为strict 1/4、CV<0.20 4/4、collision 0/4；100/125k回落为strict 0/4、CV<0.20 1/4与2/4。存在明确但非单调的积极信号，75k暂为最佳候选。
- Baseline B：54k，25k/50k均finite且storage/model-only合同正确。50k pure-CE为strict 0/4、CV<0.20 3/4、collision 0/4、CE progress +0.0642；capture仍0/4、collision 4/4、distance progress +9.83。存在coverage/接敌信号，但围捕和碰撞问题未解。
- A/B同step早期比较：Focal A在25/50k pure-CE strict分别1/4、2/4，All-Agent B均0/4；50k capture两者均0/4且collision 4/4，A/B distance progress为+14.79/+9.83。当前早期证据偏向Focal的coverage样本效率，但每点只有4 episodes且两线均未学出capture，不能提前裁决。

### ETA

- Baseline A：200k训练约今日11:40--12:00；完整final report约12:30--14:00。
- Pure-CE：200k训练约今日16:30--17:15；完整report约17:30--19:00。
- Baseline B：75k诊断约今日13:15--13:45；200k训练约08-12 07:00--09:00，完整report约09:00--12:00。
- 最近吞吐约为A 12.0k/h、Pure 10.0k/h、B 6.7k/h；GPU0/1为85°C/92°C且均满载，根盘余约49 GiB。

## 2026-08-11 04:00 完结结果、在训信号与ETA

### 已完结

- Stage4A 于01:04生成clean 200k report，48,751次更新全部finite。final 4-episode diagnostic 为 capture 4/4、collision 0/4、平均完成37.5步、平均接敌进展+19.97、平均最小敌距8.37。25k序列在50/125/150/200k均为4/4 capture，说明围捕能力可重复出现但checkpoint非单调；统计更强的既有50k eval20为capture 19/20、collision 1/20，因此50k仍是当前“最佳已充分验证”模型，200k final保留为强final候选但尚缺同规模eval20。
- Stage4C 于02:01生成clean 200k report，48,751次更新全部finite，但所有25k里程碑capture均为0/4；final为capture 0/4、collision 2/4、平均速度0.116、平均最小敌距14.55。结论是训练链路正确但任务学习失败，不应晋级为capture anchor。
- Stage4A final trainer/replay/runtime已严格验证为同一200k步；仅旧milestone replay被清理23,471,987,148 B，所有里程碑trainer/eval及final/resume replay均保留。

### 未完结

- Legacy-VorAdj Baseline A：107k，最新100k checkpoint。coverage信号明确：25/50/75/100k pure-CE strict为1/4、2/4、1/4、1/4，CV<0.20为2/4、3/4、3/4、4/4，CE progress始终约+0.061至+0.069。capture仍0/4，但50/75/100k distance progress为+14.79/+13.95/+10.85；同时collision为4/4、2/4、3/4，属于“接敌学会、围捕未成、碰撞严重”的混合信号。
- Pure-CE：73k，最新50k checkpoint。strict仍0/4，但CV<0.20由25k的0/4升至50k的2/4，collision由2/4降至1/4，CE progress由+0.0108升至+0.0524；存在弱积极信号，尚未达到严格成功。
- Baseline B all-agent：01:35在cuda:0正式启动，当前14k、2,251次更新，全部finite；自然uniform batch约mixed:pure=63:65、512 active-agent loss terms/update、约1.109 env step/s（约4.0k/h）。目前只有数值稳定、无当前窗口碰撞和采样合同正确的系统信号，尚无25k行为评估。

### ETA（按最近稳定吞吐）

- Baseline A：训练200k约08-11 13:00--14:00；完整final评估/report约14:00--15:30。
- Pure-CE：训练200k约08-11 23:30--08-12 00:30；完整report约08-12 00:30--02:00。
- Baseline B：25k首个诊断约08-11 07:00--07:30；训练200k约08-13 03:00--07:00，完整report约08-13 05:00--09:00。25k后用实测里程碑吞吐再次收紧。
- 当前GPU0/1为86°C/93°C且均100%利用；根盘余约57 GiB。以上区间已计入共享GPU与checkpoint诊断开销，但final长评估仍可能造成额外波动。

## 2026-08-10 22:17 Baseline B GPU调度修订

- Baseline B 固定在 Pure-CE 所在 `cuda:0`，不再动态选择 Stage4A/C 任一释放 GPU。
- 只等待同卡旧 Stage4A 达到 clean 200k 后执行 CUDA32/2k Gate并启动；Stage4C/cuda:1 即使先结束也不会触发。
- Baseline A 继续独占其当前 `cuda:1` 份额，不与 Baseline B 竞争。旧 supervisor 已单独停止以加载新逻辑，四条训练线全部保持运行。
- 35项相关回归通过；新 supervisor PID `542373` 已于22:17:55重新武装，状态只等待 `Stage4A`，Baseline B 尚未启动。

## 2026-08-10 22:13 五线状态

- Stage4A：PID 240128 / cuda:0 / metrics 181k，175k checkpoint 已落盘；175k diagnostic capture 3/4、collision 1/4，仍有明显围捕能力但非无碰撞。
- Stage4C：PID 240122 / cuda:1 / metrics 169k，最新完整 checkpoint 150k。
- Legacy-VorAdj old-mix Baseline A：PID 224386 / cuda:1 / metrics 52k，50k checkpoint 已落盘。50k pure-CE strict 2/4、CV<0.20 3/4；capture/mixed 都是 0/4 capture，但 distance progress 分别 +14.79/+15.85，同时 collision 4/4、3/4，说明出现接敌进展并伴随严重碰撞退化。
- Pure-CE：PID 224391 / cuda:0 / metrics 36k，最新完整 checkpoint 25k。
- 四条现有线最新窗口均 `mean_finite=1`。Baseline B 尚未启动；supervisor PID 454469 持续健康等待 Stage4A/C 任一条 clean 200k，正式 tmux/run artifact 均不存在。
- 资源快照：GPU0 85°C / GPU1 92°C、均 100% utilization；根盘余约 57 GiB。监督器按设计未抢跑 CUDA/2k Gate。

## 2026-08-10 20:34 五线状态与 All-Agent 对照队列

- Stage4A：PID 240128 / cuda:0 / metrics 170k；最新完整 checkpoint 150k，尚未到 175k。
- Stage4C：PID 240122 / cuda:1 / metrics 153k；150k checkpoint 已落盘。
- Legacy-VorAdj old-mix Baseline A：PID 224386 / cuda:1 / metrics 35k；25k checkpoint 与三场景自动评估已落盘。
- Pure-CE：PID 224391 / cuda:0 / metrics/checkpoint 25k。
- Baseline B standard all-agent 尚未启动。tmux `allagent_oldmix_ablation_supervisor` / PID 454469 健康轮询，等待 Stage4A/C 任一条 clean 200k 后，再自动执行 CUDA 32-step、2k 吞吐/显存 Gate；全部通过才启动正式 200k。
- 25k 早期信号：Baseline A 的 pure-CE 为 strict 1/4、CV<0.20 2/4、平均 CE energy progress +0.0608；capture/mixed 均 0/4 capture，但 detected 4/4、collision 0/4。独立 Pure-CE 为 strict 0/4、CV<0.20 0/4、collision 2/4、平均 CE progress +0.0108。样本仅 4 episodes，记录为早期诊断，不触发停线。
- 资源快照：GPU0 84°C / GPU1 92°C，均 100% utilization；根盘余约 63 GiB。监督器未抢跑或修改四条现有线。

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

## 7. 2026-08-12 All-Agent old-mix路线同步

- 详细证据以 `ALL_AGENT_MASAC_OLDMIX_ABLATION_20260810_ZH.md` 和 `SAC_HEALTH_AUDIT_20260811_ZH.md` 为准。
- Baseline B已于 `2026-08-12 08:45+08:00` 完整达到200k；100k后fixed Actor-Q + no-clip续训全程finite、peak VRAM约8.05 GiB，但final capture/mixed eval的capture rate仍为0，replay中无post-capture transition。工程修复成功，formal capture研究问题未解决。
- P1 no-clip from scratch的125k checkpoint/eval已于 `2026-08-12 10:18+08:00` 完整落盘，随后继续到126k；数值稳定且出现多次单机3.6--9m接敌，但125k capture/mixed eval仍为0 capture，亦无双机/三机ring或post-capture样本。B结束后普通训练窗口吞吐约13.4--13.7k step/h，预计当日16:00--16:30完成200k final artifact。
- 当前不将MASAC/CTDE/reward/MSE任一单点提前定性为根因；但B结果已降低了“只要取消`.5` clip并延长训练就会capture”的可信度。P1完成后应与P0 critic Q-ranking合并决策UTD或Formal Capture-Only诊断。

### 7.1 193k新证据（2026-08-12 15:25+08:00）

- P1在180k首次产生一个完整真实capture→post-capture coverage window：replay `post_capture_coverage=2000`，对应500 joint transitions×4 active agents，并伴随双机同时入ring；这是formal old-mix中此前未出现的明确正信号。
- 该信号仍然稀有：150k/175k deterministic diagnostic capture rate为0，尚无3+ ring或重复评估capture。P1已到193k，预计16:10--16:35完成200k final artifact，随后用final diagnostic和正式20-rollout判断可复现性。
- Baseline B 200k final的old-mix deterministic 20-rollout/5-GIF已在CPU后台启动；本轮不等待结果，下次状态更新再同步。


### 2026-08-13 01:00+08:00 自动线方向更新

C1 UTD=.5已在完整225k checkpoint/replay/4-episode diagnostic后停止：200k后无新real capture、无2+/3+ ring，不再扩到300k。GPU1已立即从scratch启动CF1 Global Enemy Broadcast 100k；GPU0的CF0 Local不再受100k Gate分支控制，100k冻结后固定原位续到200k。当前只比较local vs verified global enemy information，不改reward/horizon/LR/UTD/tau/action/network。详细合同、PID与证据见docs/ALL_AGENT_MASAC_OLDMIX_ABLATION_20260810_ZH.md第10.16节。


### 2026-08-13 06:45+08:00 CF0/CF1中期证据

CF1 Global在50k rolling replay中首次确认1个real stationary-fallback capture（transition 44892），28k出现3+ ring且多个独立窗口重复2+ ring；50k deterministic 4-episode仍为0 capture，但collision从25k的1.0降至0.5、distance progress由+7.35提高到+14.38 m。CF0 Local截至90k无capture/3+ ring，75k deterministic capture=0且distance progress退化到-1.33 m。当前证据支持global enemy information带来更强的多机几何与一次真实探索capture，但尚不足以宣称策略已稳定学会。两线合同不变并继续到既定100k/200k，详细数值见专用台账第10.18节。


### 2026-08-13 07:40+08:00 CF2 support-credit单变量启动

CF0 Local已按最新决策在完整100k trainer/replay/runtime冻结后停止，不再续200k；final formal 20-rollout capture=0且collision=1.0。GPU0随即从scratch启动CF2：完全继承CF1 Global合同，只以Legacy实际邻接定义capture/support/coverage，并让support获得`1.0*full capture + 1.0*full coverage`；global observation与任务角色明确解耦，CF1行为回归不变。人工邻接角色/reward测试、CUDA smoke及首个正式update均通过。CF1到68k再次出现3+ ring窗口，当前仍为1次stationary capture、无normal capture。详细实现、在线角色reward分量、PID、资源和ETA见专用台账第10.19节。


07:41稳态复核：CF2已到6k/251 updates，`3.782 step/s`、finite=1、peak allocated约8.05 GiB，support capture/coverage两分量持续非零；CF1到69k、`2.675 step/s`、finite=1。两条正式线与GPU/RAM正常。


### 2026-08-13 12:15+08:00 CF1 final与CF2 normal capture

CF1 Global已clean完成100k：训练内仅1次stationary-fallback capture，formal 20-rollout仍0 capture/20且collision 20/20；但全程19个2+及3个3+ ring窗口确认global信息改善多机几何。CF2 Global+Support到53k产生1次明确normal K3 capture（非stationary、非collision），截至55k已有9个2+和1个3+窗口；同55k下几何频率与capture类型优于CF1，但累计collision更高。CF2继续到75/100k验证能否重复normal capture，暂不提前扩步。详细指标与ETA见专用台账第10.20节。


### 2026-08-13 13:00+08:00 双P0更新：CF2扩200k、CF3 Local-Support

CF2已挂100k完整冻结后同trainer/replay/runtime/RNG原位续到200k的自动接力，125/150/175/200k继续诊断。CF3从scratch在GPU1启动：严格继承CF2角色/reward和全部SAC/环境合同，唯一核心变量是关闭global enemy broadcast；强制测试确认support无enemy token但正确看到capture友邻position/velocity/`is_pursuing=1`，仍拿完整capture+coverage。新增support observability/follow、角色reward分位数、distinct capture和ring hold只读诊断。当前优先级为CF2 Global-Support 200k与CF3 Local-Support 100k并行；winner达到至少3次独立normal capture或formal20 normal capture≥20%且不依赖stationary后，才恢复post-capture coverage；其后依次做capture reward与动作空间ablation。暂不启动新reward、post-capture、`vx,vy`、UTD1、LR sweep或MATD3。详见专用台账第10.21节。


12:55正式状态：CF2为63k、10.75k step/h、finite；CF3正式scratch为5k且首update finite，support无enemy token/有pursuing friend比例均100%，双reward分量与新增follow/ring-hold诊断正常。双GPU显存/RAM/storage安全，CF2自动扩200k监督器健康挂起。


### 2026-08-13 14:50+08:00 P0默认修复与Local-Max路线

- P0定位改为默认代码/语义修复，不单独训练baseline：old-mix默认`min_active_pursuers=2`、`coverage_ce_min_active_pursuers=2`；raw adjacency只作诊断/真实token可见性，K10 effective role统一驱动Actor role bit、reward三角色、replay label与coverage hold。回归验证3 active和2 active不因too-few结束，只有1 active结束；K10期间不再出现obs/replay=capture而reward降为support/coverage。
- 有normal capture与重复多机几何的CF2不重训：完整75k bundle冻结后切到P0-fixed并继续200k。CF2 0--75k标记pre-P0，75k后的新transition标记P0-fixed；首个post-switch 76k窗口finite且2.98 step/s，无OOM或吞吐退化。旧75k replay保留并随uniform新样本自然衰减。
- 局部CF3保留为主reference：pre-P0到25k后自动冻结并以P0-fixed续到100k。当前22k已有5个2+窗口，support无enemy token但持续看到pursuing friend，friend-distance delta与enemy progress均为正向；尚无capture/3+，不提前定性。
- CF3 100k完成后自动启动scratch P1 Local-Max。任务合同完整继承corrected Local CF，唯一网络变化是Actor从`self+mean`变为`self+mean+max`；hidden256/heads8/layers4和central critic不变，不恢复residual/attention/role embedding。P1先100k，只有normal capture或持续2+/3+几何Gate通过才续200k。
- 当前固定路线：P0代码修复→正信号CF continuation→Local CF reference→P1 Local-Max→capture稳定Gate通过后恢复post-capture coverage→capture reward ablation→`(a,w)` vs `vx,vy`。当前仍不启动新reward、post-capture混训、UTD1、LR sweep或MATD3。详细切换点、测试、PID、性能与ETA见专用台账第10.22节。

15:05状态补充：CF3已完整冻结pre-P0 25k bundle并成功以严格`p0_semantics`从25k续训，真实audit只有两个min-active 4→2差异；新PID 2280607在GPU1正常运行。pre-P0 25k为0 capture、6个2+窗口、0个3+。CF2 P0-fixed已到79k且finite。GitHub远端当前提交为`84ff743`，本补充将以后续仅文档提交同步；完整数值见专用台账10.22末尾。


### 2026-08-13 16:45+08:00 CF2提前收口、CF3扩200k与P1改接

- CF3 local-support在pre-P0 0--25k已有6个2+窗口；P0-fixed 26--33k又有5个2+窗口，support在100%无enemy token、100%有pursuing friend条件下，平均向friend和enemy同时靠近并持续升级capture role。尚无normal capture/3+，故这是local信息足以产生方向性双机协同的证据，而不是K3已训通。
- 据此CF2 Global-Support不再机械跑200k；到完整100k冻结后停止，保留25k P0-fixed global reference。GPU0随后立即开P1 Local-Max scratch；P1及其100k PASS extension均固定GPU0。
- CF3 local corrected reference改为100k后自动原位续200k，固定占GPU1。重接旧父supervisor时tmux HUP连带结束了33k CF3 trainer；最近完整bundle为25k，已从该完整trainer/replay/runtime/RNG以严格P0 fork在独立recovery artifact恢复。旧26--33k metrics保留作证据，但模型从25k重放，台账不重复累计。
- 当前优先级更新为：CF2 100k收口→P1 Local-Max；CF3 Local-Support corrected 200k→与P1比较mean vs mean+max；达到稳定normal capture Gate后才恢复post-capture coverage。详细PID、指标、故障和ETA见专用台账10.23。

17:15实际交接：CF2 100k frozen trainer/replay/runtime完整且finite，100k diagnostic仍0/4 capture、4/4 collision，已按决策停止；P1 Local-Max smoke通过并在GPU0 scratch启动，6k首update窗口finite、10.74k step/h。CF3 recovery在GPU1到27k，finite且再次出现0.9%的2+ ring与support同时靠近friend/enemy，100k后自动续200k。两卡资源正常，详见专用台账10.23末尾。


### 2026-08-13 19:31+08:00 Local mean vs mean+max中期趋势

- CF3 corrected Local-Support已到41k并出现本阶段首个明确3+ ring：38k的any/2+/3+ fraction=`16.4%/2.4%/0.1%`、2+ hold=12、3+ hold=1；36--41k有5/6窗口重复2+，平均any-ring由26--30k的5.26%升到11.98%，平均d1由22.76降到20.90m。support在无enemy token下持续同时靠近pursuing friend和enemy。尚无capture，3+只持续1 step，collision约由8.2升到9.8次/1k，所以结论是“局部多机几何明显转强、K3闭合仍未完成”。
- P1 Local-Max到31k，仅13k出现一次2+、无3+/capture；any-ring由6--10k的0升到26--31k的2.88%，support→friend delta由-0.008增强到-0.117m/step，表明有方向性学习但尚未形成重复多机几何。25k deterministic的min-min/progress=`18.29m/+3.26m`，弱于CF3 mean reference的`10.02m/+12.63m`，当前没有MaxPool优于mean的证据，仍按合同跑满100k。
- 两线critic/Q/TD/grad尺度随训练上升，但均finite、twin Q贴合、peak VRAM约8.05GiB且无增长；不改变clip、LR、UTD或reward。当前瓶颈进一步收窄为“把重复2+和瞬时3+稳定维持到capture，并降低碰撞”，而非support完全缺少局部方向。
- P1约`10.71k step/h`，预计100k于8月14日02:20--02:50完成；CF3约`5.77k step/h`，预计100k于06:10--06:45、final评估后07:00--08:00自动续200k，200k预计8月15日01:50--03:30。完整分段指标见专用台账10.24。


### 2026-08-13 20:31+08:00 CF3 corrected-local首次normal capture

- CF3 recovery在42k出现1个独立normal K3 capture，stationary=0；该窗10次终止中9次collision，另1次为带非零terminal reward的normal capture，排除fallback或碰撞误计。26--47k累计17个2+、2个3+窗口；45k 2+ hold=20，47k再次3+且hold=3。这是local sensing + pursuing-friend support credit首次打通真实K3的证据。
- 该证据仍不稳定：只有1个独立normal episode，42--47k仍有54次collision；尚未满足至少3个normal episodes或deterministic20≥20%的post-capture Gate。CF3保持合同继续100k并自动续200k，验证capture能否被replay放大。
- P1 Local-Max到42k只有13k、40k两次2+，无3+/capture；support follow为正但any-ring和接敌深度明显弱于CF3。当前没有MaxPool优于mean pooling的证据，仍按合同跑满100k。
- 两线均finite、VRAM稳定；Q/TD/grad尺度上升但twin Q贴合，无数值爆炸，不改clip/LR/UTD/reward。CF3 100k预计8月14日06:05--06:40并于07:00--08:00自动续200k；P1 100k预计02:20--02:50。完整数据见专用台账10.25。


### 2026-08-13 21:20+08:00 双50k换卡、MaxPool收紧Gate与PC0安全路线

- CF3/P1均已冻结完整50k trainer/replay/runtime/RNG/manifest；逐文件hash和trainer字段审计确认各自optimizer、alpha、target critics与RNG完整。旧waiter已取消，两条trainer从本线50k bundle换卡：CF3→GPU0并直达200k，P1→GPU1并到100k；CF3冻结后旧GPU多出的未checkpoint 51k窗口仅归档、不冒充可恢复状态。新51k窗口通过后才把恢复标记为100%完整。
- matched 50k下CF3有1 normal、20个2+、3个3+窗口；P1为0 capture、4个2+、0个3+。P1的100k Gate已收紧：必须至少2个独立normal，或1个normal且2+/3+频率均近似达到CF3 matched-step；单次2+或distance改善不续200k。
- CF3稳定Gate仍为至少3个独立normal，或formal deterministic 20-rollout normal capture≥20%，且不主要依赖stationary。P1释放GPU1后supervisor自动检查CF3；未满足则继续等CF3至200k。若200k仍不稳定，进入capture consolidation/collision audit，禁止post-capture、vx-vy、MATD3或UTD1。
- PC0的transition audit已确认capture-only的capture transition为terminal，而恢复300步post-capture会令其成为non-terminal并改变bootstrap。因此禁止继承旧CF3 replay/critics；Gate通过后只warm-start CF3 Actor，使用fresh replay、critics/targets、optimizer、alpha、RNG和runtime。详细hash、PID、资源与ETA见专用台账10.26。

21:27最终核验：CF3/P1换卡后均严格连续到51k/replay51k/update11501且finite，审计状态为100% complete。新实测吞吐为CF3 GPU0 `10.62k step/h`、P1 GPU1 `5.71k step/h`；P1 100k Gate约8月14日06:20--06:50执行，CF3 200k final artifact约12:30--13:30完成。PC0 Actor-only/fresh-value初始化bundle已用真实CF3 50k Actor完成hash审计和1-step strict-load smoke。


### 2026-08-14 02:30+08:00 Host restart恢复

服务器停机导致全部trainer/supervisor退出。CF3停机前metrics到89k但最新完整bundle为75k，故76--89k只归档metrics并从75k重跑；P1最新metrics与完整bundle均为75k。两条75k trainer/replay/runtime/manifest已逐文件hash冻结，CF3在GPU0继续200k、P1在GPU1继续100k，P1/CF3 Gate supervisor也已重挂。双方恢复后首个76k窗口均严格连续为`replay=76000/update=17751/finite=1`，断电审计已达到100% complete。P1在71k新增1次normal capture但仍无3+；CF3可恢复75k仍为1 normal和4个3+窗口。详细恢复点、SHA、PID、资源与新ETA见专用台账10.27。

06:20更新：P1 Local-Max已完整100k并由严格Gate自动`FAIL_STOP_100K`，其matched 25--100k只有1 normal、14个2+、0个3+，显著弱于CF3的2 normal、43个2+、4个3+，GPU1已安全释放。CF3 Local-Mean在97k出现第2个normal且stationary仍为0，当前127k/200k、125k完整bundle已保存；Stable-Gate supervisor正常等待第3个normal或200k formal20，不会提前误开PC0。完整趋势与ETA见专用台账10.28。

09:30更新：CF3 Local-Mean已到169k，150k完整bundle已保存；累计2 normal/0 stationary、90个2+、9个3+窗口。149k将3+ hold提高到11步、2+ hold提高到25步，collision由早期9.16次/1k降至约5.4--5.7次/1k，属于明确的多机几何与安全性改善。但100/125/150k deterministic 4-episode仍全为0 capture/100% collision，且150k distance progress回落，稳定capture Gate仍未通过。当前继续原合同到200k；若完整175k bundle前出现第3个normal，自动器最早在175k后安全启动Actor-only/fresh-value PC0，否则200k执行formal20或capture-consolidation/collision audit。详细阶段表、SAC尺度趋势、资源与ETA见专用台账10.29。

10:35更新：CF3在173k获得第3个独立normal capture（stationary=0），175k完整bundle通过训练Gate；自动器已冻结175k并在GPU1启动PC0，同时CF3继续GPU0到200k。PC0严格只继承Actor，replay/critics/targets/optimizer/alpha/RNG/runtime全部fresh，当前12k且5k首update后持续finite；早期有5个2+、1个3+但尚无capture/post-capture transition。supervisor因把1k无update warmup窗口的`mean_finite`缺失误判为non-finite而写`FAILED_CLOSED`，这是状态假失败，PC0 trainer正常；后续启动验证应等待`update_count>0`。CF3当前183k、累计3 normal/103个2+/11个3+，175k deterministic仍0/4 capture，故200k formal20继续执行。详见专用台账10.30。

12:18更新：CF3完整200k累计4 normal/0 stationary、113个2+和13个3+窗口，但正式deterministic capture 20-rollout仍0/20且20/20 collision。因normal在42/97/173/186k重复、175--200k几何仍活跃且GPU0空闲，已冻结200k完整bundle并原位续训到400k；201k连续性验证为replay201k/update49001/finite。若400k capture频率与formal20仍不改善，则停止扩步并转collision/observability audit。独立固定seed capture20也已完成为0/20、collision20/20、mean min distance3.84 m；5张代表性GIF仍在渲染。PC0到32k已累计3个normal capture及380条post-capture replay索引样本，闭环开始重复产生数据。详细合同、故障、资源和ETA见专用台账10.31。

14:51更新：CF3到234k，225k full bundle完整；200--234k尚无新增normal capture，虽200--225k有21/25个2+窗口、2个3+窗口且平均2+ fraction升至0.824%，实际capture仍是首要未解决指标。225k deterministic仍0/4且全collision；200k fixed-seed formal20仍0/20且5 GIF已全部完成。PC0到61k累计3个normal（27/30/32k）、49个2+和6个3+，33k后无新增capture；post-capture replay只有95条joint transition（约0.16%）。Actor初始化态vs PC0-50k同seed三场景配对诊断已完成：pure coverage的CE误差明显下降，但4/4 collision不变、平均存活1284→588步且CV015/020成功1/4→0/4；capture/mix均0/4，distance progress -1.21→-6.05 m，mix未进入post-capture。因此尚不能宣称coverage或闭环整体提升。后续CF3 formal rollout固定分开报告pure coverage、pure capture、mixed；PC0必须同时满足coverage提高且capture不明显退化。完整指标与ETA见专用台账10.32。

### 10.33 CF3/PC0 最新现场复核（2026-08-14 20:15+08:00）

#### CF3：GPU0 仍在正常连续训练

- `cf3_gpu0_200k_to400k` tmux 会话仍在，PID=`174772`，`cuda:0`，目标 `400000`；当前指标为 `306000/400000` steps、`update_count=75251`、`replay_size=250000`。最新窗口 `mean_finite=1`，速度约 `3.685 step/s`（`13.27k steps/h`），峰值显存约 `8045 MiB`，没有异常退出迹象。
- 最新完整 bundle 为 `artifacts/2026-08-13_capture_first_controls/cf3_local_support_full_p0fixed_recovery/legacy_voradj_cf3_local_support_p0fixed_recovery25k_to200k_20260813/checkpoints/step_000300000/`，滚动恢复包为同目录 `resume_latest/`。
- `300k` 诊断为 `success=0/4`、`capture=0/4`、`collision=4/4`，平均长度 `203.75`，distance progress `8.49`，最小距离 `7.67`。最新窗口无 normal/stationary capture；任一 agent 在环 `12.5%`、2+ `0.5%`、3+ `0%`，最大 hold=`23/5/0` steps。
- 以当前吞吐估算：`325k` `21:40–21:55`，`350k` `23:30–23:50`，`375k` 于 `2026-08-15 01:20–01:45`，`400k` 训练完成约 `03:15–03:45`；最终 screening/三场景评估约 `03:45–05:00`。CF3判定为正常续训，继续跑到400k。

#### PC0：自然完成，但未通过 capture/coverage 质量门

- PC0 已无运行中的 tmux/process，GPU1 已释放；于 `2026-08-14 18:11` 自然完成 `100000/100000` transitions、`23751` updates、`all_finite=true`。完整 checkpoint/replay 在 `artifacts/2026-08-13_capture_first_controls/pc0_cf3_actor_postcapture300/legacy_voradj_pc0_cf3_actor_warmstart_postcapture300_4p1e1obs_100k_aw_20260813/checkpoints/step_000100000/`，含 `trainer.pt`、`replay.pkl`，`resume_latest/` 亦完整。
- capture-only 正式20回合为 `success=0/20`、`capture=0/20`、`collision=20/20`；`coverage_cv015=1/20`、`coverage_cv020=2/20`，平均长度 `102.75`、CE energy progress `-0.0274`、area CV `0.5169`。最终4回合诊断同为 `capture=0/4`、`collision=4/4`。
- replay 虽含 `post_capture_coverage=628` 个 focal slots，但最终均匀 joint batch 的 `post_capture=0`、`pure_ce=0`，且 `915` 次 termination 全为 collision；不能据此宣称已学会 post-capture coverage。
- `control/cf3_stable_gate_pc0_status.json` 的 `FAILED_CLOSED` 是监管器把1k warmup的 `update=0` 缺失字段误判为 non-finite 的陈旧假失败；trainer实际从5k到100k全程 finite 并自然退出。PC0无需续训，是否重开由人工决定。

### 10.34 CF3/PC0最终结论与路线切换（2026-08-15）

- CF3 Local-Support已完整自然结束400k，累计8 normal+1 stationary，但normal按每100k恒为`2/2/2/2`；2+窗口增加、collision/1k从6.96降到4.51，而3+窗口后段回落。final deterministic formal20为`0/20 capture、20/20 collision`。结论是uniform replay下更多step没有把稀有随机成功固化为确定性策略，不再扩500k。
- PC0 Actor-only warm-start+post-capture300已完整自然结束100k，训练内5个normal，但只生成157条joint post-capture transition（0.157% replay），每次capture后最多存活83/300步且均被collision终止；final deterministic capture20同为0/20且全collision。PC0证明transition plumbing，不证明coverage或闭环成功，不原样扩训。
- CF3 final前5个capture GIF的14次追击者失活中11次为agent-agent，主要形态是多机争抢同一接敌点；这与共享单峰Gaussian Actor无持久slot、训练std较大而deterministic mean全失败、成功terminal在uniform replay中约每279次update才命中一次共同指向结构性瓶颈。
- 原自动链缺少CF3@400k与PC0@100k的pure-coverage/mixed正式评估；已按精确checkpoint/config补完paired三场景各20统计。两线capture/mix均0/20且collision20/20；CF3/PC0 pure-CE strict均0/20，CV.15=`3/20 vs 1/20`、collision=`7/20 vs 19/20`。PC0没有同时提升capture和coverage。两线15/15 GIF均已生成；PC0 visual完整退出，CF3 visual wrapper仍在正常完成剩余非渲染回合，但独立stats和GIF交付均完整。输出位于`artifacts/2026-08-15_final_triscene_20rollout5gif/`。
- 新优先级：先做temperature/stochastic mode、collision语义、ORCA/CBF safety-only、IQN/解析ring-slot teacher feasibility四个短探针；随后以显式slot/option+assignment、训练一致安全投影、teacher蒸馏、success episodic replay/n-step与必要时team critic/centralized-value MAPPO冲击formal deterministic高成功率。capture正式Gate建议3 seeds×100、normal≥70--80%、collision≤10--20%；过Gate后以capture/coverage双option FSM和phase-balanced fresh replay恢复post-capture。
- 完整量化与执行Gate见`docs/CF3_PC0_FINAL_AUDIT_AND_HIGH_SUCCESS_PLAN_20260815_ZH.md`。当前不再优先CF3扩步、PC0原样重跑、global/maxpool复试或LR/UTD/Huber/batch/MATD3/动作空间盲扫。
