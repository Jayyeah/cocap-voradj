# Forward Final IQN → categorical Actor bridge

更新：2026-09-09。起点及额度恢复后再次 `git pull --ff-only` 均为 `f890b33a3ae67722e470f0528fb5ed2b985cc6b7`；远端 main 仍为 `a5814f49fa29d869cdc3fb8d8e0df4722aa11f00`。本页覆盖旧台账的下一实验建议，不修改历史结果。

**当前：C0 PASS；C1 PASS；C2 完成；C3 正在运行，尚无完整迁移 PASS。PPO 未启动。** 旧 Pure-Capture A/B 仅 OPTIONAL_DIAGNOSTIC。双 GPU 仅用于三种冻结策略的明确比较，没有扩展算法树。

## 1. Forward contract 与实现边界

合同 ID：`forward-final-aw9-4v1-swept-v1`。新配置 `configs/experiments/forward_final_mappo_20260908/stage1_4v1.yaml` 只覆盖 `env.collision_semantics: synchronized_swept_v1`。使用 Final release Stage1 YAML 合并 `tasks.voradj` / `tasks.voradj_coverage`，不经过旧 AC ladder、`configure_environment` 或 Pure-Capture override。

`training/forward_final.py::preflight` 比较 canonical main 隔离审计保存的 resolved config，检查 mixed / coverage 两场景 historical→forward 的配置差异和 live facts 差异均只有 collision。实际环境 getter 验证 CR-MS、support blend、CE、sensing、friend-only 构图、AW 物理参数及 phase/terminal。手工真实围捕几何测试进一步确认 capture 后不 terminal、3名参与者获得 terminal reward、后继 CE 分支可达。此前 main 隔离审计已建立的代码/信息探针证据沿用，不再重做历史审计；有限探针并非对所有状态的形式等价证明。

| 实际项 | Forward frozen runtime |
|---|---|
| map / task | 120×120，mixed 4v1/1obs；pure coverage 4v0/1obs |
| sensing/topology | enemy、obstacle 20m surface clearance；friend-only Voronoi，free-mask projected；无 global enemy token；地图已知、友军构图共享假设继承 Final |
| direct capture | `ring_importance_ms_v0`，`omega_ring_ms=2`；旧 approach/MS/front 权重0，capture timestep0 |
| support | 0.5 × approach-only（独立 approach 权重1、clip3）+ 0.5 × CE；neighbor-visible target |
| coverage | centroid-energy / PBRS，scale10，kappa1，gamma.99；运行 reset speed penalty0，见下述调度边界 |
| physical action | integer AW9 → Robot；a±.4，w±π/6，vmax3，drag .4/3；dt.05 ×10 = .5s |
| evader | Final trainer 同款 APF `v2_fixed`，vmax3.5，直接调用 APF；不依赖旧 AC autonomous 开关 |
| capture/phase | K3，R8，角间隙≤π、比≤3；stationary min2 / speed≤.2 / hold10；capture 后继续 coverage |
| terminal | 总 horizon3000；post-capture window500；min-active4；CE RMS≤.05/max≤.10 持续30步 |
| correction | 仅把历史 legacy end-step collision 换为 synchronized swept；Historical main 仍保留原合同用于复现 |

**两个训练分布/调度边界必须保留：**

1. 本轮 pure coverage 使用 canonical task 的 `inner_random_cluster` reset；mixed 完整轨迹自然包含真实 capture→recovery。尚未复刻历史 trainer 的 capture-snapshot recovery pool / map-random 重采样比例。因此是明确的 4v1 Full-task evaluation/collection 分布，不是原课程训练分布的逐样本复现，也未覆盖8v2/12v3。
2. Final YAML reset 的 `coverage_ce_speed_weight=0`，历史 `CoCapTrainer` 在200k后设为0.0005。本轮冻结环境没有训练时钟，C1 reward 保存实际值0；固定 teacher 动作和 CE success 判据不依赖这个 reward 系数。C3 三者一致。**进入 D 前必须明确 central-V/PPO 的 CE schedule 时间锚及 recovery reset 机制，不能把 reset 默认值误称 Stage1@2m 的训练末期奖励值。** 保留 Final 调度机制与这种时钟差异是后续训练 adapter 的必做检查。

训练-only global_state 存在 dataset 中，C2 actor 只读取 `local_obs.*`。没有向 actor 增加 enemy 广播。central critic 尚未接入本轮新 runner，不能把“categorical BC 完成”写成“Final MAPPO PPO 成立”。

## 2. C0 Teacher qualification：PASS

Teacher：Final release **Stage1 4v1@2m**，不是旧 Pure-Capture BC 的 Stage2 teacher。

- checkpoint：`artifacts/2026-08-04_crms_vctls_ce_final/checkpoints/stage1_4v1_step_2000000.pt`
- SHA256：`2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89`
- greedy，32 fixed-midpoint IQN quantiles；不随机采 τ；无参数更新。
- 20 mixed + 20 pure coverage，新 seed base `2026092101`，coverage 加100000。
- 预声明 Gate：每场景≥20；mixed capture/normal/CE/safe 与 pure CE/safe 均≥.9，碰撞≤.1。明显崩坏则停止，临界才扩50。

| 指标 | mixed | pure coverage |
|---|---:|---:|
| normal / stationary / total capture | 20 / 0 / 20 | N/A |
| CE / safe full-mission | 20/20 / 20/20 | 20/20 / 20/20 |
| collision / boundary | 0 / 0 | 0 / 0 |
| full mission mean / P50 / P90，秒 | 111.7 / 105 / 161.75 | 73.8 / 69.75 / 94.4 |
| capture mean / P50 / P90，秒 | 41.225 / 38.75 / 62 | N/A |
| capture→CE mean / P50 / P90，秒 | 70.475 / 66.75 / 95.85 | N/A |

Capture 是 mixed 真实轨迹的前缀统计，不另截断 episode 再混用不同 evaluator。same-target closure/support 的逐事件完整/失效/删目标/删失记录及统计在 `c0_formal20/report.json`。完成时间为成功条件统计；同时报告全部回合观测时长、失败/删失数和 safe completion CDF，避免把失败提前终止误作更快。

**CONFIRMED：** teacher 没有因 forward collision 修正明显失效，足以进入 C1。20回合的0失败不代表总体零风险。C0未做 historical-vs-forward 配对差值实验，不能从本结果估计 collision 修正的净性能收益。

## 3. C1 完整轨迹 dataset：PASS，但不是穷尽状态覆盖

先 pilot 12对通过比例/事件 Gate，再采100对；所有轨迹完整保存，不按成功筛选。pilot seed base `2026093101`；正式数据 `2026094101`，均与 C0、C3 分离。

- 正式 200回合，146,612 active-agent rows。
- pre-capture 32,132（21.9%）；post-capture 59,036（40.3%）；pure coverage 55,444（37.8%）。
- direct 9,473；support 9,999；coverage 127,140；独立 pursuing-memory 桶0，不宣称该桶已覆盖。
- 未发现目标11,508 rows；首次本地检测322；near-capture5,532；capture transition400。
- 近障碍/边界/友机 proxy586 rows；collision transition8 rows；两条失败 episode共488 rows。proxy 的4m阈值是诊断标签，不是“即将碰撞”的物理证明。
- Q top1-top2 gap P10/P50/P90：.001275 / .008574 / .140637。动作直方图完整保留；action4没有 teacher greedy label，未人为补齐动作分布。
- 采集种子结果：mixed normal capture100/100，safe99/100，boundary1/100；pure CE/safe99/100，boundary1/100。这是采集分布描述，不替代 C3 fresh formal。

数据含 local_obs、Q[9]、greedy、reward、done/terminated/truncated、episode/timestep/agent、phase/role、direct/support/pursuing、target ID/masks、检测/捕获/近风险及整回合失败标记。global_state 按 joint transition 存一次，agent row 用 episode-local `transition_index` 引用，**不能按 agent row 拼接 global_state**。capture 事件另存 episode JSON。target_id 是 evaluator 的本地/邻居可知目标关联，不是新增 actor 输入或全局最优 assignment。Time-limit truncation 使用 native infos；post-window等任务结束仍按原 env done 语义保存，D需保持 GAE 的对应解释。

`manifest.json` 记录每个 NPZ SHA、布局和 parent；distribution audit 检查三个phase比例、direct/support、检测/捕获/近捕获及失败/近风险存在。bulk Gate 全通过。真实失败稀少、teacher-policy 外状态未覆盖；这是后续 BC 闭环偏离的可能原因，不能靠“数据量够大”排除。

## 4. C2 冻结 backbone categorical head：完成，尚非迁移 PASS

- 从该 Stage1 teacher 严格复制86个 backbone keys；每epoch逐bit检查，参数全部 frozen，所有 forward 保持 eval mode。
- 不创建旧 AC environment。head 是现有 categorical AW9 Actor。
- encoder 与 IQN decision feature 数值一致；缓存 feature 后只训练 head，缓存/full actor logits 检查通过。
- 30 epochs，batch2048，Adam lr3e-4；无超参搜索、无 PPO。CPU/GPU特征缓存使此轮约11秒完成。
- `episode_pair %5==0` 整组留出；train116,316 / validation30,296 rows，避免同回合/同种子pair跨集合。
- KL soft-Q target。仅在 train 中按phase等量抽6,144 rows，以目标entropy .35 nats标定 **T=.0031884169327952754**；全验证集teacher target实际entropy .3797，并非所有phase均等熵。未把T=1照搬过来。

验证：action agreement **89.33%**；pre-capture88.60%、post89.68%、pure89.33%；direct87.03%、support90.62%。train89.14%，没有明显的 train-val 分离；这不识别优化不足与 head 近似限制。验证 sample entropy .5960 高于 teacher target .3797。epoch30仍为所选最好点；没有据此自动解冻 backbone 或切 PPO。

Best：`c2_distillation/actor_epoch_030.pt`，SHA **7916a970226a0452a8f71a22235524f6ba4511aa4c9bda907a9c8368a60bf2cd**。其余有效epoch checkpoint均保留。这里只证明监督拟合有信号；是否保住 full-task 闭环行为由 C3 判断。

## 5. C3 formal100：RUNNING，无 PPO Gate 结论

同一个 `run_episode` 执行 IQN greedy / BC argmax / BC sample。fresh base `2026096101`，每种100 mixed +100 pure coverage；coverage加100000。BC两分支同一 SHA。actor每回合验证 eval-forward、all-action zero-update ratio与物理grid；sample按episode独立固定torch seed。配对汇总必须同时匹配 scene/seed/initial-state fingerprint/evaluator SHA。

- GPU0：IQN greedy完成后串行BC argmax。
- GPU1：BC sample。
- 所有 formal source 启动后保持固定；GPU0入下一mode前再次检查源码SHA。两队列完成后只汇总 comparison，不启动 D。
- 一级 safe/capture/CE/collision/boundary；二级 mission/capture/recovery/detection时间；三级事件、动作一致率、entropy和phase/role桶。
- 预声明工程 Gate：每场景100；BC safe/CE/capture≥.9且较teacher下降≤5百分点，碰撞≤.1且增加≤3百分点。argmax mission P90≤teacher1.25倍；sample单独效率差可标 `PASS_WITH_STOCHASTIC_EFFICIENCY_GAP`，保留用户允许的B类解释。
- 以上为点估计工程 Gate，不是统计非劣效证明。报告whole-episode paired bootstrap、共同safe子集时间差以及失败率，不能将全成功时退化的bootstrap区间解释为零总体失败率。

## 6. 工程验证与证据索引

`test_forward_final_bridge` + P0 + small-step contract：**28 passed**。覆盖唯一collision差异、错误namespace/runtime fail、capture后CE可达、dataset全局状态索引、完整轨迹留出、temperature不读取验证集、86keys冻结、zero-update概率、配对初态及sample复现、C3可靠性Gate。实际BC checkpoint另有CPU双场景2步smoke，仅标SMOKE，不混入formal。

轻量证据统一在 `artifacts/2026-09-08_forward_final/`：C0 report/runtime、C1 pilot与bulk manifest/audit、C2 report/history/temperature、C3 launch/runtime、测试输出。原始NPZ、episode JSON、全部checkpoint保留本机，Git记录manifest与SHA，不上传大文件。

C0/C1 在本次首次commit前完成，原 evaluator 的完整源码归档于 `source_snapshots/c0_c1_evaluator.py`，SHA与两次launch记录严格相同。它是源码快照，复现时需恢复到tools原路径；不直接从artifact路径运行。C3只新增策略执行/诊断入口，没有改环境；其实际source SHA随launch记录。

下一次先读本页末尾运行快照和 C3 comparison/progress。**C3 未 PASS，不接入 D，不做 teacher-KL/continuous/TD3/SAC/VXY/support奖励重训。** 如 C3失败，先用已保存的phase/role agreement、entropy、轨迹失败位置区分head欠拟合与分布偏离，不把结果直接解释为MAPPO表达能力失败。

## 7. 结束会话运行快照 / NEXT WAKE-UP

快照时间：2026-09-09T01:00:30.014942+08:00。完整记录：`artifacts/2026-09-08_forward_final/SESSION_HANDOFF_20260909_0100.json`。

- GPU0：iqn_greedy，worker PID `246769` / queue PID `246733`；105/200回合，当前step100；约12.8回合/分钟；当前mode预计2026-09-09T01:07:56.889646+08:00完成。其后自动串行BC argmax，另估16–25分钟。
- GPU1：bc_sample，worker PID `246770` / queue PID `246734`；104/200回合，当前step125；约12.7回合/分钟；当前mode预计2026-09-09T01:08:04.222246+08:00完成。完成后空闲，不自动训练。

均为detached进程，不用tmux；log/process/launch/runtime/progress保存在 `c3_formal100/<mode>/`，queue log在父目录。GPU0 teacher checkpoint和GPU1 BC checkpoint均已加载，不生成训练step checkpoint。所有source SHA仍与launch一致。

**NEXT WAKE-UP：2026-09-09 01:35 CST（或三份C3 report均完成后）。** GPU0全队列粗估01:25–01:35完成，尾部慢回合可能延后。先读取comparison；如三份report都有但comparison缺失，执行上文aggregate工具。仅按完整paired Gate判定后续；若仍只剩等待，刷新快照并结束，不长期轮询。GPU1唯一当前任务仍是完成BC sample formal100；不预启动D。
