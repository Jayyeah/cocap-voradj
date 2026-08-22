# Capture Debug 与 Pure-Capture Ablation（2026-08-15）

## 状态

- CF3@400k、PC0@100k均已结束；不再扩步。
- 两张GPU均被其他项目使用。按用户指令，本项目不使用GPU、不挂GPU waiter、不启动B正式训练。
- A1--A4采用有限、串行、自结束的CPU队列；B只完成实现、测试和CPU smoke。

## A1：collision默认合同修复

旧AW实现虽配置`collision_check_each_substep=true`，实体碰撞却只在decision末检测；同时`_refresh_collisions()`边遍历边deactivate，导致同点双机通常只失活低索引一台，且boundary/entity接触阈值与episode统计不一致。

新默认` synchronized_swept_v1`记录同步子步trace，在immutable轨迹上做swept-circle检测，再atomic apply：pursuer-pursuer双方失活；pursuer-evader保留历史“只失活pursuer”；统一接触判定，并输出`agent_agent/obstacle/evader_contact/boundary` provenance。`legacy_end_step`只保留作paired audit。旧CF3/PC0必须标记为legacy collision semantics。

回归覆盖顺序不变、双机/三机atomic、精确相切、AW swept擦碰、近平行不误报及evader/boundary统计；综合聚焦测试`61 passed`。一份修复前旁证CF3@400k legacy deterministic20为`1 normal/20、0 stationary、20 collision/20`；严格CPU pre/post paired由自动队列另行输出。

## A2--A4 CPU诊断

- `tools/probe_cf3_policy_temperature.py`：同seed、同epsilon流比较`T=0/0.25/1`，分开normal/stationary、2+/3+/hold与collision type。
- `tools/evaluate_iqn_corrected_capture.py`：公平重测IQN@125k（SHA256 `5a0ad1c1400d0004334669908c85db6f7b8496fb2987f36f992040c26bd0344d`），修复旧脚本seed+43、整数动作直传AW、APF判断错误。
- `tools/audit_cf3_success_consolidation.py`：保存normal capture、3+→collision、长2+失败最后50步，统计angular gap/separation、clearance、径向/切向速度和`(a,w)`饱和/振荡。
- `tools/run_capture_debug_cpu_queue.py`：A1 legacy/fixed→A2→A3（normal≥4/20才扩100）→A4 legacy/fixed；强制隐藏CUDA、单进程、flock、原子status/error、失败即停并自结束。

## B：Pure-Capture / All-Capture

配置：`configs/experiments/parallel_ce_legacy_voradj_20260809/legacy_voradj_b_pure_capture_local_allcapture_4p1e1obs_100k_aw.yaml`。

冻结CF3 corrected local合同：4P1E1obs、moving APF、K3+stationary fallback、K10/min-active2、`(a,w)`、all-agent MASAC self+mean、no clip、MSE、UTD=.25、LR=1e-4、tau=.005、batch128、horizon1000、capture terminal、scratch/new replay、seed`2026081304`；新实验显式使用fixed swept semantics。

唯一任务/reward变化：

- raw self-visible direct：`approach + mean-shift + front`；
- 看不到enemy但一跳友邻raw/effective pursuing且可解析真实target：`approach only`；
- uninformed：capture dense=0、coverage/PBRS=0，只保留公共safety/collision/boundary；
- normal/stationary terminal reward同值共享给所有仍active pursuers。

K10核心`task_label/reward_role`保持不变；B另设当前可观测的`capture_objective_role`，禁止K10-held agent获得oracle geometry。开关关闭时CF3回归一致。新增normal/stationary、2+/3+/hold、angular gap、agent-agent collision、三角色比例和reward decomposition。

验证：综合`61 passed`、compile/diff-check通过；32-step CPU scratch smoke为`step/replay=32/32`、updates=0（5k warmup内）、all_finite、0 termination/collision、约`13.07 step/s`；checkpoint-screen 1 episode×2 step CPU smoke也通过。未加载旧checkpoint、未启动正式B训练。未来GPU可用时必须显式`--seed 2026081304`从scratch启动。

## 2026-08-21：A1--A4结果收口与B0-K10/B1-K0双线启动

### Capture Debug结论

- CPU有限队列已自然`COMPLETED`，`error_status=NO_ERROR`，没有残留waiter/process。
- A1同seed deterministic20：legacy与fixed均`0 normal / 20`且`20/20 collision`；fixed把出现2+ ring的episode从`13/20`降为`9/20`、3+从`1/20`降为`0/20`。这说明旧decision末检测和顺序失活确实高估了存活/近成功几何；所有新线必须从scratch使用`synchronized_swept_v1`。
- A2在fixed semantics下，CF3@400k的`T=0/0.25/1`分别为normal capture `0/20、2/20、1/20`，collision `20/20、19/20、20/20`。低温噪声能偶尔打破policy mode，但不能单独解决高碰撞和K3稳定性。
- A3旧IQN@125k在同一corrected/fixed环境100回合公平重测达到`61 normal + 24 stationary = 85/100 capture`、`10/100 collision`、`65/100`出现3+ ring。它不是同训练合同算法A/B，但决定性证明当前local observation、moving APF、障碍物和`(a,w)`动力学可解，并使IQN具备teacher/anchor资格。
- A4 fixed随机策略100回合仅`1 normal`、`99 collision`；保存尾段中omega sign-flip约`0.35--0.50`，继续支持“多机同点争抢、振荡和去冲突”是连续MASAC核心瓶颈。

### K10语义复核与配对设计

B dense reward从未把K10-held且看不到enemy的agent当作direct/full-capture：它另用当前可观测的`capture_objective_role`。self丢失enemy token后，只有当当前friendly VorAdj neighbor确实仍有可解析enemy adjacency时才成为`one_hop_informed`，并用该enemy ID计算本机到enemy的真实distance progress，只给`1.0×approach`；若友邻也只是K10-held且无当前target，则为`uninformed`且dense capture=0。它不是`0.5×full capture`，也不允许fallback到任意active enemy。

Pure-Capture没有coverage/task切换，K10主要只剩role-bit hysteresis；它会令`is_pursuing=1`在enemy token消失后继续10步，且Actor看不到剩余counter。因此本轮不修改全局corrected默认，而是同seed并行隔离K10：

- B0-K10：严格CF3 task-level control，`is_pursuing_release_delay_steps=10`；
- B1-K0：Pure-Capture自然候选，`is_pursuing_release_delay_steps=0`；
- 两线除run identity、物理GPU和K10外的resolved task/training配置完全一致，seed均为`2026081304`，均scratch/new replay，固定跑满200k；100k仅诊断、不早停。

配置：

- `configs/experiments/parallel_ce_legacy_voradj_20260809/legacy_voradj_b0_pure_capture_local_k10_4p1e1obs_200k_aw.yaml`
- `configs/experiments/parallel_ce_legacy_voradj_20260809/legacy_voradj_b1_pure_capture_local_k0_4p1e1obs_200k_aw.yaml`

保存合同仍为每25k model-only milestone+deterministic20汇总JSON，只有一个原子替换的`resume_latest`持有完整replay；200k final bundle完整并与rolling/顶层兼容文件hardlink去重。中途不保存GIF/raw rollout，最终200k才另跑正式20-rollout+5代表性GIF。

### 磁盘清理、验证与正式启动

- 启动前根盘`78 GiB available / 92% used`。永久删除CF3 50/75/100/125/150/175k六份过时且link-count=1的`resume_frozen_* / replay.pkl`，共`22,751,238,825 bytes`（约`21.19 GiB`）；对应storage manifest已改为`historical_model_only_pruned_replay / contains_replay=false`，防止误resume。
- CF3@200k与@400k完整trainer/replay/runtime/manifest及所有里程碑model/metrics均保留；清理后根盘`99 GiB available / 89% used`。被删六份replay不可恢复，除非另有外部备份。
- 新增配对配置测试确认完整resolved合同只差K10/执行identity，且K0会立即清除stale pursuing bit；Pure-Capture、collision、P0聚焦回归`29 passed`，compile与`git diff --check`通过。
- 两条32-step CUDA smoke均`exit 0`、`step/replay=32/32`、all-finite、0 termination/collision；B0/B1 scene hash分别为`49422e99...`与`5a5ec052...`。
- 2026-08-21 11:35+08:00正式启动：B0-K10为PID`1956375`、tmux=`b0_purecap_k10_200k`、GPU0；B1-K0为PID`1956378`、tmux=`b1_purecap_k0_200k`、GPU1。启动时两进程日志无错、各约404 MiB VRAM，均处5k warmup；artifact根为`artifacts/2026-08-21_pure_capture_k10_k0_200k/`。
- 11:38两线均严格到达首个真实update：B0/B1=`step/replay/update=5000/5000/1`、`mean_finite=1`、alpha=`0.19998`、peak VRAM均约`8019.9 MiB`；critic loss=`8.20/9.53`、actor loss=`1.91/1.91`均有限。B0首窗出现2+ ring fraction=`0.4%`，B1尚无2+；5k仅为启动健康检查，不作K10优劣结论。
- 11:40热状态：GPU0约84°C/1770MHz且无thermal throttle；GPU1约91°C/1410MHz，driver报告`SW Thermal Slowdown=Active`、`HW Slowdown=Not Active`。B1数值与进程仍正常，但后续ETA必须使用其独立实测吞吐，不能用B0速度外推；这是资源/散热差异，不是任务合同差异。
- 11:42首个完整post-warmup窗口：B0/B1均为`step/replay/update=6000/6000/251`、finite，peak VRAM约8.05GiB；吞吐分别`3.822/3.538 step/s = 13.76k/12.74k step/h`。含25k内置20-rollout，首个完整25k bundle预计约`2026-08-21 13:15--13:40`；计入全部中间诊断后200k预计B0约`2026-08-22 02:30--03:45`、B1约`03:40--05:30`（Asia/Shanghai，热波动时取上沿）。

## 2026-08-22：B0-K10 / B1-K0 200k最终结果

### 完成状态与bundle

- B0-K10于`2026-08-22 02:28+08:00`自然完成，B1-K0于`03:47+08:00`自然完成；均为`transition/replay/update=200000/200000/48751`、`all_finite=true`，无残留trainer/tmux，GPU0/1已释放。
- B0/B1实际全程吞吐分别`3.736/3.433 step/s = 13.45k/12.36k step/h`；B1受GPU1软件热降频影响约慢8%，不把墙时差异解释为K10算法效应。
- 两条`step_000200000`均为`full_resume / contains_replay=true`；final milestone与`resume_latest`的trainer/replay/runtime分别同inode。trainer约146.37 MB，replay约6.707 GB，runtime约2.38 MB；manifest保持scratch、uniform joint、all-agent、no clip、capture-only及同一implementation hash=`1209d57f...`。
- step1及25/50/75/100/125/150/175k均为`evaluation_model_only / contains_replay=false`，只保留模型、metrics和20-rollout JSON；200k才保留完整replay。双线及smoke合计占盘约16 GiB，根盘当前`82 GiB available / 91% used`。
- 每25k deterministic20及最终200k deterministic20均已完成并写入各checkpoint的`diagnostic_eval.json`；当前GIF数量为0。不能写成“5 GIF已完成”；若需要视觉证据，只对final200k各生成5张代表性GIF，不重跑中间checkpoint GIF。

### 训练内实际事件与几何

| 指标（0--200k） | B0-K10 | B1-K0 |
|---|---:|---:|
| normal / stationary capture | 3 / 0 | 1 / 0 |
| capture窗口step | 65k、90k、149k | 180k |
| collision transitions | 875 | 987 |
| agent-agent collision | 712 | 757 |
| boundary / obstacle / evader-contact | 77 / 56 / 31 | 151 / 53 / 26 |
| 2+ independent windows | 129 | 150 |
| 3+ independent windows | 13 | 12 |
| mean 2+ / 3+ step fraction | 0.776% / 0.0300% | 0.968% / 0.0165% |
| max 2+ / 3+ hold | 29 / 13 | 41 / 7 |

B0的3次normal均为真实训练episode且stationary=0，但仅占`3/(875+3)=0.342%`的capture/collision terminal；B1为`1/(987+1)=0.101%`。两者都仍是rare stochastic success，尚未被uniform replay固化。B1虽然2+窗口/平均fraction略高，但3+ fraction、3+ hold、实际capture和碰撞均弱于B0；删除K10没有改善K3闭合。

分段趋势：

- B0在`51--75 / 76--100 / 126--150k`各1次normal；2+ fraction由首25k的`0.04%`升至76--100k的`1.30%`，后段约`0.86--1.00%`；151--200k无新增capture，3+在176--200k降为0。
- B1直至180k才出现唯一训练normal；2+ fraction在101--125k峰值`1.56%`，随后回落到176--200k的`1.03%`，support→enemy progress后段转负。K0没有带来更自然且持续的one-hop闭合，反而boundary collision为B0近两倍。
- 两线后段one-hop informed均出现friend-distance delta转正、enemy progress趋近0或转负，说明support/informed追随随训练退化；Pure-Capture去掉coverage并未消除这一现象。

### 每25k deterministic20曲线与final对比

B0在所有里程碑均`0/20 normal`；但几何从25k到200k有改善：visited-2+从`1/20`升至`10/20`，mean min-min distance从`16.81 m`降至`6.01 m`，collision从`20/20`降至`19/20`。50k和100k曾分别出现`1/20、2/20` visited-3+，但没有形成capture，200k visited-3+=0。

B1在150k和175k各出现`1/20 normal`，证明K0曾产生短暂deterministic可用mode；但200k退化回`0/20`，visited-2+从125k峰值`9/20`降至`4/20`，visited-3+=0，不能视为稳定PASS。

| final deterministic20 | B0-K10 | B1-K0 |
|---|---:|---:|
| normal / stationary capture | 0 / 0 | 0 / 0 |
| collision | 19/20 | 20/20 |
| agent-agent collision | 12/20 | 16/20 |
| visited 2+ / 3+ | 10/20 / 0 | 4/20 / 0 |
| max 2+ / 3+ hold | 14 / 0 | 12 / 0 |
| mean min-min distance | 6.01 m | 11.50 m |
| mean distance progress | +15.92 m | +11.60 m |
| mean episode length | 344.05 | 223.40 |

final碰撞分型为B0=`12 agent-agent + 2 boundary + 3 obstacle + 2 evader-contact`，B1=`16 agent-agent + 1 boundary + 3 obstacle`。失败仍由多机互撞主导，不是coverage reward残留（B中coverage reward严格为0）。

### 结论与下一步

1. **不扩B0/B1到300/400k。** B0后50k无capture且final0/20；B1短暂deterministic success已在200k丢失。继续同样uniform MASAC扩步没有强依据。
2. **不删除默认K10。** K0没有稳定收益；B0在实际normal、3+ hold、总碰撞、agent-agent碰撞和final几何上整体更好。K10保留为后续local合同默认，K0线冻结为已完成ablation。
3. **Pure-Capture没有打通MASAC。** 去掉coverage/多角色reward后仍只有`3/200k`与`1/200k`随机normal、final均0/20；因此coverage dilution不是主要根因，最多是次要干扰。
4. 当前证据继续指向：shared unimodal Gaussian的mode固化失败、2+→3+角度/slot分工不足、agent-agent去冲突和稀有团队terminal credit。terminal team sharing本身仍不足以解决这些问题。
5. 下一条正式训练优先转为**corrected Pure-Capture IQN anchor**：旧IQN在相同fixed环境zero-shot已有`61% normal / 85% total / 10% collision`，先审计并建立IQN125k warm-start+fresh replay与corrected-contract scratch对照，目标把normal提高到≥70--80%、stationary成功占比降到<20%。在该anchor稳定后，再采集normal success/near-success轨迹用于连续Actor蒸馏；不再先做普通MASAC超参或步数扩展。
