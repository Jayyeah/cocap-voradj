# IQN-VXY Full 迁移台账（2026-08-30）

## 1. 总状态与因果问题

`STATUS: CONTRACT_AND_SMOKE_PASS / FIXED-TAU_REEVAL_COMPLETE / STAGE1_PASS / STAGE2_GATE_FAILED / DECLARED_COURSE_STOPPED / USER_OVERRIDE_STAGE3_ACTIVE`

本线只回答一个问题：原 Final IQN-AW 的完整 `capture + coverage + episode mix + 4v1→8v2→12v3` 合同，仅把离散 `(a,w)` 3×3 动作替换为已经在 corrected Pure-Capture 上训练成功的 body-frame、rate-limited VXY9 后，能否继续成立。

基线是 `configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/`，不是 Pure-Capture B0。第一轮不加入新 reward、replay、LR decay、regularizer 或 early stop；25k checkpoint、严格 validation 与 rolling full resume 只属于观测/恢复基础设施。

历史基线为 `9ed9f61a462c213b93b304a6106fa5c5f083e97a`；本轮实现与审计提交为 `dda2a23f20c4b857bc958c856d76bf869b50df02`。运行产物留在本机 `artifacts/`，不上传大 checkpoint/replay。

## 2. 唯一变量与保持项

| 合同项 | Final IQN-AW | IQN-VXY Full | 分类与结论 |
| --- | --- | --- | --- |
| policy action | `(a,w)` 3×3，9 actions | body-frame desired `(vx,vy)` 3×3，9 actions | `ACTION_REQUIRED`，唯一主要变量 |
| IQN/Transformer/quantile/head | historical Final | byte-for-code-path inherited | `SAME` |
| reward/role/K10/stationary/post-capture | CR-MS + support 0.5 approach/0.5 CE | inherited | `SAME` |
| replay/recovery/task mix | `64/16/32/16` 与 historical recovery | inherited | `SAME` |
| optimizer/LR/epsilon/target/batch | historical per-stage values | inherited | `SAME` |
| map/APF/spawn/observation/VCT-LS | historical Final | inherited | `SAME` |
| collision | `legacy_end_step` | explicitly retained | `SAME`；corrected swept transfer另开控制实验 |
| checkpoint cadence | 100k historical | 25k light milestone | `IMPLEMENTATION_ONLY`，不改变 update |
| interrupted-run recovery | historical light checkpoint | one atomic rolling full-resume | `IMPLEMENTATION_ONLY` |

`test/test_iqn_vxy_full_config_contract.py` 会 flatten 三个 stage 配置并拒绝未列入白名单的差异；还逐项断言 network、replay、optimizer、reward、VorAdj、collision 与 curriculum 相等。因此本线没有为追求 VXY 成功而偷偷改变训练任务。

## 3. 已冻结的 VXY9 动作合同

实现入口为 `src/cocap_voradj/dynamics/continuous_action.py::vxy9_body_grid` 与 `src/cocap_voradj/envs/base.py`：

- action index按 body-x outer、body-y inner 排序；
- 每个非零分量为 `v_max/sqrt(2)`，故 diagonal norm 恰为 `v_max=3.0`；
- command 从当前 heading 的 body frame 旋转到 world frame；
- desired velocity 经 `0.4` acceleration limit 的 rate-limited servo 执行；
- `physics_dt=.05`，每 action 10 substeps，`decision_dt=.5`；
- 每 substep speed clip 与 boundary check，第一轮 collision 仍为 historical end-step；
- yaw 初始化保留 Final-AW 的 `legacy_random`，运行中默认 hold；
- integer `0..8` 是唯一合法输入，连续向量合同仍保持独立，不做隐式混用。

真实 2-step CUDA smoke 位于 `artifacts/2026-08-30_iqn_vxy_full/_smoke_stage1_vxy_2steps_20260830/`：model、target、optimizer、两类 replay、当前 task/environment、curriculum counters 与 RNG 均成功写入并从 `resume_latest.pt` 恢复。

## 4. Pure-Capture fixed-τ 独立复评

旧 formal “deterministic” 只关闭 epsilon，IQN forward 仍随机采样 quantile。新 protocol 使用固定 midpoint quantiles：`tau_i=(i+0.5)/N`，冻结新的 evaluation seeds，每点 100 episodes。

同时保留两个预先声明的比较：

| protocol | seed1 | seed2 | seed3 | 用途 |
| --- | --- | --- | --- | --- |
| `uniform225` | 225k | 225k | 225k | 独立复核历史跨 seed 70% 统一点 |
| `per-seed-best` | 250k | 225k | 225k | 复核历史 20-episode 各 seed 峰值；seed2/3 复用同一独立结果 |

工具：`tools/reevaluate_iqn_vxy_best_20260830.py`。输出：`artifacts/2026-08-30_iqn_vxy_fixed_tau_reeval/`。SHA-256、training/eval seed、config hash、checkpoint step 与 quantile contract 全部写入 JSON。

### 4.1 运行记录

```text
STATUS: COMPLETE / BOTH_PROTOCOLS_FINITE_AND_HASH_MATCHED
HYPOTHESIS: 225k附近的跨种子能力是真实的，但20回合事后选优数字会收缩
ONLY_CHANGED_VARIABLE: random evaluation tau -> fixed midpoint tau；不改模型与环境
CONFIG: small_step_ac_migration_20260828/iqn_vxy9_seed{1,2,3}.yaml
COMMIT: dda2a23f20c4b857bc958c856d76bf869b50df02
SEED: training 2026082803/04/05；evaluation 2026093000/4000/5000
START_STEP: uniform 225k；per-seed best 250k/225k/225k
CURRENT_STEP: four unique checkpoint evaluations均为100/100
RESULT: uniform225 capture58.33%/collision39%；per-seed-best capture64%/collision35.33%
GATE: PASS；all four unique checkpoint evaluations complete、finite、hash match
CONCLUSION: VXY能力跨seed成立；历史20回合70%是有选择偏差/方差的乐观数，不应继续引用为正式率
NEXT: 两个aggregate JSON已保留；paired reward-tail audit完成后GPU0已自动进入Full stage1
```

| protocol/seed | checkpoint | capture | normal | stationary | collision | visited2+ | visited3+ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform / seed1 | 225k | 62% | 62% | 0% | 30% | 86% | 60% |
| uniform / seed2 | 225k | 48% | 48% | 0% | 52% | 88% | 48% |
| uniform / seed3 | 225k | 65% | 62% | 3% | 35% | 85% | 40% |
| per-best / seed1 | 250k | 79% | 79% | 0% | 19% | 92% | 58% |

seed2/3 的 per-best 就是同一个225k独立结果，不重复运行。跨300回合，uniform225 的 capture/normal/stationary 为 `58.33%/57.33%/1%`，visited2+/3+为 `86.33%/49.33%`，平均每回合3+最大hold `4.97`、return `-6.92`、length `235.92`。per-seed-best 对应为 `64%/63%/1%`，visited2+/3+ `88.33%/48.67%`，平均hold `4.61`、return `+20.43`、length `208.28`。

这组独立结果同时支持两点：VXY9不是偶然单seed成功；AW动作归纳偏置仍更快、更稳。精确结果位于 `aggregate_uniform225.json` 与 `aggregate_per_seed_best.json`，而不是从日志人工抄取。

### 4.2 先升后降的 reward-fidelity 事实

历史三 seed deterministic20 合并从 225k→300k：capture `70%→8.3%`，visited-3+ `45%→81.7%`，3+ hold `27.3→75.7`，return `+33→+51`。分组后：

| group | n | mean return | mean 3+ hold | collision | mean length |
| --- | ---: | ---: | ---: | ---: | ---: |
| 225k captured | 42 | +99.08 | 3.79 | 0% | 195.7 |
| 225k failed + visited3+ | 8 | -159.14 | 11.50 | 100% | 256.1 |
| 300k captured | 5 | +170.79 | 11.80 | 0% | 310.6 |
| 300k failed + visited3+ | 46 | +51.76 | 36.85 | 45.7% | 780.4 |

为把这一现象从相关性推进到可定位证据，本轮又冻结一组独立 eval seeds，在统一225k/300k上各做 `3×20` fixed-midpoint-τ paired audit，并逐回合聚合最后100/200步：

| checkpoint/group | n | mean return | mean 3+ hold | collision | mean length |
| --- | ---: | ---: | ---: | ---: | ---: |
| 225k captured | 38 | +57.87 | 4.87 | 0% | 238.92 |
| 225k failed + visited3+ | 9 | -231.28 | 11.22 | 77.78% | 429.67 |
| 300k captured | 8 | +241.19 | 12.25 | 0% | 170.75 |
| 300k failed + visited3+ | 36 | +20.19 | 34.64 | 47.22% | 773.08 |

失败-3+组最后100步的 agent-step shaping 在225k→300k实际下降：approach `.05395→.00196`、mean-shift `.31868→.23541`、capture total `.41159→.27977`，terminal始终为0；200步窗口同样为 approach `.08392→.01500`、capture total `.40755→.31445`。与此同时，回合长度增至 `1.80×`、3+ hold增至 `3.09×`，direct-capture role占比从82.53%升至88.72%。

因此不是300k获得了更高的瞬时 shaping，而是低质量 nonterminal dwell 更长：approach几乎消失，mean-shift/ring shaping仍持续，累计 return可为正但 true terminal不发生。这是明确的 **proxy/time-horizon mismatch**；碰撞率反而下降也排除了单一“碰撞变多”解释。当前只审计、不改 reward；Full 第一轮依旧使用 Final-AW reward，并依赖独立 validation 选 checkpoint。

## 5. Full curriculum 与 gate

真实历史顺序只有三个训练 stage；所谓 mixed/generalization 是每个 stage 的 capture/coverage/mix 正式评估，不是凭空新增第四训练 stage。

### 5.1 Stage 1：4v1 scratch 2M

```text
STATUS: HISTORICAL_SNAPSHOT / ACTIVE_AT_2026-08-31_15:56 / FINAL_SEE_SECTION_8
HYPOTHESIS: strict action-only VXY9可在完整4v1 CR-MS+VCT-LS+CE mix中形成非零 capture与CE
ONLY_CHANGED_VARIABLE: AW9 -> proven VXY9；25k checkpoint/full-resume为infra
CONFIG: configs/experiments/iqn_vxy_full_migration_20260830/stage1_4p1e1obs_scratch2m.yaml
COMMIT: dda2a23f20c4b857bc958c856d76bf869b50df02
SEED: 2026080201
START_STEP: 0
CURRENT_STEP: 1.713M / 2M（2026-08-31 15:56 CST冻结快照）
RESULT: finite；68个25k checkpoint；rolling full-resume存在；截至1.700M共68个screen完成。当前严格排序最优为1.425M：capture=.85、coverage CE=.20、mix capture=.90、mix CE=.20、max collision=0，五项初筛gate全部满足
GATE: capture>=.50、mix capture>=.50、coverage CE>=.10、mix CE>=.10、max collision<=.50
CONCLUSION: 1.425M强候选继续同时跨过capture/coverage/mix/safety五项阈值；仍须跑满2M并由finalizer正式选模，不能把初筛提前写成最终PASS
ETA: 最近1.60M→1.70M用时93分35秒（约17.8 step/s）；训练到2M约还需4小时29分，预计2026-08-31 20:25 CST。计入末个screen与formal gate缓冲，预计20:55–21:40给出stage1正式决定
NEXT: 不早停、不跳stage，继续自然训练到2M；finalizer完成全量选择并仅在gate PASS后晋级8v2
```

### 5.2 Stage 2：8v2 course 700k

```text
STATUS: HISTORICAL_SNAPSHOT / BLOCKED_AT_STAGE1_GATE / FINAL_SEE_SECTION_8
HYPOTHESIS: selected 4v1 VXY policy可shape-compatible warm-start到8v2
ONLY_CHANGED_VARIABLE: historical stage size/curriculum change；action仍是同一VXY9
CONFIG: configs/experiments/iqn_vxy_full_migration_20260830/stage2_8p2e2obs_700k.yaml
COMMIT: dda2a23f20c4b857bc958c856d76bf869b50df02
SEED: 2026080202
START_STEP: 0，pretrained=stage1 selected checkpoint
CURRENT_STEP: 0
RESULT: static contract PASS；未越 gate 启动
GATE: 与stage1相同的五项阈值
CONCLUSION: 不允许因stage1失败直接跳级
NEXT: stage1 formal selection PASS后自动写只含pretrained.path的runtime wrapper并启动
```

### 5.3 Stage 3：12v3 course 700k

```text
STATUS: HISTORICAL_SNAPSHOT / BLOCKED_AT_STAGE2_GATE / FINAL_SEE_SECTION_8
HYPOTHESIS: selected 8v2 VXY policy可扩展到12v3并保留capture+CE
ONLY_CHANGED_VARIABLE: historical stage size/curriculum change；action仍是同一VXY9
CONFIG: configs/experiments/iqn_vxy_full_migration_20260830/stage3_12p3e3obs_700k.yaml
COMMIT: dda2a23f20c4b857bc958c856d76bf869b50df02
SEED: 2026080203
START_STEP: 0，pretrained=stage2 selected checkpoint
CURRENT_STEP: 0
RESULT: static contract PASS；未越 gate 启动
GATE: 与stage1相同的五项阈值
CONCLUSION: 不允许因stage2失败直接跳级
NEXT: stage2 PASS后自动启动；完成后做historical capture/coverage/mix generalization report
```

## 6. 自动化、checkpoint 与资源保护

`tools/supervise_iqn_vxy_full_20260830.py` 独占课程编排，不改算法：

- 三 stage 各自独立 train/screen/finalizer tmux；
- 每 25k light model milestone，screen capture/coverage/mix 各 20 episodes；
- stage结束后 formal20、显式 gate、只注入 selected checkpoint；
- process退出后最多3次从 rolling resume自动恢复；
- 状态持续记录 PID/tmux、step、finite metrics、checkpoint count、GPU/VRAM/温度、RAM、disk；
- disk free低于25 GiB时给训练发送 `SIGINT/Ctrl-C`，保留可恢复状态并停止晋级。

rolling `resume_latest.pt` schema 为 `cocap_iqn_full_resume_v1`，包含 model/target/optimizer、replay、所有 task env/APF、当前 observation/task/episode accumulators、curriculum与调度计数、Python/NumPy/Torch/CUDA RNG及训练合同 hash。25k milestone不复制 replay。

## 7. 验证矩阵

截至本台账创建时：

- VXY9 grid/index/body-world/rate-limit/env contract：PASS；
- 三 stage Final-AW strict-diff与历史 collision/curriculum：PASS；
- full-resume roundtrip、RNG与contract-drift rejection：PASS；
- supervisor cadence/gate/promotion：PASS；
- 2-step Full-VXY CUDA smoke/resume：PASS；
- 最终合并定向回归：`78 passed`；修改 Python 全量 `py_compile` 与 `git diff --check` 均PASS。

长训已完成并按gate停止：Stage1 4v1 PASS；Stage2 8v2 mixed CE FAIL；Stage3按合同未启动。smoke仍只证明执行与恢复合同，最终性能结论以第8节的全量筛选和formal评估为准。


## 8. 长训最终归档（2026-09-01 10:16 CST）

### 8.1 Stage 1 最终结果

```text
STATUS: COMPLETE / GATE_PASS / PROMOTED_TO_STAGE2
CURRENT_STEP: 2.000M / 2.000M，2026-08-31 20:30 CST自然完成
SELECTION: 80个25k节点全部筛选；严格排序选中step_1425000.pt
SCREEN: capture=.85、coverage CE=.20、mix capture=.90、mix CE=.20、max collision=0
FORMAL20: capture=.75、coverage CE=.15、mix capture=.85、mix CE=.15、max collision=.05
GATE: 五项全部PASS；supervisor于20:42自动晋级Stage2，formal产物于20:53完整落盘
ETA: 0
```

Stage1的screen与formal都跨过capture、coverage、mix和safety门槛。晋级使用选中的1.425M checkpoint，没有早停或人工跳级。

### 8.2 Stage 2 最终结果

```text
STATUS: COMPLETE / GATE_FAILED_ON_MIX_CE / COURSE_STOPPED
PRETRAINED: Stage1 step_1425000.pt
CURRENT_STEP: 700k / 700k，2026-09-01 09:20 CST自然完成
SELECTION: 28个25k节点全部筛选；严格排序选中step_600000.pt
SCREEN: capture=1.00、coverage CE=.10、mix capture=1.00、mix CE=0、max collision=.05
FORMAL20: capture=.90、coverage CE=.15、mix capture=.95、mix CE=.05、max collision=.10
GATE: capture/coverage/safety PASS；mixed CE < .10，FAIL
ETA: 0
```

8v2保留了强capture和单独coverage能力，但capture后的mixed CE扩展没有达到预声明门槛；screen与formal独立确认同一缺口。supervisor因此正常结束，Stage3保持0/700k且没有训练进程、checkpoint或评估产物。本条Full课程的最终结论是“4v1成功迁移，8v2捕获保持但mixed CE扩展失败”；任何修复都应另开显式变量线，不能把12v3算作本线继续。


## 9. 用户授权的 Stage2 gate override 与 paired 展示线（2026-09-01）

### 9.1 授权边界与 lineage

第8节仍是原先声明课程的不可改写归档：Stage2 的 `mix_ce` gate 确实失败，严格课程也确实在该点停止。本节记录的是用户于2026-09-01另行明确授权的执行扩展，只忽略 Stage2 promotion gate，不修改 selection、不伪造 gate 结果，也不降低 Stage3 后续选优和正式评估合同。

```text
OVERRIDE_REASON: user_authorized_20260901
STAGE2_SELECTED: step_600000.pt
STAGE2_SHA256: 1388ac6813fa6c6ee9f0cc6041446e04556a01ef611a07d0afe3071340a86940
STAGE3_START: 2026-09-01 15:16:35 CST
STAGE3_PRETRAINED_LOAD: loaded_count=119, adapted_keys=[], skipped_keys=[]
```

supervisor 先原样记录 `stage_gate_overridden`，其中原 checks 仍为 capture/coverage/collision PASS、mix CE FAIL、checkpoint PASS；随后 runtime overlay 精确注入 Stage2 600k checkpoint。Stage1 selected 1.425M 的 SHA-256 为 `51a936f04d72b062e9fb9ab631024197b115ee404cf5981f0f1a979cd2fe902c`。

### 9.2 旧 IQN 最终 rollout 的启动前逐项对齐

旧仓库与当前分支的四层正式执行代码逐字节相同：

| 文件 | SHA-256 |
| --- | --- |
| `tools/finalize_screened_run.py` | `6225fb8bb716cca41276390349563b2ca6789fd43d6d4ef79c63be5a7ba51676` |
| `tools/batch_rollouts_parallel.py` | `0656f6c9b49a18b3c866d4bc7db78e2a5e0a9925a1e905b9e05cbb197ee7786e` |
| `tools/batch_rollouts.py` | `16820f728f60b792f740f91761ddb2983eee05055289bfbdc8ddc3041abd9c7c` |
| `tools/rollout_voradj_visual.py` | `1e9540c451792727d5fdd3895037393afbae37e08878cc54fdee9c7c78b07b93` |

三阶段 resolved config 也已递归比较：排除声明过的 AW9 到 VXY9 动作/动力学差异、25k checkpoint/full-resume 基础设施、运行路径和 metadata 后，其余每个评估环境字段与旧 Final-IQN 完全相等。外置full-resume、override作用域、runtime overlay、Stage3 paired finalizer、旧配置与可视化合同定向测试合计 `21 passed`。

| Stage | 场景 | 每场景 rollout/GIF | paired seed | workers/chunks | capture/coverage/mix cap | evaders |
| --- | --- | ---: | ---: | --- | --- | ---: |
| 4p1e1obs | capture, coverage, mix | 20/10 | 2026081201..1220 | 4，5/5/5/5 | 1000/1200/2200 | 1 |
| 8p2e2obs | capture, coverage, mix | 20/10 | 2026082201..2220 | 4，5/5/5/5 | 1000/1500/2500 | 2 |
| 12p3e3obs | capture, coverage, mix | 20/10 | 2026082301..2320 | 4，5/5/5/5 | 1000/1800/2800 | 3 |

三场景共同合同：

- IQN 为 `model.eval + no_grad + epsilon=0 + fixed_midpoint_32`，不是 rollout 随机 quantile；
- 地图120乘120，Voronoi cell始终绘制，邻接线开、20m敌人/障碍surface-distance感知圈开、faded trails关；
- 100 ms每帧，最多1000帧；超过时按完整轨迹等距采样并保留首尾；
- GIF固定取每场景前10个 paired seeds，不按成功与否事后选图；
- pure coverage强制0 evader，保留1/2/3 obstacles；只有它绘制CE centroid；
- capture和mix不绘制CE centroid，即使mix已进入coverage phase；
- mix在同一环境中捕获后继续coverage，post-capture window依次为500/600/700；
- capture与mix为map-random；pure coverage按每个worker内部episode index交替map-random与inner-random-cluster，4乘5分块的旧实际分布为12次map-random和8次cluster；
- 输出路径含精确目录层 `best_20rollout10gif`，因此自动备份config和checkpoint。

旧 Final-AW 与本次唯一主要实验差异仍是动作合同：旧为unicycle AW9，本次为heading-aligned rate-limited VXY9。本次执行设备按GPU0要求为 `cuda:0`，旧 artifact 为 `cuda:1`；环境动力学仍在CPU，但不同CUDA卡不声明为bitwise一致。其余15个显式 rollout 参数已在创建tmux前逐项对照旧真实 `run_args.json` 并通过。

### 9.3 已启动后台线

| 任务 | tmux / 主PID | 输出 | 启动状态 |
| --- | --- | --- | --- |
| Stage1 best paired 20/10 | `cocap_iqn_vxy_s1_best20r10g_gpu0` / 109373 | `artifacts/2026-09-01_iqn_vxy_full_rollouts/best_20rollout10gif/stage1_4p1e1obs_step_1425000/` | 4 workers均运行，实际run_args复核PASS |
| Stage2 best paired 20/10 | `cocap_iqn_vxy_s2_best20r10g_gpu0` / 109380 | `artifacts/2026-09-01_iqn_vxy_full_rollouts/best_20rollout10gif/stage2_8p2e2obs_step_600000/` | 4 workers均运行，实际run_args复核PASS |
| Stage3 override supervisor | `cocap_iqn_vxy_full_stage3_override_gpu0` / 109566 | supervisor status/runtime overlay | ACTIVE |
| Stage3 train | `cocap_vxy_full_s3_12p3e3obs_train` / 109572 | `artifacts/2026-08-30_iqn_vxy_full/stage3_12p3e3obs_700k/` | ACTIVE，首批2k metrics已写入 |
| Stage3 screen | `cocap_vxy_full_s3_12p3e3obs_screen` | 28个25k节点，三场景各20回合 | WAITING/ACTIVE |
| Stage3 finalizer | `cocap_vxy_full_s3_12p3e3obs_finalize` / 109602 | selection后自动formal20/10 | WAITING |

Stage1/2 rollout日志位于 `logs/iqn_vxy_full_20260901/`；Stage3 train/screen/finalize仍用 `logs/iqn_vxy_full_20260830/stage3_*.log`，override supervisor另写 `logs/iqn_vxy_full_20260901/stage3_override_supervisor.log`。

启动时GPU0约3.0 GiB显存、根盘36.36 GiB空闲、RAM约109 GiB可用。为避免约10 GiB rolling full-resume把根盘压到25 GiB guard附近，Stage3仅把该单个可替换恢复包写到 `/dev/shm/cocap_iqn_vxy_full_stage3/resume_latest.pt`；25k轻量checkpoint、metrics、screen和best仍持久化在根盘。`/dev/shm`有63 GiB空间且可承受原子替换峰值，但主机重启会清空它；进程崩溃可精确恢复，主机重启后只能从持久化25k轻量checkpoint重启而不能恢复replay/RNG。

### 9.4 ETA

Stage1/2旧同合同墙钟分别约8.2分钟和27.8分钟。本次二者并行且叠加Stage3训练，保守预计在2026-09-01 15:40至16:20 CST完成；本轮不等待其全部结束。

Stage3在15:18已到2k，但该点仍处于replay warm-up，不能用瞬时速度线性外推。证据化估计采用两项历史量：本轮VXY Stage2 700k耗时12小时38分，旧AW课程从8v2到12v3的同700k规模耗时系数约1.31。由此Stage3训练中央估计约16.5小时，预计2026-09-02 07:45 CST左右结束，实际窗口约07:30至10:00。28个25k screening若能持续跟上，selection和自动20/10 formal约在08:00至12:00完成；后续应按首个25k与100k的真实吞吐收紧ETA。

### 9.5 自动收尾验收

Stage3必须自然训练到700k，等待28个25k checkpoint全部screen，按既有词典序选择历史最佳，而不是强制final700k。finalizer已实际以 `seed=2026082301 / episodes=20 / gif=10 / workers=4 / caps=1000,1800,2800 / evaders=3` 启动等待；未传 `--draw-trails`，底层固定补齐邻接线和感知圈。完成后应验收：

- 无 `failures.json`；
- 每场景20条records、10个GIF，seed范围与前10 GIF seeds精确；
- pure coverage初始化计数为12/8；
- 每个GIF记录为edges=true、circles=true、trails=false，且仅coverage为CE targets=true；
- Stage3有selection、`all_summaries.json`、`timing.json`和`FORMAL_DONE`。

## 10. GPU0后续 Pure-Coverage isolation 队列（2026-09-01）

Stage2 formal已证明standalone coverage CE `.15`但mixed CE仅`.05`。为区分VXY coverage skill难度与capture→coverage phase/replay interference，Stage3自然结束并完成formal/GIF后，GPU0固定串行：Stage2 selected 600k warm-start pure coverage 500k，再同seed scratch pure coverage 500k。

两线保持Final Stage2 reward/CE/horizon/observation/IQN/VXY9/optimizer/epsilon/target update与buffer规模，只把任务分布切为8P0E2obs pure coverage。每25k coverage-only deterministic20，选中点formal20/GIF10；gate只看CE/CV/centroid/speed/collision/boundary/length/return decomposition，不看capture。历史合同取舍、500k预算依据、paired config、指标扩展、CUDA恢复smoke以及不打断Stage3的三重放行条件详见 `docs/VXY_PURE_COVERAGE_AUDIT_20260901_ZH.md`。

### 10.1 队列已就绪，Stage3未受扰动

2026-09-01 21:58 CST，Stage3到305k，12个milestone已落盘，screen已完成至275k且300k active；train/screen/finalizer PID仍为109572/109587/109602，formal/GIF尚未开始。Pure-Coverage queue supervisor PID `264842`、tmux `cocap_vxy_pure_coverage_queue_gpu0`，状态为read-only observe且没有创建任何pure-coverage child。Stage1/2两个已完成rolling resume经terminal/selected/formal和file-handle核验后删除，释放约16.6 GiB；Stage3活跃resume未动，根盘恢复至约46 GiB空闲。

### 10.2 Stage3与Pure-Coverage最终结果

Stage3 12v3 已自然完成700k，28个screen节点、formal20/GIF10和 `FORMAL_DONE` 全部落盘。selected为 `step_150000.pt`；formal capture `.90`、standalone coverage CE `1.00`、mixed capture `.90`、mixed CE `.90`、collision `.10`（boundary `.10`），supervisor记录的六项 checks 全部PASS。screen selected节点为四项核心成功率1.00、max collision0；该结果表明在用户授权的Stage2 gate override下，12v3 VXY完整capture+coverage+mix合同得到正式PASS。

后续GPU0 isolation也已完成：warm selected75k的formal CE `1.00`、collision/boundary `0/0`；scratch selected500k的formal CE `.45`、collision/boundary `0/0`。结论是已有Stage2模型可以快速恢复coverage，但VXY9从scratch的稳定CE上限明显较低；mixed CE缺口更偏向phase/recovery exposure与初始化，而非VXY coverage skill不存在。完整指标与SHA见 `docs/VXY_PURE_COVERAGE_AUDIT_20260901_ZH.md`。

### 10.3 2026-09-03 现场状态与收尾

现场核对显示 Stage3 的 selected `step_150000.pt`、formal20/GIF10、`FORMAL_DONE` 均已落盘，Stage3 自然结束；GPU0 后续 warm-start→scratch Pure-Coverage 队列也已正常 `queue_complete`，GPU0 当前空闲。该成果线没有剩余训练，ETA 为“已完成”。完整指标、选择策略和 checkpoint SHA 分别见第10.2节及 `docs/VXY_PURE_COVERAGE_AUDIT_20260901_ZH.md`。

## 11. Stage3 selected 回迁 Stage2 formal100（2026-09-03）

本轮冻结 Stage2 600k 与 Stage3 selected 150k 权重，在同一个 Stage2 `8p2e2obs` 正式环境、fixed-midpoint-32、`epsilon=0`、seeds `2026083201..3300` 下执行 capture/coverage/mix 各100回合。Stage3 SHA-256 为 `b33dce2484f3394d1aadb13cd02ca5085929011a8bf7a2a9742ba75c0dba419b`；`pretrained_load.json` 证明它由 Stage2 600k 加载119项且无 adapted/skipped key。本节没有训练或改权重。

| 模型 | capture / collision / length | standalone CE / CV≤.15 | mixed capture / CE / collision | capture条件 survival / CE |
| --- | --- | --- | --- | --- |
| Stage2 600k | `.97/.02/235.95` | `.05/.20` | `.98/.07/.06` | `.9592/.0714` |
| Stage3 150k→Stage2 | `.98/.02/269.27` | `1.00/.49` | `.98/.97/.03` | `.9898/.9898` |

paired 结论：Stage3 capture 仅 `+1pp`，95% bootstrap interval `[-3,+5]pp`，mixed capture `0pp [-4,+4]pp`；两者均不支持更强8v2 capture，且 capture length 反而 `+33.32 [4.55,61.92]` steps。另一方面 standalone CE 为 `+95pp [90,99]pp`，mixed CE 为 `+90pp [84,96]pp`。因此准确结论是：**Stage3 学到了可回迁的更强 capture→coverage joint skill，但没有学到更强或更快的8v2 capture。**

Stage3 在 capture-only 的2+/3+ episode rate 为 `.98/.40`，高于 Stage2 的 `.90/.25`；2+/3+ time fraction 为 `.1439/.01283`，高于 `.0761/.00636`，但首次3+（仅统计出现回合）为197.90步而非126.64步。这支持“更持久的 coverage-like ring geometry”，不支持“更早完成 capture”。所有 stationary capture 均为0。

完整 formal100、support-latency、paired GIF 和合同/lineage 边界见 `docs/IQN_VXY_STAGE2_CROSS_RETENTION_AUDIT_20260903_ZH.md`。后续 matched Final-AW/VXY 与 IQN→MAPPO distillation 只登记为 TODO，本轮未启动长训。
