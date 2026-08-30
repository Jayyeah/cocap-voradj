# IQN-VXY Full 迁移台账（2026-08-30）

## 1. 总状态与因果问题

`STATUS: CONTRACT_AND_SMOKE_PASS / FIXED-TAU_REEVAL_COMPLETE / STAGE1_ACTIVE`

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
STATUS: ACTIVE / TRAIN_SCREEN_FINALIZER_RUNNING
HYPOTHESIS: strict action-only VXY9可在完整4v1 CR-MS+VCT-LS+CE mix中形成非零 capture与CE
ONLY_CHANGED_VARIABLE: AW9 -> proven VXY9；25k checkpoint/full-resume为infra
CONFIG: configs/experiments/iqn_vxy_full_migration_20260830/stage1_4p1e1obs_scratch2m.yaml
COMMIT: dda2a23f20c4b857bc958c856d76bf869b50df02
SEED: 2026080201
START_STEP: 0
CURRENT_STEP: 524k / 2M（2026-08-30 22:15 CST冻结快照）
RESULT: finite；20个25k checkpoint；rolling full-resume存在（约4.3 GiB）；截至500k共20个screen完成。当前严格排序最优仍为275k：capture=.20、coverage CE=.10、mix capture=.20、mix CE=0、max collision=.45，尚未通过最终gate
GATE: capture>=.50、mix capture>=.50、coverage CE>=.10、mix CE>=.10、max collision<=.50
CONCLUSION: contract可执行，性能待2M及25k screening
NEXT: 不早停、不跳stage，继续自然训练到2M；finalizer完成全量选择并仅在gate PASS后晋级8v2
```

### 5.2 Stage 2：8v2 course 700k

```text
STATUS: BLOCKED_BY_STAGE1_GATE（按设计）
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
STATUS: BLOCKED_BY_STAGE2_GATE（按设计）
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

正式 capture/coverage/generalization 结论必须等待长训；smoke 只证明执行与恢复合同成立。
