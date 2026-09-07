# Final-IQN-AW ↔ VXY matched 与 IQN→MAPPO BC 台账（2026-09-03）

## 1. 范围与冻结合同

本轮基于远端最新 `0f72614b6cd15d0cfc8c295337abe310e3ae976d`。GPU0 只做冻结 checkpoint formal100；未重训 AW、未改 checkpoint、未按 GIF 改 reward。AW/VXY 均为 fixed midpoint 32 quantiles、epsilon=0、相同 scene/seed/horizon；Stage2 seeds `2026083201..3300`，Stage3 seeds `2026083301..3400`。完整 paired JSON：

`artifacts/2026-09-03_iqn_aw_vxy_matched_formal100/paired_report.json`

## 2. Final-AW ↔ VXY matched formal100

### 2.1 Stage2（8p2e2obs）

| 场景/指标 | Final-AW 300k | VXY 600k |
| --- | ---: | ---: |
| capture / normal / stationary | `.97/.97/0` | `.97/.97/0` |
| capture collision | `.03`（agent/boundary/obstacle各1） | `.02`（boundary 2） |
| capture mean length | `87.76` | `235.95` |
| first detect→capture | `83.92` | `224.71` |
| support→direct / pursuing | `34.50/34.29` | `84.19/84.19` |
| 2+→3+（出现条件下） | `18.96` | `59.64` |
| direct / support speed | `1.851/1.983` | `1.207/1.032` |
| pure CE / CV≤.15 / collision | `1.00/.33/0` | `.05/.20/.01` |
| mixed capture / CE / collision | `.97/.91/.06` | `.98/.07/.06` |
| mixed settled / post-capture steps | `.25/206.13` | `.94/529.40` |

同 seed paired：capture差 `0pp [-5,+5]pp`，collision `-1pp [-6,+3]pp`；但 VXY length `+148.19 [119.72,180.72]` steps、detect→capture `+142.00 [115.85,171.15]`、support→direct `+49.63 [36.84,64.13]`。Stage2 成功率相同，但 VXY 的接敌后收尾和 support 升级明确更慢，同时这一 checkpoint 尚未保住 AW 的 pure/mixed CE。

### 2.2 Stage3（12p3e3obs，主判据）

| 场景/指标 | Final-AW 700k | VXY 150k |
| --- | ---: | ---: |
| capture / normal / stationary | `1.00/1.00/0` | `.90/.90/0` |
| capture collision | `0` | `.10`（boundary 10） |
| capture mean length | `81.87` | `200.85` |
| first detect→capture | `80.63` | `200.39` |
| support→direct / pursuing | `27.71/27.69` | `50.07/50.07` |
| 2+→3+（出现条件下） | `5.80` | `67.91` |
| direct / support speed | `1.681/1.467` | `1.184/1.091` |
| pure CE / CV≤.15 / collision | `1.00/.15/0` | `.99/.36/.01` |
| mixed capture / CE / collision | `1.00/.96/.01` | `.90/.90/.10` |
| mixed settled / post-capture steps | `0/305.90` | `0/60.70` |

同 seed paired：VXY capture `-10pp [-16,-4]pp`，capture collision `+10pp [+4,+16]pp`，length `+118.98 [106.08,132.78]`，detect→capture `+118.61 [105.41,132.48]`，support→direct `+22.36 [18.13,26.92]`，2+→3+ `+46.57 [19.10,76.91]`。Pure CE 基本保持（`.99` 对 `1.00`），mixed CE仅 `-6pp [-13,+1]pp`；主要缺口是 capture latency、Stage3成功率和 boundary safety。

`post_capture_settled` 不作优劣 Gate：Final 合同关闭 settling terminal/reward，且 CE 达标后的早结束会改变该诊断的可观察窗口。

### 2.3 第一条 correction 结论

Matched 证据本身把 **VXY servo/action dynamics** 标为强机制候选：

- direct 与 support 同时显著降速，不是 support 独有；
- VXY direct/support rate-limit pressure 均约 `.95–.99`；
- first direct/target 出现时间与 AW 接近，排除初始信息到达慢为主因；
- Stage3 高 CE 已保持，而 capture latency 和 boundary collision 同时恶化。

用户于16:21明确授权的首条长训改为 **support reward scale**：capture/coverage从`.5/.5`同时改为`1/1`，三阶段各1M，其余合同不变。该改动保持capture:coverage相对比例1:1，只把support总信号放大2倍；它是单变量因果实验，不能预先写成已被matched数据证明优于servo方案。servo候选保留但不并行改动，避免归因混杂。

## 3. IQN teacher → MAPPO-9-v2 BC Actor

### 3.1 Teacher qualification

三个 Final-AW checkpoint 均在 `configs/experiments/mappo9_v2_20260830/seed1.yaml` exact target formal100：capture `1.00`、collision `.01`。Stage1/2/3 的2+/3+分别为 `.45/.02`、`.66/.14`、`.48/.02`，mean length `60.65/73.25/63.31`。因此选择 Stage2 300k 作为 teacher：成功/安全并列，但目标状态覆盖最强。SHA：

`ef58ae9bdd018633afee0d16f2242d2f157a6cc18611e41474f115aed254b87e`

轻量选择记录：`artifacts/2026-09-03_iqn_aw_teacher_qualification_formal100/qualification_decision.json`。三份大 raw formal JSON 留在本机，不提交 Git。

### 3.2 实现与 Gate

- `collect_iqn_aw_teacher_dataset_20260903.py`：正常 deterministic teacher rollout；固定 midpoint-Q；分 shard/resume；保存 local obs、Q[9]、greedy、episode/timestep/agent、role/pursuing、reward/done、central global state，并统计 support/near-capture/2+/3+ 覆盖。
- `distill_mappo_actor_from_iqn_20260903.py`：严格复制86个 Legacy backbone keys并冻结；只训练 policy head；主 loss 为 `KL(p_T||pi)`，同时报告 hard-BC CE sanity；checkpoint 为 weights-only-safe schema。
- `evaluate_distilled_mappo_actor_20260903.py`：PPO update=0 的 formal100 Gate，检查 action agreement、capture、collision、2+/3+、length。
- `supervise_iqn_mappo_bc_20260903.py`：只串行 dataset→distillation→Gate；Gate PASS/FAIL 后均停在 PPO 之前，不猜后续分支。

相关 contract tests：`12 passed`（dataset/distillation、VXY evaluator、MAPPO-v2）。

### 3.3 最终 Gate（2026-09-03 16:20 CST）

- dataset完成：600个正常teacher episodes、171,052 active-agent rows；capture 600/600，support rows 37,113，near-capture rows 12,494，2+/3+ episodes 439/76。没有只保留成功回合。
- distilled Actor：严格复制86个backbone keys并冻结；best epoch 20；train/validation agreement `.96914/.96910`，validation support/pursuing/near-capture为`.96306/.97082/.96609`；checkpoint SHA `bb8f971201f6e0a55af3ccb62bcae117f96afa1b07fad2c95c3da1dc552fa35e`。
- pre-PPO formal100 Gate：`PASS_TO_PPO_BRANCHES`。Student capture/collision/2+/3+/length为`1.00/.03/.64/.12/71.16`，teacher为`1.00/.01/.66/.14/73.25`；rollout agreement `.96605`，support agreement `.95278`，四项Gate全PASS，PPO updates=`0`。
- 原dataset/Gate tmux均自然结束；外部 OmniVLA PID `1276759` 未触碰。
- GPU0 matched 已完成并释放；没有 VXY correction 训练。

当前 PPO 状态：`GATE_PASS / TWO_BRANCHES_ACTIVE`。两支均从同一 SHA 的 BC Actor 启动，见第4节。

## 4. 用户授权长训与 BC 两分支启动（2026-09-03 22:15 CST）

### 4.1 VXY support11 三阶段课程

用户授权首条唯一变量实验已启动：support capture/coverage reward weight 由 `.5/.5` 同时改为 `1/1`，保持二者比例 `1:1`；其余算法、环境、VXY9 servo/action dynamics、阶段继承、fixed-midpoint评估与25k checkpoint合同不变。Stage1/2/3预算均为1,000,000步，严格resolved-config diff测试PASS。Stage1完成后按既有selection与formal合同自动进入Stage2，再进入Stage3；不覆盖历史checkpoint，rolling resume使用独立`/dev/shm/iqn_vxy_support11_1m_20260903/`路径。

```text
GPU: physical GPU0 / cuda:0
SUPERVISOR: cocap_vxy_support11_supervisor_gpu0 / PID 1510158
ACTIVE: cocap_vxy_support11_1m_20260903_s1_4p1e1obs_train / PID 1510169
SCREEN/FINALIZE: matching Stage1 tmux active；25k screening已完成
SNAPSHOT: 2026-09-03 22:31 CST, step=32,000/1,000,000, finite_metrics=true
CHECKPOINT: 1 milestone；`stage1_4p1e1obs_support11_1m/checkpoints/step_25000.pt`
GPU0: 739 MiB，约7–16% utilization，66°C
```

首个25k deterministic20仅作早期基线：capture `.10` / collision `.90`，coverage CE `0` / collision `.85`，mix capture `.10` / CE `0` / collision `.95`；这是scratch早期节点，不据此判定support11有效或失败。启动后16分钟到32k的gross throughput约33 step/s；考虑后续训练态与screen开销，Stage1暂估9–14小时，Stage2约18–24小时，Stage3约22–30小时，全课程约2–2.8天（中央约2026-09-06清晨）。任何阶段失败均由supervisor保留真实Gate记录，不改写为PASS。

### 4.2 同一 BC Actor 的 Direct PPO / critic warm-up 对照

两支严格共享 `distilled_actor.pt` 与 SHA `bb8f971201f6e0a55af3ccb62bcae117f96afa1b07fad2c95c3da1dc552fa35e`，均使用MAPPO-9-v2 exact target环境、fresh central critic、25k checkpoint与resume。Direct预算100k；warm-up总预算110k，其中Actor冻结至少10k，EV≥.20连续2次后可切PPO，最迟25k强制结束warm-up，因此后续PPO预算85k–100k。critic target始终是MAPPO return/value，不复制IQN Q。

```text
GPU: physical GPU1（CUDA_VISIBLE_DEVICES=1，进程内cuda:0）
SUPERVISOR: cocap_bc_ppo_supervisor_gpu1 / PID 1510162
DIRECT: cocap_bc_direct_ppo_gpu1 / PID 1510173 / step 20,000/100,000
WARM-UP: cocap_bc_warmup_ppo_gpu1 / PID 1510251 / step 19,000/110,000
DIRECT throughput/ETA: 23.34 step/s / ~57 min
WARM throughput/ETA: 22.45 step/s / ~68 min
WARM CONTRACT: step 12,032达到EV≥.20连续2次，Actor随后解冻；19k时ppo_env_steps=6,912
CHECKPOINT: 两支均0 milestones；首个25k尚未产生
```

GPU1同时存在不属于本任务的外部PID `1502230`（约2.45 GiB），未触碰。本任务两支约占6.92/6.03 GiB，总显存约15.49/49.14 GiB；22:29瞬时90°C，但最近核验的`HW/SW Thermal Slowdown`均未激活，slowdown/shutdown阈值95/98°C。supervisor每60秒记录资源；本轮不因单次温度读数打断健康训练。

训练终点原生评估为deterministic/stochastic各20回合。为与冻结BC Gate公平比较，另启 `cocap_bc_ppo_formal100_supervisor_gpu1` / PID `1523062`：当前为`WAITING_FOR_TRAINING`，待两支自然完成后，按Direct→Warm顺序从各自终点full-resume只读恢复，使用同一seeds `2026090301..0400`串行执行deterministic formal100。该路径不训练、不修改checkpoint，全部完成后停在`WAITING_FOR_RESULT`，不自动猜测下一路线。实现经24项合同测试与历史resume CPU 1回合烟测PASS。

在两分支结束并完成同合同deterministic formal前不作优劣结论。判读锚点保持冻结BC Gate：capture `1.00`、collision `.03`、2+/3+ `.64/.12`、length `71.16`。若Direct退化而warm-up保持，默认主线转为distillation→critic warm-up→PPO；若二者都退化，下一阶段才讨论annealed teacher KL。

## 5. 2026-09-07 终态复核

BC两支与同seed deterministic formal100均自然完成：Direct为capture/collision/2+/3+/length `.96/.04/.87/.28/80.32`，Warm为`.92/.08/.92/.27/72.18`，冻结BC为`1.00/.03/.64/.12/71.16`。Direct与Warm都出现capture保持损失，warm-up没有改善且point estimate更差；当前默认保留BC为冻结上限、Direct为最佳PPO分支，teacher-KL只登记为待决候选。

VXY support11线经事后合同审计确认配置层级错误：候选写入`reward.support_reward_*`，环境实际读取`voradj.support_reward_*`，训练仍为`.5/.5`。Stage1 850k/1M checkpoint与旧VXY bitwise一致，故整条线不能作为1/1因果实验。Stage1/2已完成，Stage2 950k formal20四项成功率均1且collision0，可作为原合同延长训练资产；Stage3因2026-09-06 19:53主机重启中断于日志472425，最后持久化checkpoint为450k，`/dev/shm`精确resume已丢失。本轮不恢复。

完整宏观坐标、paired区间、checkpoint SHA与中断证据见 `docs/PROJECT_STATUS_20260907_ZH.md`。2026-09-07现场没有本项目进程或tmux。

NEXT WAKE-UP: first verify `docs/PROJECT_STATUS_20260907_ZH.md` and choose one valid G0 correction plus whether G1 should stop at Direct PPO or test annealed teacher KL; do not resume the mislabeled support11 Stage3 as a valid 1/1 experiment.
