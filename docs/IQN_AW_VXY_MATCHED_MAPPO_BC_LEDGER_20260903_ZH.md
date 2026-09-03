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

当前 PPO 状态：`GATE_PASS / READY_FOR_TWO_BRANCHES`。下一步从同一 SHA 的BC Actor实现/启动 Direct PPO control与critic warm-up主线。

NEXT WAKE-UP: after the two PPO branches start, first compare their earliest deterministic eval against the frozen BC Gate (`capture=1.00`, `collision=.03`, `length=71.16`).
