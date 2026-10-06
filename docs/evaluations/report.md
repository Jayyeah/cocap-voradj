# CoCap IQN-Z Z05/Z07 最终性能评估

日期：2026-10-06。Stage3 新评估每场景 20 rollouts；泛化地图为 120×120，固定 3 个障碍，pursuer 数量为 12 / 18 / 24。没有训练，也没有改 reward、模型或训练合同。

## Checkpoint registry

| Arm | Stage | Step | SHA256 | Runtime 路径 |
|---|---:|---:|---|---|
| Z05 | 1 | 1,300,000 | `d35b7984d8ca03fa7b536811794578ff402e9b8ad48589e3318f2cb1077d702b` | `/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05/stages/stage1/training/checkpoints/step_1300000.pt` |
| Z05 | 2 | 100,000 | `f137f4fcd302c5ff8b7282609344a02e4c1702c84d15cefff9deda7c23d56f8c` | `/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05/stages/stage2/training/checkpoints/step_100000.pt` |
| Z05 | 3 | 600,000 | `8ee5c162c32883984f72aa4be4b86e82338ae1e8d8c912011d181987a476d095` | `/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05/stages/stage3/training/checkpoints/step_600000.pt` |
| Z07 | 1 | 1,600,000 | `377f90587debb7538dd34b0354559e49b54ff07b51b70c8a91e6efee530f6a00` | `/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z07/stages/stage1/training/checkpoints/step_1600000.pt` |
| Z07 | 2 | 200,000 | `a6db76539c96d6f9c6d3b33abc002aa06021189146aa7ecf352b51fbfb526f35` | `/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z07/stages/stage2/training/checkpoints/step_200000.pt` |
| Z07 | 3 | 300,000 | `0180d96644c2abadb7c6a5183ecbd0ce23cc68347f15a7152485edeb68e53fd3` | `/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z07/stages/stage3/training/checkpoints/step_300000.pt` |

DAG 当前记录的 Stage3 700k 训练终点与 registry 选中的 final-stage checkpoint 分开列出。`step_700000.pt` 与兼容封装 `final_step_700000.pt` 是不同文件，均按 SHA256 单独记录。

- Z05 Stage3 step 700,000 `step_700000.pt` — `dcf326c18f577ae0608b12017f914c3be06af485409e87a8daf3a30b4d7e8a7e` — `/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05/stages/stage3/training/checkpoints/step_700000.pt`
- Z05 Stage3 step 700,000 `final_step_700000.pt` — `eb8a0da3a9b10e2f1538acff03964571f99c75ec2e87e56a78ee6039a3adf552` — `/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05/stages/stage3/training/checkpoints/final_step_700000.pt`
- Z07 Stage3 step 700,000 `step_700000.pt` — `4accabc6f320db0209e8b73e142d92e242b6f6c0e3539f47e7b25fcc1f6e04e3` — `/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z07/stages/stage3/training/checkpoints/step_700000.pt`
- Z07 Stage3 step 700,000 `final_step_700000.pt` — `a67823a299c3087540dbcc0071968a33a71574cf793fd96b387cfc80675ab71f` — `/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z07/stages/stage3/training/checkpoints/final_step_700000.pt`

## Stage1 / Stage2 / Stage3 比较

来源为各 checkpoint 的 registry formal evaluation：每场景 20 episodes。完成步数按成功 rollout 的 `length` 统计；capture time 按已捕获 rollout 统计。CE RMS 和 Area CV 是可用 episode 的均值；有效样本数记录在 `results.json`。失败类型按逐 episode flags 分类。

| Arm | Stage | Scene | Success | Completion steps mean / median | Collision | CE RMS mean | Area CV mean | Capture time mean (s) | Failure counts |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Z05 | 1 | coverage | 20/20 (100.0%) | 366.000 / 213.000 | 0/20 | 0.041 | 0.073 | — | {"success":20} |
| Z05 | 1 | capture | 20/20 (100.0%) | 71.100 / 70.000 | 0/20 | 0.388 | 0.827 | 35.550 | {"success":20} |
| Z05 | 1 | mixed | 18/20 (90.0%) | 297.944 / 279.000 | 0/20 | 0.040 | 0.058 | 39.575 | {"success":18,"post_capture_CE_or_safe_completion_failure":2} |
| Z05 | 2 | coverage | 18/20 (90.0%) | 851.333 / 723.000 | 1/20 | 0.049 | 0.165 | — | {"success":18,"strict_CE_not_reached":1,"collision":1} |
| Z05 | 2 | capture | 20/20 (100.0%) | 80.150 / 77.500 | 0/20 | 0.339 | 0.957 | 40.075 | {"success":20} |
| Z05 | 2 | mixed | 4/20 (20.0%) | 427.250 / 452.000 | 5/20 | 0.081 | 0.301 | 40.026 | {"post_capture_CE_or_safe_completion_failure":11,"collision":5,"success":4} |
| Z05 | 3 | coverage | 20/20 (100.0%) | 369.450 / 231.500 | 0/20 | 0.044 | 0.194 | — | {"success":20} |
| Z05 | 3 | capture | 20/20 (100.0%) | 75.600 / 71.500 | 0/20 | 0.295 | 1.011 | 37.800 | {"success":20} |
| Z05 | 3 | mixed | 17/20 (85.0%) | 348.882 / 382.000 | 2/20 | 0.055 | 0.227 | 32.650 | {"success":17,"post_capture_CE_or_safe_completion_failure":1,"collision":2} |
| Z07 | 1 | coverage | 20/20 (100.0%) | 192.500 / 194.000 | 0/20 | 0.036 | 0.046 | — | {"success":20} |
| Z07 | 1 | capture | 20/20 (100.0%) | 92.100 / 81.500 | 0/20 | 0.400 | 0.685 | 46.050 | {"success":20} |
| Z07 | 1 | mixed | 20/20 (100.0%) | 286.750 / 256.500 | 0/20 | 0.038 | 0.050 | 54.625 | {"success":20} |
| Z07 | 2 | coverage | 20/20 (100.0%) | 147.450 / 123.000 | 0/20 | 0.040 | 0.153 | — | {"success":20} |
| Z07 | 2 | capture | 20/20 (100.0%) | 78.150 / 75.000 | 0/20 | 0.364 | 1.190 | 39.075 | {"success":20} |
| Z07 | 2 | mixed | 20/20 (100.0%) | 220.400 / 220.000 | 0/20 | 0.040 | 0.141 | 41.775 | {"success":20} |
| Z07 | 3 | coverage | 17/20 (85.0%) | 577.588 / 349.000 | 0/20 | 0.049 | 0.181 | — | {"success":17,"strict_CE_not_reached":3} |
| Z07 | 3 | capture | 19/20 (95.0%) | 78.474 / 75.000 | 1/20 | 0.276 | 1.012 | 39.237 | {"success":19,"collision":1} |
| Z07 | 3 | mixed | 13/20 (65.0%) | 347.154 / 260.000 | 1/20 | 0.061 | 0.200 | 32.450 | {"success":13,"post_capture_CE_or_safe_completion_failure":6,"collision":1} |

## Stage3 原始规模与泛化结果

下表 metrics 按 arm / 规模 / 场景给出。Small=12 pursuers（原始规模），Medium=18（约 1.5×），Large=24（约 2×）。Enemy count、障碍数、有效面积、解析 sensing radius 与 coverage 初始化半径均从运行时 resolver / 配置自动读取；Small native evaluator 将 coverage_spawn_radius 留空时取自 Stage3 有效配置值 13.0。

| Arm | Scale | Scene | P / E / Obs | Map | Sensing R mean [min,max] | Coverage init R | Success | Completion steps mean / median | Collision | CE RMS mean | Area CV mean | Capture time mean (s) | Failure counts |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Z05 | Small | coverage | 12/0/3 | 120.0×120.0 | 30.064 [30.047,30.076] | 13.000 | 20/20 (100.0%) | 307.650 / 226.500 | 0/20 | 0.043 | 0.182 | — | {"success":20} |
| Z05 | Small | capture | 12/3/3 | 120.0×120.0 | 30.064 [30.047,30.076] | n/a | 20/20 (100.0%) | 71.500 / 69.500 | 0/20 | 0.313 | 0.967 | 35.750 | {"success":20} |
| Z05 | Small | mixed | 12/3/3 | 120.0×120.0 | 30.064 [30.047,30.076] | n/a | 15/20 (75.0%) | 359.333 / 306.000 | 1/20 | 0.057 | 0.247 | 35.342 | {"success":15,"post_capture_CE_or_safe_completion_failure":4,"collision":1} |
| Z05 | Medium | coverage | 18/0/3 | 120.0×120.0 | 24.547 [24.533,24.557] | 24.548 | 20/20 (100.0%) | 263.100 / 138.000 | 0/20 | 0.045 | 0.195 | — | {"success":20} |
| Z05 | Medium | capture | 18/3/3 | 120.0×120.0 | 24.547 [24.533,24.557] | n/a | 20/20 (100.0%) | 53.700 / 53.500 | 1/20 | 0.240 | 0.881 | 26.850 | {"collision":1,"success":19} |
| Z05 | Medium | mixed | 18/3/3 | 120.0×120.0 | 24.547 [24.533,24.557] | n/a | 17/20 (85.0%) | 336.882 / 360.000 | 0/20 | 0.046 | 0.203 | 24.075 | {"success":17,"post_capture_CE_or_safe_completion_failure":3} |
| Z05 | Large | coverage | 24/0/3 | 120.0×120.0 | 21.259 [21.246,21.267] | 21.260 | 19/20 (95.0%) | 452.684 / 323.000 | 0/20 | 0.045 | 0.205 | — | {"success":19,"strict_CE_not_reached":1} |
| Z05 | Large | capture | 24/3/3 | 120.0×120.0 | 21.259 [21.246,21.267] | n/a | 20/20 (100.0%) | 36.550 / 33.000 | 0/20 | 0.236 | 0.812 | 18.275 | {"success":20} |
| Z05 | Large | mixed | 24/3/3 | 120.0×120.0 | 21.259 [21.246,21.267] | n/a | 15/20 (75.0%) | 264.667 / 234.000 | 1/20 | 0.047 | 0.222 | 17.100 | {"success":15,"collision":1,"post_capture_CE_or_safe_completion_failure":4} |
| Z07 | Small | coverage | 12/0/3 | 120.0×120.0 | 30.064 [30.047,30.076] | 13.000 | 14/20 (70.0%) | 500.286 / 179.000 | 1/20 | 0.058 | 0.245 | — | {"success":14,"strict_CE_not_reached":5,"collision":1} |
| Z07 | Small | capture | 12/3/3 | 120.0×120.0 | 30.064 [30.047,30.076] | n/a | 20/20 (100.0%) | 83.600 / 79.000 | 0/20 | 0.255 | 1.044 | 41.800 | {"success":20} |
| Z07 | Small | mixed | 12/3/3 | 120.0×120.0 | 30.064 [30.047,30.076] | n/a | 9/20 (45.0%) | 357.556 / 327.000 | 1/20 | 0.057 | 0.215 | 38.275 | {"post_capture_CE_or_safe_completion_failure":10,"success":9,"collision":1} |
| Z07 | Medium | coverage | 18/0/3 | 120.0×120.0 | 24.547 [24.533,24.557] | 24.548 | 5/20 (25.0%) | 460.200 / 423.000 | 1/20 | 0.054 | 0.196 | — | {"strict_CE_not_reached":14,"collision":1,"success":5} |
| Z07 | Medium | capture | 18/3/3 | 120.0×120.0 | 24.547 [24.533,24.557] | n/a | 20/20 (100.0%) | 70.700 / 68.000 | 1/20 | 0.243 | 0.903 | 35.350 | {"success":19,"collision":1} |
| Z07 | Medium | mixed | 18/3/3 | 120.0×120.0 | 24.547 [24.533,24.557] | n/a | 5/20 (25.0%) | 239.400 / 231.000 | 4/20 | 0.055 | 0.225 | 29.500 | {"post_capture_CE_or_safe_completion_failure":11,"success":5,"collision":4} |
| Z07 | Large | coverage | 24/0/3 | 120.0×120.0 | 21.259 [21.246,21.267] | 21.260 | 5/20 (25.0%) | 143.400 / 127.000 | 2/20 | 0.055 | 0.222 | — | {"strict_CE_not_reached":14,"success":4,"collision":2} |
| Z07 | Large | capture | 24/3/3 | 120.0×120.0 | 21.259 [21.246,21.267] | n/a | 20/20 (100.0%) | 55.850 / 51.000 | 0/20 | 0.196 | 0.791 | 27.925 | {"success":20} |
| Z07 | Large | mixed | 24/3/3 | 120.0×120.0 | 21.259 [21.246,21.267] | n/a | 1/20 (5.0%) | 162.000 / 162.000 | 7/20 | 0.057 | 0.282 | 22.050 | {"post_capture_CE_or_safe_completion_failure":12,"collision":7,"success":1} |

Radius policy: `R = 0.8715 × sqrt(A_eff / N)`, lower bound 20; `A_eff` changes by generated obstacle layout. Every map was 120×120 and no run crossed the 120×120 limit. Coverage initialization radius for Medium/Large is resolved by the evaluator at runtime.

## GIFs for manual review

- [z05_stage3_capture_2026200601.gif](gifs/z05_stage3_capture_2026200601.gif) (1,491,865 bytes)
- [z05_stage3_capture_2026200603.gif](gifs/z05_stage3_capture_2026200603.gif) (2,895,335 bytes)
- [z05_stage3_capture_2026200605.gif](gifs/z05_stage3_capture_2026200605.gif) (2,027,653 bytes)
- [z05_stage3_coverage_2026100601.gif](gifs/z05_stage3_coverage_2026100601.gif) (5,730,641 bytes)
- [z05_stage3_coverage_2026100603.gif](gifs/z05_stage3_coverage_2026100603.gif) (11,214,031 bytes)
- [z05_stage3_coverage_2026100605.gif](gifs/z05_stage3_coverage_2026100605.gif) (7,671,717 bytes)
- [z05_stage3_mixed_2026300601.gif](gifs/z05_stage3_mixed_2026300601.gif) (10,425,017 bytes)
- [z05_stage3_mixed_2026300603.gif](gifs/z05_stage3_mixed_2026300603.gif) (10,080,780 bytes)
- [z05_stage3_mixed_2026300605.gif](gifs/z05_stage3_mixed_2026300605.gif) (10,488,793 bytes)
- [z07_stage3_mixed_2026300601.gif](gifs/z07_stage3_mixed_2026300601.gif) (10,440,650 bytes)

## Failure taxonomy

- `success`: coverage reached strict CE; capture captured target; mixed safely completed the post-capture CE mission.
- `collision`: collision flag; classified before other failure causes.
- `boundary`: boundary flag without collision.
- `strict_CE_not_reached`: pure coverage did not reach strict CE, without collision/boundary.
- `capture_not_achieved_or_censored`: pure capture ended without target capture.
- `capture_failure`: mixed mission ended before capture.
- `post_capture_CE_or_safe_completion_failure`: mixed mission captured but did not safely finish.

Per-rollout success, completion steps, collision, CE, capture time, failure type and auto-resolved environment values are in [`results.json`](results.json). Stage3 new-run source reports and all stage comparison official report locations are enumerated there.
