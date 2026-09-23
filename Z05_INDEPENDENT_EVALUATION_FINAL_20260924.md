# IQN Z05 independent evaluation

日期：2026-09-24（Asia/Shanghai）  
分支：`evaluation/iqn-z05-independent-20260923`  
评测原则：只评估；未训练新 checkpoint，未修改 policy/reward，未处理 AC。

## 结论先行

- Stage1/2/3 native selected-best 均完成；Stage3 selected 固定为 `step_600000.pt`。
- Stage3 向下泛化：4p 是“捕获兼容、完整任务不兼容”（Coverage strict CE 65%，Mixed safe-complete 50%）；8p 是“基本兼容但有退化”（Coverage 95%，Capture 95%，Mixed safe-complete 85%）。
- Stage3 向上 fixed-map zero-shot：G1/G2/G3 均可运行，未触发 Density-Normalized Sensing V2 的 20m floor；Coverage 分别为 100%/100%/95%，但 Mixed safe-complete 为 75%/60%/65%，碰撞率为 15%/25%/25%。
- 24-agent 运行时 policy 实际 friend token cap 仍是 8；G1/G2/G3 均使用原 checkpoint 的 physical-only local-adjacency top-k/truncation 语义，没有改网络架构。截断在 G1/G2/G3 分别为 90/352/330 events，涉及 11/31/31 个 episode。
- 发现并修复一个 evaluator-only blocker：N>12 时跳过 central diagnostic-only padding tensor；该 tensor 不是 policy input，未改变 checkpoint、policy tensor shape、reward 或训练语义。

## Selected checkpoint 与复现对象

| stage | contract | selected checkpoint | SHA256 |
|---|---|---|---|
| Stage1 | 4p/1e/1 obstacle | `step_1300000.pt` | `d35b7984d8ca03fa7b536811794578ff402e9b8ad48589e3318f2cb1077d702b` |
| Stage2 | 8p/2e/2 obstacles | `step_100000.pt` | `f137f4fcd302c5ff8b7282609344a02e4c1702c84d15cefff9deda7c23d56f8c` |
| Stage3 | 12p/3e/3 obstacles | `step_600000.pt` | `8ee5c162c32883984f72aa4be4b86e82338ae1e8d8c912011d181987a476d095` |

Stage1/2 selected 值均直接来自各自 `selection_report.json`，没有猜 checkpoint。

正式 evaluator 使用：`run_episode`、`ZDiagnostics`、`efficiency_timing_summary`、fixed-midpoint-32 greedy quantiles；每个 scene 20 个完整 rollout，GIF 使用同一批 20 中的 episode index `0,2,...,18`，每 scene 10 个，无额外 GIF 分布。

## Part A — Native selected-best

下表中的时间格式为 `mean / median / p90`；Coverage 的 CE time 同时给 steps 与 seconds。`cens` 是 censored 或失败 episode 数，单独计数，不当作成功。

### Stage1 native — 4p/1e/1 obstacle

- Coverage：strict CE `1.00`，collision `0`；CE RMS `0.0415/0.0430/0.0491`，CE max `0.0637/0.0652/0.0770`，area CV `0.0734/0.0717/0.1102`；time-to-CE `366/213/840 steps`、`183/106.5/420 s`；cens `0`。
- Capture：success `1.00`，normal `1.00`，stationary `0`，collision `0`；capture `71.1/70/94.4 steps`、`35.55/35/47.2 s`；cens `0`。
- Mixed：capture `1.00`，collision `0`，post-capture CE `0.90`，safe-complete `0.90`；capture `79.15/73.5/113.2 steps`、`39.575/36.75/56.6 s`；recovery `218.67/207.5/307.5 steps`、`109.33/103.75/153.75 s`；mission `297.94/279/414.2 steps`、`148.97/139.5/207.1 s`；cens `2/20`。

### Stage2 native — 8p/2e/2 obstacles

- Coverage：strict CE `0.90`，collision `0.05`；CE RMS `0.0489/0.0476/0.0520`，CE max `0.0800/0.0714/0.0953`，area CV `0.1649/0.1459/0.2369`；time-to-CE `851.33/723/1475.2 steps`、`425.67/361.5/737.6 s`；cens `2/20`。
- Capture：success `1.00`，normal `1.00`，stationary `0`，collision `0`；capture `80.15/77.5/109.8 steps`、`40.075/38.75/54.9 s`；cens `0`。
- Mixed：capture `0.95`，collision `0.25`，post-capture CE `0.20`，safe-complete `0.20`；capture `80.05/80/106.4 steps`、`40.03/40/53.2 s`；recovery `341.75/366/496.6 steps`、`170.88/183/248.3 s`；mission `427.25/452/600.4 steps`、`213.63/226/300.2 s`；cens `16/20`。

### Stage3 native — 12p/3e/3 obstacles

- Coverage：strict CE `1.00`，collision `0`；CE RMS `0.0443/0.0437/0.0489`，CE max `0.0793/0.0790/0.0918`，area CV `0.1940/0.1903/0.2354`；time-to-CE `369.45/231.5/597.1 steps`、`184.73/115.75/298.55 s`；cens `0`。
- Capture：success `1.00`，normal `1.00`，stationary `0`，collision `0`；capture `75.6/71.5/101.1 steps`、`37.8/35.75/50.55 s`；cens `0`。
- Mixed：capture `1.00`，collision `0.10`，post-capture CE `0.85`，safe-complete `0.85`；capture `65.3/63.5/93.9 steps`、`32.65/31.75/46.95 s`；recovery `283.18/318/509.8 steps`、`141.59/159/254.9 s`；mission `348.88/382/566.2 steps`、`174.44/191/283.1 s`；cens `3/20`。

三个 native run 的 z-state direct violations 和 update-count violations 均为 `0`，`update_count_per_decision_exact=true`。

## Part B — Stage3 selected 向下泛化

仍使用 Stage3 `step_600000.pt`，4p/8p 保留历史 scene/spawn/sensing/task 合同，不使用大规模动态 spawn。

### Stage3 → 4p/1e

- Coverage：strict CE `0.65`，collision `0`；CE RMS `0.0546/0.0488/0.0778`，CE max `0.0690/0.0680/0.0909`，area CV `0.0550/0.0517/0.0804`；time-to-CE `848.23/754/1540.2 steps`、`424.12/377/770.1 s`；cens `7/20`。
- Capture：success `1.00`，normal `1.00`，stationary `0`，collision `0`；capture `67.05/65.5/91.9 steps`、`33.53/32.75/45.95 s`；cens `0`。
- Mixed：capture `1.00`，collision `0`，post-capture CE `0.50`，safe-complete `0.50`；capture `60.2/61.5/78.9 steps`、`30.1/30.75/39.45 s`；recovery `210.7/165/287.9 steps`、`105.35/82.5/143.95 s`；mission `271.4/230.5/369.0 steps`、`135.7/115.25/184.5 s`；cens `10/20`。

结论：4p 只保留 capture 能力，不能称为完整任务向下兼容。

### Stage3 → 8p/2e

- Coverage：strict CE `0.95`，collision `0`；CE RMS `0.0458/0.0463/0.0488`，CE max `0.0682/0.0668/0.0798`，area CV `0.1412/0.1322/0.1933`；time-to-CE `314.84/304/526.8 steps`、`157.42/152/263.4 s`；cens `1/20`。
- Capture：success `0.95`，normal `0.95`，stationary `0`，collision `0.05`；capture `67.32/60/97 steps`、`33.66/30/48.5 s`；cens `1/20`。
- Mixed：capture `1.00`，collision `0.05`，post-capture CE `0.85`，safe-complete `0.85`；capture `72.85/72/92.1 steps`、`36.43/36/46.05 s`；recovery `254.94/251/380 steps`、`127.47/125.5/190 s`；mission `327.59/315/446 steps`、`163.79/157.5/223 s`；cens `3/20`。

结论：8p 基本向下兼容，但有轻微 capture/collision/mission 退化。

## Part C — Fixed-map zero-shot，120m × 120m

公式固定为 `R = 0.8715 * sqrt(A_eff / N)`，floor `20m`，surface-distance semantics=`surface_clearance`。`A_eff` 来自该 episode 的真实 obstacle mask、robot radius、`free_mask_projected`；每个 episode 保存 N/E/O、map size、A_eff、raw/resolved R、floor、map scale、free-mask/obstacle-mask hash。Coverage 的 `spawn_cluster_radius` 等于该 episode 的 `resolved_R`。

### G1 — 16p/4e/4 obstacles，post window 800

- resolver：`A_eff 14216..14252`，`R 25.9774..26.0103m`，floor 未触发，map scale `1`，spawn radius 同 R。
- Coverage：strict CE `1.00`，collision `0`；CE RMS `0.0451/0.0452/0.0496`，CE max `0.0824/0.0832/0.0987`，area CV `0.1972/0.1973/0.2741`；time-to-CE `360.25/218.5/752 steps`、`180.13/109.25/376 s`；cens `0`。
- Capture：success `1.00`，normal `1.00`，stationary `0`，collision `0`；capture `70.85/69.5/87.2 steps`、`35.43/34.75/43.6 s`；cens `0`。
- Mixed：capture `1.00`，collision `0.15`，post-capture CE `0.85`，safe-complete `0.75`；capture `63.4/60.5/80 steps`、`31.7/30.25/40 s`；recovery `327.88/322/583.4 steps`、`163.94/161/291.7 s`；mission `409.27/422/686 steps`、`204.63/211/343 s`；cens `5/20`。
- token audit：friend candidate `4.121/4/6`（mean/p50/p90），occupancy mean `4.121`，max candidate `10`，policy occupancy max `8`；truncation `90 events / 11 episodes`。

### G2 — 20p/5e/5 obstacles，post window 900

- resolver：`A_eff 14176..14216`，`R 23.2022..23.2349m`，floor 未触发，map scale `1`，spawn radius 同 R。
- Coverage：strict CE `1.00`，collision `0`；CE RMS `0.0448/0.0449/0.0483`，CE max `0.0823/0.0830/0.0883`，area CV `0.2001/0.1920/0.2395`；time-to-CE `324/253.5/621.5 steps`、`162/126.75/310.75 s`；cens `0`。
- Capture：success `1.00`，normal `1.00`，stationary `0`，collision `0`；capture `68.25/63/89.5 steps`、`34.13/31.5/44.75 s`；cens `0`。
- Mixed：capture `1.00`，collision `0.25`，post-capture CE `0.85`，safe-complete `0.60`；capture `62.55/55.5/82 steps`、`31.28/27.75/41 s`；recovery `317.76/289/588.4 steps`、`158.88/144.5/294.2 s`；mission `342.5/360/517.5 steps`、`171.25/180/258.75 s`；cens `8/20`。
- token audit：friend candidate `4.323/4/6`，occupancy mean `4.322`，max candidate `12`，policy occupancy max `8`；truncation `352 events / 31 episodes`。

### G3 — 24p/6e/5 obstacles，post window 1000

- resolver：`A_eff 14176..14216`，`R 21.1806..21.2105m`，floor 未触发，map scale `1`，spawn radius 同 R；仍高于 20m sanity floor，正式值以 resolver 为准。
- Coverage：strict CE `0.95`，collision `0`；CE RMS `0.0453/0.0456/0.0495`，CE max `0.0864/0.0862/0.0953`，area CV `0.2031/0.2075/0.2561`；time-to-CE `513.95/225/1702.2 steps`、`256.97/112.5/851.1 s`；cens `1/20`。
- Capture：success `1.00`，normal `1.00`，stationary `0`，collision `0.05`；capture `59.15/56/75.3 steps`、`29.58/28/37.65 s`；cens `0`。
- Mixed：capture `1.00`，collision `0.25`，post-capture CE `0.80`，safe-complete `0.65`；capture `57.8/56.5/75.5 steps`、`28.9/28.25/37.75 s`；recovery `440.81/353.5/820.5 steps`、`220.41/176.75/410.25 s`；mission `448.77/312/905.6 steps`、`224.38/156/452.8 s`；cens `7/20`。
- token audit：friend candidate `4.428/4/6`，occupancy mean `4.427`，max candidate `11`，policy occupancy max `8`；truncation `330 events / 31 episodes`。

## Capacity / exact z-state audit

- 原 checkpoint 的 policy/environment caps：friend `8`、evader `8`、obstacle `5`。因此 16/20/24 环境人数不等于 policy 一次看到全部 friends；实际使用 `physical_only sorted local adjacency then first max_pursuer_num`，padding masked，role/z 不影响 truncation priority。
- Stage3 native 的 truncation 为 `75/6 episodes`；G1/G2/G3 为 `90/11`、`352/31`、`330/31`。这保持了训练时既有 local/top-k/padded 语义；`architecture_modified=false`。
- 所有正式 run 的 exact z-state checks：direct violations `0`、update-count violations `0`；visible-z-one rate `1`；post-capture never-release `0`。大规模 run 分别使用 post window 800/900/1000，失败仍保留为 censored/unsuccessful。

## Scalability 判断

Density resolver 趋势符合合同：Stage3 native 约 `30.05m` → G1 `25.99m` → G2 `23.22m` → G3 `21.20m`，没有偷偷触发 floor 或 map scaling。Coverage 在 G1/G2 保持 100%，G3 降至 95%；但 Mixed collision 从 15% 升至 25%，safe-complete 在 G1/G2/G3 为 75%/60%/65%，并且 friend truncation 在大规模显著出现。

因此本轮的第一处工程容量压力是 friend-token truncation（G1 已出现，G2 达到更高 event count），但它不是 schema/architecture blocker；第一处可观察的 policy-performance failure 是从 G1 开始的 Mixed collision/safe-complete 退化，而不是 Coverage 或 Capture 单项失败。G3 没有触发 floor，也没有触发 N>=28 map-scaling regime。

## Artifacts

每个 run 目录均包含 `report.json`、`compact_summary.json`、`episodes.jsonl` 和 `gifs/`。G3 已校验 `60` episode rows、`30` GIF；其余每个 run 也为 20×3 episodes、30 GIF。原始大文件未纳入 git commit。

核心代码：`tools/evaluate_iqn_z05_independent_20260923.py`。  
本轮 evaluator-only 修复：`tools/run_forward_final_bridge_20260908.py`。

## 资源与 git

评测前 root disk 约 29GB free、inode 约 7% used；完成 G3 后约 23GB free、inode 7% used。全程单 GPU 串行大规模 GIF 评测，没有训练新 checkpoint。

