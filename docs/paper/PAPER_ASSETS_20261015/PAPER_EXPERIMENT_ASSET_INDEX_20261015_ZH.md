# CoCap RA-L 论文实验资产总索引

**盘点日期：** 2026-10-15

**用途：** 论文实验事实源、checkpoint 核对、历史评估与可视化素材查阅。

**范围：** 只盘点既有资产；本轮没有训练、没有启动 rollout、没有重渲染 GIF，也没有覆盖、迁移或删除旧文件。

## 1. 版本事实源

| 资料线 | 本地仓库 / 分支 | 本次核验的 HEAD 或记录版本 | 说明 |
|---|---|---|---|
| Z05 统一衰减双课程训练 | /home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919；experiment/iqn-z-unified-decay-dual-curriculum-20260919 | f0c0dfca8a508233f30ddbb5158699400d92a00f | Z05、Z07 训练配置和正式 selected checkpoint 所属训练线 |
| Z05 独立评估 | /home/yjq/rl/CoCap1/iqn-z05-independent-20260923；evaluation/iqn-z05-independent-20260923 | 当前 checkout 1a9da054ecb5652726005d00d08ab05a2188dd8e；9/23 主要报告所记 evaluator/source SHA 5c146f74752073414af416b0db6c686e1c588a07；10/6 补充报告 SHA 2eeec7e6638640295a8469583500e5d3ad265682 | 每份结果以其 report 的 source SHA 为准，不能把当前 checkout SHA 冒充单次运行 SHA |
| repeated-arrival 演示/审计 | /home/yjq/rl/CoCap1/z05-repeated-arrival-demo-20261006；evaluation/z05-repeated-arrival-demo-20261006 | 751ee2fdf8e0134504e9b4f033904cc610dd6891 | strict 正式数据在相邻的 z05-repeated-arrival-formal-20261007-heldout20x3；relaxed robustness 数据在 z05-repeated-arrival-casualty-robustness-20261006。strict evaluator 记录版本 b308050f4ad99d49180538af4fe0159340f430c2；relaxed 功能版本 906e2bcfd205d6d0f21072e2eed00a7409b2f532 |
| Evidence Local/Global 对照 | /home/yjq/rl/CoCap1/evidence-local-global-20261006 | 审计 checkout 6b7aab28546e1a30c60f84d12de3bde70930f670（有本地运行产物改动） | 这是审计目录版本，不是 50×3 运行 source SHA；旧报告没有记录真实 evaluator/training source SHA |
| 历史 main 比对 | /home/yjq/rl/cocap-voradj；main | a5814f49fa29d869cdc3fb8d8e0df4722aa11f00 | 只作历史比对；工作树已有 staged 删除，本轮没有提交或改动它们 |

方法源代码审计与人工说明已读取：/home/yjq/rl/cocap-voradj/docs/paper/COCAP_METHOD_SOURCE_AUDIT_20261015_ZH.md、/home/yjq/rl/cocap-voradj/docs/paper/COCAP_METHOD_HUMAN_GUIDE_20261015_ZH.md。后者解释方法和传感语义；训练/评估事实以各自分支、checkpoint、配置和报告为准。

## 2. Checkpoint 盘点结论

checkpoint registry 共 16 项：每个二进制 checkpoint 都记录了路径、字节数、SHA256、配置/hash、stage/step、训练来源记录、评估来源和保留优先级。盘点时对现存权重采用 PyTorch weights_only=True 读取，并使用对应网络配置进行 strict=True state-dict 加载，核对结构与兼容性。所有 registry 中列出的文件都通过 SHA256 和兼容性核验。

| 方法/用途 | 正式阶段与 step | checkpoint SHA256 | 论文保留解释 |
|---|---|---|---|
| **Z05 主方法** | Stage1 / 1,300,000 | d35b7984d8ca03fa7b536811794578ff402e9b8ad48589e3318f2cb1077d702b | 课程早期检查点；不是论文最终 policy |
| **Z05 主方法** | Stage2 / 100,000 | f137f4fcd302c5ff8b7282609344a02e4c1702c84d15cefff9deda7c23d56f8c | 课程中间检查点；不是论文最终 policy |
| **Z05 主方法** | **Stage3 / 600,000 selected** | **8ee5c162c32883984f72aa4be4b86e82338ae1e8d8c912011d181987a476d095** | **论文主方法正式 selected；与预期 SHA 完全一致** |
| Z05 未选择候选 | Stage3 / 700,000 | dcf326c18f577ae0608b12017f914c3be06af485409e87a8daf3a30b4d7e8a7e | 文件存在、可加载，但未正式 selected；不与 600k 混用 |
| Z07 对照 | Stage1 / 1,600,000 | 377f90587debb7538dd34b0354559e49b54ff07b51b70c8a91e6efee530f6a00 | 历史方法版本；Stage1/2 native 独立评估报告缺失 |
| Z07 对照 | Stage2 / 200,000 | a6db76539c96d6f9c6d3b33abc002aa06021189146aa7ecf352b51fbfb526f35 | 同上 |
| Z07 对照 | Stage3 / 300,000 | 0180d96644c2abadb7c6a5183ecbd0ce23cc68347f15a7152485edeb68e53fd3 | 有 20×3 native 报告；不可写成单变量因果消融 |
| Local Binary Evidence 臂 | Stage1 / 1,500,000 | 18de0e289efc0fb07bd4b890c505c21da92f4092f6d2be155faa4265ffa938c6 | Local Binary；checkpoint 权重已核验，训练源 SHA 未登记 |
| Local Binary Evidence 臂 | Stage2 / 300,000 | f6e0cf2172fc66e43a18bd1a77d6559a6c3ce6797c296f533703c3717e1ead5c | 同上 |
| Local Binary Evidence 臂 | Stage3 / 600,000 | 86d6e30aa9d6e615df9da4d4a255550464b0ca8feb5ed64de9fc805e6b6f0a4d | 旧 held-out 报告引用的最终权重 |
| Global Oracle Evidence 臂 | Stage1 / 1,700,000 | f16b90fedd377e496f6a02e43e7eb6068d48446f6dc84af67845184107d74640 | privileged/global 信息条件；checkpoint 权重已核验，训练源 SHA 未登记 |
| Global Oracle Evidence 臂 | Stage2 / 700,000 | 97e5561b0938b71c4698be3d7dff701fd332a3f2caa5e5da5cfd030e71309ca4 | 同上 |
| Global Oracle Evidence 臂 | Stage3 / 300,000 | b4e6566c4ac7860cdd802320f28c103b36b083e2651e1e335527c2782e16fbe0 | 旧 held-out 报告引用的最终权重 |
| Final is_pursuing 历史发布 | Stage1 / 2,000,000 | 2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89 | release checkpoint；无 rollout/GIF，source/config 不全 |
| Final is_pursuing 历史发布 | Stage2 / 300,000 | ef58ae9bdd018633afee0d16f2242d2f157a6cc18611e41474f115aed254b87e | 同上 |
| Final is_pursuing 历史发布 | Stage3 / 700,000 | 5e4173eac94c921685091d60033bffb8798180e974b72ec02437e3de265848ee | 同上 |

Z05/Z07 是同一核心维度配置：117 tensors、4,463,580 参数、hidden 256、8 heads、4 layers、32 quantiles；输入 self 9 维、最多 8 个友机与 8 个目标（各 7 维/体）、最多 5 个障碍 token（各 5 维/体），9 个 AW9 动作。Z05/Z07 开启 Z-state、不含 is_pursuing。Evidence Local/Global 结构同维，但关闭 Z-state；Global Oracle 的信息条件更强。Final is_pursuing release 使用不同结构（119 tensors、4,465,580 参数、pursuing embedding 8）。详细 config JSON、文件绝对路径、selection report SHA、hash 一致性及 eval 状态见 PAPER_CHECKPOINT_REGISTRY.csv。

## 3. 既有 rollout 与结果

### Z05 独立评估主线

主目录：/home/yjq/rl/CoCap1/iqn-z05-independent-20260923/artifacts/2026-09-23_iqn_z05_independent/

9/23 主要评估 source SHA：5c146f74752073414af416b0db6c686e1c588a07。每个批次 20 seeds × 3 场景，独立 JSON 报告和 episodes.jsonl 均在。8 个批次累计 480 局，有 240 个历史 GIF（每批每场景 10 个）；仅为 GIF 局保留完整 trajectory source JSON，事件逐局信息嵌入 episode 记录。评估 renderer 使用默认渲染配置，无独立 render config 文件。

| Batch | policy / 场景规模 | Pure Coverage CE 成功率 | Pure Capture normal 成功率 | Mixed 捕获率 | Mixed 存活完成率 | Mixed 碰撞率 |
|---|---|---:|---:|---:|---:|---:|
| stage1_native | S1，4P1E1O | 1.00 | 1.00 | 1.00 | 0.90 | 0.00 |
| stage2_native | S2，8P2E2O | 0.90 | 1.00 | 0.95 | 0.20 | 0.25 |
| stage3_native_run | S3，12P3E3O | 1.00 | 1.00 | 1.00 | 0.85 | 0.10 |
| stage3_to_4p | S3 固定策略，4P1E1O | 0.65 | 1.00 | — | 0.50 | 0.00 |
| stage3_to_8p | S3 固定策略，8P2E2O | 0.95 | 0.95 | — | 0.85 | 0.05 |
| G1_16p4e4o_retry | S3 固定策略，16P4E4O | 1.00 | 1.00 | 1.00 | 0.75 | 0.15 |
| G2_20p5e5o | S3 固定策略，20P5E5O | 1.00 | 1.00 | 1.00 | 0.60 | 0.25 |
| G3_24p6e5o | S3 固定策略，24P6E5O | 0.95 | 1.00 | 1.00 | 0.65 | 0.25 |

Native 的 S1/S2/S3 分别对应课程内生规模 4/1/1、8/2/2、12/3/3；跨规模组均使用正式 Z05 Stage3 checkpoint。S3 native capture 平均 37.8 s；mixed capture 平均 32.65 s、recovery 141.59 s、mission 174.44 s，20 局中 3 局删失。指标口径、逐局数值和 collision 子类型请读原 report，不从汇总率反推。

补充结果位于 /home/yjq/rl/CoCap1/iqn-z05-independent-20260923/artifacts/2026-10-06_iqn_z_final/stage3_z05/，source SHA 2eeec7e6638640295a8469583500e5d3ad265682：native 12/3/3 有 20×3 局及 9 GIF，CE/capture 各 1.00、mixed safe completion 0.75、collision 0.05；18/3/3 有 20×3 局、无 GIF，CE/capture 1.00、mixed safe 0.85；24/3/3 由 4 个 shard 合并为 20×3 局、无 GIF。它们使用不同目标密度/seed 家族，不能和 24/6/5 合并为同一点。

### Z07 与 Evidence

Z07 Stage3 native：/home/yjq/rl/CoCap1/iqn-z05-independent-20260923/artifacts/2026-10-06_iqn_z_final/stage3_z07/native/，20×3 逐局报告，checkpoint SHA 为 Z07 Stage3 selected；evaluator/source SHA 2eeec7e6638640295a8469583500e5d3ad265682。Coverage CE 0.70、collision 0.05；Pure Capture normal 1.00、stationary 0、平均 41.8 s；Mixed capture 1.00、safe 0.45、collision 0.05、recovery 平均 141.5 s。报告索引了固定 episode index 0 的 GIF；不能把它解释为均值或把 Z07 与 Z05 的差异归因于单一因素。

Evidence 旧版 50×3 的三个报告确实存在于 /home/yjq/rl/CoCap1/iqn-evidence-comparison-20261006-runtime/final_heldout_50/{z05,local_binary,global_oracle}/report.json。每臂 150 行、三臂 450 行，同 seed 和初态 fingerprint 配对；零 optimizer/replay update，checkpoint 未变。旧版摘要结果如下，**只可作为 provisional**：

| 臂 | Coverage CE 成功 / RMS CE / CE max | Pure Capture 正常成功 / 平均秒 | Mixed 捕获 / safe / collision |
|---|---|---|---|
| Z05 | 1.00 / 0.04453 / 0.07828 | 0.98 / 36.255 | 0.96 / 0.80 / 0.06 |
| Local Binary | 0.96 / 0.04423 / 0.07550 | 0.86 / 38.83 | 0.96 / 0.62 / 0.10 |
| Global Oracle | 1.00 / 0.03555 / 0.06089 | 0.98 / 33.38 | 0.96 / 0.96 / 0.04 |

旧报告对成功 CE episode 的 time_to_ce_seconds 全为空；episode duration 又包含 CE 后 grace，不能补算首次 CE 时间。运行报告也没有真实 training/evaluator source SHA，没有独立 episodes 文件、逐局 event、trajectory 或 GIF。final_heldout_50_corrected 尚不存在，故 corrected 50×3 **尚未闭合**。旧结果不得作为最终主表的完整复现证据。详见 readiness 文档的资源记录与命令位置。

### Repeated-arrival

Strict held-out：/home/yjq/rl/CoCap1/z05-repeated-arrival-formal-20261007-heldout20x3/。60 局、60 trajectory、60 events、15 GIF，contract diagnostics 60/60 通过；verification 文件核对轨迹/event SHA 以及抽样 GIF 解码和 seed 映射。PERSIST-A/B/C strict service completion 分别 9/20、13/20、14/20；全目标捕获分别 12/20、16/20、16/20。A/B/C mission time 均值分别 467.075、218.45、227.075 秒。strict service completion 要求三目标全捕获、最终恢复成功，且无碰撞、终止失败、active-pursuer loss。未证明可靠性或泛化。

Relaxed casualty robustness：/home/yjq/rl/CoCap1/z05-repeated-arrival-casualty-robustness-20261006/。A′/B/C 各 20 局，共 60 trajectories、60 events、15 GIF；完成服务 13/20、13/20、14/20，三目标捕获各 20/20。它维持至少 12 台机器人 strict CE 门槛，但采用 relaxed survival stop；A′ 还改了 timeout-continue 语义，因此只能作鲁棒性补充，不能替代 strict 结果。另有 3×3 demo：9 局、9 trajectory/event、3 个 regimen GIF，样本量小且 GIF 未绑定具体 episode seed，只作演示。

## 4. Rollout / GIF 归档与选择规则

PAPER_ROLLOUT_GIF_REGISTRY.csv 有 1,305 行机器可读逐局/资产记录，登记 seed、checkpoint SHA、配置路径与 hash、评估 source SHA、报告 hash、trajectory/GIF 路径、碰撞/结果字段及缺失状态。当前索引记录 285 个 GIF 路径，其中 Z05 主独立评估 240 个、10/6 补充 native 9 个、Z07 3 个、strict repeated 15 个、relaxed repeated 15 个、未绑 seed 的 3×3 demo 3 个。未复制任何 GIF、未生成新 GIF。

主图候选 33 例：Z05 native/跨规模按既有 GIF 中每批每场景最小 seed 的成功局选取 25 例；另选 G3 mixed 碰撞失败案例 1 个作失败说明；Z07 固定报告 index-0 GIF 只选了索引中可用的两个场景；strict 与 relaxed repeated 各按 regimen 的 episode 0 选 3 个。所有候选均在 CSV 写有真实 seed 与选择规则。它们是展示性单局，绝不代表平均表现。3×3 demo GIF 保留索引，但不把无法绑定 episode seed 的文件当作论文定量证据。失败案例只有在可解释 collision/censoring 时展示。

## 5. 论文图表落点

| 论文表/图 | 可用事实源与输入 | 写作限制 |
|---|---|---|
| Table I：selected policy、stage、参数规模、权重身份 | PAPER_CHECKPOINT_REGISTRY.csv | Evidence 臂训练 source SHA 未登记，需在表注说明 |
| Table II：Z05 curriculum native 三任务 | 9/23 的 S1/S2/S3 report 与 episodes.jsonl | 保留每个 stage 的原生规模；不可把 CE 写成“搜索成功” |
| Table III：Z05 固定 Stage3 跨规模 | 9/23 4/8/16/20/24P 批次 report；10/6 18P/24P补充单列 | 24/6/5 与 24/3/3 分开呈现；只称 zero-shot/cross-scale evaluation |
| Fig. 1：Pure Coverage/Pure Capture/Mixed 轨迹示例 | Z05 Stage3 native 的三类 seed-mapped GIF | 标具体 seed/checkpoint；仅为典型单局 |
| Fig. 2 或附录图：跨规模轨迹 | G1/G2/G3 现存 GIF 及对应轨迹 JSON | 增加失败例时明确是 G3 的 collision/censored case |
| Table IV：repeated-arrival service | strict A/B/C summary 与 60 个 per-episode/event/trajectory；relaxed另表/附录 | strict 是主结果；relaxed 是不同停机合同 |
| Evidence paired comparison | 旧 450 行 report 可用于 provisional 描述 | corrected 时间、run source SHA 及独立逐局资产闭合前不放入最终主结论 |
| Global Oracle 对比 | 既有 Global Oracle report/checkpoint | 作为 privileged upper-bound 时标明额外信息，不能默认完全公平 |

## 6. 六份交付件

本目录 /home/yjq/rl/cocap-voradj/docs/paper/PAPER_ASSETS_20261015/：

1. PAPER_EXPERIMENT_ASSET_INDEX_20261015_ZH.md：本文件，人类阅读版总览。
2. PAPER_CHECKPOINT_REGISTRY.csv：16 个 checkpoint 的路径、SHA、配置与来源登记。
3. PAPER_ROLLOUT_GIF_REGISTRY.csv：1,305 行逐局/文件关联、seed 与案例选择记录。
4. PAPER_BENCHMARK_CONTRACT_DRAFT_ZH.md：由真实 Z05 配置出发的经典合同草案、指标定义、删失规则及信息公平性。
5. PAPER_VISUALIZATION_TOOLKIT_ZH.md：现存 evaluator、renderer、plotter、summary 脚本索引和参考命令。
6. PAPER_EXPERIMENT_READINESS_ZH.md：READY / MISSING_ARTIFACT / NEED_EVALUATION / NEED_DECISION 状态、Evidence 缺口、资源估计和补实验顺序。

源事实中的大型 checkpoint、trajectory、GIF 均保留服务器原路径；Git 资料目录只存 Markdown/CSV，不放大文件。本轮没有发现已有发布授权，没有 commit/push；上述文件是本地整理稿，待后续显式发布流程。
