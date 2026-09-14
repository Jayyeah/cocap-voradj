# Forward Final Overnight bounded diagnostics — 2026-09-14

两条均为 **EXPLORATORY_NON_GATE**。用户显式授权本次非正式 bounded diagnostic，覆盖旧文档/AGENTS 中对本次短预算训练的禁止；不提升科学 Gate，不改变 canonical recipe。恢复时本地/远端均为 `8f6f4f3e3206f0d3a5da6898197918f2fc43a090`。已完整读取指定五份文档/JSON及 B 独立报告；B 源码/checkpoint SHA、文档 raw 引用已核验，独立提交并 push 为 `06f1128`。训练入口提交为 `08b2506`。

P1 transition/return：PASS_AFTER_FOUR_CONFIRMED_BUG_FIXES；P1 central-V state aliasing：UNRESOLVED；P2：FAIL/HOLD；P3：INCONCLUSIVE/HOLD；formal PPO、formal Scratch：HOLD。两条锁定 `terminal-priority-truncation-bootstrap-weighted-ce-v2`。今晚任何正结果只提出下一 Gate；任何负结果不能自动证明算法不可行。

## 已启动的两条任务

| GPU | PID / tmux | run | 硬预算 |
|---|---|---|---|
| 0 | 622682 / `cocap_overnight_scratch_20260914` | `artifacts/2026-09-14_overnight/scratch_seed1` | canonical seed 2026091401，100000 env steps |
| 1 | 622686 / `cocap_overnight_bc_ppo_20260914` | `artifacts/2026-09-14_overnight/bc_ppo_seed1` | seed 2026097101，5000 后 operational screen，最多10000 |

启动前两张 A6000 无 compute 进程；未停止任何非本项目进程。两进程分别设置物理 `CUDA_VISIBLE_DEVICES=0/1`，均使用本地 `cuda:0`。各自 output、checkpoint、RNG、训练stream与 capture pool 独立。启动时两条均先保存 `step_000000.pt`，随后固定种子基准评估，评估完成后自动进入训练；**step0 evaluation 不等于已经执行 Actor update**。最新阶段、PID、吞吐、ETA、checkpoint/eval在 [MASTER_SUMMARY.json](../artifacts/2026-09-14_overnight/MASTER_SUMMARY.json)。

## 实现及不变合同

入口：[Scratch](../tools/train_forward_final_scratch_mappo_20260914.py)、[BC→PPO](../tools/train_forward_final_ppo_diagnostic_20260914.py)；[共同预算/记录代码](../tools/forward_final_overnight_20260914.py)。原 production launcher 的 25k assert 完全保留，没有伪装 non-smoke 正式实验。

复用当前 `FinalMissionStream` / `collect_transition`、`MAPPOTrainer.update`、scratch construction 与 RNG/state restore。Actor LR3e-5、critic LR1e-4、rollout256、epochs3/minibatches2、γ.99/λ.95、clip.2、entropy.01、targetKL.02、原 Adam/ValueNorm/geometry V；无 warm-up。不是后来的3e-6 probe。没有 teacher-KL、BC loss、context critic、two-head、counterfactual-Q、reward改动或 sweep。

Scratch仅读取 canonical YAML / preflight 合同，直接 random Actor、fresh V/Adam/ValueNorm；guard 禁止 IQN 构造/load/forward、legacy trained tensor transfer、当前 run 之外的 artifact 读取。smoke 与正式 diagnostic 进程均启用此 guard，`teacher_dependency=0`。eval同样不执行teacher-Q。网络形状来自 canonical 标量，不加载训练tensor。

checkpoint保存整个 trainer（Actor/V/两个Adam/ValueNorm/update_count）、stream/env/在线capture pool、partial rollout、Python/NumPy/Torch/CUDA RNG、日志状态、source SHA、config、seed与版本。只恢复本 run 的新 schema，拒绝旧 PPO/checkpoint和源码漂移。不覆盖已有 checkpoint；5k partial136、25k partial168，100k partial160均保留，不为了cadence提前更新。截止预算末尾未满256的经验保持在checkpoint，不额外采集或缩短rollout。无 `--steps`、`--seed`、`--formal` 可绕过预声明预算。

## Matched evaluation 与记录口径

每checkpoint包括step0：argmax/sample × mixed/coverage ×20完整native episodes（80局）。固定 eval seed base2026092401，coverage偏移100000；训练seed分离。独立场景reset、不使用训练pool；每条对照断言 seed / initial fingerprint相同。eval前后保存/还原训练RNG，Actor hash不变，CE speed=.0005，完整native horizon/termination。训练occupancy和eval occupancy分表，不混淆初态分布。eval成本另记，不计入100k训练。

immutable `eval_step_000000.json` 与后续逐episode records包括 safe completion、capture / normal / stationary、CE/post CE、collision/boundary、mission/recovery时间、discounted/undiscounted mean-agent return、2+/3+ visitation与最长3+hold、support、pre/post/pure、CE RMS/CV、episode length/entropy、reset source。失败不写成短mission time。共同safe配对时间按相同episode子集计算。训练另记真实capture snapshot数量、reset分布、phase/support/ring/collision统计；每update记录Value EV/loss、entropy、KL/clip、Actor/V grad、raw GAE/return extrema与zero-update error。

BC 5k operational screen同时要求四组scene/mode：所有数值finite、无semantic/log-prob/resume失败；safe下降≤5pp、collision上升≤5pp、共同safe mission均值恶化≤5%，且每组共同safe≥10对。return的“明显一致恶化”前瞻操作化为：mean delta低于 `-0.05*max(abs(baseline mean),1)`，至少60% pairs为负且median delta为负。post-recovery collapse前瞻操作化为post CE下降>5pp或共同safe recovery均值恶化>10%。全部满足才续到10k；缺样为INCONCLUSIVE也不续。5k失败写 `STOP_BC_PPO_AT_5K`，不改LR/entropy/clip。10k无条件停止；不自动25k。该screen不宣称统计显著或formal PASS。

Scratch在0/25k/50k/75k/100k完整保存/评估；25k/50k zero capture不早停，只有工程/数值/合同错误硬停。100k按sample分布分类，argmax另报：连续两个checkpoint相对step0有2+≥15pp、3+≥10pp且hold不退化的前驱改善，或出现capture/post；同时coverage/support不得崩坏，才为 `POSITIVE_LEARNABILITY_SIGNAL`。coverage/support检查为CE成功不掉超过1/20、collision不升超过1/20、CE RMS/CV中位数不恶化超过15%、support active-row占比不低于step0的75%。局部、非连续或缺post改善为 `WEAK_OR_EARLY_SIGNAL`；无持续前驱或局部改善为 `NEGATIVE_DIAGNOSTIC_NOT_ALGORITHM_FAILURE`。粗阈值是本次operational判读，不是formal preflight未来Gate，不声称总体显著。禁止今晚200k或3seeds。

## 验证与 morning handoff

四项新入口 CPU/CUDA smoke全部 PASS：每条40个unique training steps、4步恢复重放，resume后transition/RNG/optimizer及下一update逐位一致；zero-update log-prob、finite与eval RNG隔离通过。Scratch teacher dependency=0。四份完整报告在 `artifacts/2026-09-14_overnight/smoke_*_v1/report.json`。回归 **38 passed**，包括新预算边界/return/post停止规则、日志对native状态/RNG无扰动、已有terminal/truncation/weighted CE/inactive Actor、scratch与MAPPO/PPO合同。仅既有protobuf弃用警告，见 [tests](../artifacts/2026-09-14_overnight/regression_tests.txt)。smoke只证明工程执行，不代表value校准或算法可学。

[低频交接工具](../tools/handoff_forward_final_overnight_20260914.py)每300秒读取两条状态，绝不训练/调参/抢卡；两条结束后生成完整JSON timeline、ledger、SESSION_HANDOFF和stage_gate，压缩learning/episode日志并定向commit/push。所有大 `.pt` 继续本地保存。若进程异常退出，只记录工程停止，不自动恢复/扩预算。若Git已有他人staged变更或分支改变，保留结果并拒绝误提交；不会force push。

明早读取 `MASTER_SUMMARY.json`：BC step0→5k→若有10k的所有 matched delta及SURVIVED/REGRESSED/INCONCLUSIVE；Scratch 0/25/50/75/100k趋势及POSITIVE/WEAK/NEGATIVE-DIAGNOSTIC。尚未到达的节点明确 `NOT_REACHED`，不得编造结果。formal HOLD始终保留。
