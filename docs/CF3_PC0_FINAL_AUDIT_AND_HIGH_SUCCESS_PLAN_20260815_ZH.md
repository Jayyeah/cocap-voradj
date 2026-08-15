# CF3 / PC0 最终审计与高成功率围捕路线（2026-08-15）

> 本文冻结截至 2026-08-15 的 CF3 Local-Support 400k 与 PC0 post-capture 100k 证据，并据此替换“继续扩训练步数”的旧默认路线。原始历史不回写、不删除；总台账只追加本结论或链接本文。

## 1. 最终状态与恢复合同

### CF3 — corrected Local-Support

- run：`artifacts/2026-08-13_capture_first_controls/cf3_local_support_full_p0fixed_recovery/legacy_voradj_cf3_local_support_p0fixed_recovery25k_to200k_20260813/`
- final：`checkpoints/step_000400000/`；rolling：`resume_latest/`。
- `transition/replay/update = 400000/250000/98751`，`all_finite=true`；训练自然结束，无残留 trainer。
- final bundle 为 `full_resume`，trainer/replay/runtime、optimizer、alpha、target critics 与 RNG 均完整。milestone、rolling 和顶层 final 文件通过 inode/manifest/step 交叉核对。
- 合同：corrected K10 effective role、`min_active=2`、local Legacy-VorAdj、capture/support/coverage 三角色、support=`full capture + full coverage`、self+mean Actor、all-agent MASAC、no clip、MSE、UTD=.25、`(a,w)`。

### PC0 — CF3 Actor warm-start + post-capture 300

- run：`artifacts/2026-08-13_capture_first_controls/pc0_cf3_actor_postcapture300/legacy_voradj_pc0_cf3_actor_warmstart_postcapture300_4p1e1obs_100k_aw_20260813/`
- final：`checkpoints/step_000100000/`；rolling：`resume_latest/`。
- `transition/replay/update = 100000/100000/23751`，`all_finite=true`；训练自然结束，无残留 trainer。
- 初始化严格为 CF3@175k **Actor-only**；critic/target/replay/optimizer/alpha/RNG/runtime 全部 fresh。原因是 capture-only 中 capture transition 为 terminal，而 PC0 中它变为 non-terminal，旧 value/replay 的 Bellman 语义不兼容。
- `control/cf3_stable_gate_pc0_status.json=FAILED_CLOSED` 是 1k warmup `update=0` 时缺少 `mean_finite` 的陈旧监管器假失败；真实 5k--100k 全程 finite，不影响 final bundle。

## 2. 训练结果

### 2.1 CF3 400k：几何与碰撞改善，但 capture 没有被训练步数放大

| 区间 | normal / stationary | 2+ 独立窗口 | 3+ 独立窗口 | 2+ fraction | 3+ fraction | collision / 1k |
|---|---:|---:|---:|---:|---:|---:|
| 0--100k | 2 / 0 | 49 | 4 | 0.500% | 0.011% | 6.96 |
| 100--200k | 2 / 0 | 64 | 9 | 0.574% | 0.031% | 5.72 |
| 200--300k | 2 / 1 | 65 | 6 | 0.674% | 0.024% | 5.03 |
| 300--400k | 2 / 0 | 80 | 5 | 0.940% | 0.013% | 4.51 |

- 累计 `8 normal + 1 stationary`，normal 分布在约 42/97/173/186/281/298/321/331k；最后 69k 没有新 capture。
- 累计 258 个2+与24个3+窗口，最长2+/3+ hold=`31/11`。2+越来越常见、collision持续下降，但3+在100--200k后回落，说明第三机补位/保持没有随训练继续改善。
- 全程 `terminated=2231`、`truncated=6`；其中 collision=2222、normal=8、stationary=1。以所有结束回合计，normal约`0.358%`、collision约`99.33%`。
- support reward合同确实工作，角色长期约70% capture、29% support、1% coverage，support两类reward component均非零；但行为后段退化：每100k support→friend distance delta 从`-0.051`变为`+0.001/+0.023/+0.053 m/step`，support→enemy progress 从`+0.059`降至`+0.047/+0.030/+0.002 m/step`。
- final SAC仍 finite：critic/actor loss约`139.5/259.9`、alpha=`0.214`、Q1/Q2/target约`-258.30/-258.31/-259.22`、TD abs=`5.31`、critic/actor raw grad约`1861/8.71`。这排除了简单的 NaN/OOM 解释，但不等于 critic ranking 或策略目标正确。
- final stochastic action std 仍约`a=0.590,w=0.468`，而正式评估使用 `tanh(mean)`；训练稀有成功与确定性策略完全失败之间存在明确 mode gap。

结论：**400k consolidation control 已经否定“uniform replay 下再多给一些 step 就会稳定放大 capture”的主要解释。CF3 不再继续扩到 500k。**

### 2.2 PC0 100k：产生了真实闭环样本，但没有学成闭环

| 区间 | normal / stationary | 2+窗口 | 3+窗口 | 2+ fraction | collision / 1k |
|---|---:|---:|---:|---:|---:|
| 0--25k | 0 / 0 | 15 | 2 | 0.460% | 10.08 |
| 25--50k | 3 / 0 | 23 | 2 | 0.952% | 10.88 |
| 50--75k | 2 / 0 | 24 | 4 | 1.074% | 7.88 |
| 75--100k | 0 / 0 | 21 | 4 | 1.184% | 7.76 |

- 累计5个独立normal capture，集中在27/30/32/72/75k；最后25k无新增capture。累计83个2+、12个3+窗口，最长hold=`23/8`。
- capture后实际只留下157条 joint post-capture transition（628 active-agent slots），占100k replay的`0.157%`；五次成功后约只存活`7/5/83/6/56`步，均远小于目标300步并被collision提前终止。
- uniform batch128每批期望只有`0.201`条post-capture transition，零post-capture样本概率约`81.8%`。final batch实际为0。因此单一uniform buffer下，coverage loss几乎看不到闭环数据。
- 50k paired pure-coverage 诊断出现CE几何改善：Actor-init→PC0-50k的final CE energy `0.0496→0.00451`、CE-center RMS/max `0.302/0.335→0.134/0.134`；但collision仍4/4、平均存活`1284→588`、CV015/020成功`1/4→0/4`。capture/mix同为0/4且全collision，不能把CE局部改善写成coverage或闭环成功。

结论：**PC0证明了“capture→coverage transition能真实写入”，但没有证明capture稳定提高、300步coverage学成或mixed闭环成功。原样扩PC0或单纯加post-capture reward没有数据基础。**

## 3. 正式 rollout 状态

### 已完整完成

- CF3@400k runner内置 deterministic formal capture 20：`0/20 capture`、`20/20 collision`，mean length=`217.55`、mean min-min distance=`6.15 m`、distance progress=`+25.47 m`。
- CF3@200k独立 capture-only 20+5GIF：`0/20 capture`、`20/20 collision`，5个GIF完整。
- PC0@100k runner内置 deterministic formal capture 20：`0/20 capture`、`20/20 collision`，mean length=`102.75`、mean min-min distance=`13.07 m`、distance progress=`+18.62 m`。

### 2026-08-15 补齐的最终三场景评估

- 原自动链并没有生成 CF3@400k 与 PC0@100k 的 `capture/pure_ce/mixed` 各20回合+各5GIF，不能写成“原rollout全部完成”。
- 已从上述精确final checkpoint/effective-config完成 paired seed `2026081201--2026081220`、deterministic、old-mix三场景stats-only评估：每线60回合、4个CPU worker；CF3/PC0分别耗时约560/354秒，无报错。
- 输出：`artifacts/2026-08-15_final_triscene_20rollout5gif/{cf3_step_000400000,pc0_step_000100000}*`。

| line / scene | capture | strict CE | CV≤.15 / .20 | collision | mean length | 关键几何/CE |
|---|---:|---:|---:|---:|---:|---|
| CF3@400 capture | 0/20 | — | — | 20/20 | 636.35 | min-min 5.34 m；progress +14.17 m |
| CF3@400 pure_ce | — | 0/20 | 3/20 / 5/20 | 7/20 | 1500.00 | CE progress +0.0724；RMS 0.2150 |
| CF3@400 mixed | 0/20 | 0/20 | 0/20 / 0/20 | 20/20 | 636.35 | 因0 capture，从未进入coverage phase |
| PC0@100 capture | 0/20 | — | — | 20/20 | 436.95 | min-min 4.39 m；progress +18.90 m |
| PC0@100 pure_ce | — | 0/20 | 1/20 / 2/20 | 19/20 | 858.30 | CE progress +0.0792；RMS 0.2733 |
| PC0@100 mixed | 0/20 | 0/20 | 0/20 / 0/20 | 20/20 | 436.95 | 因0 capture，从未进入coverage phase |

- paired三场景进一步确认：两线的确定性capture与mixed均完全失败；PC0 pure-coverage虽有CE energy下降，但安全显著弱于CF3（collision 95% vs 35%），strict仍均为0。因此PC0没有同时提升capture和coverage。
- visual版本也使用相同20 seeds，但每场景固定渲染前5个seed，并非事后挑选代表性episode。PC0与CF3均已生成15/15 GIF；PC0 visual 60回合完整落盘，CF3 visual wrapper仍在CPU低优先级完成同批剩余非渲染回合，但独立stats-only 60回合与全部GIF已经齐全，主进程无错误，最终结论不再依赖wrapper退出。

## 4. 根因判断：当前真正缺的不是更多 SAC step

### 4.1 确定性均值没有固化随机探索中的成功 mode

当前共享Actor是无记忆、factorized、单峰对角Gaussian；CF3使用`self_token + mean_context`，输出一个mean与log-std，部署取`tanh(mean)`。训练始终保留较大std并偶发成功，所有正式 deterministic rollout 却为0。这更像“噪声偶尔打破对称/补对槽位”，而不是均值策略学会K3。

P1仅增加max pooling在matched100k反而更弱，说明问题不是少一个池化统计量。历史IQN 125k在相关local `(a,w)`合同曾达到15/20 capture，但它有target attention、summary attention、pursuing embedding/专用head；必须先在**精确corrected CF3环境**复评该checkpoint，再判断能否作为teacher，不能直接跨合同宣称优越。

### 4.2 四台共享Actor争抢同一接敌点，缺少持久slot与去冲突结构

CF3@400k最终前5个paired capture GIF共14次pursuer失活：11次agent-agent（78.6%）、2次boundary、1次obstacle；5/5均无capture且触发collision。反复出现P0/P1/P2贴撞P3，属于同目标堆叠，不是障碍主导。

现有reward虽有approach/front/mean-shift与flat emergency项，但没有显式指定三台应占据的不同角度槽位，也没有连续、可执行的双机避碰约束。共享、无持久身份/slot的Actor在确定性模式下输出相似动作是结构性风险。

### 4.3 成功样本与团队credit都太稀

CF3 final replay只保留150--400k，期间最多含6个normal+1个stationary terminal joint transition。uniform batch128命中任一capture terminal的期望频率约每279次optimizer update一次，而且成功前64--128步没有episode-level success标签。

当前all-agent update给每台agent自己的`dQ_i/da_i`，terminal reward主要给ring participants；K3却是“第三台补位使整个团队成功”的强外部性。即使显存修复后的梯度数学正确，per-agent value与一步uniform replay也不擅长把稀有团队事件向前归因。既有200k Q-ranking中严格`Qseek>Qpolicy>Qrandom`仅7.75%，`Qpolicy>Qrandom`仅53.5%，也说明critic动作排序尚不可靠。

### 4.4 collision语义与时间离散必须先审计

当前实体碰撞在完整0.5s decision结束后刷新，而boundary每substep检查；`_refresh_collisions()`又边扫描边deactivate。友机重叠时，先被扫描者deactivate后可能改变后续clearance，存在只杀索引较早一方的顺序偏置。新大线前必须用同时碰撞集合与swept collision测试核验；若是模拟器bug，应修成新默认合同并保留旧结果标签。

### 4.5 post-capture不应继续挤在单Actor、单buffer里

PC0里post phase仅0.157%，且成功后约31步就撞毁。即使coverage reward正确，单Actor/单critic/单uniform replay会让capture与coverage彼此干扰，并让稀有phase完全淹没。历史P1-200k Pure-CE已达到strict/CV015/CV020=`0.45/0.80/0.95`且collision=0，说明coverage模块有可复用anchor；更合理的是两option/FSM组合，而不是从0.157%样本重新把一切学一遍。

## 5. 下一阶段 TODO：以 rollout 高成功率为目标

### P0 — 不开长训，先做四个因果探针

1. **paired stochastic-temperature eval**：CF3@400k与PC0@100k各用相同100 seeds比较 deterministic 与temperature/noise scale `0.1/0.25/0.5/1.0`；同步记录成功轨迹上的mean/std/alpha。低温随机若显著优于mean，即直接坐实mode-collapse/均值问题。
2. **collision forensic与模拟器修复Gate**：按agent-agent/obstacle/boundary/evader分型，保存失败前后±50步的TTC、clearance、角间隙、role与2+→3+转换；补“同时重叠双方一致处理”和swept collision测试。任何顺序偏置先修再训。
3. **Safety-only paired100 probe**：不重训，给CF3动作加仅约束友机/障碍/边界的ORCA/CBF/QP projection，且不把8m ring目标挡掉。若collision显著下降并带来capture，下一线训练时让Actor穿过同一projection或蒸馏它；不能只在eval临时加shield后声称策略学会。
4. **teacher feasibility Gate**：先在精确corrected formal任务上评历史IQN125k；若不兼容或低于20%，实现解析teacher：预测evader轨迹，生成3个约120° ring slots+1个support slot，Hungarian assignment带hysteresis，unicycle MPC/ORCA输出原`(a,w)`。teacher必须先在formal deterministic100达到至少80% normal capture、collision不高于10%，否则先修控制/环境，不把失败teacher蒸馏给Actor。

### P1 — 高成功率capture主线

按可归因顺序推进，不把所有变化一次堆上：

1. **显式slot/option表征**：保留local sensing与`(a,w)`，加入episode-persistent capture slot/role；优先蒸馏IQN或解析teacher。Actor恢复target/summary attention或一跳图消息；动作头改为少量几何option/mixture中心加连续residual，而不是只用一个Gaussian mean。
2. **成功轨迹固化**：增加episode/trajectory buffer，保留capture与3+ near-success前64--128步；训练batch建议先试75% uniform +25% success/near-success reservoir，critic使用importance weight，并加入5--20 step return。Actor对正advantage成功动作增加AWR/BC，使随机成功动作移动到deterministic mode。
3. **reverse curriculum但formal终评不放宽**：从3+ hold附近reset→2+ near-ring→完整map-random逐级退火；始终保持moving APF、obstacle、local sensing、K3和真实collision。课程只解决探索，最终只以原formal random-spawn评估。
4. **团队credit ablation**：若上述仍不能固化，优先试scalar team return/team critic或centralized-value MAPPO，再做counterfactual/difference credit；理论上它比继续扫UTD/LR更贴近K3外部性。
5. **local可部署通信**：capture agent只广播自身enemy estimate/velocity/confidence与slot状态；support用一跳message+GRU持久记忆，不长期保留global oracle。central state显式加入effective role与K10 remaining counter，消除POMDP/critic状态缺项。

P1阶段Gate：

- 快速Gate：paired deterministic20 normal capture≥50%、collision≤25%；
- 正式Gate：3 seeds ×100 deterministic formal rollouts，normal capture≥70%（目标80%），95%置信区间下界≥60%，collision≤10--20%，stationary占成功<20%；
- 未通过不得恢复post-capture，也不以训练中的随机capture替代。

### P2 — capture稳定后再恢复coverage闭环

1. capture option冻结或小LR；coverage option从历史最佳Pure-CE Actor warm-start；显式phase signal/FSM在capture后硬切换。
2. 因terminal语义变化，默认fresh critic/value/replay；通过near-capture合法reset使25--50%采样来自真实post-capture phase，按phase分buffer/value head，禁止再让0.157%数据被uniform淹没。
3. post phase使用同一训练一致的CBF/ORCA safety，先保证绝大多数capture后能存活完整300步。
4. mixed正式Gate建议：normal capture≥70%、capture-conditioned coverage success≥70%、joint episode success≥50%、collision≤20%，并单列pure coverage与capture-only性能防止互相掩盖。

## 6. 当前明确不做

- CF3继续到500k或原样重跑PC0；
- 再做global broadcast、mean+max、单纯terminal reward加权；
- 在结构探针前盲扫LR/tau/batch/Huber/UTD/replay容量、MATD3或`vx,vy`；
- 仅凭训练随机capture或CE误差改善宣称formal围捕/coverage成功。

当前优先级已经从“把现有MASAC再训久一点”切换为：

```text
collision/温度/teacher feasibility probes
→ 显式slot + safety + teacher distillation
→ success episodic replay / n-step / team credit
→ deterministic formal capture Gate
→ capture/coverage双option闭环
```
