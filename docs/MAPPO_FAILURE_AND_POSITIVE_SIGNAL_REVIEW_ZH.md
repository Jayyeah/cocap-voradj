# MAPPO 正信号、退化机制与未解决问题（2026-10-15）

置信等级：**CONFIRMED**=运行/合同记录直接确认；**SUPPORTED**=方向一致但尚未因果识别；**HYPOTHESIS**=可检验但证据不足；**UNKNOWN**=缺少能区分解释的数据。单seed小screen均为描述性结果。

## A. Scratch MAPPO学到了什么

| 能力层 | 判读 | 证据与限制 |
|---|---|---|
| TERL Stage1 | **CONFIRMED：**原生TERL任务上学到capture行为。 | selected 775k正常capture45/50、collision5/50；last 1M退化。其observation/reward与3P1E0O合同均是TERL，不是CoCap。 |
| CoCap coverage | **CONFIRMED：**pure CE子任务出现过高成功率。 | M-COV与DAG A0复现有CE信号；没有pursuit、capture→recovery。 |
| CoCap capture | **SUPPORTED：**能进入capture/ring状态，但完成率低且碰撞高。 | M-CAP capture4/40、collision36/40；R2 selected capture completion62%/mode而last为32/40%，collision last70/74%。 |
| CoCap mixed/full task | **CONFIRMED：**已闭合的MAPPO mixed评估中full-task success为0。 | N2/R2/R3/N3/V3与continuation的episode_success均为0。N3/V3 final closure缺失，故其最终总体表现仍UNKNOWN；现有selection screen无success。 |

TERL success不能写成CoCap success；capture event不能写成capture后recovery/CE completion。IQN/BC在full task成功而scratch MAPPO失败，也不是同初始化算法比较。

## B. 后期退化和潜在机制

### 观察到的事实

- **CONFIRMED：TERL Stage1末期PPO统计与eval共同变差。**KL early-stop由早期约0%升至775k→1M约13.15%；entropy从2.164降至1.052；末期post-update KL约.0273、clip fraction约31.35%、EV约−.0058至−.043；同时capture由45/50降至约30%，collision上升。共变不能单独确定原因。
- **CONFIRMED：**CoCap R2/R3 continuation中target_kl .02→.01将KL early-stop从约1.4%提高至18%，说明配置被真实消费并改变early-stop频率。
- **CONFIRMED：**该continuation没有改变actor/critic LR（3e-5 / 1e-4），没有LR scheduler；因此观察到的退化不是本续训人为降低/提高LR造成。不过Adam的实际步长仍随梯度和二阶矩变化，固定LR不等于固定更新大小。
- **SUPPORTED：**低target KL没有使CoCap late-window综合指标改善。R2/R3 800–950k treatment的capture/collision均值33.6/85.0%，baseline为45.0/63.6%；1M treatment argmax/sample capture completion50/20%、collision40/90%、full success0。screen样本小，不说明低KL普遍有害，但没有正向能力证据。
- **CONFIRMED：**R2/R3 stability是配对shadow，不是两个独立复现。R3仅加read-only reward ledger，provider原reward数组仍进入PPO；两支screen逐项同值，账本重构残差约1e-14。
- **CONFIRMED：**reward/return可能和任务能力背离。R1改变reward后数值return上升但capture/safety变差；Forward-Final BC PPO 5k探索run的reward增加而CE/full completion掉至零。不能用return替代任务终点。
- **CONFIRMED（telemetry合同）：**V3 progress在998,808 decisions的latest_metrics ledger列出capture +1.540M、support −0.105M、safety −3.880M、geometry −0.185M、auxiliary shaping +0.0036M、terminal +0.037M，重建残差约1.4e-14。字段的时间聚合范围没有独立分层说明；无论按何种汇总窗，它都没有phase/成功失败分层，且V3未完成final，不能据此把碰撞归因给reward scale或建议重加权。
- **CONFIRMED（跨checkpoint行为变化）：**R2 selected 550k→last 1M capture completion从62/62%降至32/40%，collision从38/58%升至70/74%；R2/R3 stability late screens反复在argmax/sample间大幅波动。说明策略行为随checkpoint显著变化，但没有保存到能隔离单个动作机制的trajectory分析，策略漂移的具体因果路径仍UNKNOWN。

### 机制的证据等级

| 解释 | 分类 | 依据与缺口 |
|---|---|---|
| 后期PPO update drift | **SUPPORTED** | KL/clip/entropy/EV与TERL行为同时恶化；TERL P1降低target KL有正向endpoint。无跨seed因果拆分，CoCap续训也未证实净收益。 |
| target KL是退化唯一原因 | **UNKNOWN /不支持** | 确实改变stop频率，但不同task与窗口不一致。 |
| 熵耗尽是CoCap capture失败主因 | **UNKNOWN** | M-CAP decomposition将entropy/KL collapse列为not supported；CoCap没有匹配熵处理。 |
| M-CAP几何/ring3受限及碰撞 | **SUPPORTED** | 4/40捕获、ring3访问低、strict hold 0；多数失败在detection后进入或维持ring时碰撞。terminal trace及训练分层有限。 |
| Critic/reward scale放大M-CAP失败 | **HYPOTHESIS** | failure decomposition列为amplifier；缺逐episode V/GAE及成功/失败对照。 |
| global V缺特定协调信息 | **UNKNOWN** | 没有相同actor、相同数据、仅改变critic input的对照。 |

## C. BC Actor之后继续PPO：发生了什么

### Teacher动作一致性与完整任务成功

- **CONFIRMED：**canonical Final IQN teacher mixed 100 episodes：capture100%、collision1%、CE99%、safe completion99%。canonical BC actor SHA 7916a970…bf2cd：argmax capture99%、collision1%、CE98%、safe98%；sample capture99%、collision2%、CE98%、safe98%；coverage CE100%。
- Teacher greedy与BC argmax transition agreement约88.64%，capture前87.75%、后88.76%；Q gap中位数0、p90约2.08e-4。多数不一致接近tie，不应把约11%动作不一致直接解释成明显策略损失。
- **CONFIRMED（exploratory run）：**Forward-Final PPO从成功BC actor初始化，fresh critic/Adam/ValueNorm，5k steps后capture仍100%、collision0%，但mixed full safe capture+CE completion从step0的100%降到5k的0%；argmax CE 20/20→0/20、sample 20/20→3/20。该run标为EXPLORATORY_NON_GATE，common-safe n=0，不能写成formal causal gate。
- 更早MAPPO-9 Direct/warm-up属于不同pure-capture合同；旧matched因果结论已因seed、dropout、log-prob合同问题撤回，只保留描述性记录。

### Critic、advantage与action credit

- **SUPPORTED：**P2 fixed-policy calibration的10个held-out episodes中，early recovery V mean −3.21而MC mean −14.00，bias +10.79、RMSE14.99、EV .097；late recovery RMSE2.37。pure coverage early RMSE12.93，说明恢复阶段不是唯一难阶段；样本小。
- **不支持GAE系统性鼓励慢恢复：**P3 raw advantage和MC−V符号一致约91.3%，positive-bad约2.5%；capture事件样本n=20，覆盖有限。
- **双Value Head有明确失败结论：**共享trunk、100次matched updates下，recovery预测有移动但没有recovery time profile校准收益，判定停止head splitting。无依据增加更多heads。
- **SUPPORTED但非因果：**共享critic 16个minibatch方向审计显示，joint raw negative-gradient的一阶方向会提高phase-specific loss：pre-capture 0/16、post-capture 6/16、pure coverage约6–7/16（记录汇总值7/16）。这提示post-capture梯度冲突可能存在；样本少、梯度范数/一阶loss不是实际更新后的因果行为。
- **SUPPORTED：**其他BC critic audit中local LQ/NQ优于central CQ/V的realized-policy MC拟合；central RMSE差约13.6%/11.3%，EV低约.157/.127。该合同不同于Batch02训练，属于反证提示而不是Batch02直接因果对照。
- **SUPPORTED：**action-ranking质量弱。LQ top1约.1875、Spearman约.178、sign accuracy .594（64 anchors）。B2直接counterfactual branches alternative top1约.773，但有27 collision branches、成功状态分层不足且phase差异明显。之后natural action support改善相对排序，绝对ranking仍弱。
- **UNKNOWN：**弱Q-ranking是否导致5k PPO后CE坍塌。IQN Q是teacher自身的action排序，不等于当前MAPPO联合策略下的team advantage；许多teacher/BC差异又接近tie。

## 已排除、未支持与仍未知

- DAG A1的final evaluator加载不兼容造成0 rollout/update，是工程兼容失败；no-entropy算法结论incomplete。
- M-CAP detection及时，保存的collision tick发生在detection后，detection-limited不获支持；主要方向是geometry/ring3与collision。
- R3是telemetry，不改reward；PPO始终消费provider reward。
- MC、bootstrap和iterative Bellman audit未发现严重target drift；这不证明所有value target都正确。
- **UNKNOWN：**central V具体缺少什么协调字段；actor/critic语义差是否造成不可识别；mixed任务主要瓶颈是actor表示、探索/数据分布、安全冲突还是critic/advantage，或多项交互。

## 最窄的待检验问题

最值得留作未来问题的是：**现有action-free V是否把几何接近但phase/recovery clock不同的状态混合，导致capture transition与early recovery的return calibration/advantage排序偏差？**P2给出可检验信号，但P3大体正确、coverage early误差同样高、双head未解决恢复校准；目前不能把它写成根因。若重开，先做phase-stratified calibration和teacher-Q ranking离线诊断；两者不一致即停止。见 [Critic选项](MAPPO_CRITIC_RESEARCH_OPTIONS_ZH.md)。
