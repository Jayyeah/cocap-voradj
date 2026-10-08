# TERL-MAPPO 累计1m Stage1自动线

更新时间：2026-10-09T01:43:36.021307+08:00。
自动线状态：`RUNNING_TO_1M`；总目标1,000,000 joint environment decisions；已完成累计步数：225,768。
科学分类：`PENDING`。100k gate结果不控制续训。

100k原run为精确续训parent；从100k checkpoint恢复actor/critic、Adam、ValueNorm、环境、全部RNG与计数。只改变累计budget：100k→1m；原native Stage1、256/8/4 backbone、seed9/109、PPO超参及25k保存切点全部保持。

| step | domain | mode | normal/capture/n | ring2/ring3/strict | collision | capture mean/median/p90 s | censored |
|---:|---|---|---|---|---|---|---|
| 0 | screen | argmax | 0/0/20 | 0/0/0 | 1/20 | —/—/— | 20 |
| 0 | screen | sample | 0/0/20 | 0/0/0 | 2/20 | —/—/— | 20 |
| 25000 | screen | argmax | 0/0/20 | 0/0/0 | 1/20 | —/—/— | 20 |
| 25000 | screen | sample | 0/0/20 | 0/0/0 | 3/20 | —/—/— | 20 |
| 50000 | screen | argmax | 0/0/20 | 0/0/0 | 0/20 | —/—/— | 20 |
| 50000 | screen | sample | 0/0/20 | 0/0/0 | 9/20 | —/—/— | 20 |
| 75000 | screen | argmax | 0/0/10 | 0/0/0 | 0/10 | —/—/— | 10 |
| 75000 | screen | sample | 0/0/10 | 0/0/0 | 1/10 | —/—/— | 10 |
| 100000 | screen | argmax | 1/1/20 | 9/1/1 | 19/20 | 29.50/29.50/29.50 | 19 |
| 100000 | screen | sample | 0/0/20 | 5/0/0 | 20/20 | —/—/— | 20 |
| 125000 | screen | argmax | 0/0/10 | 1/0/0 | 1/10 | —/—/— | 10 |
| 125000 | screen | sample | 0/0/10 | 3/0/0 | 10/10 | —/—/— | 10 |
| 150000 | screen | argmax | 0/0/10 | 8/0/0 | 10/10 | —/—/— | 10 |
| 150000 | screen | sample | 0/0/10 | 7/2/0 | 10/10 | —/—/— | 10 |
| 175000 | screen | argmax | 0/0/10 | 10/1/0 | 10/10 | —/—/— | 10 |
| 175000 | screen | sample | 0/0/10 | 6/1/0 | 10/10 | —/—/— | 10 |
| 200000 | screen | argmax | 0/0/10 | 6/2/0 | 10/10 | —/—/— | 10 |
| 200000 | screen | sample | 1/1/10 | 7/3/1 | 9/10 | 261.50/261.50/261.50 | 9 |
| 225000 | screen | argmax | 1/1/10 | 8/3/1 | 9/10 | 75.00/75.00/75.00 | 9 |
| 225000 | screen | sample | 0/0/10 | 4/0/0 | 10/10 | —/—/— | 10 |

原100k的screen结果复用且记录来源，后续每25k regular各10局，100/250/500/750/1000k各20局。screen/selection seed沿用原配置；最终独立seed2046101800起，各50局，与原100k final隔离。
最终selection候选为screen预声明排序的前三个checkpoint，held-out各20局，再按normal、低collision、strict、ring3、较早step排序。保留latest、固定milestones、当前前三候选和待评估点，其余仅在完整screen/hash验证后删除本自动线文件。

best checkpoint/hash：`PENDING` / `PENDING`。
最新数值健康：`True`；training PID：`1045822`。
最新PPO指标：`{"steps": 225768, "agent_transitions": 677304, "optimizer_steps": 5309, "reward_mean": 5.192773503831542, "reward_min": -1.6235022864232445, "reward_max": 193.63764820950252, "actor_loss": -0.019609064950297277, "value_loss": 0.03311613450447718, "entropy": 1.7580334345499675, "clip_fraction": 0.03515625248352686, "approx_kl": 0.0030723184948631874, "actor_grad_norm": 1.095782478650411, "value_grad_norm": 1.8032986323038738, "explained_variance": 0.5164151191711426, "value_norm_mean": -28.32597541809082, "value_norm_std": 213.00506591796875, "kl_early_stop": 0.0, "ppo_epochs_completed": 3.0, "minibatch_updates": 6.0, "actor_update_l2": 0.058018337478315946, "actor_update_relative_l2": 0.0006110829819948382, "approx_kl_max": 0.0055435895919799805, "update_count": 885.0, "post_update_ratio_mean": 0.9985802173614502, "post_update_ratio_min": 0.6991433501243591, "post_update_ratio_max": 1.2982337474822998, "post_update_clip_fraction": 0.0234375, "post_update_kl": 0.0032357971649616957, "episodes": 1, "training_capture": 1, "training_collision": 0}`。

每checkpoint完整reward/entropy/type/time统计及相同步数PPO指标见artifact summary/evaluation JSON。低collision不等于低boundary；训练episode ring仅是终止时刻，学习判断使用独立整局visitation。

100k报告只描述早期gate，不是1m终态结论。1m自动线不会因no-capture/partial/no-convincing-signal或CPU评估失败而停止训练。真实训练异常保留诊断，不静默重启或拼接科学版本。中央DAG仍只读。
