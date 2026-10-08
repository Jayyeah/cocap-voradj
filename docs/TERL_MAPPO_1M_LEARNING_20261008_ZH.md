# TERL-MAPPO 累计1m Stage1自动线

更新时间：2026-10-09T07:34:35.795780+08:00。
自动线状态：`RUNNING_TO_1M`；总目标1,000,000 joint environment decisions；已完成累计步数：652,304。
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
| 250000 | screen | argmax | 6/6/20 | 20/12/6 | 14/20 | 88.50/80.00/146.75 | 14 |
| 250000 | screen | sample | 2/2/20 | 18/4/2 | 17/20 | 83.00/83.00/85.00 | 18 |
| 275000 | screen | argmax | 0/0/10 | 10/1/0 | 4/10 | —/—/— | 10 |
| 275000 | screen | sample | 3/3/10 | 9/4/3 | 5/10 | 453.00/461.00/722.20 | 7 |
| 300000 | screen | argmax | 4/4/10 | 10/8/4 | 6/10 | 79.75/86.75/109.80 | 6 |
| 300000 | screen | sample | 3/3/10 | 10/6/3 | 7/10 | 89.50/102.00/120.40 | 7 |
| 325000 | screen | argmax | 7/7/10 | 10/9/7 | 3/10 | 180.86/159.50/309.10 | 3 |
| 325000 | screen | sample | 4/4/10 | 10/8/4 | 6/10 | 173.50/146.50/305.35 | 6 |
| 350000 | screen | argmax | 3/3/10 | 10/7/3 | 4/10 | 762.83/559.50/1149.50 | 7 |
| 350000 | screen | sample | 1/1/10 | 10/5/1 | 9/10 | 83.50/83.50/83.50 | 9 |
| 375000 | screen | argmax | 4/4/10 | 10/9/4 | 5/10 | 168.00/38.00/413.40 | 6 |
| 375000 | screen | sample | 2/2/10 | 10/4/2 | 8/10 | 238.75/238.75/372.15 | 8 |
| 400000 | screen | argmax | 7/7/10 | 10/10/7 | 0/10 | 670.71/558.00/1316.50 | 3 |
| 400000 | screen | sample | 6/6/10 | 10/10/6 | 3/10 | 405.08/220.50/853.25 | 4 |
| 425000 | screen | argmax | 8/8/10 | 10/10/8 | 2/10 | 39.12/37.50/58.45 | 2 |
| 425000 | screen | sample | 6/6/10 | 9/7/6 | 4/10 | 179.17/133.75/356.25 | 4 |
| 450000 | screen | argmax | 10/10/10 | 10/10/10 | 0/10 | 77.60/85.00/111.50 | 0 |
| 450000 | screen | sample | 9/9/10 | 9/9/9 | 1/10 | 266.22/208.00/541.70 | 1 |
| 475000 | screen | argmax | 9/9/10 | 10/9/9 | 1/10 | 275.22/204.00/661.50 | 1 |
| 475000 | screen | sample | 1/1/10 | 10/8/1 | 6/10 | 71.00/71.00/71.00 | 9 |
| 500000 | screen | argmax | 19/19/20 | 20/20/19 | 1/20 | 108.61/74.00/186.40 | 1 |
| 500000 | screen | sample | 13/13/20 | 20/20/13 | 4/20 | 396.54/392.50/786.00 | 7 |
| 525000 | screen | argmax | 9/9/10 | 10/9/9 | 1/10 | 46.22/42.00/72.40 | 1 |
| 525000 | screen | sample | 7/7/10 | 10/9/7 | 3/10 | 167.29/162.00/307.60 | 3 |
| 550000 | screen | argmax | 7/7/10 | 10/10/7 | 1/10 | 329.00/98.50/873.60 | 3 |
| 550000 | screen | sample | 6/6/10 | 10/9/6 | 3/10 | 498.08/459.25/972.00 | 4 |
| 575000 | screen | argmax | 3/3/10 | 9/7/3 | 5/10 | 43.67/22.00/75.60 | 7 |
| 575000 | screen | sample | 2/2/10 | 9/6/2 | 8/10 | 240.75/240.75/391.35 | 8 |
| 600000 | screen | argmax | 6/6/10 | 9/6/6 | 4/10 | 42.17/31.25/72.25 | 4 |
| 600000 | screen | sample | 5/5/10 | 9/7/5 | 5/10 | 471.30/449.50/853.10 | 5 |
| 625000 | screen | argmax | 8/8/10 | 9/8/8 | 2/10 | 43.88/45.25/66.65 | 2 |
| 625000 | screen | sample | 8/8/10 | 9/8/8 | 2/10 | 138.44/76.00/294.80 | 2 |
| 650000 | screen | argmax | 8/8/10 | 8/8/8 | 2/10 | 86.44/59.25/173.75 | 2 |
| 650000 | screen | sample | 3/3/10 | 9/9/3 | 3/10 | 677.17/711.50/714.30 | 7 |

原100k的screen结果复用且记录来源，后续每25k regular各10局，100/250/500/750/1000k各20局。screen/selection seed沿用原配置；最终独立seed2046101800起，各50局，与原100k final隔离。
最终selection候选为screen预声明排序的前三个checkpoint，held-out各20局，再按normal、低collision、strict、ring3、较早step排序。保留latest、固定milestones、当前前三候选和待评估点，其余仅在完整screen/hash验证后删除本自动线文件。

best checkpoint/hash：`PENDING` / `PENDING`。
最新数值健康：`True`；training PID：`1045822`。
最新PPO指标：`{"steps": 652304, "agent_transitions": 1956912, "optimizer_steps": 15106, "reward_mean": 7.909849913270894, "reward_min": 3.3445192069165293, "reward_max": 9.0, "actor_loss": -0.010286132991313934, "value_loss": 0.000811033183708787, "entropy": 1.330742073059082, "clip_fraction": 0.08645833833143116, "approx_kl": 0.00964598196248212, "actor_grad_norm": 2.6968070030212403, "value_grad_norm": 0.182954416051507, "explained_variance": -0.14409565925598145, "value_norm_mean": 428.0920715332031, "value_norm_std": 381.3612060546875, "kl_early_stop": 1.0, "ppo_epochs_completed": 3.0, "minibatch_updates": 5.0, "actor_update_l2": 0.028638558413647656, "actor_update_relative_l2": 0.00030118116858788907, "approx_kl_max": 0.02013435587286949, "update_count": 2557.0, "post_update_ratio_mean": 0.9992672801017761, "post_update_ratio_min": 0.3610897362232208, "post_update_ratio_max": 2.2306032180786133, "post_update_clip_fraction": 0.1393229216337204, "post_update_kl": 0.013131365180015564, "episodes": 0, "training_capture": 0, "training_collision": 0}`。

每checkpoint完整reward/entropy/type/time统计及相同步数PPO指标见artifact summary/evaluation JSON。低collision不等于低boundary；训练episode ring仅是终止时刻，学习判断使用独立整局visitation。

100k报告只描述早期gate，不是1m终态结论。1m自动线不会因no-capture/partial/no-convincing-signal或CPU评估失败而停止训练。真实训练异常保留诊断，不静默重启或拼接科学版本。中央DAG仍只读。
