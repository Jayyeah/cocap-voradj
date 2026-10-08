# TERL-MAPPO 累计1m Stage1自动线

更新时间：2026-10-09T06:16:01.790687+08:00。
自动线状态：`RUNNING_TO_1M`；总目标1,000,000 joint environment decisions；已完成累计步数：552,816。
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

原100k的screen结果复用且记录来源，后续每25k regular各10局，100/250/500/750/1000k各20局。screen/selection seed沿用原配置；最终独立seed2046101800起，各50局，与原100k final隔离。
最终selection候选为screen预声明排序的前三个checkpoint，held-out各20局，再按normal、低collision、strict、ring3、较早step排序。保留latest、固定milestones、当前前三候选和待评估点，其余仅在完整screen/hash验证后删除本自动线文件。

best checkpoint/hash：`PENDING` / `PENDING`。
最新数值健康：`True`；training PID：`1045822`。
最新PPO指标：`{"steps": 552816, "agent_transitions": 1658448, "optimizer_steps": 12816, "reward_mean": 8.617153926983447, "reward_min": 3.420507972860179, "reward_max": 9.0, "actor_loss": -0.014634773135185242, "value_loss": 0.0004225789016345516, "entropy": 1.466744840145111, "clip_fraction": 0.03428819612599909, "approx_kl": 0.002619075024591022, "actor_grad_norm": 2.5380502343177795, "value_grad_norm": 0.0693384949117899, "explained_variance": -0.2008683681488037, "value_norm_mean": 378.8388366699219, "value_norm_std": 387.59423828125, "kl_early_stop": 0.0, "ppo_epochs_completed": 3.0, "minibatch_updates": 6.0, "actor_update_l2": 0.03522572973264277, "actor_update_relative_l2": 0.00037058747793862343, "approx_kl_max": 0.009781923145055771, "update_count": 2167.0, "post_update_ratio_mean": 0.9989454746246338, "post_update_ratio_min": 0.5223002433776855, "post_update_ratio_max": 1.7835330963134766, "post_update_clip_fraction": 0.1497395932674408, "post_update_kl": 0.011442459188401699, "episodes": 0, "training_capture": 0, "training_collision": 0}`。

每checkpoint完整reward/entropy/type/time统计及相同步数PPO指标见artifact summary/evaluation JSON。低collision不等于低boundary；训练episode ring仅是终止时刻，学习判断使用独立整局visitation。

100k报告只描述早期gate，不是1m终态结论。1m自动线不会因no-capture/partial/no-convincing-signal或CPU评估失败而停止训练。真实训练异常保留诊断，不静默重启或拼接科学版本。中央DAG仍只读。
