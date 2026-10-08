# TERL-MAPPO 累计1m Stage1自动线

更新时间：2026-10-09T03:56:57.675797+08:00。
自动线状态：`RUNNING_TO_1M`；总目标1,000,000 joint environment decisions；已完成累计步数：376,536。
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

原100k的screen结果复用且记录来源，后续每25k regular各10局，100/250/500/750/1000k各20局。screen/selection seed沿用原配置；最终独立seed2046101800起，各50局，与原100k final隔离。
最终selection候选为screen预声明排序的前三个checkpoint，held-out各20局，再按normal、低collision、strict、ring3、较早step排序。保留latest、固定milestones、当前前三候选和待评估点，其余仅在完整screen/hash验证后删除本自动线文件。

best checkpoint/hash：`PENDING` / `PENDING`。
最新数值健康：`True`；training PID：`1045822`。
最新PPO指标：`{"steps": 376536, "agent_transitions": 1129608, "optimizer_steps": 8796, "reward_mean": 8.325279893369183, "reward_min": 3.3320680474396562, "reward_max": 9.0, "actor_loss": -0.0169022916816175, "value_loss": 0.0012172978992263477, "entropy": 1.5366912086804707, "clip_fraction": 0.10286458705862363, "approx_kl": 0.006953792193801035, "actor_grad_norm": 1.6494509776433308, "value_grad_norm": 0.2995450223485629, "explained_variance": -0.21735107898712158, "value_norm_mean": 209.22608947753906, "value_norm_std": 351.406982421875, "kl_early_stop": 0.0, "ppo_epochs_completed": 3.0, "minibatch_updates": 6.0, "actor_update_l2": 0.04436187965559414, "actor_update_relative_l2": 0.00046695558723598045, "approx_kl_max": 0.01194455474615097, "update_count": 1476.0, "post_update_ratio_mean": 0.9996511340141296, "post_update_ratio_min": 0.8268160223960876, "post_update_ratio_max": 1.3256511688232422, "post_update_clip_fraction": 0.01302083395421505, "post_update_kl": 0.0019048459362238646, "episodes": 0, "training_capture": 0, "training_collision": 0}`。

每checkpoint完整reward/entropy/type/time统计及相同步数PPO指标见artifact summary/evaluation JSON。低collision不等于低boundary；训练episode ring仅是终止时刻，学习判断使用独立整局visitation。

100k报告只描述早期gate，不是1m终态结论。1m自动线不会因no-capture/partial/no-convincing-signal或CPU评估失败而停止训练。真实训练异常保留诊断，不静默重启或拼接科学版本。中央DAG仍只读。
