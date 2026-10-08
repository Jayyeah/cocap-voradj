# TERL-MAPPO 累计1m Stage1自动线

更新时间：2026-10-08T23:42:10.206894+08:00。
自动线状态：`ARMED_WAITING_PARENT_100K`；总目标1,000,000 joint environment decisions；已完成累计步数：86,520。
科学分类：`PENDING`。100k gate结果不控制续训。

100k原run为精确续训parent；从100k checkpoint恢复actor/critic、Adam、ValueNorm、环境、全部RNG与计数。只改变累计budget：100k→1m；原native Stage1、256/8/4 backbone、seed9/109、PPO超参及25k保存切点全部保持。

| step | domain | mode | normal/capture/n | ring2/ring3/strict | collision | capture mean/median/p90 s | censored |
|---:|---|---|---|---|---|---|---|
| — | — | pending | — | — | — | — | — |

原100k的screen结果复用且记录来源，后续每25k regular各10局，100/250/500/750/1000k各20局。screen/selection seed沿用原配置；最终独立seed2046101800起，各50局，与原100k final隔离。
最终selection候选为screen预声明排序的前三个checkpoint，held-out各20局，再按normal、低collision、strict、ring3、较早step排序。保留latest、固定milestones、当前前三候选和待评估点，其余仅在完整screen/hash验证后删除本自动线文件。

best checkpoint/hash：`PENDING` / `PENDING`。
最新数值健康：`True`；training PID：`992932`。
最新PPO指标：`{"steps": 86520, "agent_transitions": 259560, "optimizer_steps": 2034, "reward_mean": -3.465035909414772, "reward_min": -5.972301890765709, "reward_max": -0.7023795231802097, "actor_loss": -0.02220171069105466, "value_loss": 0.0394454225897789, "entropy": 2.1471630334854126, "clip_fraction": 0.0, "approx_kl": 0.00012143903465281862, "actor_grad_norm": 0.7137047251065572, "value_grad_norm": 2.7545048793156943, "explained_variance": 0.799144983291626, "value_norm_mean": -235.0149688720703, "value_norm_std": 138.48390197753906, "kl_early_stop": 0.0, "ppo_epochs_completed": 3.0, "minibatch_updates": 6.0, "actor_update_l2": 0.0699350283892506, "actor_update_relative_l2": 0.0007371283565939221, "approx_kl_max": 0.0003148008545394987, "update_count": 339.0, "post_update_ratio_mean": 0.9996382594108582, "post_update_ratio_min": 0.9276962876319885, "post_update_ratio_max": 1.0959550142288208, "post_update_clip_fraction": 0.0, "post_update_kl": 0.0005668214871548116, "episodes": 0, "training_capture": 0, "training_collision": 0}`。

每checkpoint完整reward/entropy/type/time统计及相同步数PPO指标见artifact summary/evaluation JSON。低collision不等于低boundary；训练episode ring仅是终止时刻，学习判断使用独立整局visitation。

100k报告只描述早期gate，不是1m终态结论。1m自动线不会因no-capture/partial/no-convincing-signal或CPU评估失败而停止训练。真实训练异常保留诊断，不静默重启或拼接科学版本。中央DAG仍只读。
