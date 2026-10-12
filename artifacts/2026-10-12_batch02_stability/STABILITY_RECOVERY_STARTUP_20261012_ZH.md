# Batch02 stability recovery 启动交付

观测时间 2026-10-12T09:55:52.054950+08:00；PROVISIONAL。三线已启动且实际 PPO update/梯度/参数 finite，无启动 failure，科学稳定性结论仍待500k→1M完整配对。

代码SHA `b479a869027262ad60bf30b0604263f5b99b5341`；历史P1 treatment `5b53c3f`，唯一科学delta `target_kl .02→.01`；environment/reward/observation/network/architecture不变。

| 线 | 实际GPU | PID | decisions | continuation | last checkpoint | throughput | ETA |
|---|---:|---:|---:|---:|---:|---:|---:|
| R2-stability | 0 | 388183 | 504096 | 4096 | 500000 | 21.84/s | 6.31h |
| R3-stability | 0 | 388347 | 503584 | 3584 | 500000 | 21.71/s | 6.35h |
| V3-stability | 1 | 388572 | 503072 | 3072 | 500000 | 22.21/s | 6.22h |

全部从各自原500k完整actor/V/Adam/ValueNorm/env/APF/RNG状态恢复，KL覆盖前后状态hash一致；新进程16-step PPO恢复一致，原evaluator对应episode完全相同。每25k保留全部checkpoint并screen，固定到1M；公共500k评估原样保留。

GPU1旧R2短尝试最后保存500k，未保存2304decision已明确排除/归档；原五条baseline继续，源码/配置未改。

R3实际是TERL observation/backbone+reward telemetry，与R2关联，V3为CoCap backbone+原critic；不能作为三独立seed的证据。

已有baseline R2/R3的550k screen capture(argmax/sample)=80%/50%、collision=20%/70%；700k两模式capture=0、collision=100%。这只构成退化现象，不证明原因。新的A/B/C结论待完整配对和KL真实触发，不用best代替last；无改善也不单独证明reward/network故障。

NEXT WAKE-UP：2026-10-12T10:25:52.054950+08:00，或任一新线首525k checkpoint。输出每线paired_screen_curve.csv、optimization_curve.csv、paired_screen_curves.png、stability_report.json；不自动启动下一批。
