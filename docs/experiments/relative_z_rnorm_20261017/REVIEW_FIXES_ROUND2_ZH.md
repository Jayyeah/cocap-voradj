# 独立审核首轮阻断修复

首轮REQUEST_CHANGES完整保存在INDEPENDENT_SOL_XHIGH_REVIEW_ROUND1_ZH.md及原始11-fail/13-pass诊断；没有删除或放宽失败断言。

| 阻断 | 修复 | 内部证据 |
|---|---|---|
| B1 Supervisor状态覆盖 | 统一raise_if_paused覆盖启动、adoption、启动/重试、选模等待、BalancedFloor返回后、每次handoff/worker重建、尾队列与闭合。scientific_failure转持久pause；仅等待被识别PID/start的训练器完整退出，无杀保存超时；正常回收worker | 原5个失败断言通过；tests/test_relative_z_supervisor_20261017.py |
| B2 审核依赖遗漏 | source hash涵盖整个src/tools/tests所有py和configs的yaml/yml，包含所有本地导入及extends父配置；正式train入口也核验review+smoke+hold | 原4个依赖变更断言通过 |
| B3 科学失败分类 | 统一识别runtime assertion、AssertionError、数值/信息映射/配置/资源合同失败；持久scientific_failure+pause，worker直接failed不普通重试；trainer也停止自动恢复 | 原2个worker失败断言通过 |
| B4 smoke实际控制路径 | 官方smoke源码增加真实supervise进程、实际CPU workers、真实CUDA短训子进程：先owned-child暂停，再从exact-resume启动并由真实Supervisor接管后再次暂停；断言无重复训练/worker重启/Stage2 handoff，SQLite pending持久。selection/handoff/tail/adoption科学路径另执行实际supervise函数的显式替身隔离回归；记录低Z .5/.25/.125真实环境transition及alpha/CE权重/自然衰减 | 源码待独立APPROVE后执行，尚未宣称smoke通过 |

119项相关内部回归全部通过，见review_fix_full_tests.txt/xml。新增科学健康记录含实际batch reward统计、各task最近capture/collision/Coverage与normal/stationary事件计数；不额外调用forward，不消耗额外RNG、不改优化或奖励。

非阻断建议同时修正：尚不足两个不重叠50k吞吐窗口时provisional=true。正式训练配置仍2M/700k/700k，只有两项授权科学变量；诊断子进程1000步上限、小Replay/采样门槛仅在smoke脚本隔离runtime，绝不作为正式训练父状态。尚未执行整套官方smoke，也未启动正式Stage1。
