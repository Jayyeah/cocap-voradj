V3r2科学candidate `9cc2d47c532c76239a61a0f6a19c8603358c92bf`，canonical lock `384b9ba587b905b879a08f01fa1073ca470bdc0a836459067d193d97b03a84ab`。CORE_V3_SELFTEST_PASS；独立QA仍QA_PENDING，BASE未冻结。

V3r1自身Guard同size/mtime旧缓存漏洞已复现，旧六pilot隔离，源码/QA/历史checkpoint均保留。所有科学入口必须 `python -S tools/batch01_python.py --lock-sha 384b9ba587b905b879a08f01fa1073ca470bdc0a836459067d193d97b03a84ab --delta <ARM delta> --python-args -m <module> ...`；不可用plain python或仅-B启动。trusted脚本验证canonical及自身pin，SourceFileLoader从同一已验证bytes编译本地module，永不读本地pyc；子进程继承固定pin启动器。SourceGuard实际code比对、启动witness、lazy import、漂移、失败latch与每optimizer完整源码hash仍保留。

CPU87通过/3CUDA跳过，CUDA3通过；A3复用12探针通过，执行人仍为同一Commander而非独立签字。补充自身Guard/site旧缓存、合法缓存、无bootstrap拒绝、预载旧实际code拒绝；775k strict load、原T0 parity、256/8/4正式rollout/update、full-state exact恢复均通过。未重跑历史1200局。完整diff及JUnit/source/seed/probe证据见artifacts/2026-10-09_batch01_v3r2_selftest。

未来ARM共同screen2056100900、selection2066100900、盲final2076100900；sample offset100000、scene offsets0/1000/2000，训练env9/actor109及历史180已观察保留域均不重叠。历史T0协议不改，pilot不使用final。

完整Guard检查中位18.95ms，A3 V2约268ms；单样本CPU8决策有/无Guard对比仅工程参考，不能据此声称精确吞吐收益。仍每optimizer step检查全部102来源。并发容量须以ARM共享GPU benchmark与实时lease判定；若一GPU完全空闲则优先该卡，若两卡均有外部任务每卡最多两条自己的线。
