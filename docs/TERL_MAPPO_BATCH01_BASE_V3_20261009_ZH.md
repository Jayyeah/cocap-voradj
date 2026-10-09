# Batch01 V3 科学修复交付

Candidate `0b2686a06e900a090a34fd3be4e0143ba3f794f0`，canonical lock `9ca6c24e881e4e13de58513bde32e2008805f3e1fd8fe55645c66a9c413cdf19`。状态 **CORE_V3_SELFTEST_PASS / QA_PENDING / BASE未冻结**；同一Commander复用A3独立编写脚本不构成A3独立签字。

实际执行code object包含nested constants、opcode、flags、行及exception tables，与锁定源码bytes在同Python优化级别编译产物比对；源码hash或pyc timestamp不能代替。启动时`src/sitecustomize.py`捕获guard之前执行对象；缺少执行witness的本地模块fail closed。所有入口必须在解释器启动前将该worktree的`src`置于PYTHONPATH首位。合法源码/cache和声明动态import保留；同size/mtime陈旧cache及提前加载旧代码拒绝，运行中漂移拒绝。failure latch阻止后续hook/update/save/load与checkpoint promotion。pytest显式使用plain assertions，以免将源码重写产物作为科学hook。

最终同candidate CPU **83 passed / 3 CUDA skipped / 0 failures**；另外CUDA三个节点 **3 passed / 0 failures**，包括正式256/8/4、T0逐值parity、256 rollout/update、精确full-state resume和B3 CUDA RNG。CPU保留775k strict load、B1/B2/B3/B4/Q1全部回归。A3原脚本只做pin、archive inventory、输出目录及stale-pyc预期的可审查适配，12个探针通过；两旧漏洞均在不合格动态模块体执行前拒绝，optimizer=0，save/load锁住。原QA工作树与证据未修改。

完整hash检查仍位于每个actor/value optimizer step和rollout/update/checkpoint边界。只缓存外部模块遍历和已验证code identity；完整检查中位数16.52ms（A3 V2为268.27ms），动态hook短样本中位4.34ms。实际ARM CUDA并发吞吐另测，不把这项短采样当训练吞吐。

T0历史协议不变。未来五ARM共同锁定screen2056100900、selection2066100900、final2076100900，sample动作offset100000；3P1E0O4C/4P1E1O6C/7P2E2O8C的scene offset为0/1000/2000。逐一读取历史实际episode种子并核验所有用途/场景/动作流与新域不相交，保留训练seed9/actor109。future final不得用于pilot、训练或selection；共享新物理seed用于配对比较。

TERL原MarineEnv/actor、critic、PPO/GAE/ValueNorm、原reward/动力学、原runner/evaluator/config与T0 Git对象逐字一致。R1可选只读pre/post几何snapshot及分量记录仅在显式reward hook上开启；T0默认路径完全不变。P1 config allowlist仅target_kl；N1不能通过actor宽度参数连带修改critic。

轻量完整diff、source audit、最终JUnit、12探针原始输出、适配patch、新seed合同均在`artifacts/2026-10-09_batch01_v3_selftest/`。未重复历史1200局evaluation。独立审核人仍须对该candidate/lock签字；正式共同parent仍为空。有界25k PROVISIONAL pilots独立登记，不能自动升级或追认为正式证据。
