# C0 V3工程与pilot合同

状态：独立BASE QA_PENDING；只允许最多25k joint decisions PROVISIONAL。科学parent候选0b2686a，正式scratch1M需最终冻结和独立验收。

保留TERL原始encoder、critic、MarineEnv、reward与PPO；actor末端为双维Gaussian。a范围[-0.4,0.4]，omega范围[-pi/6,pi/6]。联合density包含两维Normal、稳定tanh Jacobian和两个physical scale Jacobian；checkpoint/rollout保存原pre-tanh latent，零更新ratio核验在optimizer前执行。

通过PhysicalActionList将连续物理AW直接交给未修改的Robot.update_state与MarineEnv.update_rob_state；不量化回AW9。九点AW9 transition/reward/event parity、off-grid实数积分、洋流/阻力/限速/碰撞时序均有测试。

独立evaluator使用mean/sample，physical differential MC entropy与categorical entropy分别标记；entropy诊断隔离RNG，不消耗sample动作seed流。使用锁定新screen/selection/final域，pilot禁止final。

验收文件：artifacts/2026-10-09_arm_preflight。包含CPU/CUDA256实际update、Jacobian解析检查/饱和有限性、错误behavior density在optimizer前拒绝、scratch Adam/ValueNorm、完整状态精确恢复、连续evaluator smoke及GPU1并发benchmark。
