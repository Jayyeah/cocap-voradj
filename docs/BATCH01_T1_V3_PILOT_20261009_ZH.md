# T1 V3工程与pilot合同

状态：独立BASE QA_PENDING；只允许最多25k joint decisions PROVISIONAL。科学parent候选0b2686a；不能据此宣称正式冻结。

从selected775k严格导入actor参数，固定容量critic、两个Adam、ValueNorm、Stage2环境/APF/RNG/计数从新状态开始。跨规模迁移不称bit-exact resume；同一规模完整checkpoint恢复逐项精确核验。

Stage2原生4P1E1O6C；Stage3原生7P2E2O8C。actor保留原生self4/friend5x7/enemy8x7/obstacle5x5，padding为inactive；central critic容量7P/2E/2O/8C，包含所有current的位置、方向、Gamma、core radius和presence。MarineEnv、AW9、PPO和reward源码保持原样。

Stage2首review100k，可显式登记延长至200k；Stage3为100k/200k/300k review窗口，不能自动追加。升级要求Stage2与Stage1 retention的selection各20局/模式，两模式normal_capture>=0.8且collision<=0.2。保留原阈值率，将用于训练升级的用途绑定selection，final不参与升级或调参。Stage3入口要求已冻结BASE、正式完整selection+retention报告、同一checkpoint、manifest和报告hash的受控promotion bundle。未达到条件不启动正式Stage3。

评估按规模分别记录：Stage1 offset0、Stage2 offset1000、Stage3 offset2000，共用锁定新seed域2056100900/2066100900/2076100900。pilot只用screen；Stage3短工程恢复测试不属于Stage3训练。

验收文件：artifacts/2026-10-09_arm_preflight。覆盖三规模state/mask/current、实际collision后部分agent继续、第二次collision终止、多目标部分capture继续与all-capture终止、原生Stage1 parity、Stage2/Stage3完整恢复、升级门禁负测试、CUDA256/8/4和独立各规模evaluator smoke。
