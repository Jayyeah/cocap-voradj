# N3 representation implementation

PROVISIONAL。代码历史 parent=`8ce559335d656b7338a23886eb9cc958fa1a96cc`；算法父线 N2，冻结 E1=`8827c11`。成功 IQN 唯一来源=`1a9da054ecb5652726005d00d08ab05a2188dd8e`，模型文件逐字 hash 对照。新增代码在 batch02_native；旧 E1、N2/R2、PPO/provider/evaluator 文件不修改。

| 项 | TERL / N2 | N3 | 相对成功 IQN |
|---|---|---|---|
| token 来源 | E1 完整 CoCap observations | 不变：self9、P8×7、E8×7、O5×5，22 slots | 成功 Z05 VorAdj/NormSense/self Z，原 provider |
| encoder | 原 TERL entity_encoders | 原 CoCapIQN.encoders/type_embedding | 原类构造，256/8/4与成功实现相同 |
| attention/pooling | transformer + self/max + TERL target_selection | 原 features：masked mean/max、target Q/K/V、3 summary roles、summary attention/fusion、single_action_feature | 直接继承成功 features/voradj_single_feature |
| policy head | hidden_layer/layer_norm/output_layer，AW9 categorical | 同一原 TERL head，初始化保持 | MAPPO adaptation：不使用 quantile Q head；无 teacher 权重 |
| mask | TERL pooling 允许 padding query 影响 max | 原 CoCap pooling 排除 masked slots；target mask/type=2 | 原 IQN mask；inactive全空行加 self placeholder，仅原 active_mask 排除行 |
| critic | E1 action-free central V self9/4P | 类/参数/初始权重/训练逻辑保持 | 原迁移合同，非 IQN Q learner |

actor 只接受 focal local observation，不接收 central state 或其他 actor batch 行。没有新增 friendly/target visibility，未修改环境、reward、action 或 terminal。CoCap模块原dropout=.1；继承现有 PPO 始终 eval 模式，避免新随机训练路径；这是 MAPPO adaptation。未使用 quantile/coverage/dual-head/gate 参数，删除 dormant heads。

工程验证：tools/verify_batch02_native_20261011.py。固定64步动作比较 observation/reward/done/next state，包含原200k reward callback两侧；256 on-policy steps / PPO，16-step fresh-process exact resume，原 IQN feature输出、masked padding、batch行独立、inactive finite。工程通过不代替 capability evaluation。

训练准备：scratch，pilot1024；用户已授权 long1M，每25k保存并评估，固定成功4P1E1O mixed合同，无新curriculum/lifecycle/repeated arrival/multi-target/coverage扩展。保留 Batch02-A evaluator的任务、seed域、动作模式和 selection/final 合同；只增加新actor factory binding。

参数量：N2 actor=3,568,137；N3 actor=3,963,913（+395,776，约+11.09%）；critic均3,236,865。工程验证 PASS；evaluator兼容由后续pilot核验，尚无学习结论。
