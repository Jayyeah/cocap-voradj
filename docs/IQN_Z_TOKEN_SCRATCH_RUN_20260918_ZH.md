# IQN-Z-TOKEN scratch 运行记录（2026-09-18）

更新时间：2026-09-18 18:17（Asia/Shanghai）

## 当前状态

**BLOCKED_PRELAUNCH：未启动正式训练。**

Z 线独立工作树与正式监督器已经就绪，但实际 ROLE scratch 配置仍使用历史固定 20m 感知，而本任务明确要求两臂都使用 NormSense-V2。自动 ROLE-vs-Z 门禁因此按设计停止；没有创建 Z 训练进程、没有占用 GPU0，也没有生成可误认为正式运行的 PID、step、吞吐或 ETA。

## 固定实验合同

- 分支：`experiment/iqn-z-token-scratch-20260918`
- 训练预算：scratch IQN，严格止于 200,000 environment steps
- 里程碑：25k、50k、75k、100k、125k、150k、175k、200k
- self token：`[physical, z_i]`
- friend token：`[physical, z_j]`
- friend ordering：physical-only
- 已删除 `is_pursuing` 输入、`pursuing_embed`、late fusion 和 role-dependent ordering
- z-v1：`lambda=0.95`、`eta=0.85`、previous-step neighbor、同步更新、direct 当步置 1、reset 置 0、无 hard floor
- 感知：NormSense-V2
- 训练初始化：完全随机；无 BC、Final IQN、no-role 或 teacher 权重迁移
- 正式评估：每场景 20 episodes，seed base 2026092801，与 ROLE 约定一致

## 已通过的启动 sanity

使用当前 seed 2026080201、physical-only ordering 和 NormSense-V2 参考 ROLE 配置完成了真实 GPU0 预检：

- ROLE/Z 随机初始化 state hash bit-exact 相同：`22a7deaa4b7265909a8b6c340f84d8741b69194fd3cc5c643f487ddeb3908ec8`
- Z observation shape：self `[9]`、friend `[8,7]`
- physical friend rows 不受 role token 翻转影响
- z 同步传播、direct z=1、pure coverage z=0、自然衰减语义通过
- full-resume 恢复 z 完整状态通过
- 真实环境 16 步产生 15 次 optimizer update；参数发生变化；loss 有限
- target network 更新 15 次，最后同步 step=16
- coverage/capture/mixed evaluator smoke 完成
- GPU：NVIDIA RTX A6000

机器证据：`artifacts/2026-09-18_iqn_z_token_scratch/reference_preflight/startup_sanity.json`。

聚焦回归：25 passed，2 skipped；跳过项为条件性测试，不是失败。

## 实际 ROLE 门禁结果

实际 ROLE commit：`14702f5e1d2c2ec49ced78efb84c3f4b3642f612`。

机器证据：

- `artifacts/2026-09-18_iqn_z_token_scratch/preflight/config_diff.json`
- `artifacts/2026-09-18_iqn_z_token_scratch/status.json`

门禁发现的非允许差异全部属于感知配置：

- ROLE 缺少 `normsense_v2.enabled/policy/schema_version`
- ROLE 仍有 `voradj.enemy_sensing_radius=20`
- ROLE 仍有 `voradj.obstacle_sensing_radius=20`
- ROLE 缺少 `voradj.onboard_sensing.{policy,schema_version,k,radius_floor}`

这不是 evidence token 差异，故不能开始可归因的 ROLE-vs-Z 对照。

## 解阻条件与后续动作

等待 ROLE 线采用相同 NormSense-V2 有效配置并重新建立可比较基线。Z 线不会修改、停止或重启 ROLE 工作树/进程。ROLE 配置更新后，重新运行监督器会强制完整重跑预检，不复用缓存 sanity；通过后才会在独立 GPU 上启动，并在确认 optimizer update、loss、replay、target sync 正常后记录实际 PID、step、吞吐、ETA 与远端 head。

## Z-v2 TODO（本轮禁止实施）

1. 降低 lambda，缩短 half-life。
2. 在 `z_tilde` 后应用小阈值 `epsilon_z`，使指数尾巴有限时间归零。

只有 Z-v1 RL 证明“能学但 release/recovery 偏慢”后，才考虑 Z-v2。
