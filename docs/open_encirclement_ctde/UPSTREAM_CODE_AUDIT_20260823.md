# Open Encirclement CTDE 上游代码审计

审计对象由 `upstream_manifest.json` 的逐文件 SHA256 冻结。三个仓库均保留 MIT license；它们是软件 donor，不是某篇围捕论文的官方逐项复现代码。

## 固定来源

| 项目 | URL | branch | commit | 本轮角色 |
|---|---|---|---|---|
| light_mappo | https://github.com/tinyzqh/light_mappo.git | main | `c503d89b6f28c9687ce9e45304fe66b57322ce1e` | MAPPO 工程与算法底座 |
| MADDPG_Multi_UAV_Roundup | https://github.com/reinshift/MADDPG_Multi_UAV_Roundup.git | main | `15309de231f639c62d2049b0ad5b07b8975309c9` | raw 环境、MADDPG、随仓库权重 |
| KF_AA_MARL | https://github.com/reinshift/KF_AA_MARL.git | main | `c8d68cab016ce9e6b18f8435b2faa0048d59da41` | 延后多目标/MATD3 donor；L0 不使用 |

抓取日期为 2026-08-23。clone 中 `git lfs ls-files` 均为空；Roundup 的 16 个无扩展名 PyTorch state dict 和 KF_AA 的 `.pth` 模型均为仓库普通 blob。快照导出时剥离 `.git`，源码不在 snapshot 内修改。

## Roundup/light_mappo 已证实缺陷

- stage/terminal reward 与训练 score 使用 `0:2`，遗漏第三名 hunter；
- 速度限幅计算整个 `multi_current_vel` 矩阵的 norm，而非单 agent norm；
- success 使用 `Sum_S == S4` 浮点精确相等；
- raw action space 无界，但 Actor 返回前按向量 norm 裁到 0.04；light_mappo wrapper 又声明逐元素 ±0.04，而原物理层不强制该界；
- Roundup `main.py` 默认 `evaluate=True`；
- raw target 为第四个 learned agent；light_mappo 实际 target 脚本是“逃离最近 hunter + 墙壁排斥”，与 README 的 idle 描述不一致；
- upstream checkpoint 只有网络权重，没有 optimizer、replay、RNG 或 runtime；
- light_mappo argparse 的实际 `max_grad_norm` 默认值是 10.0，help 字符串仍写 0.5；L0 明确采用并记录 10.0；
- light_mappo 连续动作头是无界 Normal。本地 bridge 保留其 actor/critic、PPO、GAE、Huber 与 ValueNorm，只替换为 log-prob 一致的 tanh 有界物理动作头。

## KF_AA 边界

本轮只冻结来源，不把其 MATD3 或多目标环境接入 L0。其 README 明示 target 未充分训练、assignment/role allocation 尚有 TODO；进入 R6 前仍须按主合同完成 twin critic、joint input、stop-gradient、delayed update、target smoothing、done/bootstrap、replay 和 checkpoint 硬审计。
