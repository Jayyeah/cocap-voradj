<div align="center">

# `light_mappo`

### 一个下午就能训练出合作多智能体策略。

PyTorch 实现的极简、可读、自带 BYO-环境模板的 MAPPO。
**没有 SMAC。没有 GFootball。没有 wandb。约 30 个 Python 文件。**

[![License](https://img.shields.io/badge/license-MIT-1f6feb.svg?style=flat-square)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-1f6feb?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=flat-square)
[![English](https://img.shields.io/badge/lang-English-1f6feb?style=flat-square)](README.md)
[![bilibili](https://img.shields.io/badge/视频解析-bilibili-00a1d6?style=flat-square)](https://www.bilibili.com/video/BV1bd4y1L73N)

<br>

<img src="assets/demo.gif" width="460" alt="3 个 hunter 在 UAV 围捕环境中包围逃逸 target。">

<sub>3 个 hunter · 1 个 target · 3 个障碍物 · 训练 15 万步 · 49 步内完成围捕。</sub>

</div>

---

## 你能拿到什么

```
✓ 一个能跑的 MAPPO 基线   ✓ 中心化 critic / GAE / value norm / PopArt
✓ 循环网络或前馈可选        ✓ Box / Discrete / MultiDiscrete / MultiBinary
✓ 共享 / 分离两种多智能体训练模式
✓ 一个现成可跑的 UAV 围捕 demo + 把推理结果导出成 GIF 的渲染脚本
✓ 接入你自己的环境，只需要改一个文件
```

<div align="center">
<table>
  <tr>
    <td align="center" width="33%"><img src="assets/start.png" width="240"><br><b>① 起始</b><br><sub>Hunter 出生在角落，target 在捕获圈内</sub></td>
    <td align="center" width="33%"><img src="assets/encircle.png" width="240"><br><b>② 包围</b><br><sub>策略展开协同追击队形</sub></td>
    <td align="center" width="33%"><img src="assets/capture.png" width="240"><br><b>③ 捕获</b><br><sub>包围三角形闭合，~50 步抓住</sub></td>
  </tr>
</table>
</div>

---

## 一句话快开

```bash
# 安装
pip install -r requirements.txt

# 训练（现代 CPU 约 3 分钟）
python train/train.py --env_name uav --experiment_name uav_demo \
    --num_env_steps 150000 --episode_length 100 \
    --n_rollout_threads 8 --hidden_size 64 --layer_N 1

# 看看它学到了什么
python scripts/render_uav.py \
    --model_dir results/uav/MyEnv/mappo/uav_demo/run1/models \
    --video_path videos/uav_demo.mp4 \
    --hidden_size 64 --layer_N 1 --num_episodes 3
```

打开 `videos/uav_demo.mp4`，就是你在 GIF 里看到的画面，在你自己机器上跑出来。

---

## 为什么再造一个 MAPPO 仓库？

原版 [`marlbenchmark/on-policy`](https://github.com/marlbenchmark/on-policy) 内置了五套环境（SMAC、MPE、Hanabi、GFootball、MAMuJoCo），互相耦合在 runner / buffer / config 各处。要接入**你自己的**环境，得把所有这些地方都改一遍。

`light_mappo` 反过来：训练栈固定，环境是唯一的变量。

|  | `on-policy` 原版 | `light_mappo` |
|---|---|---|
| Python 文件数 | 200+ | ~30 |
| 内置环境 | 5 套 | 1 个 demo + 你自己的 |
| 接入新环境工作量 | 改 scenario / registry / runner | 填一个文件 |
| 外部依赖 | wandb、SC2 二进制、RoboSchool | 都不需要 |
| 在你的环境上跑出第一条策略 | 几个小时 | 几分钟 |

如果你想跑标准 benchmark 对比，用原版。如果你想**把 MAPPO 装进一个新东西里**，从这里开始。

---

## 算法结构

```
        ┌──────────────────────────────────────────────┐
        │           中心化 Critic                       │
        │         V( concat(o_1 … o_N) )               │
        └──────────────────────────────────────────────┘
            ▲              ▲              ▲
            │              │              │
   ┌────────┴────┐ ┌───────┴────┐ ┌──────┴─────┐
   │  Actor π_θ  │ │ Actor π_θ  │ │ Actor π_θ  │   ← 参数共享
   │   (o_1)     │ │   (o_2)    │ │   (o_3)    │     （或独立）
   └─────────────┘ └────────────┘ └────────────┘
         │              │              │
         ▼              ▼              ▼
       agent 1        agent 2        agent 3
              \         |         /
               \        ▼        /
              ┌──────────────────┐
              │      环境         │
              └──────────────────┘
```

- **去中心化执行**：推理时每个 actor 只看自己的 obs
- **中心化训练**：critic 拿到联合状态，credit assignment 更准
- **PPO 全家桶**：GAE · clipped ratio · value clip · 可选 Huber · 可选 PopArt / ValueNorm

---

## 目录速览

```
light_mappo/
├── algorithms/        MAPPO 训练器 + actor-critic 网络
│   ├── algorithm/      r_mappo.py  rMAPPOPolicy.py  r_actor_critic.py
│   └── utils/          act / mlp / rnn / cnn / popart / distributions
├── runner/
│   ├── shared/         所有 agent 共享一个策略
│   └── separated/      每个 agent 独立策略
├── envs/
│   ├── env_wrappers.py  ── DummyVecEnv（共用基础设施）
│   │
│   ├── custom_env/      ──▶ 在这里接入你的环境
│   │   ├── env_core.py        写 step / reset / spaces
│   │   ├── env_continuous.py  连续动作空间外壳
│   │   └── env_discrete.py    离散动作空间外壳
│   │
│   └── uav/             ──▶ 现成可跑的 demo 环境
│       ├── uav_env.py         2D UAV 物理 + lidar
│       ├── uav_roundup_env.py framework wrapper（3 hunters）
│       └── uav_utils.py       几何辅助
│
├── train/train.py     入口
├── scripts/render_uav.py  载入 checkpoint → 输出 MP4 / GIF
├── config.py          ~80 个 CLI 参数 (lr, ppo_epoch, hidden_size, …)
└── utils/             SharedReplayBuffer, ValueNorm, 杂项
```

---

## 接入你自己的环境

`envs/custom_env/env_core.py` 是你**唯一**需要写的文件。规格：

```python
class EnvCore:
    def __init__(self):
        self.agent_num  = 2      # 几个 agent
        self.obs_dim    = 14     # 单 agent 观测维度
        self.action_dim = 5      # 单 agent 动作维度

    def reset(self):
        # 返回长度 agent_num 的 list，每个元素 shape (obs_dim,)
        ...

    def step(self, actions):
        # actions: 长度 agent_num 的 list，每个元素 shape (action_dim,)
        # 返回 [obs_list, reward_list, done_list, info_list]
        ...
```

连续动作直接用 `env_continuous.py`；离散动作在 `train/train.py:_build_single_env` 里换一个 import。

然后用 `--env_name <随便你叫>` 跑训练就行，和 demo 一样。

`envs/uav/` 里的 UAV demo 是一个稍复杂的样例：异构观测空间、脚本化对手、matplotlib 渲染——可以当模板抄。

---

## 算法特性

| 组件 | 状态 |
|---|---|
| 共享 / 分离策略 | ✓ 两套 runner |
| 循环网络 | ✓ GRU / naive recurrent (`--use_recurrent_policy`) |
| 动作空间 | ✓ Box · Discrete · MultiDiscrete · MultiBinary |
| Advantage estimation | ✓ GAE + 归一化 |
| Value loss | ✓ MSE 或 Huber，可选 clip |
| Value 缩放 | ✓ PopArt 或 running ValueNorm |
| 学习率调度 | ✓ 线性衰减 |
| 梯度裁剪 | ✓ max-grad-norm |
| 并行 rollout | ✓ DummyVecEnv |
| 日志 | ✓ TensorBoard (via `tensorboardX`) |

---

## 引用

如果 `light_mappo` 帮到你，请顺手点个 ⭐ 并引用：

```bibtex
@software{light_mappo,
  author = {Zhiqiang He},
  title  = {light\_mappo: Lightweight MAPPO implementation},
  year   = {2025},
  url    = {https://github.com/tinyzqh/light_mappo},
  note   = {Version v0.1.0}
}
```

<details>
<summary><b>用过本代码的论文</b></summary>

```bibtex
@inproceedings{he2024intelligent,
  title  = {Intelligent Decentralized Multiple Access via Multi-Agent Deep Reinforcement Learning},
  author = {He, Yuxuan and Gang, Xinyuan and Gao, Yayu},
  booktitle = {2024 IEEE Wireless Communications and Networking Conference (WCNC)},
  pages = {1--6}, year = {2024}, organization = {IEEE}
}
@article{qiu2024enhancing,
  title   = {Enhancing UAV Communications in Disasters: Integrating ESFM and MAPPO for Superior Performance},
  author  = {Qiu, Wen and Shao, Xun and Loke, Seng W and He, Zhiqiang and Alqahtani, Fayez and Masui, Hiroshi},
  journal = {Journal of Circuits, Systems and Computers},
  year = {2024}, publisher = {World Scientific}
}
@article{qiu2024optimizing,
  title   = {Optimizing Drone Energy Use for Emergency Communications in Disasters via Deep Reinforcement Learning},
  author  = {Qiu, Wen and Shao, Xun and Masui, Hiroshi and Liu, William},
  journal = {Future Internet}, volume = {16}, number = {7}, pages = {245},
  year = {2024}, publisher = {MDPI}
}
@inproceedings{yu2024path,
  title  = {Path Planning for Multi-AGV Systems Based on Globally Guided Reinforcement Learning Approach},
  author = {Yu, Lanlin and Wang, Yusheng and Sheng, Zixiang and Xu, Pengfei and He, Zhiqiang and Du, Haibo},
  booktitle = {2024 IEEE International Conference on Unmanned Systems (ICUS)},
  pages = {819--825}, year = {2024}, organization = {IEEE}
}
```

</details>

---

## 致谢

- 算法核心参考自 [`marlbenchmark/on-policy`](https://github.com/marlbenchmark/on-policy)（MAPPO 原作者实现）
- UAV 环境改编自一个公开 MAPPO 训练仓库，重写成独立的 benchmark 环境
- 英文翻译 [@tianyu-z](https://github.com/tianyu-z)

由 [@tinyzqh](https://github.com/tinyzqh) 维护 · [MIT License](LICENSE)

<br>

<div align="center">
<sub>如果 <code>light_mappo</code> 帮你省了一个下午，给个 ⭐ 让别人也能找到它。</sub>
</div>
