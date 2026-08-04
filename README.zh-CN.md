# CoCap Voronoi-Adjacency

这是一个面向多追捕者协同围捕与围捕后覆盖的多智能体强化学习项目。当前正式主线是 **CR-MS + VCT-LS + CE**：从 `4v1` scratch 训练，自动筛选并依次晋级到 `8v2`、`12v3`。

完整发布与复现说明见 [最终三合一发布说明](docs/CRMS_VCTLS_CE_FINAL_RELEASE_20260804_ZH.md)。

## 当前正式主线

- **VCT-LS**：友方 Voronoi 一阶通信；敌人和障碍物只按局部表面距离感知。
- **CR-MS**：直接发现敌人的追捕者使用紧凑、速度加权的环形 mean-shift 围捕奖励。
- **Support attraction**：未直接看到敌人、但与追捕邻居一跳连通的 agent，接收 `0.5` approach-only 吸引奖励与 `0.5` CE coverage 奖励。
- **CE coverage**：使用 centroid-energy PBRS、严格中心误差成功条件，并保留 `CV<0.15` 作为宽松诊断。
- **筛选顺序**：围捕/碰撞优先；之后依次比较 mix CE strict、pure CE strict，再比较 mix/pure CV。

最终配置仅依赖已有 A3 基线和下列四个文件，不依赖 CE、VCT-LS、CR-MS 的过程实验配置：

```text
configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/
  common.yaml
  stage1_4p1e1obs_scratch2m.yaml
  stage2_8p2e2obs_700k.yaml
  stage3_12p3e3obs_700k.yaml
```

## 安装

要求 Python 3.10+。建议新建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## 从 scratch 跑完整课程

默认使用 `cuda:0` 训练、`cuda:1` screening/正式 20-rollout/10-GIF：

```bash
python3 tools/supervise_crms_support_approach_curriculum.py \
  --train-device cuda:0 \
  --eval-device cuda:1
```

脚本会执行：`4v1 2M scratch → 全程每 100k screening → 自动选优 → 8v2 700k → 自动选优 → 12v3 700k`。每个大阶段选出最佳模型后自动做对应 `20 rollout / 10 GIF`。渲染尾迹默认关闭，只有显式传入 `--draw-trails` 才开启。

只启动 stage1 训练：

```bash
python3 train.py \
  --config configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/stage1_4p1e1obs_scratch2m.yaml \
  --device cuda:0
```

## 随仓库提供的模型

`artifacts/2026-08-04_crms_vctls_ce_final/` 只保留三个最优 checkpoint 和校验清单：

- `stage1_4v1_step_2000000.pt`
- `stage2_8v2_step_300000.pt`
- `stage3_12v3_step_700000.pt`

不包含实际 rollout、GIF、screening worker 输出或训练日志。注意：当前 CE-first 重排后 `8v2 300k` 优于 `500k`；现有 `12v3 700k` 的真实历史训练 lineage 使用当时选中的 `8v2 500k` warm start，文档没有把它伪装成从 300k 重训所得。

## Zone 泛化分支

ZoneDemo 代码、测试、评估配置和 B1 adaptation 训练配置随仓库备份，但它目前只用于检查“敌人从外区进入内区”时的泛化性能，**不直接作为正式主线的训练任务场景**。正式三合一配置不启用 `zone_demo`，场景仍只有 `voradj` 与 `voradj_coverage`。

- 泛化评估配置：`configs/demos/zonedemo_v0/`
- Zone B1 训练/对照配置：`configs/experiments/zonedemo_adapt_20260728/`
- 批量评估：`tools/run_zonedemo_batch.py`
- 训练监督：`tools/supervise_zonedemo_adapt.py`
- 契约测试：`tests/test_zone_demo_contract.py`

## 上传边界

本次 GitHub 同步包含核心代码、最终三合一配置、自动训练/筛选工具、必要测试与文档、三个最终 checkpoint，以及 Zone 泛化分支的代码/配置/测试。明确不包含：

- `runs/`、`logs/`、W&B 本地文件；
- 过程实验的 checkpoint、screening、rollout、GIF；
- 最终三阶段的实际 rollout 和 GIF；
- `.orig`、worker 临时文件和本机运行时注入配置。

所有 GIF 工具默认关闭 faded trails；显式 `--draw-trails` 才开启。
