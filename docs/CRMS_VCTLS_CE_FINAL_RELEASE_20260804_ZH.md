# CR-MS + VCT-LS + CE 最终发布与复现说明

## 1. 发布目标与边界

本发布包把当前最强线整理为一个可独立复现的三阶段课程：

```text
4v1 scratch 2M
  -> 每 100k 独立 screening、全程训练后选历史最优
8v2 warm course 700k
  -> 每 100k screening、自动选优晋级
12v3 warm course 700k
  -> 每 100k screening、选终态最优
```

主配置已经把最终 CR-MS、VCT-LS、CE 与 support attraction 的 override 扁平化到 `common.yaml`，不再继承过程实验目录。仓库只上传三个最终 checkpoint，不上传训练过程 checkpoint、实际 rollout、GIF、JSONL 或后台日志。

## 2. 最终逻辑

### 2.1 VCT-LS：局部感知与一阶通信

围捕者之间用 pursuer-only Voronoi 邻接建立一阶通信；敌人与障碍物 token 只能由局部 surface-distance 半径产生。Support agent 可以知道相邻追捕队友处于 pursuing 状态，但不会获得全局敌人坐标。

### 2.2 CR-MS：直接探测者的紧凑围捕奖励

直接看到敌人的 agent 使用 `ring_importance_ms_v0`。核心设置为首选中心半径 `8.0`、外半径 `10.5`、径向尺度 `2.0`、环形权重 `2.0`，并关闭旧的全局 approach、mean-shift、front 三项，避免重复或冲突塑形。

### 2.3 Support attraction：局部感知缺口的桥接

局部感知下，support agent 常看不到围捕圈与敌人。最终奖励保持：

```latex
r_i^{\mathrm{support}}=0.5\,r_i^{\mathrm{approach}}+0.5\,r_i^{\mathrm{CE}}
```

其中 approach 目标仅来自“一跳 pursuing 邻居直接看到的敌人”，只使用旧 mix-capture 的吸引进度项：

```latex
r_i^{\mathrm{approach}}=\operatorname{clip}\!\left(d_{i,t-1}-d_{i,t},-3,3\right)
```

这使 support agent 学会跟随围捕队友接近敌人，进入自己的局部感知范围后再切入 CR-MS 直接围捕，同时仍保留一半 CE coverage 目标。

### 2.4 CE coverage 与判定

CE 使用 centroid energy、PBRS 和延迟的轻量速度代价。严格成功要求中心误差 RMS、最大误差与 hold 同时满足：

```latex
\sqrt{\frac{1}{N}\sum_{i=1}^{N}d_i^2}\le 0.05,\qquad \max_i d_i\le 0.10
```

并连续保持 30 steps。`CV<0.15` 仅作为宽松旧指标诊断，不是 CE strict。本轮 checkpoint 排序在围捕/碰撞主指标之后，依次比较 `mix CE strict`、`pure CE strict`，再比较 mix/pure CV。

## 3. 最终文件

```text
configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/
  common.yaml
  stage1_4p1e1obs_scratch2m.yaml
  stage2_8p2e2obs_700k.yaml
  stage3_12p3e3obs_700k.yaml

artifacts/2026-08-04_crms_vctls_ce_final/
  manifest.json
  checkpoints/
    stage1_4v1_step_2000000.pt
    stage2_8v2_step_300000.pt
    stage3_12v3_step_700000.pt
```

哈希见 artifact 内 `manifest.json`。`8v2 300k` 是 CE-first 正式重评后的当前最佳；`12v3 700k` 是已完成的历史终态模型，其真实训练 lineage 使用当时选中的 `8v2 500k` warm start。本发布没有声称 12v3 已由 300k 重新训练。

## 4. 新环境安装与兼容性

要求 Python 3.10+、PyTorch 2.4+，自动流程另需 `tmux`。在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
pytest -q
```

## 5. 一条命令从 scratch 完成课程

双 GPU 推荐命令：

```bash
python3 tools/supervise_crms_support_approach_curriculum.py \
  --train-device cuda:0 \
  --eval-device cuda:1 \
  --poll-seconds 120
```

监督器会自动启动训练、每 100k screening、阶段选优和正式 `20 rollout / 10 GIF`，再把最佳 checkpoint 注入下一阶段。注入只写入被 Git 忽略的 `runs/_curriculum_runtime/` 覆盖配置，不修改仓库中的最终 YAML。

单 GPU 环境也可把两个参数都设为 `cuda:0`，但 training 与 screening 并行时会竞争显存；更稳妥的是手工错峰。CPU 仅适合 smoke/契约测试，不建议正式训练。

### 手工启动 stage1

```bash
python3 train.py \
  --config configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/stage1_4p1e1obs_scratch2m.yaml \
  --device cuda:0
```

stage2/3 的源码 YAML 有意不固定 `pretrained.path`，以防新环境误用本机路径；推荐由监督器生成 runtime overlay 并自动注入上一阶段 selection。

## 6. Screening、正式展示与尾迹

`tools/watch_screened_run.sh` 负责 100k 间隔的 capture、coverage、mix screening；`tools/finalize_screened_run.py` 负责 CE-first 选优并自动跑正式 20-rollout/10-GIF。

所有 rollout/GIF 入口的 faded trails 默认关闭。只有显式输入下列参数才开启：

```text
--draw-trails
```

## 7. Zone 泛化分支

ZoneDemo 暂时只用于泛化性能测试：检查策略在敌人由外部环带进入原内区时的行为。它不是当前 CR-MS+VCT-LS+CE 的直接训练任务场景，最终主配置也不含 Zone scene。

仓库仍备份以下可复用内容：

- `configs/demos/zonedemo_v0/`：B0--B5 与不同规模/目标设置的泛化评估配置；
- `configs/experiments/zonedemo_adapt_20260728/`：B1 的 8v2、12v3 adaptation 训练与 old-mix 对照配置；
- `tools/run_zonedemo_batch.py`、`tools/supervise_zonedemo_adapt.py`；
- `tests/test_zone_demo_contract.py` 及环境中的 Zone opt-in 代码路径。

Zone rollout/GIF 和 adaptation 训练产物不上传。若之后正式把 Zone 纳入任务训练，应另开配置系列并重新定义晋级门槛，不能把当前泛化展示结果直接当作主线训练指标。

## 8. Git 上传边界核对

应上传：最终核心代码、四个最终课程 YAML、自动课程与 screening/finalizer 工具、必要测试、发布文档、三阶段 selected checkpoints、Zone 分支代码/配置/测试。

不应上传：`runs/`、`logs/`、W&B、过程实验 artifacts、所有实际 rollout/GIF、`.orig`、临时 worker/selection 文件、除三个 selected checkpoint 外的 `.pt`。
