# CoCap Voronoi-Adjacency

CoCap Voronoi-Adjacency 是一个多智能体强化学习项目，目标是在二维连续场景中完成多追捕者协同围捕，并在围捕后切换到 Voronoi coverage，使追捕者尽量形成稳定、均匀、低速的覆盖布局。

当前整理版聚焦最新主线 A3：CenterSqrtN 归一化、hard coverage motion gate、几何后速度收敛判定，以及从 `4v1` 逐步扩展到大规模场景的课程学习。

## 项目结构

GitHub 上传与日常同步方案见 [docs/GIT_WORKFLOW.zh-CN.md](docs/GIT_WORKFLOW.zh-CN.md)。


- `src/cocap_voradj/`：核心代码包。
- `src/cocap_voradj/envs/`：CoCap 基础环境和 Voronoi-adjacency 环境。
- `src/cocap_voradj/dynamics/`：pursuer、evader、robot、perception 等运动与感知基础模块。
- `src/cocap_voradj/control/`：APF evader 控制器。
- `src/cocap_voradj/models/`：IQN 策略/价值网络。
- `src/cocap_voradj/training/`：trainer、replay buffer、配置加载和训练主循环。
- `configs/experiments/`：实验配置文件。
- `tools/`：训练监督、checkpoint 筛选、rollout、GIF 渲染工具。
- `artifacts/`：精选实验产物，包括 A3 各阶段 best checkpoint、对应配置、summary 和少量 GIF。
- `runs/`：本地训练输出目录，默认不纳入 git。

## 当前主线

A3 主线的核心配置：

- `train_mode: voradj_mixed_coverage`：混合围捕与 coverage 训练。
- `apf.version: v2_fixed`：使用修正后的 APF evader 控制逻辑。
- `voradj.center_sqrt_n_normalization_enabled: true`：对子区域中心向量做随智能体数量变化的尺度归一化。
- `coverage_motion_gate_mode: hard`：coverage 几何条件满足后才进入正式速度 settle 阶段。
- `iqn.update_rule: distributional_iqn`：使用分布式 IQN 更新。
- `output_root: runs`：训练输出统一写入 `runs/`。

A3 已纳入的精选 artifact 为 stage1-stage6，每个阶段保留：

- 选中的 best checkpoint；
- checkpoint 对应训练配置；
- `all_summaries.json` 和各场景 summary；
- 每阶段 3 个 mix GIF，用于快速查看策略行为。

## 安装

建议使用 Python 3.10+。依赖可按当前 `requirements.txt` 安装：

```bash
pip install -r requirements.txt
```

无需安装包也可以直接运行：

```bash
PYTHONPATH=src python3 -c "import cocap_voradj; print(cocap_voradj.__version__)"
```

## 训练

查看训练入口参数：

```bash
python3 train.py --help
```

启动 A3 `4v1` scratch 训练：

```bash
python3 train.py \
  --config configs/experiments/voradj_a3_apfnew_sqrtn_20260723/a3_apfnew_sqrtn_4v1_scratch_mix_2m.yaml \
  --device cuda:1
```

短训练 smoke test：

```bash
python3 train.py --config configs/smoke/a3_4v1_cpu_smoke.yaml --device cpu
```

训练输出会写入：

```text
runs/<run_name>/
  effective_config.yaml
  episodes.jsonl
  metrics.jsonl
  checkpoints/
```

## 课程训练

A3 课程监督脚本：

```bash
python3 tools/supervise_a3_curriculum.py --help
```

该脚本用于自动推进 A3 课程阶段、周期性筛选 checkpoint、启动正式 rollout，并把阶段 handoff 写入 `artifacts/2026-07-23_a3_apfnew_sqrtn_curriculum/`。

## Rollout 与可视化

并行 rollout，适合 CPU 多进程评估：

```bash
python3 tools/batch_rollouts_parallel.py \
  --config configs/experiments/voradj_a3_apfnew_sqrtn_20260723/a3_apfnew_sqrtn_4v1_scratch_mix_2m.yaml \
  --checkpoint artifacts/2026-07-23_a3_apfnew_sqrtn_curriculum/best_20rollout10gif/stage1_4p1e1obs_step_2000000/step_2000000.pt \
  --output-root artifacts/local_rollout_debug/stage1 \
  --episodes 20 \
  --gif-count 3 \
  --scenarios mix coverage \
  --device cuda:1 \
  --workers 4
```

单进程 rollout：

```bash
python3 tools/batch_rollouts.py \
  --config configs/experiments/voradj_a3_apfnew_sqrtn_20260723/a3_apfnew_sqrtn_4v1_scratch_mix_2m.yaml \
  --checkpoint artifacts/2026-07-23_a3_apfnew_sqrtn_curriculum/best_20rollout10gif/stage1_4p1e1obs_step_2000000/step_2000000.pt \
  --output-root artifacts/local_rollout_debug/stage1_single \
  --episodes 5 \
  --gif-count 2 \
  --scenarios mix \
  --device cpu
```

输出目录会包含 `batch_summary.json`、episode 记录和 GIF 文件。

## Checkpoint 筛选

对某个训练 run 的 checkpoint 做快速筛选：

```bash
python3 tools/evaluate_checkpoints.py \
  --run-dir runs/<run_name> \
  --checkpoint-step 500000 \
  --checkpoint-step 600000 \
  --episodes 12 \
  --max-steps 500 \
  --device cuda:1 \
  --output artifacts/local_screening/<run_name>.jsonl
```

## 调试命令

检查配置能否加载并 reset 环境：

```bash
PYTHONPATH=src python3 -c "from cocap_voradj.training.trainer import load_config, set_global_config; from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv; cfg=load_config('configs/experiments/voradj_a3_apfnew_sqrtn_20260723/a3_apfnew_sqrtn_4v1_scratch_mix_2m.yaml'); set_global_config(cfg); env=VorAdjEnv(cfg); obs=env.reset(); print(type(obs).__name__, len(obs), type(obs[0]).__name__)"
```

加载精选 checkpoint 并做一次 CPU forward：

```bash
PYTHONPATH=src python3 -c "from cocap_voradj.training.trainer import load_config, set_global_config; from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv; from cocap_voradj.training.replay import stack_obs; from cocap_voradj.models.iqn import CoCapIQN; cfg=load_config('configs/experiments/voradj_a3_apfnew_sqrtn_20260723/a3_apfnew_sqrtn_4v1_scratch_mix_2m.yaml'); set_global_config(cfg); env=VorAdjEnv(cfg); obs=env.reset(); model=CoCapIQN.load('artifacts/2026-07-23_a3_apfnew_sqrtn_curriculum/best_20rollout10gif/stage1_4p1e1obs_step_2000000/step_2000000.pt', device='cpu'); out=model(stack_obs(obs, 'cpu'), num_tau=4, mode='voradj'); print(tuple(out['q_values'].shape))"
```

## 上传边界

当前整理版不包含核心内部文档，也不包含大规模训练输出。默认不追踪 `runs/`、`logs/`、大规模 `.jsonl/.log`、非精选 checkpoint 和大量 GIF；只保留 A3 当前精选 checkpoint 和少量可视化 GIF。
