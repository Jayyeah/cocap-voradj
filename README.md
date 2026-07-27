# CoCap Voronoi-Adjacency

CoCap Voronoi-Adjacency is a multi-agent reinforcement learning project for cooperative capture and post-capture coverage. Pursuer agents learn with an IQN-based value network, while evaders can be controlled by an APF baseline. The current main line focuses on APF-new, CenterSqrtN-normalized Voronoi observations/rewards, hard coverage motion gating, and curriculum scaling from small to larger multi-agent scenes.

中文说明见 [README.zh-CN.md](README.zh-CN.md). Git/GitHub workflow: [docs/GIT_WORKFLOW.zh-CN.md](docs/GIT_WORKFLOW.zh-CN.md).

## Layout

- `src/cocap_voradj/`: package code for environments, dynamics, APF control, IQN models, replay, and training.
- `configs/experiments/`: experiment YAML files. The current curated line is A3 APF-new + CenterSqrtN.
- `tools/`: rollout, parallel evaluation, checkpoint screening, and curriculum supervision tools.
- `artifacts/`: selected A3 checkpoints, configs, summaries, and a small number of GIFs.
- `runs/`: local training outputs; ignored by git.

## Quick Start

```bash
python3 train.py --help
python3 train.py --config configs/experiments/voradj_a3_apfnew_sqrtn_20260723/a3_apfnew_sqrtn_4v1_scratch_mix_2m.yaml --device cuda:1
```

Run a short CPU smoke training:

```bash
python3 train.py --config configs/smoke/a3_4v1_cpu_smoke.yaml --device cpu
```

Render/evaluate selected rollouts:

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

## Current Artifacts

The included artifact subset keeps the current A3 best stages `stage1` through `stage6`. Each stage keeps the selected checkpoint, the config used with that checkpoint, summary files, and three mix GIFs for visual inspection.
