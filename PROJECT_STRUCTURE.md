# Project Structure

## Main Layout

- `src/cocap_voradj/envs/`: base CoCap environment and Voronoi-adjacency environment.
- `src/cocap_voradj/dynamics/`: pursuer, evader, robot, and perception dynamics.
- `src/cocap_voradj/control/`: APF evader controller.
- `src/cocap_voradj/models/`: IQN policy/value network.
- `src/cocap_voradj/training/`: replay buffer and trainer.
- `src/cocap_voradj/utils/`: logger and small shared helpers.
- `tools/`: command-line rollout, screening, and A3 curriculum supervision tools.
- `configs/experiments/`: current A3 experiment configs.
- `artifacts/`: curated A3 best checkpoints/configs/summaries/GIFs.

## Not Included

- Historical experiment-generation scripts not needed for the current A3 line.
- Large training outputs and screening worker details.
- Core internal docs.

## Project Conventions

- Configs use `output_root: runs`.
- Configs use `iqn.update_rule: distributional_iqn`.
- Runnable tools resolve paths relative to the repository root.
- Curated artifacts keep only selected checkpoints and a small visual sample.

## Next Cleanup Candidates

- Add a small pytest suite for config loading, environment reset, APF action selection, and one checkpoint forward pass.
- Decide whether curated checkpoints/GIFs should be tracked directly or moved to release assets later.
