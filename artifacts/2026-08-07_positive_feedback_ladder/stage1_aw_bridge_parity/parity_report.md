# Stage 1 AW Bridge Parity Report

- config: `/home/yjq/rl/CoCap1/cocap-voradj/configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/stage1_4p1e1obs_scratch2m.yaml`
- checkpoint: `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-08-02_three_line_stage_best_20rollout10gif/crms_supportapproach_4v1_s1_step_2000000/step_2000000.pt`
- seeds: [2026081201, 2026081202, 2026081203]
- max steps: 400

## Gate

- event parity rate: 1.000
- capture parity rate: 1.000
- collision parity rate: 1.000
- state error max mean: 0.0000
- state error max p95: 0.0000
- reward abs diff max mean: 0.0000
- gate_passed: True

## Notes

- Action indices are identical by construction (legacy IQN -> exact legacy action list -> continuous API).
- Old discrete path integrates position explicitly then speed/theta; bridge path uses the 10-substep trapezoidal AW integrator, so small numerical drift is expected and must not change event-level behavior.
