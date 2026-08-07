# Legacy IQN Encoder Compatibility Report

- checkpoint: `/home/yjq/rl/CoCap1/cocap-voradj/artifacts/2026-08-04_crms_vctls_ce_final/checkpoints/stage1_4v1_step_2000000.pt`
- SHA256: `2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89`
- exact_shape_transfer_possible: **True**
- loaded keys: 65
- missing: []
- unexpected: []
- shape mismatch: []
- loaded parameter ratio: 1.000000

Only `encoders.*`, `type_embedding.*`, `transformer.*` are eligible. IQN quantile/action/gate heads are not loaded.
