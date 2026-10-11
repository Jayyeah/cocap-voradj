# Batch02-A N2/R2 consumption pilot

E1 independently validated and frozen at `8827c11`. Both full contracts already
enter the learner, so this is one shared consumption-qualification trajectory,
with separate N2/R2 ledgers, no causal ablation.

One1024-decision CPU continuation from original E1 checkpoint256; four PPO calls,
24 minibatch updates,4096 active-agent transitions,1024 provider replay decisions.
See [report](../../docs/experiments/BATCH02_A_N2_R2_PILOT_REPORT_20261011_ZH.md),
`manifest.json`, `training_curve.json`, `n2_consumption.json`,
`r2_consumption.json`, `result.json`, and `run_registry.json`.

Frozen runtime source hashes and independent reviewer hash are in
`frozen_E1_binding.json`. Only the pilot tool/config are new implementation;
`pilot_source_delta.patch` records that diff. No src/vendor source changes.

Raw rollout batches and last checkpoint stay in ignored
`runs/batch02_a_pilot_20261011/`. Checkpoint SHA is recorded in receipts; weights
are not published. Four completed episodes ended at boundary, zero capture
events, no recovery phase; one unfinished episode remains censored. No long run.

Execution, after independent E1 PASS:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.:/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/.runtime-deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /usr/local/anaconda3/bin/python tools/run_batch02_a_consumption_pilot_20261011.py --qualification /home/yjq/rl/CoCap1/ac-master-dag-20260921/artifacts/2026-10-11_batch02_e1_independent/independent_report.json --output runs/batch02_a_pilot_20261011
```

The tool refuses to overwrite an existing output. Do not automatically repeat
or extend this completed pilot. This report is not held-out evaluation or
independent pilot validation.
