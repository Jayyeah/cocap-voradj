# E1 CPU evidence

Result: `E1_ENGINEERING_CLOSED_LOOP_PASS`; independent validation absent, formal `E1_PASS=false`. See [Chinese report](../../docs/experiments/BATCH02_E1_BRIDGE_REPORT_20261011_ZH.md).

`source_provenance.json` pins successful IQN task code and unchanged learner sources. `provider_parity.json` preserves same-state fixture inputs/results. `ppo_update.json` preserves one real256-decision update. `expected_continuation.json` / `checkpoint_restore.json` prove fresh-process16-decision continuation and update equality. `boundary_wiring.json` checks CE hold and pre-reset time-limit bootstrap without PPO. `network_binding.json` lists the two authorized input-weight shape changes. The original fixture failure and its single expectation correction are retained.

The checkpoint is local and ignored by Git: `runs/e1_cpu_closed_loop_20261011/resume.pt`, SHA256 `e69025598651c38c01eb2600ca6d476ee472d3c550b7d2a9efcfeac5e0775190`.

The verifier reads the original successful IQN worktree and the previously delivered central fixture archive by absolute paths; these are local verification dependencies, not training dependencies. Runtime only requires this repository's existing CoCap sources/config and ordinary TERL/MAPPO dependencies.

From `/home/yjq/rl/CoCap1/terl-mappo-batch02-e1-20261011`:

```bash
PYTHONPATH=src:.:/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/.runtime-deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /usr/local/anaconda3/bin/python tools/verify_batch02_e1_20261011.py --output artifacts/2026-10-11_batch02_e1
PYTHONPATH=src:.:/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/.runtime-deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /usr/local/anaconda3/bin/python tools/verify_batch02_e1_boundaries_20261011.py
```

Do not repeat the finite optimizer smoke automatically or treat it as formal training/evaluation evidence. No long-run entry point or budget is provided.
