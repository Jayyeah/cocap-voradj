# Project execution constraints (user, 2026-09-07)

- Use physical GPU1 only for GPU eval, smoke, and training, in a serial queue.
  Set CUDA_VISIBLE_DEVICES=1 and use process-local cuda:0. Do not claim GPU0
  or stop unrelated processes on either GPU. CPU-only work may run in parallel.
- Do not launch long training, teacher-KL, TD3/SAC, new Transformers, support
  reward retraining, or MAPPO-VXY without a new user decision after P0.
- Default to no sub-agents. If explicitly needed and authorized, use Luna Max.
- If only a GPU job remains, record progress, ETA and NEXT WAKE-UP in the
  ledger, commit/push the reviewable changes, and end the session instead of
  long polling. Never mistake a queued/running evaluation for a completed Gate.
- Runtime semantic assertions, effective forward mode and likelihood checks
  are required before formal experiments. YAML diffs alone are insufficient.
