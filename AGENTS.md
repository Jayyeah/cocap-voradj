# Project execution constraints (user, 2026-09-08, supersedes GPU1-only)

- Both physical GPU0 and GPU1 may be used for explicitly gated independent
  CoCap tasks. Set CUDA_VISIBLE_DEVICES to the assigned physical GPU and use
  process-local cuda:0. Do not claim or stop unrelated processes.
- Current route: C0 Forward Final qualification -> C1 full-task dataset audit
  -> C2 frozen-backbone categorical distillation -> C3 paired full-task gate.
  Old Pure-Capture A/B are OPTIONAL_DIAGNOSTIC, not required gates.
- Forward Final inherits canonical main reward/information/topology/AW9/phase
  and changes collision only to synchronized_swept_v1. Historical main remains
  the legacy-collision reproduction reference.
- D Direct PPO / critic warm-up may run only after C3 PASS, same BC parent,
  first 25k -> 50k budgets. No automatic long extension, teacher-KL, continuous
  MAPPO, TD3/SAC, support reward retraining, or MAPPO-VXY.
- Default to no sub-agents. If explicitly needed and authorized, use Luna Max.
- Once independent CPU work is complete, do not wait/poll indefinitely for
  GPU jobs. Record PID/step/checkpoint/throughput/ETA/NEXT WAKE-UP for each job,
  commit/push and end the session. A running job is never a completed Gate.
- Formal jobs require resolved/runtime semantic assertions for reward,
  sensing, action physics and collision; requested settings that do not reach
  live consumers must fail before training. Preserve valid checkpoints.
