# IQN ROLE-token vs Z-token matched scratch 200k report

Date: 2026-09-19

## Executive summary

Both matched arms completed 200,000 environment steps and all eight formal evaluations. The dual supervisor completed without contract drift, non-finite metrics, checkpoint corruption, wrong-GPU execution, or restart.

- Z-token is decisively stronger for capture at 200k: 85% pure-capture success and 90% mixed-scene capture, versus 0% for ROLE-token.
- ROLE-token is substantially stronger on final pure-coverage geometry: CE RMS 0.129 versus 0.235 and area CV 0.366 versus 0.502.
- Neither arm solves capture -> release -> coverage recovery. Both have 0% mixed safe-complete and 0% post-capture CE at 200k.
- Both curves are non-monotonic, so checkpoint trends matter more than terminal metrics alone.

## Contract and completion status

| Item | ROLE-token | Z-token |
|---|---:|---:|
| Branch | `experiment/iqn-role-token-scratch-20260918` | `experiment/iqn-z-token-scratch-20260918` |
| Training HEAD | `0d9df2bb5f1cbda2cf9af57173f474d94dccfec3` | `fd7f7f89d540d4bfa25ff093cc33fe94d2f23ad5` |
| Final step | 200,000 | 200,000 |
| Completed | 2026-09-19 06:09:41 +08:00 | 2026-09-19 06:51:53 +08:00 |
| Wall time | 4h 24m 42s | 5h 17m 28s |
| Optimizer updates | 49,304 | 49,278 |
| Final replay size | 800,000 | 800,000 |
| Final loss / EMA | 27.725 / 24.985 | 17.321 / 18.814 |
| Final epsilon / LR | 0.38 / 3e-5 | 0.38 / 3e-5 |
| Final resumable checkpoint | PASS, step 200k | PASS, step 200k |

NormSense-V2, environment, reward, action space, trainer, replay, evaluator, seeds, and physical-only friend ordering are matched. Non-token config difference count is zero. Initial parameters are bit-exact with SHA `22a7deaa4b7265909a8b6c340f84d8741b69194fd3cc5c643f487ddeb3908ec8`. Supervisor final state is `complete`, `fail_reasons=[]`, with zero restarts.

## Formal evaluation trend

Each checkpoint uses identical fixed seeds and 20 episodes per scene. CE RMS is lower-is-better.

### ROLE-token

| Step | Strict CE | CE RMS | Capture | Mixed capture | Safe complete |
|---:|---:|---:|---:|---:|---:|
| 25k | 0% | 0.240 | 0% | 0% | 0% |
| 50k | 0% | 0.261 | 5% | 0% | 0% |
| 75k | 0% | 0.233 | 30% | 35% | 0% |
| 100k | 0% | 0.228 | 55% | 45% | 0% |
| 125k | 0% | 0.177 | 30% | 40% | 0% |
| 150k | 5% | 0.113 | 45% | 55% | 0% |
| 175k | 0% | 0.147 | 15% | 15% | 0% |
| 200k | 0% | 0.129 | 0% | 0% | 0% |

ROLE learns useful coverage geometry and briefly learns capture, but capture regresses after 150k. The best coverage checkpoint is 150k; the best pure-capture checkpoint is 100k. The terminal policy is coverage-biased and no longer captures in formal evaluation.

### Z-token

| Step | Strict CE | CE RMS | Capture | Mixed capture | Safe complete |
|---:|---:|---:|---:|---:|---:|
| 25k | 0% | 0.236 | 0% | 0% | 0% |
| 50k | 0% | 0.166 | 0% | 10% | 0% |
| 75k | 0% | 0.208 | 35% | 45% | 0% |
| 100k | 0% | 0.120 | 35% | 20% | 0% |
| 125k | 0% | 0.304 | 85% | 70% | 0% |
| 150k | 0% | 0.273 | 20% | 5% | 0% |
| 175k | 0% | 0.252 | 25% | 30% | 0% |
| 200k | 0% | 0.235 | 85% | 90% | 0% |

Z develops a strong capture mode at 125k and 200k, while coverage quality degrades after its 100k minimum CE RMS. This supports capture specialization, not stable monotonic learning.

## Final 200k comparison

| Metric | ROLE-token | Z-token | Result |
|---|---:|---:|---|
| Coverage strict CE | 0% | 0% | Neither passes strict coverage |
| Coverage CE RMS | **0.129** | 0.235 | ROLE better |
| Coverage area CV | **0.366** | 0.502 | ROLE better |
| Coverage collision | 0% | 0% | Tied |
| Pure-capture success | 0% | **85%** | Z better |
| Ring-3 reached | 0% | **90%** | Z better |
| Capture collision | 100% | **15%** | Z better |
| Mixed capture | 0% | **90%** | Z better |
| Mixed post-capture CE | 0% | 0% | Unsolved |
| Mixed safe-complete | 0% | 0% | Unsolved |

Z semantics remain correct at 200k: direct-visible `z=1` is 100%, pure-coverage z is exactly zero, maximum lineage hop is 2, and measured post-capture release is 22 seconds in 17 episodes. One captured episode never released. Correct propagation therefore does not by itself produce coverage recovery.

## Policy behavior and conclusion

The final greedy policies are highly concentrated. ROLE actions 3 and 5 account for about 79.2% of decisions; Z action 4 alone accounts for 85.9%. This is consistent with specialization and checkpoint instability. Z's mode is effective for capture but weak for coverage; ROLE retains better coverage geometry but loses capture.

Under this exact matched contract, continuous propagated Z evidence is substantially more learnable for capture than binary local ROLE evidence. Binary ROLE evidence retains a better terminal coverage policy. Neither representation learns the complete lifecycle, so neither is an overall winner.

Recommended checkpoint interpretation:

- Z 200k: strongest capture checkpoint.
- ROLE 150k: strongest coverage checkpoint; prefer it over ROLE 200k for coverage.
- Do not label either as a full-lifecycle solution.
- Before changing the scientific variable, run additional seeds and investigate action-mode collapse and post-capture recovery credit assignment.

## Sources

- `artifacts/2026-09-18_iqn_role_token_scratch/trend_ledger.json`
- `artifacts/2026-09-18_iqn_z_token_scratch/trend_ledger.json`
- `artifacts/2026-09-18_iqn_role_token_scratch/evaluations/step_*/report.json`
- `artifacts/2026-09-18_iqn_z_token_scratch/evaluations/step_*/report.json`
- `artifacts/2026-09-18_iqn_role_z_matched_supervisor/status.json`
- `artifacts/2026-09-18_iqn_role_z_matched_supervisor/matched_ledger.jsonl`
