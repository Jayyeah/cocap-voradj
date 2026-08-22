# Matched 25k pilot audit

Recorded: 2026-08-23 (Asia/Shanghai)
Training seed: `2026082301`
Budget: 25,000 real environment steps per variant
Warm-up: 10,000 transitions; updates thereafter: one batch of 1,024 per step

The pilots used the same environment, target policy, replay, network,
exploration, optimizer, seed, and step budget. The only algorithmic difference
was the Paper-primary PFM/adaptive gate in MAADPG. Training-episode outcomes
include OU exploration (and, for MAADPG, adopted guidance) and are not formal
policy results.

## Numerical and replay health

| Measure | MADDPG | MAADPG |
|---|---:|---:|
| Environment steps | 25,000 | 25,000 |
| Gradient steps | 15,001 | 15,001 |
| Episodes completed | 269 | 295 |
| Non-finite monitored values | 0 | 0 |
| Final critic losses (agents 0/1/2) | 1.513 / 0.927 / 1.014 | 1.970 / 1.156 / 1.799 |
| Final replay-Q means | 14.310 / 12.337 / 12.703 | 18.523 / 17.124 / 18.232 |
| Peak sampled absolute TD error | 32.923 | 33.725 |
| Peak sampled critic gradient norm | 49.135 | 93.657 |
| Peak sampled actor gradient norm | 1.514 | 2.137 |
| Silent gradient clipping | none | none |
| Maximum executable action norm | <= 1 | <= 1 |

The gradient peaks are large enough to remain a formal-run diagnostic, but they
are finite and did not produce parameter, Q, loss, TD, or action non-finites.
Adding the historical `0.5` clip would silently clip nearly the entire useful
signal and is not justified by this pilot.

Offline Q-ranking on a restored batch of 1,024 transitions gave:

| Variant | Policy minus replay Q (agents 0/1/2) | Policy minus shuffled-action Q |
|---|---|---|
| MADDPG | +0.825 / +0.733 / +0.758 | +0.989 / +0.916 / +0.907 |
| MAADPG | +0.853 / +0.883 / +0.920 | +0.837 / +1.020 / +0.989 |

Thus the critics were not merely finite: they ranked current policy actions
above replay and state-mismatched shuffled actions on the audited batches.

## Behavioral outcomes

| Measure | MADDPG | MAADPG |
|---|---:|---:|
| Training strict captures | 0 | 0 |
| Mean training episode length | 92.90 | 84.73 |
| Boundary-collision episodes | 170 | 171 |
| Obstacle-collision episodes | 48 | 28 |
| Teammate-collision episodes | 27 | 3 |
| Target-hard-collision episodes | 24 | 107 |
| PFM adoptions / comparisons | 0 / 0 | 14,416 / 75,000 (19.22%) |
| Actor-only tuning success | 0/20 | 0/20 |
| Actor-only tuning collision | 20/20 | 20/20 |
| 0/20 Wilson 95% upper bound | 16.11% | 16.11% |

The early gate reduced obstacle and teammate collision episodes but sharply
increased hard target contacts. This is a material warning consistent with an
aggressive target-attraction PFM and a myopic one-step reward comparison. It is
not hidden or reclassified as capture.

## Restore evidence

Both final rolling checkpoints were loaded onto their assigned CUDA device and
advanced in isolated copies:

- MADDPG: step 25,000/gradient 15,001 -> step 25,001/gradient 15,002;
- MAADPG: step 25,000/gradient 15,001 -> step 25,001/gradient 15,002.

This CUDA audit exposed and fixed one infrastructure defect: `map_location`
moved the stored CPU RNG byte tensor to CUDA, while `torch.set_rng_state`
requires a CPU byte tensor. Restore now explicitly moves CPU and CUDA RNG state
payloads to CPU before calling the PyTorch RNG restoration APIs. A CUDA
regression test covers the corrected path.

## Freeze decision

The pilot does **not** establish a successful deterministic policy and no result
is promoted to the formal table. It also does not meet a pre-launch blocker:

- the strict environment has a safe oracle capture in 86 steps;
- both variants crossed warm-up and completed 15,001 stable updates;
- actor, critic, replay, target, gate, and checkpoint contracts passed;
- both restored checkpoints produced the exact next executable training step;
- the observed behavioral failure is precisely what the 450k matched long run
  is meant to resolve or falsify.

Therefore `maadpg-paper-v1` is frozen without post-hoc reward, gate, PFM, target,
or collision changes. The formal run retains explicit non-finite tripwires,
unclipped gradient/Q/TD/action diagnostics, rolling full checkpoints, and
actor-only validation. If long training remains collision-only, that negative
result will be reported under the disclosed assumptions rather than repaired by
importing CoCap mechanisms.
