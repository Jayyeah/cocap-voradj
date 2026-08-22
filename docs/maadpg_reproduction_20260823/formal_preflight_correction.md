# Formal preflight correction audit

The first queues started from commit `2a96e14` were intentionally reclassified
as **formal preflight**, not formal evidence. A final line-by-line acceptance
audit found three objective requirements that the otherwise healthy queues did
not yet satisfy:

1. `rolling_latest` was a full runtime bundle, but `final` was model-only rather
   than an additional full runtime bundle;
2. checkpoint load verified tensor/config shapes but did not explicitly fail
   closed when trainer, replay, learner, environment, and current-observation
   counters disagreed;
3. the queue stopped at a matched 450k environment-step budget instead of
   prioritizing the paper's disclosed 1,500 completed episodes.

Both preflights were allowed to cross warm-up and finish an atomic rolling
checkpoint, then were stopped through the CLI's SIGTERM handler. No checkpoint
was interrupted and no artifact was deleted.

| Variant | Last environment step | Last gradient step | Rolling full size |
|---|---:|---:|---:|
| MADDPG | 31,910 | 21,910 | 30,067,539 bytes |
| MAADPG | 25,969 | 15,969 | 24,896,275 bytes |

The preserved artifacts were moved as a unit from `formal/` to
`formal_preflight_2a96e14/`. Their manifests say
`stopped_with_checkpoint`; they must never enter the formal result table.

## Corrected launch contract

- stop after 1,500 completed episodes, with 450k environment steps retained as
  a fail-safe cap derived from the assumed 300-step horizon;
- every rolling and final checkpoint is a full runtime bundle and explicitly
  declares `contains_replay=true`;
- milestones and convenience final-model files declare
  `contains_replay=false`;
- save/load fails closed unless runtime environment steps, replay total
  insertions/cursor/size, learner update count, environment episode step, and
  current observation all agree;
- every artifact carries an explicit observation/action schema SHA-256;
- the formal evaluator separately emits the assumed Eq.33 paper-terminal
  diagnostic, strict Eq.18, instantaneous Eq.18, 10-step hold, every geometry
  clause, collision provenance, time to capture, action saturation, speed, and
  deterministic/stochastic/guidance modes;
- a per-variant `flock` makes each three-seed supervisor queue idempotent.

## Deferred-signal correction

A second short launch on `43a00e4` validated the new fail-closed counters and
exposed an asynchronous-stop edge case. The original signal handler raised
inside `env.step`; one MADDPG stop landed after the environment advanced but
before runtime/replay accounting (`episode_step=47` versus `runtime=46`). The
full save correctly refused this inconsistent state instead of emitting a false
checkpoint. MAADPG happened to stop at a safe boundary and produced a valid
7,532-step full bundle. Both artifact directories are preserved under
`formal_signal_preflight_43a00e4/` and are not formal evidence.

Signals now only set a deferred request. The trainer completes the current real
transition, replay insertion, optional update, metrics, and periodic checkpoint,
then raises at the loop boundary. An integration test sent SIGTERM during a
MAADPG rollout and recovered a consistent 829-step bundle with
`runtime_steps=replay_insertions=829` and
`episode_length=environment_episode_step=16`; the isolated restored copy
advanced exactly to step 830.

## Storage bound

The observed preflight full bundles were 30.1 MB at 31,910 transitions and
24.9 MB at 25,969 transitions. Linear replay growth therefore implies roughly
425 MB at the 450k safety cap. Keeping both rolling and final full bundles is
about 0.85 GB per run; all six worst-case runs plus model milestones remain
well below the audited 84 GiB free-space headroom. The expected 1,500-episode
stop is earlier, but storage safety is calculated from the cap.

This correction does not change the frozen environment, reward, target policy,
PFM, adaptive gate, network, optimizer, seed partitions, or action semantics.
