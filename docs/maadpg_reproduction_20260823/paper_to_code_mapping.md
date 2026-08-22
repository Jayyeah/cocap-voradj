# Paper-to-code ownership map

This map is prospective during R-1. Paths become binding when R0 starts. The
MAADPG package must remain self-contained under `src/maadpg_reproduction/` so a
reviewer can audit it without traversing the legacy CoCap/MASAC training stack.

| Contract area | Planned module | Public object | Required tests |
|---|---|---|---|
| immutable typed config | `config.py` | `MAADPGConfig` | schema, hash, paper constants |
| geometry | `geometry.py` | hull/gap/clearance/ray helpers | analytic and threshold tables |
| dynamics | `dynamics.py` | `integrate_usv` | hand-computed golden transitions |
| lidar | `lidar.py` | `cast_lidar` | boundary/circle/tangent/no-hit cases |
| strict success | `success.py` | `evaluate_capture` | each Eq.18 clause toggled independently |
| reward | `reward.py` | `compute_reward_terms` | every branch and aggregate golden values |
| environment | `env.py` | `MAADPGPursuitEnv` | reset/step/API/termination invariants |
| target plugin | `target_policy.py` | `TargetPolicy` protocol and frozen plugins | seed, snapshot, restore, determinism |
| PFM | `pfm.py` | `potential_field_actions` | analytic signs, symmetry, singularity floors |
| counterfactual gate | `gate.py` | `AdaptiveDifferenceGate` | sign, threshold, mutation, RNG, order |
| replay | `replay.py` | `JointReplayBuffer` | fixed slots, executed action, terminal storage |
| actors/critics | `networks.py` | `Actor`, `JointCritic` | shapes, locality, cross-agent critic dependence |
| MADDPG learner | `maddpg.py` | `MADDPGLearner` | target equations, gradient ownership, updates |
| rollout | `trainer.py`; `evaluation_contract.py` | train and full evaluation modes | actor-only eval, no lost terminal transition |
| checkpoint | `checkpoint.py` | atomic rolling/final full and model-only checkpoints | split-run bitwise/numerical equivalence |
| diagnostics | `diagnostics.py` | structured health records | schema and finite/non-finite tripwires |
| CLI | `cli.py` | train/evaluate/inspect commands | smoke invocation and config provenance |

## Data-flow contract

```text
seeded scenario + target/RNG state
               |
               v
      local obs[3,26] --actors--> base joint action[3,2]
               |                         |
               |                         +-- PFM candidates[3,2]
               |                                   |
               |                            snapshot counterfactuals
               |                                   |
               +------------------------- gate decisions
                                                   |
                                      canonical executed action[3,2]
                                                   |
                             real environment step exactly once
                                                   |
                     joint replay transition with fixed agent slots
                                                   |
                          three independent joint critics + actors
```

During formal validation/test the PFM and gate branch is absent: actor outputs
are canonically projected and passed directly to the real environment.

## Anti-contamination boundary

Paper-primary modules may not import legacy environment, reward, observation,
policy, replay, rollout, or controller code. The only permitted reuse is a small
generic utility whose behavior is independently contract-tested (for example,
atomic file replacement or RNG state serialization). Any such reuse must be
listed here before use.

Initially permitted external dependencies:

- Python standard library;
- NumPy;
- PyTorch;
- PyYAML only for reading frozen configuration if already installed;
- pytest for tests.

Explicitly forbidden imports/configuration in Paper-primary:

- Voronoi or communication topology;
- ORCA, action shield, feasibility or safety projection;
- IQN, SAC/MASAC entropy machinery;
- CoCap capture oracle, reward, observation compression, or curriculum;
- legacy continuous-action adapters whose replay action differs from the action
  actually applied by the environment.

## Source-to-object links

| Paper item | Code field/object |
|---|---|
| Eq. 12 | `dynamics.integrate_usv`; `config.dynamics` |
| Eq. 13 | `dynamics.project_unit_disk` |
| Eq. 18 | `success.evaluate_capture` |
| Eq. 19 | `pfm.potential_field_actions` |
| Eq. 20 | `pfm.normalize_force` |
| Eqs. 21-22 | `gate.compute_delta`; `gate.decide` |
| Eq. 24 | `env.pursuer_observation` |
| Eq. 25 | `env.target_observation` |
| Eqs. 26-27 | `dynamics.integrate_usv` |
| Eqs. 28-33 | `reward.compute_reward_terms` |
| Algorithm 1 | `rollout.training_action_with_gate` |
| Table 1 | `config.networks`; `config.training` |
| Tables 2-3 | `config.scenario`; `config.dynamics` |

## Version provenance

Every runtime artifact must carry:

- `spec_version` and canonical config SHA-256;
- git commit and dirty flag;
- run ID, algorithm variant, training seed, scenario seed partition;
- paper PDF SHA-256;
- exact target-policy ID;
- checkpoint kind (`full_runtime` or `model_only`);
- environment step, gradient step, and episode counters.
