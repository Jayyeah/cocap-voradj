# MAADPG independent reproduction contract

Status: R-1 paper contract, frozen before any MAADPG training.

This directory is the evidence and decision boundary for the independent
reproduction of Du et al., *An Adaptive Difference Policy Gradient Method for
Cooperative Multi-USV Pursuit...*, JMSE (2026). It does not inherit the CoCap
task, reward, observation, action, oracle, replay, or training semantics merely
because the implementation lives in the same repository.

## Claim layers

Every material choice belongs to exactly one layer:

1. **Paper-primary**: explicitly stated in the paper or directly implied by a
   displayed equation. This is the formal reproduction claim.
2. **Audit-corrected**: a disclosed repair needed to make contradictory or
   incomplete paper text executable. These runs must never be relabelled as
   literal paper reproduction.
3. **CoCap-port**: an optional later portability experiment. CoCap mechanisms
   are prohibited from the Paper-primary implementation and baselines.

## R-1 files

- `paper_spec.md`: facts transcribed from the paper, including equation-level
  semantics.
- `ambiguities.yaml`: contradictions and missing specifications, each with an
  impact and resolution policy.
- `assumptions.yaml`: executable v1 assumptions. Values here are hypotheses,
  not reconstructed paper facts.
- `contract_matrix.csv`: requirement/source/status/implementation contract.
- `paper_to_code_mapping.md`: planned code ownership and anti-contamination
  boundary.
- `evidence_index.md`: immutable source identity and historical safeguards.

## Freeze rule

After the first matched pilot starts, any change to an environment, reward,
observation, target-policy, training, evaluation, or gate field in these files
requires:

- a new `spec_version`;
- a new run family (no in-place reinterpretation);
- a reason and affected run IDs in the run ledger;
- rerunning contract tests before training resumes.

The first executable version is `maadpg-paper-v1`. It is deliberately strict:
fixed three-pursuer slots, continuous normalized world-frame acceleration,
independent local actors, joint centralized critics, replay of the exact action
executed by the environment, and actor-only formal evaluation.

## Non-claims

The paper does not disclose the trained target checkpoint, target training
procedure, reward coefficients, several geometry thresholds, episode horizon,
random seeds, exploration schedule, or complete update schedule. Therefore no
run using `maadpg-paper-v1` may claim bitwise or checkpoint-level replication of
the authors' experiment. The supported claim is an independent, traceable
reproduction under the explicitly frozen assumptions in this directory.
