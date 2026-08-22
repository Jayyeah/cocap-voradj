# Evidence index and initial fact snapshot

Recorded: 2026-08-23 (Asia/Shanghai).

## Primary evidence

| Evidence | Identity | Coverage |
|---|---|---|
| Local JMSE paper PDF | SHA-256 `956fb59036dcba020b75bda0bc12d75c50204aaf6cfe49bf53c1eaf48d28091f`; 18,007,359 bytes; 29 pages | equations, tables, figures, algorithm |
| `docs/MAADPG_REPRODUCTION_PRINCIPLES_AND_START_PROMPT_20260823_ZH.md` | repository source at task start | reproduction boundary and execution ladder |

The official MDPI page was unreachable from the execution environment (`Proxy
CONNECT aborted`). The local PDF is therefore the primary paper artifact for
this run. Its checksum is pinned above; a changed PDF must create a new spec
version.

All 29 PDF pages were text-extracted and reviewed. Pages containing dynamics,
success, PFM/gate, observations, reward, parameters, evaluation, and overhead
were additionally rendered for layout-aware inspection. This matters because
several equation labels and Algorithm 1 expressions are internally inconsistent.

## Historical failure evidence reviewed in full

| Document | Safeguard carried forward |
|---|---|
| `docs/CONTINUOUS_ACTION_MARL_EXPERIMENT_TRACKER_20260804_ZH.md` | keep action semantics aligned end-to-end; use fixed slots and executable diagnostics |
| `docs/CONTINUOUS_ACTION_AXAY_REFACTOR_20260805_ZH.md` | world-frame normalized acceleration must be identical at actor, critic, replay, and environment boundaries |
| `docs/CONTINUOUS_ACTION_MARL_EXECUTION_LOG_20260805_ZH.md` | distinguish terminated/truncated and retain terminal transitions |
| `docs/ALL_AGENT_MASAC_OLDMIX_ABLATION_20260810_ZH.md` | stochastic rare capture is not evidence of a deterministic actor policy; report collision modes |
| `docs/SAC_HEALTH_AUDIT_20260811_ZH.md` | finite losses are insufficient; monitor Q ordering, TD error, gradients, saturation, and clipping |
| `docs/CAPTURE_DEBUG_AND_PURE_CAPTURE_ABLATION_20260815_ZH.md` | explicitly separate paper-discrete from swept collision and test step ordering |

No legacy result is evidence that MAADPG works. It is used only to prevent
known implementation and interpretation failures.

## Repository isolation

| Field | Value |
|---|---|
| Source worktree at audit | `/home/yjq/rl/CoCap1/cocap-voradj-allagent-oldmix` |
| Source branch/commit | `ablation/all-agent-oldmix-20260810` at `3ee4d94` |
| Independent worktree | `/home/yjq/rl/CoCap1/cocap-voradj-maadpg` |
| Independent branch | `repro/maadpg-20260823` |
| Base commit | `3ee4d94` |
| Training process at audit | none detected |

The source worktree contained user-owned untracked artifacts and was not
modified. The reproduction is developed only in the independent worktree.

## Compute/storage snapshot

- two NVIDIA RTX A6000 GPUs, each approximately 49 GiB and idle at audit;
- PyTorch `2.9.1+cu128`, CUDA available;
- approximately 114 GiB RAM free and swap unused;
- root filesystem approximately 85 GiB free at audit;
- legacy artifacts occupy roughly 96 GiB in the source tree.

Replay/checkpoint retention must be bounded because the paper replay capacity is
one million joint transitions and root-disk headroom is limited. Do not delete or
move user artifacts without explicit authorization.

## Internal consistency observations

- Eq. 12 is semi-implicit although called explicit Euler.
- Eq. 13 norm bounds conflict with component-wise action clipping.
- Eq. 24 defines 26 inputs while network prose states 28.
- a 28-wide critic conflicts with the centralized joint critic definition.
- Algorithm 1 swaps actor/PFM labels and reverses the Eq. 21 improvement sign.
- Eq. 18 and Eq. 33 do not fully specify the same success predicate.
- the target policy and many reward/geometry/training constants are unreleased.
- reported evaluation timing/overhead values are internally inconsistent and are
  treated as diagnostic context rather than a reproduction acceptance target.

## R-1 acceptance evidence

R-1 is complete only when:

1. YAML and CSV artifacts parse;
2. every ambiguity has a resolution and materiality;
3. every assumed material field links to an ambiguity;
4. every paper-primary invariant has a planned code owner and test;
5. the isolated branch contains no training artifact and no training was started.
