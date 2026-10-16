# CoCap baseline candidate search log

**Screening date:** 2026-10-16  
**Repository fact source:** `Jayyeah/cocap-voradj`, branch `main` at `a5814f49fa29d869cdc3fb8d8e0df4722aa11f00`  
**Scope:** literature and public-code screening only; no port, reproduction, or training started.

## Search method

1. Started from CoCap's stated goal: local discovery and shared-policy search/spread, cooperative capture/encirclement after discovery, then return to search for further targets.
2. Checked CoCap's `main` README and sensing/topology note for the current CR-MS + VCT-LS + CE direction and its local sensing/neighbor communication context.
3. Searched scholarly pages (arXiv, IEEE/AAAI/Elsevier proceedings and journal pages) for RL multi-target pursuit/encirclement, explicit scout/exploitation or heterogeneous role policies, and target-allocation/coalition plus geometric encirclement.
4. Checked exact-title and author/project GitHub searches for candidate implementations. “Official code” is assigned only when a paper/project author or lab directly identifies the repository as the implementation. Generic MADDPG/MAPPO packages and unrelated forks are not counted as an official implementation.
5. Recorded mismatches explicitly: full-state/global target knowledge, single-target capture, collision-as-capture, absent search, simulator dependence, and lack of repeated task cycles.

## Query log

Representative queries (search engines may return overlapping results):

- `TERL multi-agent pursuit evasion reinforcement learning encirclement paper`
- `role-based heterogeneous multi-agent reinforcement learning pursuit evasion encirclement target search exploration exploitation`
- `multi UAV reinforcement learning search and encirclement heterogeneous roles pursuer scout paper code`
- `multi-agent reinforcement learning pursuit evasion encirclement target allocation official code`
- `multi UAV multi target encirclement task allocation coalition formation pursuit control paper`
- `multi-target cooperative encirclement pursuit task assignment multiple pursuers geometric control open access`
- `multi-robot multi-target pursuit task allocation coalition encirclement algorithm`
- Exact-title GitHub searches for TERL, Role-based MADDPG, OPEN, DualCL, the k-WTA multi-target encirclement paper, and potential-game multi-target encirclement.
- Second-pass searches for 2026 heterogeneous multi-target encirclement/role-structured RL, Ex-MADDPG target assignment, and exact-title GitHub repositories for each.

## Main findings

- **TERL (IROS 2025):** author-linked public implementation found at `ApricityZ/TERL`. This is the closest published RL task match for multi-target encirclement, target selection, and continuing after one target is encircled. Key contract issue: all non-encircled evaders are included in each pursuer's observation regardless of distance, even though teammates/obstacles are range-limited. Treat this as global target-state access, not matched local search.
- **Role-based MADDPG (ICRA 2023):** exact role mechanism found: a pursuer/scout split, Voronoi-rewarded scouting, target tracking, local neighbor communication, and a centralized critic in training. No official paper implementation was found in exact-title/author GitHub searches. The reported task tracks/collides with targets; it does not establish CoCap's geometric ring capture or post-capture multi-round objective.
- **OPEN (IEEE RA-L 2025):** official public repository found at `thu-uav/Multi-UAV-pursuit-evasion`. It uses a shared MAPPO policy, partial observation, and a prediction-enhanced policy. The task is one evader and multi-UAV capture, with a high-cost Isaac Sim 2022.2.0/OmniDrones stack; no persistent search or multi-target reassignment contract.
- **DualCL (arXiv preprint, 2023/2024):** author-linked public repository found. Shared policy for homogeneous UAVs, multi-UAV capture of one evader, dual training curriculum. Isaac Sim 2022.2.0/OmniDrones makes reproduction cost high; no explicit multi-target encirclement/search cycle.
- **LIPPO (Designs, 2026):** new heterogeneous multi-target encirclement candidate with a three-hunter triangle capture condition and LSTM interception. Episode-start greedy assignment uses distances from all hunters to uncaught targets; no explicit scout role or post-capture search/reassignment. Open paper, but no official code was found.
- **PRTDA (Drones, 2026):** new role-structured CTDE candidate for an airship-guided UAV swarm with multi-target assignment/reassignment. No official repository found. The extra airship provides global coverage/assignment support and changes the information and platform contract, so it is not a fair primary CoCap baseline.
- **Ex-MADDPG (Drones, 2023):** local communication and local target detection; assumes at least two UAVs per target and learns task assignment. Paper reports a Boids formation controller rather than a defined geometric encirclement baseline. No official code was found.
- **Multi-target k-WTA + obstacle-aware particle model (Robotics and Autonomous Systems, 2025):** a strong conventional candidate. It assigns a required number of pursuers to each target and geometrically approaches/surrounds multiple targets, with physical robot validation. No official code was found in exact-title GitHub searches. The method's distance matrix/assignment inputs indicate global or broadly shared target-state information; exact decentralization and sensing assumptions require paper-level confirmation before preflight.
- **Potential-game multi-UAV multi-target encirclement (CCC 2023):** DBSCAN target clustering, potential-game task assignment, and coupled encirclement/tracking control are described. No official repository was found. The public abstract is insufficient to settle sensing/communication and repeated-round details, so it remains a maybe rather than a preferred reproduction.
- **Leader-follower coalition formation (ACC 2018):** targets need not be known before discovery; the first UAV to find one forms a coalition. It addresses task coalition allocation, not encirclement guidance, so it is a component candidate only.
- **Distributed RMPC encirclement-guaranteed pursuit (Robotics and Autonomous Systems, 2025):** useful geometric controller reference for one evader; no multi-target task allocation or official repository found. It is not a standalone CoCap baseline.

## Search exclusions / scope decisions

- Generic OpenAI MADDPG/MPE code is not counted as a reproduction of the role-based ICRA method; it lacks that paper's role rewards, exploration mechanism, and scenario.
- Single-evader pursuit or capture methods are retained only as auxiliary RL/controller candidates, not as primary CoCap end-to-end baselines.
- Multi-UAV coalition papers for generic sensing/communications and unrelated task allocation are excluded unless they also support target detection/tasking and multiple agents per target.
- A 2026 Voronoi “whole-group encirclement” paper was not shortlisted: its target abstraction encircles the target group as a whole, which does not establish per-target cooperative capture or a search-after-capture cycle.
- A 2025 option-based hierarchical SAC encirclement paper was logged as a hybrid lead only: learned assignment and path planning are not a traditional non-RL control baseline, and no author repository was found.

## Screening limitation

Code status is a dated public-web check, not a guarantee that no private, unindexed, or subsequently released code exists. A candidate marked **no official code found** must not be described later as open source without a new source check. The k-WTA candidate's global-information/decentralization contract should be confirmed from the full paper before any formal baseline preflight.

