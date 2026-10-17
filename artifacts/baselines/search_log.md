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

---

## Round 2 — focused traditional review, 2026-10-17

### Recovery and branch lineage

- Refetched `docs/baseline-screening-20261016`; remote HEAD remains `94753f9fd82d137af7ff9930634788ffeba6f3e2`. Inspected history: 94753f9 → fb70b53 → a5814f4 → e851e17.
- Created isolated `docs/baseline-focused-review-20261017` from that exact parent. Read all four Round 1 deliverables in full and preserved all 12 IDs and candidate snapshots in the same registry. New traditional leads use TR-06 onward; no second candidate registry.
- Refetched `docs/20261016-iqn-z-solid-asset-audit`, HEAD `3038ef5cd728acdd302772ca08bf10060685af16`. Read Benchmark Contract, Asset Index and Readiness in full. Latest recovered contract is still a draft; historical main a5814f4 is not Z05 runtime.
- No applicable AGENTS.md was present in the baseline tree or workspace ancestors. Read repository `docs/GIT_WORKFLOW.zh-CN.md`. Work remains documentation/metadata only; no baseline implementation, training, rollout or preflight execution.

### Directed search coverage

Focused on 2022–2026 traditional assignment/coalition plus genuine geometric encirclement, with classic sources only where needed. Queries were run through public web search, publisher/author pages, arXiv, Crossref publisher deposits, and public GitHub repository metadata search. Search coverage is targeted, not an assertion of exhaustive indexing of every named journal.

Representative exact queries and follow-ups:

- `"A competition-based scheme" "k-WTA" pdf`; exact title plus `filetype:pdf`, author Ning Tan, `OAPM`, `distance matrix`, software/code.
- `"Cooperative Control of Multi-UAV" "Potential Game" pdf`; exact title, DOI, IEEE document 10240696 and CCC official contents.
- `"Accelerated K-Serial" coalition`; exact title plus DOI/GitHub; read arXiv v3 and v1 separately.
- `"Distributed cooperative pursuit with encirclement guarantee" pdf`; author/project/code; recovered IROS2021 predecessor arXiv:2108.07445, kept separate from RAS2025.
- `"multi-robot" "search" "encirclement" decentralized algorithm 2024 2025`; `"multi-target" "encirclement" "github" -reinforcement`.
- `"distributed" "search" "encirclement" "2024" -reinforcement -learning -researchgate -survey`; analogous 2025 query.
- `"Multitarget Pursuit-Evasion Based on Distributed and Competitive Mechanisms" pdf`; exact title and DOI → TSMC2024 DMK-WTA/PPM.
- `"Distance-based Multiple Non-cooperative" code`; arXiv2409.15840 full text, exact-title Crossref → TCST2025 formal record, distinct reading version.
- `"Multi-Robot Target Monitoring and Encirclement" github`; author/institution publication pages and Crossref → TRO2026 formal publication; read arXiv2024 version, not claimed typeset final.
- `"Self-Organized Multitarget Pursuit"` plus PDF/GitHub; Wiley DOI metadata; full-text403.
- `"Wolf-Pack-Inspired Distributed Encirclement"` → Biomimetics2026 full PDF and supplementary-results description. Examined Algorithms1–3 and global Γ/argmax/TopK requirements.
- `"Dynamic Robust Pursuit of Multiple Evaders"` → JAS2026 official PDF plus supplement; exact-title code searches; followed supplement's SDQP link.
- `"multi-target" "encirclement" "TASE" search`; `"multi-target" "encirclement" "Automatica" assignment`; final follow-ups on collaborative hunting and target guarding. No new RL training candidates were sought.
- Exact titles of all three requested surveys; official Crossref/publisher metadata; PDF/section reading as available.

### Full-text access outcomes and limits

- Downloaded and inspected: ACC2018 author PDF, KS-COAL arXiv v3 and v1, distance-based arXiv manuscript, triggered-feedback arXiv2024, RMPC2021 predecessor, GB-VCE formal PDF, JAS2026 main and supplement, and Robotics2025 coalition survey.
- JAS2026 PDF was obtained through its normal public journal `article/exportPdf` link, and its supplement from the publisher's link. No paywall or access check was bypassed.
- MK-WTA2025, CCC2023 potential-game, DMK-WTA2024 and RMPC2025 full text were not obtained. Publisher snippets/abstracts are not sufficient for complete formulas, centralization or repeated lifecycle claims.
- Elsevier API HTTP200 default responses contained only metadata. Explicit `view=FULL` returned401; CJA PDF request406 and website PDF403. Neurocomputing survey relevant full chapters/PDF and CJA review PDF pages remain unverified. CJA accessible HTML sections beyond the abstract were read; no fake PDF pagination was assigned.
- Wiley PGM PDF403. Exact metadata was subsequently verified against Crossref; no full mechanism inferred from its abstract.
- PDF byte hashes, download statuses, source versions and metadata are in `focused_review_source_manifest_20261017.json`. Cached PDFs remain outside Git at `/tmp/cocap-round2-sources`; they are not redistributed.

### Code evidence

- Exact GitHub repository metadata searches included `"k-WTA" pursuit`, `"KS-COAL"`, `"DMK-WTA"`, `"Dynamic Robust Pursuit"`, `"125684" pursuit`, exact CCC title, exact PGM title, `"encirclement" "Pichierri"`, `"encirclement" "Notarstefano"`, `"coalition" "Junfeng"`, and `"GB-VCE"`. Zero results are bounded repository-search evidence, not global source-code absence. Generic OAPM string results were unrelated and rejected.
- Cloned official TERL read-only at `143359b2722d49c29b4fecc0ad1fd8d46326e45a`. Static source shows no target range check in `robots/pursuer.py`307–330, unlike teammate/obstacle loops; entity-count cap still applies. MIT, configs/entry points and trained model files verified as present, without running/loading them.
- Cloned independent SDQP at `78e9860508cc7c191870dea79e63b1e7c45e2296`. MIT is in the header; C++11/Eigen3/example present. This only supplies a low-dimensional strict QP component, not the JAS controller LP or full method.
- No complete official traditional implementation was verified. No generic allocation repo, simulation-results CSV, middleware framework or custom ring was promoted to a complete paper implementation.

### Decisions and corrections to Round 1

- Global Strong first recommendation changed to JAS2026 Dynamic Robust Pursuit: complete formal equations/supplement, dynamic assignment, multi-target slots and ≥3-pursuer simulation. Whole-method code NOT_FOUND; conditional native/AW9 preflight only.
- Final author-PDF follow-up added **TR-13 CEO-DMPC, RA-L2024** as Global Strong #2: read the 8-page paper, including exact grouping/slot MILPs, winding-number capture and DMPC QP. It assumes full real-time pursuer communication (p.6); circular patrol is not autonomous area search; Gazebo is not hardware. Python3/Quadprog/OR-Tools dependencies are verified in the paper, whole-method code NOT_FOUND.
- MK-WTA remains a Global full-text reserve behind those two. Round1 all-pair distance-matrix claim is now explicitly `UNVERIFIED_ROUND1_LEAD`; neither a central server nor full global matrix was established from a read equation.
- KS-COAL non-learned backbone is closest in continuous task organization. Full Accelerated variant includes supervised HGAN; scout-team communication is unrestricted, SWAT is local. Resource encirclement is not target normal capture; v1 pure-pursuit details are not silently substituted for missing final supplement.
- GB-VCE is a local role/geometric component reserve, not a multi-target release/search method; global reduction/message details need explicit resolution.
- Distance-based method was initially a preprint lead; final exact-title metadata check confirmed **TCST33(1):261–273, 2025**, DOI10.1109/TCST.2024.3469032. Registry and documents corrected before publication. Its two-drone geometry still fails the CoCap three-pursuer contract.
- TERL mandatory decision preserved. Final Crossref check resolved IROS2025 DOI10.1109/IROS60139.2025.11246832, pp.13438–13445. Fixed-role RL stays Related Work unless reliable implementation/contract is found.
- TR-05 lifecycle is **planned, not implemented/validated**, correcting overly affirmative Round1 wording. All GO decisions refer only to a recommended future preflight.
- Late full texts screened out: Chaib et al. RA-L2026 socially aware multi-target method encloses the group QuickHull rather than allocating ≥3 per target; Zhou et al. JAS2024 OAVC/Hungarian method concerns a single evader and assumes all real-time robot positions. They are excluded source leads, not silently promoted to full CoCap candidates. Their metadata/hash are in the same source manifest. Exact CEO-DMPC/title repository searches returned no matching public repo metadata; this is a bounded negative search.

### Reporting and synchronization

Six requested deliverables are under `docs/baselines/`; current facts remain in the original registry, with Round1 archival snapshots and new `focused_review_v2` fields. Old master/cards carry links to the new decision. Verification covers JSON/CSV consistency, citation/link targets, old-ID preservation and documentation-only changes. Publication commit and remote HEAD are verified separately and reported in the delivery response; no main merge is requested or performed.

