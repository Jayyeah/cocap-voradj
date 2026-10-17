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

## Round 3 — Architecture-first traditional replacement (2026-10-17)

### Authorization, recovery and branch lineage

- User requested another survey/recent-method round, a non-RL hierarchical replacement for the actual CoCap scene, alternatives at every layer, follow-up literature/code research, documents and remote synchronization. This authorizes documentation commit/push; no baseline implementation, preflight execution or training was started.
- Refreshed remote refs/history at startup. Round-2 branch `docs/baseline-focused-review-20261017` remained at `ec15f16eca08f9071f35ba4cbcccfcf1542a5d2a`; audit branch remained at `3038ef5cd728acdd302772ca08bf10060685af16`. Round-1 parent `94753f9fd82d137af7ff9930634788ffeba6f3e2` and earlier history retained.
- Created isolated branch/worktree `docs/baseline-hierarchical-design-20261017`, `/home/yjq/rl/CoCap1/baseline-hierarchical-design-20261017`, from the current Round-2 remote HEAD. Reused V2 shortlist/feasibility/evidence, original registry and logs, and re-read the audit Benchmark Contract. Local workflow `docs/GIT_WORKFLOW.zh-CN.md` supports normal feature-branch documentation synchronization; applicable ancestor/baseline AGENTS files were not found during recovery.

### Architecture-driven search questions and sources

- Search questions: persistent search rather than static deployment; local discovery/tracking vs global task lists; minimum-three coalition rather than independent single-robot tasks; geometric normal capture rather than tracking; AW9 execution/swept safety; capture-release-search and method-independent repeated arrival.
- New relevant-section full texts: **2026 TAMP survey** (Artificial Intelligence Review 59:198), **2026 CPP survey preprint** (arXiv2607.10649v1), **2025 distributed optimization tutorial** (Proceedings IEEE113(1):40–65, read arXiv2309.04257v2). Reused the already-read **2025 Robotics coalition survey**. Page/section evidence is in `TRADITIONAL_SURVEYS_AND_RECENT_METHODS_V3_ZH.md`.
- Followed the CPP survey references to **C* TRO2026** (10.1109/TRO.2026.3661719, read arXiv2505.13782v3), **Multi-CAP** (arXiv2509.14941v2; formal venue for exact title not verified), and **LS-MCPP TRO2025** (10.1109/TRO.2025.3567476, author PDF). C* multi-robot uses central server/shared RCG; Multi-CAP explicitly assumes base station and spatial connectivity. Neither was relabeled local-distributed from its title.
- Read **HEDAC-maze formal 2026-10-05 Mathematics14(19):3613** (10.3390/math14193613), using publisher-res PDF after ordinary publisher PDF download returned403. Its centralized sequential grid exploration and stop-at-exit lifecycle remain separate from the paper's proposed decentralized extension in §4.3.
- Read **GRAPE TRO2018** arXiv1711.06871v2. Crucial finding: §V p.9 already treats minimum participation Rj via auxiliary/dummy utilities, Condition1 and Proposition1. Basic SPAO claim alone is insufficient. This corrected the preliminary interpretation of the basic implementation during research; final documents give the paper's actual MTR mechanism and its conditions. Inspected official Python/MATLAB utility branches do not implement that extension.
- Read **SPACE arXiv2409.04230v1** beyond abstract, §§III–V. It has real local sensing, exploration, completion, dynamic tasks and plugins, but workloads are single-robot-feasible, collision avoidance is omitted, and source uses global task-ID initialization/live task handles. These are explicit adaptation gaps, not hidden capabilities.
- Read **ICRA2024 sensor-based spatial-separation coverage** arXiv2403.01710v3. Formal DOI10.1109/ICRA57147.2024.10611020, exact formal title uses Multi-Robot and includes Yulong Ding. Its single-integrator/local obstacle safe cells with information-density/static convergence are not persistent unknown moving-target search.
- Revisited classic **Cortés et al. TRA2004** using older arXiv math/0212212 for locational-cost/CVT convergence, explicitly labeling the read version.

### Latest-method and survey evidence gates

- Verified Crossref/publisher metadata for CFG-BAP dynamic MTR **IoTJ13(6):12501–12518, 2026** (10.1109/JIOT.2026.3651713) and TNSE minimum-requirement game **11(1):1202–1213, 2024** (10.1109/TNSE.2023.3321605). Full formulas/communication/code not obtained; keep as allocator reserves, not implemented capture methods.
- Added latest survey metadata: hunting survey **RAS197:105296, March2026** (10.1016/j.robot.2025.105296), MRTA survey **IoTJ13(19):44021–44058, October2026** (10.1109/JIOT.2026.3699696). No full PDFs; no chapter/page claims inferred. Previous Neurocomputing target-survey and CJA RL-review PDF gaps remain.
- Screened publisher previews of whole-group Voronoi encirclement **Matcom2026 online/pre-proof** (10.1016/j.matcom.2026.08.028), elliptical AUV encircling **Ocean Engineering366:127487, October2026** (10.1016/j.oceaneng.2026.127487), and Chinese AUV sliding-MPC lead. Full text unavailable; no claim of per-target three-pursuer capture or continuous release/search was promoted.
- Web query groups covered exact titles/DOIs and combinations of 2025/2026 + encirclement + unknown/search + decentralized + code/GitHub, then layer-specific C*/Multi-CAP/GRAPE/MTR sources. Recorded GitHub repository queries for CFG-BAP,3651713,dynamic-task+encirclement,sensor+spatial-separation,3321605 all returned200 with0 matching repository metadata results. This is bounded negative evidence, not proof no code exists.

### Seven static official-repository audits

- pyGRAPE `7f4b619b188ffb8b4257b51168c460a4957826d7`: author Python equivalent, **GPLv3**; basic utility/mutex/demo, not §V MTR or pursuit lifecycle.
- GRAPE MATLAB `0d3aa265cbf6d97f1b933b9781985563d3643fb5`: author explicitly partial Fig7 codes, **MIT**; do not transfer this license to Python equivalent.
- SPACE `b5286bbb1f99f81caf3f0ee50212bc58d4348bea`: **GPLv3**, configs/MC/behavior tree/plugins; identified full-task-ID and live-object concerns and NumPy not explicit in requirements.
- LS-MCPP `9dcba0b16a3a7669e7632f1b81a04a029d9ba304`: **GPLv3**, master TRO algorithms/deconfliction; setup declares gurobipy but static tracked `.py` search found no API calls outside setup. Do not assert either mandatory Gurobi optimization or proven removable dependency without running.
- Robotarium Python `2b1b6136a242c164e53081b794ebf10d904cbaae`: **MIT**, CVXOPT, SI/velocity-unicycle safety examples; not AW9 or distributed coalition code. Platform-wide libraries.txt is not the minimal simulator requirements.
- HEDAC maze `c7905951141b1683e06731175dfa242731d859f7`: algorithm/configs/batch scripts and pyproject/uv.lock exist; no LICENSE/COPYING; README references absent requirements.txt. Declared package versions/lock reproduction untested.
- multirobot-tutorial `ccde5f0d55d69ad97ec40d49ebf83f8af102c37a`: official paper-linked repo has **README only**, no algorithm/config/license. Separate toolboxes are not treated as audited complete tutorial software.
- Repo URL, scoped category, core files, dependencies, entry points, gaps, difficulty and static hashes are recorded in the same registry's components and V3 source manifest. No external source code or paper PDFs added to Git.

### Decisions, deliverables and checks

- Best complete-paper comparator remains **TR-11 / H-G1 CONDITIONAL_GO**, with **TR-13 / H-G2** backup. They remain privileged, including their strengths; code absence did not remove them.
- New closest-scene proposal **H-L CONDITIONAL_GO**: persistent age search + GRAPE §V minimum requirement + project geometric/AW9/lifecycle modules. This is a literature-informed project composite, not original GRAPE encirclement. **H-S** is the simpler auxiliary sharing lower layers; TR-05 project identity is preserved.
- Deliverables: V3 entry decision, hierarchical blueprint, 28-option CSV, surveys/recent-method evidence, seven-repo component audit, and staged preflight recommendation. These do not claim implementation/performance.
- Updated the **same** registry to schema1.2.0: original20 candidate IDs, Round1 snapshots and V2 focused fields preserved; V2 recommendation/repository snapshots retained;18 literature sources,7 components and4 configurations added. Historical entry documents now link to V3.
- Validation checks JSON/CSV parsing and references, old-field preservation, PDF page counts/hash provenance, commit pins, relative document links, LF formatting and documentation-only changes. Synchronization uses a normal commit/push on the new branch; resulting remote HEAD is verified and reported with delivery, without merging main.



## Round 4 — 2026-10-17：全局敌人信息与集中式传统分层基线

- 用户追加范围：不限局部分布式，允许全局敌人状态和集中协调；延续同一 registry，完成文档并同步远端。未授权/未启动正式复现、preflight 或训练。
- 启动重新 fetch/history：V1 `94753f9fd82d137af7ff9930634788ffeba6f3e2`，V2 `ec15f16eca08f9071f35ba4cbcccfcf1542a5d2a`，V3 `a3726025352a3cf0a688ea79d04b097e68c4b8b0`；IQN-Z 审计 `3038ef5cd728acdd302772ca08bf10060685af16`。从 V3 新建 `docs/baseline-global-hierarchical-review-20261017`。
- 重读 TR-11 正文/补充、TR-13 pp.3–6；核查纯中央执行与原分布式分配的区别。式(7)固定配额分组可采用 min-cost-flow，式(8)为槽位线性指派；两阶段与联合槽位目标不同。
- 新读 AFRA，Drones10(6):406，2026，DOI10.3390/drones10060406：全局人数经验库+MCKP-DP+APF；目标是预期捕获数，不是全目标联合成功。21页正式 PDF 重点 pp.5–6、9–11、13–14。
- AFRA 由论文定位 Gitee 官方仓库，固定 `72440fdf0aa9516e87fcd52f282ce2a612c2df68`；静态审计发现主捕获路径跳过角度、经验库变手写 logistic、动态脚本缺源、余量追加可能破坏人数选项，根许可/环境锁缺失。OFFICIAL_PARTIAL；未执行第三方代码/pyc/权重。
- 新读 ICRA2021 Liao cooperative hunting 7页及 TCyb2022 Fang faster-evader arXivv2 10页，作为单目标几何/速度比理论，不升格为完整多目标基线。新读 FITEE2023 Mu pursuit-evasion survey 24页相关章节；沿用 V3 TAMP、distributed optimization、Robotics2025 coalition survey 的已读页码。
- 新核 ICUS2024 Dynamic Alliance DOI10.1109/ICUS61736.2024.10840019，Crossref/作者记录一致，PDF418；PGM full/pdf403，MK-WTA 作者页/OpenAlex仍无正文。缺口保留。下载的 Sewlia IJRNC2022 PDF 未作为本轮阅读证据。
- GitHub 精确标题/缩写仓库查询（详见 manifest）未找到 TR11/TR13 完整实现；该否定限于检索范围，不忽略无代码的已发表强方法。
- 新审 OR-Tools/SciPy/PythonRobotics 固定 commit、真实核心路径、许可和现有测试；纠正猜测文件404；只作组件可用性审计，没有安装/跑测试/性能测量。
- 决策：完整论文强基线仍选 TR-11；集中式分配主要依据 TR-13；新增项目组合 TR-15 G-CAP-GE，配额>=3（四机优先验证）→分组/槽位→几何→AW9→捕获释放重分配；AFRA TR-14 为自适应人数备选。所有推荐仅为下一阶段 preflight 决策。
- 信息分 GE 与额外 full-map 的 GEM；未来轨迹不授予。V3 H-L/TERL 决策保留，GE 不能将 oracle 发现作为自主搜索性能。重复到达采用共同外生 manifest，不复用 Z 归零触发。
- 输出：V4 总览、蓝图、方法/逐层 CSV、文献证据、代码审计、preflight 可行性共六份；统一 registry 20→22候选，sources18→23，components7→11。源 manifest / 验证 JSON 与本节追加供追溯。
