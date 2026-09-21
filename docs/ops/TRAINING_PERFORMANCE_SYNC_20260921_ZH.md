# LOCAL-RUNTIME-SYNC — IQN/AC Training Performance Recovery

- Evidence cutoff: `2026-09-21T15:59:58+08:00` (Asia/Shanghai).
- Scope: read-only runtime audit; no live process was stopped, paused, restarted, attached with control input, or modified.
- Checkpoint/replay/full-resume binaries and large logs are not copied.

## Overall status

| Line | Stage | Current step | Latest formal | Selected checkpoint | Key performance | Status |
| --- | --- | ---: | ---: | --- | --- | --- |
| IQN-Z05 | Stage2 | 100k | 100k | `step_1300000.pt` | Stage1 selected; Stage2 strict CE 0.9 / CE RMS 0.0489383 | IN_PROGRESS |
| IQN-Z07 | Stage2 | 200k | 200k | `step_1600000.pt` | Stage1 selected; Stage2 strict CE 1 / CE RMS 0.0402727 | IN_PROGRESS |
| AC-COV | Stage1 | 500k | 500k | N/A — training incomplete | strict CE 0; CE RMS 0.281068; area CV 0.4793237 | NOT_LIVE_LATEST_FORMAL_500K |
| AC-CAP | Stage1 | 300k | 300k | N/A — training incomplete | normal capture 0; 2+ ring windows 0; collision 0 | IN_PROGRESS_OR_POST_FORMAL |
| AC-MIX | Stage1 | 293k observed | 150k | N/A — training incomplete | capture 0 at 150k; first capture telemetry 260752 | IN_PROGRESS |

## IQN Stage1 selection

| arm | selected Stage1 step | coverage strict CE | pure capture | mixed safe complete | mixed post CE | collision | CE RMS | capture time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Z05 | 1300000 | 1 | 1 | 0.9 | 0.9 | 0 | 0.0414763 | 35.55 |
| Z07 | 1600000 | 1 | 1 | 1 | 1 | 0 | 0.0359177 | 46.05 |

Selection result: `BALANCED_FLOOR_DIRECT_SELECTION` for both arms; `fallback_used=false`. Tie-break order is recorded in `checkpoint_selection_summary.json`. Stage2 warm-start provenance is shape-compatible, 117 keys loaded, with no adapted or skipped keys.

## IQN selected formal metrics

### Z05 — `/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05/stages/stage1/training/checkpoints/step_1300000.pt`

| metric | value |
| --- | ---: |
| coverage_strict_ce | 1 |
| coverage_ce_rms | 0.0414763 |
| coverage_area_cv | 0.0733936 |
| coverage_collision | 0 |
| coverage_time_to_ce | 183 |
| pure_capture_rate | 1 |
| pure_normal_capture_rate | 1 |
| pure_stationary_capture_rate | 0 |
| pure_ring2_seen | 1 |
| pure_ring3_reached | 1 |
| pure_capture_collision | 0 |
| pure_capture_time | 35.55 |
| mixed_capture_rate | 1 |
| mixed_collision | 0 |
| mixed_post_capture_ce | 0.9 |
| mixed_safe_complete | 0.9 |
| mixed_capture_time | 39.575 |
| mixed_recovery_time | 109.333 |
| mixed_mission_time | 148.972 |
| direct_visible_z_one_rate | 1 |
| support_z_mean | 0.480141 |
| coverage_z_mean | 0.00176516 |
| pure_coverage_exact_zero | True |
| neighbor_dominant_count | 1837 |
| neighbor_dominant_z_mean | 0.489521 |
| max_lineage_hop | 2 |
| release_seconds | 1.5 |
| never_release_episodes | 0 |
| update_violations | 0 |

### Z07 — `/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z07/stages/stage1/training/checkpoints/step_1600000.pt`

| metric | value |
| --- | ---: |
| coverage_strict_ce | 1 |
| coverage_ce_rms | 0.0359177 |
| coverage_area_cv | 0.0464559 |
| coverage_collision | 0 |
| coverage_time_to_ce | 96.25 |
| pure_capture_rate | 1 |
| pure_normal_capture_rate | 1 |
| pure_stationary_capture_rate | 0 |
| pure_ring2_seen | 1 |
| pure_ring3_reached | 1 |
| pure_capture_collision | 0 |
| pure_capture_time | 46.05 |
| mixed_capture_rate | 1 |
| mixed_collision | 0 |
| mixed_post_capture_ce | 1 |
| mixed_safe_complete | 1 |
| mixed_capture_time | 54.625 |
| mixed_recovery_time | 88.75 |
| mixed_mission_time | 143.375 |
| direct_visible_z_one_rate | 1 |
| support_z_mean | 0.67228 |
| coverage_z_mean | 0.00661101 |
| pure_coverage_exact_zero | True |
| neighbor_dominant_count | 1802 |
| neighbor_dominant_z_mean | 0.691143 |
| max_lineage_hop | 2 |
| release_seconds | 3 |
| never_release_episodes | 0 |
| update_violations | 0 |

## IQN Stage1 curves

Each row is a completed formal report; no interpolation was used. Full compact curve data is in the two `iqn_*_stage1_curve.json` files.

### Z05

| step | strict CE | CE RMS | area CV | pure capture | capture time | mixed capture | mixed post CE | safe complete | mixed collision | recovery time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100000 | 0 | 0.211401 | 0.284468 | 0.75 | 68.6667 | 0.65 | 0 | 0 | 0.45 | — |
| 200000 | 0 | 0.171095 | 0.382254 | 0 | — | 0 | 0 | 0 | 0.8 | — |
| 300000 | 0 | 0.0971257 | 0.211096 | 0.9 | 61.5833 | 0.9 | 0 | 0 | 0.35 | — |
| 400000 | 0 | 0.0973241 | 0.273904 | 0.75 | 73.5 | 0.85 | 0 | 0 | 0.3 | — |
| 500000 | 0 | 0.0863705 | 0.202692 | 1 | 60.1 | 1 | 0 | 0 | 0 | — |
| 600000 | 0.85 | 0.0459416 | 0.0676017 | 1 | 81.425 | 1 | 0.8 | 0.8 | 0 | 138.844 |
| 700000 | 0.05 | 0.0646794 | 0.146428 | 1 | 48.15 | 1 | 0.1 | 0.1 | 0 | 62.75 |
| 800000 | 0.4 | 0.0569905 | 0.119476 | 1 | 80.025 | 1 | 0.25 | 0.25 | 0.05 | 94.4 |
| 900000 | 0 | 0.107897 | 0.292532 | 1 | 43.025 | 1 | 0 | 0 | 0 | — |
| 1000000 | 0 | 0.0812792 | 0.18346 | 1 | 38.55 | 1 | 0 | 0 | 0 | — |
| 1100000 | 0 | 0.0987636 | 0.196558 | 1 | 37.7 | 1 | 0 | 0 | 0 | — |
| 1200000 | 0.95 | 0.0380609 | 0.0520975 | 1 | 33.425 | 1 | 0.8 | 0.8 | 0 | 118.156 |
| 1300000 | 1 | 0.0414763 | 0.0733936 | 1 | 35.55 | 1 | 0.9 | 0.9 | 0 | 109.333 |
| 1400000 | 0.25 | 0.0651207 | 0.149433 | 1 | 32.875 | 1 | 0.15 | 0.15 | 0 | 143 |
| 1500000 | 0.95 | 0.032387 | 0.0560242 | 1 | 33.375 | 1 | 0.9 | 0.9 | 0.1 | 137.444 |
| 1600000 | 0 | 0.0741504 | 0.164915 | 1 | 37.175 | 1 | 0 | 0 | 0 | — |
| 1700000 | 0.05 | 0.0707294 | 0.219495 | 1 | 36.65 | 1 | 0 | 0 | 0 | — |
| 1800000 | 0.15 | 0.0634008 | 0.143653 | 1 | 35.15 | 1 | 0.2 | 0.2 | 0 | 191.125 |
| 1900000 | 0 | 0.0905001 | 0.219979 | 1 | 33.825 | 1 | 0 | 0 | 0 | — |
| 2000000 | 0 | 0.105117 | 0.305825 | 1 | 33.25 | 1 | 0 | 0 | 0.1 | — |

### Z07

| step | strict CE | CE RMS | area CV | pure capture | capture time | mixed capture | mixed post CE | safe complete | mixed collision | recovery time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100000 | 0 | 0.242773 | 0.699753 | 0.1 | 116.25 | 0.15 | 0 | 0 | 0.65 | — |
| 200000 | 0 | 0.127292 | 0.315874 | 0.1 | 333.25 | 0.35 | 0 | 0 | 0.7 | — |
| 300000 | 0.15 | 0.0646851 | 0.119868 | 0.5 | 252.2 | 0.45 | 0 | 0 | 0.55 | — |
| 400000 | 0 | 0.106299 | 0.285311 | 0.85 | 53.2059 | 0.95 | 0 | 0 | 0.35 | — |
| 500000 | 0.25 | 0.0697009 | 0.111474 | 0.75 | 89.6 | 0.8 | 0.05 | 0.05 | 0.4 | 134 |
| 600000 | 0.05 | 0.0767399 | 0.134256 | 1 | 49.425 | 0.95 | 0 | 0 | 0.25 | — |
| 700000 | 0 | 0.107553 | 0.29006 | 1 | 43.725 | 1 | 0 | 0 | 0 | — |
| 800000 | 0.05 | 0.0774228 | 0.194674 | 1 | 42.675 | 1 | 0 | 0 | 0.05 | — |
| 900000 | 0 | 0.0844371 | 0.179345 | 0.95 | 39.3421 | 1 | 0 | 0 | 0.25 | — |
| 1000000 | 0.5 | 0.0578657 | 0.138445 | 1 | 45.425 | 1 | 0.45 | 0.45 | 0.05 | 106.222 |
| 1100000 | 0.95 | 0.0310089 | 0.0664624 | 1 | 37.625 | 1 | 0.85 | 0.85 | 0.15 | 65.4412 |
| 1200000 | 0.05 | 0.0675306 | 0.182034 | 1 | 39.675 | 1 | 0 | 0 | 0.05 | — |
| 1300000 | 0.15 | 0.0599411 | 0.142653 | 1 | 41.25 | 1 | 0.2 | 0.2 | 0 | 59.5 |
| 1400000 | 0.95 | 0.0340677 | 0.0486763 | 1 | 35.175 | 1 | 0.45 | 0.45 | 0.05 | 196.389 |
| 1500000 | 0.7 | 0.0479824 | 0.0604561 | 1 | 46.025 | 1 | 0.7 | 0.7 | 0 | 132.214 |
| 1600000 | 1 | 0.0359177 | 0.0464559 | 1 | 46.05 | 1 | 1 | 1 | 0 | 88.75 |
| 1700000 | 0.25 | 0.0667296 | 0.0872965 | 1 | 36.95 | 1 | 0 | 0 | 0 | — |
| 1800000 | 0.1 | 0.0685225 | 0.165931 | 1 | 33.75 | 1 | 0 | 0 | 0 | — |
| 1900000 | 0 | 0.0927507 | 0.270931 | 1 | 32.275 | 1 | 0 | 0 | 0 | — |
| 2000000 | 0.05 | 0.0726076 | 0.182839 | 1 | 34.425 | 1 | 0 | 0 | 0.05 | — |

## IQN Stage2 current formal

| line | current live/durable step | latest formal step | strict CE | CE RMS | area CV | collision | mixed post CE | safe complete |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Z05 | 100000 | 100000 | 0.9 | 0.0489383 | 0.164934 | 0.05 | 0.2 | 0.2 |
| Z07 | 200000 | 200000 | 1 | 0.0402727 | 0.153295 | 0 | 1 | 1 |

## AC milestone status

### COV

| milestone | status |
| ---: | --- |
| 0 | COMPLETE |
| 25000 | COMPLETE |
| 50000 | COMPLETE |
| 75000 | COMPLETE |
| 100000 | COMPLETE |
| 150000 | COMPLETE |
| 200000 | COMPLETE |
| 300000 | COMPLETE |
| 400000 | COMPLETE |
| 500000 | COMPLETE |

### CAP

| milestone | status |
| ---: | --- |
| 0 | COMPLETE |
| 25000 | COMPLETE |
| 50000 | COMPLETE |
| 75000 | COMPLETE |
| 100000 | COMPLETE |
| 150000 | COMPLETE |
| 200000 | COMPLETE |
| 300000 | COMPLETE |
| 400000 | NOT_REACHED |
| 500000 | NOT_REACHED |

### MIX

| milestone | status |
| ---: | --- |
| 0 | COMPLETE |
| 25000 | COMPLETE |
| 50000 | COMPLETE |
| 75000 | COMPLETE |
| 100000 | COMPLETE |
| 150000 | COMPLETE |
| 200000 | INCOMPLETE |
| 300000 | NOT_REACHED |
| 400000 | NOT_REACHED |
| 500000 | NOT_REACHED |

## AC-COV curve

Pure coverage scratch has no strict CE success at any completed milestone. CE RMS and area CV show a transient improvement at 200k, but it is not sustained at 300–500k; classification: `NO_CLEAR_SUSTAINED_SIGNAL` (transient 200k metric signal only).

| step | strict CE | CE RMS | area CV | collision | return | actor entropy | epsilon | actor loss | critic loss | Q mean | Q std | replay |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0.310787 | 0.529509 | 0 | -12240.9 | — | 0.6 | — | — | — | — | 0 |
| 25000 | 0 | 0.294301 | 0.47727 | 1 | -384.625 | 0.181923 | 0.5725 | 9.7239 | 0.911542 | -9.7396 | 6.1896 | 100000 |
| 50000 | 0 | 0.254289 | 0.538138 | 0.15 | -9760.91 | 0.0263612 | 0.545 | 19.8186 | 1.85214 | -20.0404 | 18.8793 | 200000 |
| 75000 | 0 | 0.284239 | 0.531124 | 0.3 | -13426.2 | 7.96348e-05 | 0.5175 | 20.8645 | 2.07579 | -21.1579 | 25.1951 | 300000 |
| 100000 | 0 | 0.276544 | 0.492163 | 0.6 | -8132.27 | 8.37473e-06 | 0.49 | 21.4329 | 2.03789 | -21.75 | 24.7772 | 400000 |
| 150000 | 0 | 0.281763 | 0.557941 | 0.1 | -14864.2 | 3.39071e-06 | 0.435 | 23.2732 | 2.08237 | -23.6519 | 25.7656 | 600000 |
| 200000 | 0 | 0.123759 | 0.246472 | 0.15 | -3287.65 | 2.08415e-06 | 0.38 | 23.2564 | 1.78771 | -23.8358 | 26.219 | 800000 |
| 300000 | 0 | 0.278476 | 0.513047 | 0.5 | -12377.1 | 1.74662e-06 | 0.27 | 26.4168 | 1.80811 | -26.8016 | 30.2364 | 1000000 |
| 400000 | 0 | 0.281068 | 0.479324 | 0.6 | -9033.26 | 1.65074e-06 | 0.16 | 50.7747 | 3.0743 | -51.0653 | 54.0412 | 1000000 |
| 500000 | 0 | 0.281068 | 0.479324 | 0.6 | -9033.26 | 1.48254e-06 | 0.05 | 72.7284 | 4.09613 | -73.1857 | 69.4855 | 1000000 |

## AC-CAP curve

Classification: `RING_SIGNAL_ONLY`; 2+ ring windows appear at 25k/75k/150k, but normal and stationary capture remain zero and 3+ remains zero.

| step | normal capture | 2+ | 3+ | collision | capture time | return | actor entropy | epsilon | critic loss | replay |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 | 0 | — | -5.05561 | 9.64191e-05 | — | — | 0 |
| 25000 | 0 | 3 | 0 | 1 | — | -208.763 | 0.000495582 | 0.572501 | 0.482338 | 100000 |
| 50000 | 0 | 0 | 0 | 0 | — | -5.05561 | 5.25172e-05 | 0.545001 | 0.731774 | 200000 |
| 75000 | 0 | 1 | 0 | 0.9 | — | -681.764 | 0.0106528 | 0.517501 | 0.867106 | 300000 |
| 100000 | 0 | 0 | 0 | 0 | — | -5.05561 | 0.00664876 | 0.490001 | 0.917151 | 400000 |
| 150000 | 0 | 8 | 0 | 0.65 | — | -3752 | 0.00145975 | 0.435001 | 1.02848 | 600000 |
| 200000 | 0 | 0 | 0 | 0 | — | -5.05561 | 9.64191e-05 | 0.380001 | 1.16206 | 800000 |
| 300000 | 0 | 0 | 0 | 0 | — | -5.05561 | 2.53935e-05 | 0.270001 | 0.908517 | 400000 |

## AC-MIX curve

The 150k formal report has no post-capture recovery success. Runtime telemetry later shows first post-capture row at 260253, first post>=32 at 260260, strict 64/16/32/16 batch at 260260, and first capture at 260752.

| step | capture | post CE | safe complete | pure CE | collision | capture time | replay classes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 0 | 0 | 0 | 0.6 | — | `{"post_capture_real": 0, "pre_capture_cover": 0, "pursuing": 0, "recovery_pure": 0}` |
| 25000 | 0 | 0 | 0 | 0 | 1 | — | `{"post_capture_real": 0, "pre_capture_cover": 39796, "pursuing": 4452, "recovery_pure": 55752}` |
| 50000 | 0.1 | 0 | 0 | 0 | 0.85 | 270 | `{"post_capture_real": 0, "pre_capture_cover": 66658, "pursuing": 8770, "recovery_pure": 124572}` |
| 75000 | 0 | 0 | 0 | 0 | 0.95 | — | `{"post_capture_real": 0, "pre_capture_cover": 102331, "pursuing": 17485, "recovery_pure": 180184}` |
| 100000 | 0 | 0 | 0 | 0 | 0 | — | `{"post_capture_real": 0, "pre_capture_cover": 142287, "pursuing": 25493, "recovery_pure": 232220}` |
| 150000 | 0 | 0 | 0 | 0 | 0.8 | — | `{"post_capture_real": 0, "pre_capture_cover": 225564, "pursuing": 56876, "recovery_pure": 317560}` |

## Live process snapshot

| line | PID | PPID | tmux | GPU | command/cwd | observed status |
| --- | ---: | ---: | --- | --- | --- | --- |
| IQN-Z05 | 17097 | 9488 | iqn_z05_recovery_20260921 | cuda:0 | `tools/iqn_z_unified_decay_curriculum_20260919.py supervise --arm z05` / `.../iqn-z-unified-decay-dual-curriculum-20260919` | LIVE |
| IQN-Z07 | 19555 | 9488 | iqn_z07_recovery_20260921 | cuda:0 | `tools/iqn_z_unified_decay_curriculum_20260919.py supervise --arm z07` / `.../iqn-z-unified-decay-dual-curriculum-20260919` | LIVE |
| AC-COV | — | — | — | — | `.../ac-capability-cov-stage1-20260920/runs/ac_capability_cov_stage1` | NOT_LIVE; latest formal 500k |
| AC-CAP | 19532 | 9488 | ac_cap_recovery_20260921 | cuda:0 | `tools/run_shared_local_ac_capability.py --run-name ac_capability_cap_stage1` / `.../ac-capability-cap-stage1-20260920` | LIVE/post-formal 300k |
| AC-MIX | 19542 | 9488 | ac_mix_recovery_20260921 | cuda:1 | `tools/run_shared_local_ac_capability.py --run-name ac_capability_mix_stage1` / `.../cocap-voradj-critic-audit` | LIVE; 293k observed |

Start/elapsed values are retained verbatim in `current_runtime_status.json` from the same `ps` snapshot.

## Resume continuity audit

| line | classification | pre durable | resume step | first post-resume observed | optimizer | replay | LR | RNG path |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| AC-COV | RESUME_PROVENANCE_INCOMPLETE | — | 500000 | — | 124813 | RESET/REWARM | 0.0001 | COV loader lacks CPU normalization; formal_report says resumed=false and replay_policy=scratch |
| AC-CAP | FUNCTIONAL_RESUME_NOT_BIT_EXACT | 200000 | 200000 | 300000 | 74626 | RESET/REWARM | 0.0001 | fixed loader: torch_rng.cpu() and [state.cpu() for state in cuda_rng] |
| AC-MIX | FUNCTIONAL_RESUME_NOT_BIT_EXACT | 200000 | 200000 | 200392 | 49951 | RESET/REWARM | 0.0001 | fixed loader: torch_rng.cpu() and [state.cpu() for state in cuda_rng] |

Replay reset is a real discontinuity for CAP/MIX, so neither is `EXACT_RESUME_CONFIRMED`. COV is `RESUME_PROVENANCE_INCOMPLETE`: its formal report explicitly records `resumed=false`, `replay_policy=scratch`, and its branch lacks the CPU-normalizing RNG loader fix used by CAP/MIX.

## Contract drift / restart audit

- `CONTRACT_DRIFT_FOUND`: yes, limited to the AC resume-runtime contract. COV commit `2832e77` directly passes checkpoint RNG tensors to PyTorch; CAP commit `640a26d` and MIX commit `9e9a432` normalize them to CPU before restore.
- `RESTART_RESET_DISCONTINUITY_FOUND`: yes for replay state on CAP/MIX; COV has incomplete resume provenance rather than a confirmed repaired resume.
- No live process was stopped, paused, restarted, or modified by this audit.

## Files

- JSON artifact directory: `artifacts/2026-09-21_training_performance_sync/`
- Summary: `docs/ops/TRAINING_PERFORMANCE_SYNC_20260921_ZH.md`
- The exact committed file list is the seven JSON files plus this Markdown document.

## Final classifications

- `IQN_SELECTION_RECOVERED`
- `IQN_CURVES_SYNCED`
- `AC_CURVES_SYNCED`
- `RESUME_CONTINUITY_AUDITED`
- `PARTIAL_RESUME_CONTINUITY`: COV provenance incomplete; CAP/MIX are functional but not bit-exact because replay was re-warmed.
