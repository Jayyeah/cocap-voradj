# 测试输入、预期与实际回执

完整组合：64 passed（JUnit XML）。所有选择用例的完整输入 token 和实际输出另见 selection_test_inputs_results.json。

| 用例 | 输入 | 预期 | 实际 |
|---|---|---|---|
| selection/zero | selfZ=0.0, IDs=[]；完整token见JSON | target=0, alpha=0.0 | target=0, alpha=0.0；PASS |
| selection/self-max | selfZ=0.5, IDs=[1]；完整token见JSON | target=0, alpha=0.0 | target=0, alpha=0.0；PASS |
| selection/self-equal | selfZ=0.5, IDs=[1]；完整token见JSON | target=0, alpha=0.0 | target=0, alpha=0.0；PASS |
| selection/self-near-equal | selfZ=0.5, IDs=[1]；完整token见JSON | target=0, alpha=0.0 | target=0, alpha=0.0；PASS |
| selection/unique-high | selfZ=0.25, IDs=[1]；完整token见JSON | target=1, alpha=1.0 | target=1, alpha=1.0；PASS |
| selection/far-high-vs-near-low | selfZ=0.0, IDs=[1, 2]；完整token见JSON | target=2, alpha=1.0 | target=2, alpha=1.0；PASS |
| selection/highest-nearest | selfZ=0.0, IDs=[1, 2, 3]；完整token见JSON | target=2, alpha=1.0 | target=2, alpha=1.0；PASS |
| selection/geometry-opposite | selfZ=0.0, IDs=[1, 2]；完整token见JSON | target=2, alpha=0.5 | target=2, alpha=0.5；PASS |
| selection/geometry-velocity | selfZ=0.0, IDs=[1, 2]；完整token见JSON | target=2, alpha=0.5 | target=2, alpha=0.5；PASS |
| selection/stable-id-fallback | selfZ=0.0, IDs=[3, 2]；完整token见JSON | target=2, alpha=0.5 | target=2, alpha=0.5；PASS |
| selection/not-slot-order | selfZ=0.0, IDs=[2, 1]；完整token见JSON | target=1, alpha=0.5 | target=1, alpha=0.5；PASS |
| selection/invalid-friend | selfZ=0.0, IDs=[1, 2]；完整token见JSON | target=2, alpha=0.5 | target=2, alpha=0.5；PASS |
| T0 ×2 seeds ×3 scenes | seed 2026101601/02；coverage/capture/mixed；固定双边AW9动作最多80步 | 基线全部obs/reward/分量/CE/capture/done/terminal/Z/replay keys逐transition一致 | 6/6 PASS |
| token cap / inactive / adjacency | 11P；高Z ID9被截断；ID10失活；邻接随后只见9 | 第一次选7，再选9；不允许截断节点偷入；token/mask映射相等 | PASS |
| 2/3-hop + decay + two sources | 4P链图source0；清空direct；source0/3去掉0 | 同步传播1,.5,.25,.125；四步归0；另一source保持1 | PASS |
| frozen anchor / clip / NaN | self[0,0]→[1,0]；anchor[3,0]；动作后friend可[100,0] | progress=1；self不动0；clip下界-3；NaN拒绝 | PASS |
| post-capture CE / self tie | pure coverage，Z全.5，强制support但无敌人 | gate禁止support，两臂完整CE一致；alpha0 | PASS |
| dynamic ledger | 真实4P；Z=[.25,1,.5,.125]；单元测试强制support | CE权重1-alpha；CE与总账一致且finite | PASS |
| observable selection API | self .25；两友机Z=.5，距离5与2 | 选2；API无enemy坐标、ID或reward role | PASS |
| real normal capture ×2 | 3P半径7包围目标；4P远处；mixed/capture terminal | loose event；peak .5；weighted reset一次，w0不消项；self全CE；mixed下步CE/.25 | 2/2 PASS |
| inactive NaN | friend1失活NaN且Z1；friend2有效 | 只打包2，token finite | PASS |
| two actual enemy sources | 两组友机/敌人近[12,12]和[108,108]；移除第一个 | 第二source direct/Z1继续；第一组无direct，Z衰减 | PASS |
| matched YAML | control/treatment resolved config | 只差run_name和新reward开关；Z05合同有效 | PASS |
| engineering gate | 50episodes；baseline cap .9/collision .1 | 相同通过；cap .7或collision .3暂停 | PASS |
| optimizer/full resume ×2 | 共同selected；tiny16step batch8/replay256 | fresh runtime；真实update>=10；loss finite；映射恢复后step finite | 各15 updates/15 sync；PASS |
| T2 archived replay | 15条冻结Z05轨迹；9636 active agent-transitions | 两臂obs/物理/Z/capture/done一致；总账finite且加总正确 | PASS；326有效Support |
| T2 all-role PBRS | 同一归档；9576相邻转移 | 测量实际CE权重边界残差/reset，不增加补偿奖励 | PASS；15处额外残差 maxabs .70597 |

## 全部 JUnit 实例

| 测试 | 实际 |
|---|---|
| test_selection[zero] | PASS |
| test_selection[self-max] | PASS |
| test_selection[self-equal] | PASS |
| test_selection[self-near-equal] | PASS |
| test_selection[unique-high] | PASS |
| test_selection[far-high-vs-near-low] | PASS |
| test_selection[highest-nearest] | PASS |
| test_selection[geometry-opposite] | PASS |
| test_selection[geometry-velocity] | PASS |
| test_selection[stable-id-fallback] | PASS |
| test_selection[not-slot-order] | PASS |
| test_selection[invalid-friend] | PASS |
| test_t0_historical_fixed_transitions[2026101601-coverage] | PASS |
| test_t0_historical_fixed_transitions[2026101601-capture] | PASS |
| test_t0_historical_fixed_transitions[2026101601-mixed] | PASS |
| test_t0_historical_fixed_transitions[2026101602-coverage] | PASS |
| test_t0_historical_fixed_transitions[2026101602-capture] | PASS |
| test_t0_historical_fixed_transitions[2026101602-mixed] | PASS |
| test_token_mapping_sort_cap_inactive_and_changed_adjacency | PASS |
| test_synchronous_two_three_hop_decay_and_multiple_enemies | PASS |
| test_ally_anchor_frozen_and_own_motion | PASS |
| test_support_self_tie_full_ce_and_gating | PASS |
| test_dynamic_ce_scales_ledger_and_terminal_reset | PASS |
| test_new_selector_ignores_hidden_enemy_positions | PASS |
| test_real_capture_boundary_and_weighted_reset[False] | PASS |
| test_real_capture_boundary_and_weighted_reset[True] | PASS |
| test_inactive_nan_friend_never_packed | PASS |
| test_two_actual_enemy_sources_capture_one_preserves_other | PASS |
| test_matched_configs_differ_only_reward_switch_and_operational_name | PASS |
| test_paired_gate_rejects_capture_and_collision_regressions | PASS |
| test_dual_line_contract_diff_has_only_alpha | PASS |
| test_all_stages_keep_registered_scientific_contract[z05-0.5] | PASS |
| test_all_stages_keep_registered_scientific_contract[z07-0.7] | PASS |
| test_balanced_floor_selection_prefers_joint_strength | PASS |
| test_zero_strict_coverage_fallback_selects_maximin_compromise | PASS |
| test_final_dual_comparison_contains_all_stage_metrics_and_chinese_report | PASS |
| test_dual_supervisor_launch_policy[runtime0-False] | PASS |
| test_dual_supervisor_launch_policy[runtime1-True] | PASS |
| test_dual_supervisor_launch_policy[runtime2-True] | PASS |
| test_dual_supervisor_launch_policy[runtime3-False] | PASS |
| test_dual_supervisor_launch_policy[runtime4-False] | PASS |
| test_dual_supervisor_pid_alive_detects_current_process | PASS |
| test_dual_supervisor_arm_runtime_falls_back_to_launch_fields | PASS |
| test_dual_supervisor_arm_runtime_reads_heartbeat_when_status_is_absent | PASS |
| test_b3_config_changes_observation_only_and_preregisters_lambda_eta | PASS |
| test_final_teacher_loads_and_z_student_has_no_late_role_branch | PASS |
| test_initial_decision_has_current_direct_evidence_and_plain_token_dimensions | PASS |
| test_z_update_is_synchronous_and_uses_previous_neighbor_state | PASS |
| test_z_decay_half_life_and_exact_snapshot_restore | PASS |
| test_replay_transition_records_current_and_next_z | PASS |
| test_masked_voronoi_excludes_obstacles_from_all_cells | PASS |
| test_disconnected_cell_centroid_projects_to_site_component | PASS |
| test_ce_pbrs_terminal_correction_removes_terminal_potential | PASS |
| test_config_extends_and_ce_success_ignore_speed_and_cv | PASS |
| test_legacy_config_keeps_legacy_voronoi_and_objective | PASS |
| test_ce_step_success_uses_hold_without_legacy_terminal_bonus | PASS |
| test_ce_mix_uses_legacy_capture_graph_and_free_mask_coverage_target | PASS |
| test_final_capture_resets_pbrs_and_post_capture_starts_next_transition | PASS |
| test_ce_timeout_keeps_potential_for_external_bootstrap | PASS |
| test_ring_ms_angle_weight_reverts_to_uniform_when_evader_static | PASS |
| test_ring_ms_target_uses_compact_reward_ring_not_detection_radius | PASS |
| test_ring_ms_vct_ls_direct_visibility_candidates_are_self_local | PASS |
| test_ring_ms_reward_is_positive_when_moving_toward_pre_action_target | PASS |
| test_ring_ms_phase_occupancy_can_mask_occupied_front_sector | PASS |
