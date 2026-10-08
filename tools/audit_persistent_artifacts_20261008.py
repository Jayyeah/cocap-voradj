#!/usr/bin/env python3
"""Verify existing persistent artifacts; no policy inference or env.step calls."""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
from PIL import Image
from tools import run_iqn_repeated_arrival_demo_20261006 as demo

STRICT = Path('/home/yjq/rl/CoCap1/z05-repeated-arrival-formal-20261007-heldout20x3')
RELAXED = Path('/home/yjq/rl/CoCap1/z05-repeated-arrival-casualty-robustness-20261006')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_source_fields_equal(summary_value, event_value):
    # Formal endpoint audit adds per-wave contract diagnostics and worker
    # hashes to summary records after the original event file was written.
    if isinstance(event_value, dict):
        for key, value in event_value.items():
            assert key in summary_value
            assert_source_fields_equal(summary_value[key], value)
    elif isinstance(event_value, list):
        assert len(summary_value) == len(event_value)
        for left, right in zip(summary_value, event_value):
            assert_source_fields_equal(left, right)
    else:
        assert summary_value == event_value


def near_miss_geometry(episode, trajectory, config):
    cfg = demo.formal.formal.resolved(config)
    cfg = demo.formal.configure_large_case(cfg, pursuers=12, evaders=3, obstacles=3, post_window=700)
    cfg = demo.formal.scene_config(cfg, 'mixed')
    demo.set_global_config(cfg)
    env = demo.VorAdjEnv(copy.deepcopy(cfg), seed=episode['initial_world_seed'])
    env.reset()
    obstacle_hash = demo.stable_hash([[float(o.x), float(o.y), float(o.r)] for o in env.obstacles])
    assert obstacle_hash == episode['obstacle_hash_initial']
    capture_step = episode['waves'][-1]['capture_step']
    values, hold, maximum_hold = [], 0, 0
    for row in trajectory:
        if row['wave_id'] != 3 or row['global_step'] <= capture_step:
            continue
        for p, state in zip(env.pursuers, row['pursuers']):
            p.x, p.y, p.theta, p.speed = state[:4]
            p.deactivated = not state[4]
        for e, state in zip(env.evaders, row['targets']):
            e.x, e.y = state[3:5]
            e.deactivated, e.collision = not state[5], state[6]
        env._invalidate_voronoi_cache()
        metrics = env._ce_coverage_geometry(env._coverage_voronoi_map(), strict=True)
        good = metrics['converged_now']
        hold = hold + 1 if good else 0
        maximum_hold = max(maximum_hold, hold)
        values.append((row['global_step'], metrics['ce_center_rms'], metrics['ce_center_max'], bool(good)))
    assert len(values) == 700
    best = min(values, key=lambda x: max(x[1] / metrics['ce_rms_threshold'], x[2] / metrics['ce_max_threshold']))
    return {
        'regime': episode['regime'], 'episode_index': episode['episode_index'],
        'seed': episode['initial_world_seed'], 'post_capture_snapshots': len(values),
        'rms_threshold': metrics['ce_rms_threshold'], 'max_threshold': metrics['ce_max_threshold'],
        'required_hold_steps': env.reward_cfg.get('coverage_ce_success_hold_steps', 30),
        'best_joint_threshold_snapshot': {'step': best[0], 'ce_rms': best[1], 'ce_max': best[2]},
        'minimum_ce_rms': min(x[1] for x in values), 'minimum_ce_max': min(x[2] for x in values),
        'final_ce_rms': values[-1][1], 'final_ce_max': values[-1][2],
        'geometry_qualifying_steps': sum(x[3] for x in values),
        'maximum_contiguous_geometry_hold_upper_bound': maximum_hold,
        'trajectory': episode['trajectory_path'],
        'method': 'Static snapshot geometry reconstruction; no dynamics/policy rollout; hold ignores other eligibility gates and is an upper bound.',
    }


def audit():
    result = {'schema': 'persistent-artifact-verification-v1',
              'observed_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'policy_inference_calls': 0, 'environment_step_calls': 0,
              'arms': [], 'representative_gifs': [], 'full_team_recovery_misses': []}
    sources = [('strict', STRICT)] + [('relaxed', RELAXED / name) for name in
               ('a_timeout_continue', 'b_survival_relaxed', 'c_survival_relaxed')]
    for kind, directory in sources:
        summary = json.loads((directory / 'summary.json').read_text())
        assert summary['status'] == 'complete'
        assert summary['selected_policy']['checkpoint_sha256'] == demo.EXPECTED_CHECKPOINT_SHA256
        assert summary['selected_policy']['model_state_sha256_before'] == summary['selected_policy']['model_state_sha256_after']
        for regime, reported in summary['regimes'].items():
            episodes = [e for e in summary['episodes'] if e['regime'] == regime]
            assert len(episodes) == 20
            collisions, phases, source_files, failures = collections.Counter(), collections.Counter(), [], []
            casualty_episodes = post_casualty_all3 = post_casualty_continue = 0
            active_reached = collections.Counter()
            post_casualty_targets = post_casualty_waves = w2_degraded = w3_degraded = 0
            for episode in episodes:
                events = json.loads(Path(episode['events_path']).read_text())
                assert_source_fields_equal(episode, events)
                assert all(v for v in episode['contract_diagnostics'].values() if isinstance(v, bool))
                assert episode['contract_diagnostics']['evaluator_parameter_updates'] == 0
                assert episode['worker_model_state_sha256_before'] == episode['worker_model_state_sha256_after']
                assert episode['initial_world_seed'] == 2026100701 + episode['episode_index']
                trajectory = [json.loads(line) for line in open(episode['trajectory_path'])]
                assert trajectory[-1]['global_step'] == episode['total_mission_steps']
                assert max(t['wave_id'] for t in trajectory) <= 3
                active = [sum(bool(p[4]) for p in t['pursuers']) for t in trajectory]
                casualty = min(active) < 12
                casualty_episodes += casualty
                for n in (11, 10, 9, 0): active_reached[n] += n in active
                previous = None
                event_count = 0
                for row in trajectory:
                    wave = episode['waves'][row['wave_id'] - 1]
                    for event in row['collision_events']:
                        event_count += 1
                        collisions[event['type']] += 1
                        after_capture = wave['capture_step'] is not None and row['global_step'] > wave['capture_step']
                        phases['post_capture_recovery' if after_capture else 'pre_capture'] += 1
                        if row['wave_id'] > 1 and row['global_step'] - wave['wave_start_step'] <= 20:
                            phases['early_refresh_first_20_steps'] += 1
                        state = previous or row
                        if any(np.linalg.norm(np.asarray(state['pursuers'][i][:2]) - np.asarray(t[3:5])) <= 8.0
                               for i in event.get('pursuer_ids', []) for t in state['targets'] if t[5]):
                            phases['capture_region_radius8'] += 1
                    previous = row
                assert event_count == episode['cumulative_collision_count']
                if kind == 'relaxed':
                    post_casualty_all3 += bool(episode['all_3_captured_despite_casualty'])
                    post_casualty_continue += bool(episode['continued_after_first_casualty'])
                    post_casualty_targets += episode['target_captures_after_first_casualty']
                    post_casualty_waves += episode['waves_captured_after_first_casualty']
                    w2_degraded += episode['wave2_capture_with_lt12_active']
                    w3_degraded += episode['wave3_capture_with_lt12_active']
                    if not episode['final_recovery_success']:
                        failures.append('casualty_strict_coverage_infeasible' if casualty else 'full_team_recovery_timeout')
                        if not casualty:
                            result['full_team_recovery_misses'].append(near_miss_geometry(episode, trajectory, Path(summary['source_contract']['config'])))
                if episode['episode_index'] == 0:
                    path = Path(episode['gif_path'])
                    assert path.is_file() and path.stat().st_size > 0
                    with Image.open(path) as gif:
                        frame_count = gif.n_frames
                        for frame in range(frame_count): gif.seek(frame); gif.load()
                    source_frames = [t for t in trajectory if t['global_step'] % 20 == 0 or t['capture_events'] or t['global_step'] == 1]
                    assert frame_count == episode['gif_frames']['rendered_frames']
                    assert len(source_frames) == episode['gif_frames']['source_frames']
                    assert {t['wave_id'] for t in source_frames} == {1, 2, 3}
                    assert episode['all_3_captured']
                    result['representative_gifs'].append({'kind': kind, 'regime': regime, 'seed': episode['initial_world_seed'],
                        'gif': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path), 'frames': frame_count,
                        'source_waves': [1, 2, 3], 'last_sampled_step': source_frames[-1]['global_step'],
                        'trajectory_final_step': trajectory[-1]['global_step'],
                        'trajectory': episode['trajectory_path'], 'gif_decode': 'PASS', 'seed_event_manifest_match': True})
                source_files.append({'seed': episode['initial_world_seed'], 'events': episode['events_path'],
                                     'events_sha256': sha(episode['events_path']), 'trajectory': episode['trajectory_path'],
                                     'trajectory_sha256': sha(episode['trajectory_path'])})
            counts = {key: sum(bool(e[key]) for e in episodes) for key in
                      ('all_3_captured', 'final_recovery_success', 'persistent_service_complete')}
            assert all(reported[k] == v for k, v in counts.items())
            times = demo.summarize([e['final_recovery_time'] for e in episodes if e['final_recovery_success']])
            assert times == reported['final_recovery_time']
            result['arms'].append({'kind': kind, 'regime': regime, 'episodes': 20, **counts,
                'final_recovery_time_seconds': times, 'recovery_unobserved': 20 - times['n'],
                'casualty_episodes': casualty_episodes, 'post_casualty_continued': post_casualty_continue,
                'post_casualty_all3': post_casualty_all3, 'post_casualty_target_captures': post_casualty_targets,
                'post_casualty_waves_captured': post_casualty_waves, 'w2_capture_lt12': w2_degraded, 'w3_capture_lt12': w3_degraded,
                'exact_active_counts_reached': dict(active_reached), 'collision_types': dict(collisions),
                'collision_phase_diagnostics': dict(phases), 'failure_taxonomy': dict(collections.Counter(failures)),
                'contract_pass_episodes': 20, 'summary': str(directory / 'summary.json'), 'summary_sha256': sha(directory / 'summary.json'),
                'sources': source_files})
    result['collision_phase_definition'] = 'Events; pre/post capture exclusive. Other tags overlap: early refresh=first 20 decision steps (10 s) after W2/W3 spawn; capture-region=involved pursuer within canonical radius 8 of an active target in previous trajectory snapshot.'
    result['status'] = 'PASS'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'arms': len(result['arms']),
                      'gifs': len(result['representative_gifs']), 'full_team_misses': len(result['full_team_recovery_misses'])}))
