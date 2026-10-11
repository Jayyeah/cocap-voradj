#!/usr/bin/env python3
"""Engineering verification of new R2 view and long-run orchestration.

This neither reopens frozen E1 nor substitutes for held-out capability eval.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'tools')]

import numpy as np
import torch

from terl_mappo.batch02.cocap_env import CoCapAdapter
from terl_mappo.batch02.runtime import sources as e1_sources
from terl_mappo.batch02_a_long.runtime import CapabilityAdapter, build, collect, save_checkpoint, load_checkpoint, sources, terl_observations
from terl_mappo.batch02_a_long.evaluation import evaluate, selection_key
from terl_mappo.run import atomic_json, rng_state, restore_rng
from verify_batch02_e1_20261011 import same, digest_graph, robot_state

CONFIGS = ROOT / 'configs/experiments/terl_mappo_batch02_a_long_20261011'
FREEZE = Path('/home/yjq/rl/CoCap1/ac-master-dag-20260921/artifacts/2026-10-11_batch02_e1_independent/freeze.json')


def continuation(trainer, adapter):
    batch, episodes = collect(trainer, adapter, 16)
    metrics = trainer.update(batch, categorical=True)
    return dict(batch=digest_graph(batch), episodes=digest_graph(episodes), metrics=metrics,
                trainer=digest_graph(trainer.state_dict()), runtime=digest_graph(adapter.state_dict()),
                rng=digest_graph(rng_state()))


def main(args):
    torch.set_num_threads(1)
    config = json.loads((CONFIGS / f'{args.arm.lower()}.json').read_text())
    adapter = CapabilityAdapter(config['learner']['seed'], args.arm)
    trainer = build(config, adapter, 'cpu')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.restore:
        load_checkpoint(args.output / 'resume.pt', trainer, adapter, config)
        actual = continuation(trainer, adapter)
        expected = json.loads((args.output / 'expected.json').read_text())
        same(actual, expected)
        atomic_json(args.output / 'restored.json', dict(exact=True, continuation=actual))
        return
    frozen = json.loads(FREEZE.read_text())
    same(e1_sources(), frozen['source_manifest'])
    # New wrapper must preserve provider transitions before the original
    # schedule boundary. Compare every original obs/reward/info/done/state.
    native = CoCapAdapter(config['learner']['seed'])
    decision_rng = np.random.RandomState(2116111900)
    for _ in range(64):
        old = adapter.fingerprint()
        before_rng = digest_graph(rng_state())
        first = adapter.packed_observations()
        same(first, adapter.packed_observations())
        same(old, adapter.fingerprint())
        same(before_rng, digest_graph(rng_state()))
        actions = decision_rng.randint(0, 9, len(adapter.env.pursuers))
        a, b = adapter.step(actions), native.step(actions)
        for index in range(1, 5):
            same(a[index], b[index])
        same(a[5]['task_infos'], b[5]['task_infos'])
        same(a[5]['evader_actions'], b[5]['evader_actions'])
        same(adapter.observations, native.observations)
        same(robot_state(adapter.env), robot_state(native.env))
        same(adapter.env.z_state, native.env.z_state)
        if a[4]:
            adapter.reset(); native.reset()
    # Original callback, exact switch clock/value; no invented coefficients.
    schedule_checks = []
    original_step = adapter.env.total_steps
    for clock in [0, 199999, 200000, 1000000]:
        adapter.env.total_steps = clock
        value = adapter.apply_original_schedule()
        expected = next(value for step, value in reversed(adapter.coverage_ce_speed_weight_schedule) if step <= clock)
        same(value, expected)
        schedule_checks.append(dict(clock=clock, value=value))
    adapter.env.total_steps = original_step
    adapter.apply_original_schedule()
    fixture = None
    if args.arm == 'R2':
        # Off-line fixed state tests actual native units/visibility/rotation.
        sample = copy.deepcopy(adapter.env)
        p = sample.pursuers[0]
        p.x, p.y, p.theta, p.velocity, p.is_pursuing = 10., 10., np.pi/2, np.array([0., 2.]), True
        p.deactivated = p.collision = False
        for q in sample.pursuers[1:]:
            q.x, q.y, q.deactivated = 90., 90., False
        e = sample.evaders[0]
        e.x, e.y, e.deactivated = 100., 10., False
        for obstacle in sample.obstacles:
            obstacle.x, obstacle.y = 80., 80.
        before = digest_graph(sample)
        row = terl_observations(sample)[0]
        same(before, digest_graph(sample))
        np.testing.assert_allclose(row['self'], [2., 0., 20., 1.], atol=1e-12)
        assert row['masks'][6] and not row['masks'][1:6].any()
        np.testing.assert_allclose(row['evaders'][0, :2], [0., -90.], atol=1e-12)
        fixture = dict(self=row['self'].tolist(), distant_enemy_visible=bool(row['masks'][6]),
                       distant_friends_visible=bool(row['masks'][1:6].any()), physical_state_unmodified=True)
    # Genuine on-policy rollout/update and fresh-process exact continuation.
    adapter = CapabilityAdapter(config['learner']['seed'], args.arm)
    trainer = build(config, adapter, 'cpu')
    batch, episodes = collect(trainer, adapter, 256)
    logp = trainer.assert_behavior_log_probs(batch)
    metrics = trainer.update(batch, categorical=True)
    assert all(np.isfinite(value) for value in metrics.values())
    checkpoint_hash = save_checkpoint(args.output / 'resume.pt', trainer, adapter, config, 256,
                                      int(batch['active_mask'].sum()), int(metrics['minibatch_updates']))
    expected = continuation(trainer, adapter)
    atomic_json(args.output / 'expected.json', expected)
    child = subprocess.run([sys.executable, str(Path(__file__)), '--arm', args.arm,
                            '--output', str(args.output), '--restore'], cwd=ROOT,
                           env=os.environ.copy(), capture_output=True, text=True)
    (args.output / 'restore_stdout.log').write_text(child.stdout)
    (args.output / 'restore_stderr.log').write_text(child.stderr)
    if child.returncode:
        raise RuntimeError('fresh-process resume failed: ' + child.stderr)
    # Smoke of actual evaluator; separate seed/output, not performance evidence.
    evaluation = evaluate(args.output / 'resume.pt', args.output / 'evaluator_smoke.json', config,
                          'screen', 'cpu', smoke_episodes=1)
    same(e1_sources(), frozen['source_manifest'])
    atomic_json(args.output / 'report.json', dict(arm=args.arm, classification='ENGINEERING_VERIFICATION',
        performance_conclusion=None, independent_validation=False, frozen_E1_unchanged=True,
        provider_transitions_exact=64, pure_observation_queries=True, native_TERL_fixture=fixture,
        original_schedule_checks=schedule_checks, behavior_log_prob_max_error=logp,
        rollout_decisions=256, PPO_metrics=metrics, resume_checkpoint_sha256=checkpoint_hash,
        fresh_process_resume_exact=True, evaluator_executed=True,
        evaluator_smoke_episodes=len(evaluation['episodes']),
        selection_key=[float(value) if np.isfinite(value) else None for value in selection_key(evaluation)],
        sources=sources(), task_reference=config['task_reference']))
    print(json.dumps(dict(arm=args.arm, status='ENGINEERING_VERIFICATION_PASS', metrics=metrics)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['N2', 'R2'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--restore', action='store_true')
    main(parser.parse_args())
