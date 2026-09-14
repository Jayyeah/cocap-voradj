#!/usr/bin/env python3
"""Teacher-free scratch contract, random occupancy and bounded engineering smoke.

This entry point intentionally has NO formal-training mode. Reuses the current
Forward Final stream/transition collector and MAPPO update without calling their
BC factories, checkpoint exporters, teacher preflight or teacher evaluator.
"""
from __future__ import annotations
import argparse
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import numpy as np
import torch
import yaml
from cocap_voradj.models.continuous.local_entity_token_encoder import (
    LegacyVorAdjFeatureBackbone, LegacyVorAdjFeatureBackboneConfig,
)
from cocap_voradj.models.small_step_ac import CategoricalGridActor, CentralValueNetwork
from cocap_voradj.training.small_step_ac import MAPPOConfig, MAPPOTrainer, tensor_tree
from cocap_voradj.training.forward_final import CONFIG as ENV_CONFIG, check_env, make_env, scene_config
from cocap_voradj.training.runtime_semantics import assert_runtime, initial_state_fingerprint
from cocap_voradj.training.trainer import CoCapTrainer, set_global_config
from cocap_voradj.control.apf import ApfAgent
from tools.train_forward_final_ppo_20260909 import (
    FinalMissionStream, collect_transition, empty_rollout, stack_rollout,
    tensor_hash, rollout_log_probs, exact_update_diagnostics,
)

CONFIG = ROOT / 'configs/experiments/forward_final_scratch_mappo_20260914/canonical.yaml'
SCHEMA = 'forward-final-scratch-preflight-v1'
PHASES = ('pre_capture', 'post_capture', 'pure_coverage')
COMPONENTS = ('reward_capture', 'reward_coverage', 'reward_safety', 'reward_terminal')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_contract(path=CONFIG):
    c = yaml.safe_load(Path(path).read_text())
    if c['schema'] != SCHEMA or c['formal_training'] != 'HOLD':
        raise ValueError('Preflight only: FORMAL TRAINING: HOLD')
    assert c['initialization'] == 'random'
    assert all(c[k] is None for k in ('teacher_checkpoint', 'bc_actor', 'bc_dataset'))
    assert not any(c[k] for k in ('teacher_q', 'teacher_kl', 'imitation_loss'))
    assert (ROOT / c['environment_config']).resolve() == ENV_CONFIG.resolve()
    resolved = scene_config('mixed')
    iqn, perception = resolved['iqn'], resolved['perception']
    # Read ONLY architecture scalars, never teacher parameters/data.
    expected = dict(hidden_dim=iqn['hidden_dim'], num_heads=iqn['num_heads'],
                    num_layers=iqn['num_layers'], self_feature_dim=iqn['self_feature_dim'],
                    max_pursuers=perception['max_pursuer_num'],
                    max_evaders=perception['max_evader_num'],
                    max_obstacles=perception['max_obstacle_num'],
                    pursuing_embed_dim=iqn.get('pursuing_embed_dim', 8), dropout=.1)
    assert c['actor'] == expected, (c['actor'], expected)
    # Explicit baseline: production Direct recipe (not later diagnostic LR probes).
    assert c['ppo'] == dataclasses.asdict(MAPPOConfig(
        ppo_epochs=3, minibatches=2, actor_lr=3e-5, critic_lr=1e-4,
        target_kl=.02, value_norm=True))
    assert c['critic'] == dict(hidden_dim=256, num_heads=8, num_layers=4,
                              self_feature_dim=9, max_agents=4, max_evaders=8, max_obstacles=5)
    assert c['actor_head'] == dict(orthogonal_policy_head=True, output_gain=.01)
    assert c['rollout_length'] == 256
    assert c['reward_clock_offset'] == 2000000 and c['ce_speed_weight'] == .0005
    assert c['reset'] == dict(cycle=['mixed', 'coverage'], pool_initial_size=0,
                             pool_capacity=1000, captured_state_ratio=.75,
                             map_random_ratio_within_non_capture=.5)
    return c


def make_actor(c, seed, device='cpu'):
    torch.manual_seed(seed)
    backbone = LegacyVorAdjFeatureBackbone(LegacyVorAdjFeatureBackboneConfig(**c['actor']))
    grid = np.asarray([(a, w) for a in (-.4, 0., .4) for w in (-np.pi/6, 0., np.pi/6)])
    return CategoricalGridActor(backbone, grid, **c['actor_head']).to(device).eval()


def make_trainer(c, seed, device='cpu'):
    actor = make_actor(c, seed, device)
    torch.manual_seed(seed)  # Same initial V as the Direct branch at the same seed.
    value = CentralValueNetwork(**c['critic'])
    trainer = MAPPOTrainer(actor, value, MAPPOConfig(**c['ppo']), device)
    assert not trainer.actor_optimizer.state and not trainer.value_optimizer.state
    assert trainer.update_count == 0
    assert all(float(getattr(trainer.value_norm, k)) == 0 for k in
               ('running_mean', 'running_mean_sq', 'debiasing_term'))
    assert float(trainer.value_norm.mean) == 0 and float(trainer.value_norm.std) == 1
    assert all(p.requires_grad for p in trainer.actor.parameters())
    return trainer


def forbid_teacher_dependencies(output):
    """Fail closed on teacher execution or reads from any existing artifact bank."""
    from cocap_voradj.models.iqn import CoCapIQN
    def deny(*args, **kwargs):
        raise RuntimeError('Forbidden teacher / transferred-weight dependency')
    CoCapIQN.__init__ = deny
    CoCapIQN.load = deny
    CoCapIQN.forward = deny
    LegacyVorAdjFeatureBackbone.load_legacy_iqn_state_dict = deny
    artifact_root, allowed = (ROOT / 'artifacts').resolve(), Path(output).resolve()
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(artifact_root) and not path.is_relative_to(allowed):
            raise RuntimeError(f'Existing artifact dependency forbidden: {path}')
    sys.addaudithook(audit)


def validate_stream(c, stream):
    assert stream.global_step == c['reward_clock_offset']
    assert stream.current_coverage_ce_speed_weight == c['ce_speed_weight']
    assert len(stream.recovery_init_pool) == 0
    assert stream.recovery_init_pool.maxlen == c['reset']['pool_capacity']
    assert stream.recovery_from_capture_ratio == c['reset']['captured_state_ratio']
    assert stream.recovery_map_random_ratio_within_non_capture == c['reset']['map_random_ratio_within_non_capture']
    assert all(e.transition_semantics == c['transition_semantics'] for e in stream.envs.values())
    return {task: check_env(env) for task, env in stream.envs.items()}


def stats(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(n=0, mean=None, p50=None, p90=None)
    return dict(n=len(a), mean=float(a.mean()), p50=float(np.quantile(a, .5)), p90=float(np.quantile(a, .9)))


def bucket():
    return dict(rows=0, nonzero=0, positive=0, total=0., absolute=0.,
                components={k: dict(nonzero=0, total=0.) for k in COMPONENTS})


def add_reward(b, reward, meta):
    b['rows'] += 1
    b['nonzero'] += int(abs(reward) > 1e-8)
    b['positive'] += int(reward > 1e-8)
    b['total'] += float(reward)
    b['absolute'] += abs(float(reward))
    for k in COMPONENTS:
        v = float(meta[k])
        b['components'][k]['nonzero'] += int(abs(v) > 1e-8)
        b['components'][k]['total'] += v


def finish_bucket(b):
    n = b['rows']
    return {**b, 'nonzero_density': b['nonzero']/n if n else None,
            'positive_density': b['positive']/n if n else None,
            'mean_reward': b['total']/n if n else None,
            'mean_absolute_reward': b['absolute']/n if n else None,
            'components': {k: {**v, 'nonzero_density': v['nonzero']/n if n else None,
                              'mean_reward': v['total']/n if n else None}
                           for k, v in b['components'].items()}}


@torch.no_grad()
def observe_episode(actor, env, observations, step_fn, scene, mode='sample', max_steps=None):
    """No central V / teacher Q; occupancy labels never enter actor inputs."""
    fingerprint = initial_state_fingerprint(env)
    check_env(env)
    assert_runtime(env, actor, categorical=True)
    device = next(actor.parameters()).device
    phases = {p: bucket() for p in PHASES}
    phase_steps = dict.fromkeys(PHASES, 0)
    support = bucket()
    histogram = np.zeros(9, dtype=np.int64)
    entropy_sum = 0.
    active_rows = 0
    capture_types = []
    capture_step = None
    ring_steps = {2: 0, 3: 0}
    ring_run = ring_max = 0
    collision = boundary = False
    collision_types = {}
    horizon = env.episode_max_length if max_steps is None else min(max_steps, env.episode_max_length)
    for step in range(1, horizon + 1):
        active = [i for i, o in enumerate(observations) if o is not None]
        local = tensor_tree({k: np.stack([observations[i][k] for i in active])
                             for k in observations[active[0]]}, device)
        d = actor.distribution(local)
        indices = d.logits.argmax(-1) if mode == 'argmax' else d.sample()
        entropy_sum += float(d.entropy().sum())
        active_rows += len(active)
        histogram += np.bincount(indices.cpu().numpy(), minlength=9)
        phase = 'pure_coverage' if scene == 'coverage' else ('post_capture' if capture_step is not None else 'pre_capture')
        phase_steps[phase] += 1
        command = [None] * len(observations)
        for i, a in zip(active, indices.cpu().tolist()):
            command[i] = int(a)
        outcome = step_fn(command)
        # Successor ring occupancy, reconstruct removed targets from event participants.
        events = list(env.last_capture_events)
        sizes = [sum(np.hypot(p.x-e.x, p.y-e.y) <= 8. for p in env.pursuers if not p.deactivated)
                 for e in env.evaders if not e.deactivated]
        sizes += [len(e['participants']) for e in events]
        size = max(sizes, default=0)
        for k in (2, 3):
            ring_steps[k] += int(size >= k)
        ring_run = ring_run + 1 if size >= 3 else 0
        ring_max = max(ring_max, ring_run)
        capture_types.extend(e['capture_type'] for e in events)
        if events and capture_step is None:
            capture_step = step
        for i in active:
            meta = outcome.infos[i]['replay_metadata']
            add_reward(phases[phase], outcome.rewards[i], meta)
            if meta['support_candidate']:
                add_reward(support, outcome.rewards[i], meta)
        for event in env.last_collision_events:
            collision = True
            kind = event.get('type', 'unknown')
            boundary |= kind == 'boundary'
            collision_types[kind] = collision_types.get(kind, 0) + 1
        observations = outcome.observations
        if all(outcome.dones):
            break
    record = env.episode_record(task='coverage' if scene == 'coverage' else 'mix')
    collision |= bool(record.get('collision_event', False))
    boundary |= bool(record.get('boundary_collision_event', False))
    captured = bool(record['captured']) if scene == 'mixed' else False
    ce = bool(env.post_capture_coverage_success)
    safe = ce and (scene == 'coverage' or captured) and not collision and record['all_pursuers_active']
    return dict(scene=scene, mode=mode, length=step, native_done=bool(all(outcome.dones)),
                initial_state_fingerprint=fingerprint, active_rows=active_rows,
                phase_steps=phase_steps, phases=phases, support=support,
                captured=captured, normal_capture=captured and 'loose' in capture_types,
                stationary_capture=captured and 'stationary' in capture_types,
                capture_step=capture_step, ce_success=ce, safe_complete=bool(safe),
                collision=collision, boundary=boundary, collision_types=collision_types,
                ring2_steps=ring_steps[2], ring3_steps=ring_steps[3], ring3_max_hold=ring_max,
                entropy_sum=entropy_sum, action_histogram=histogram.tolist(),
                ce_rms=stats([record['coverage_ce_center_rms']])['mean'],
                area_cv=stats([record['coverage_strict_area_cv']])['mean'],
                episode_seconds=.5*step, mission_seconds=.5*step if safe else None,
                recovery_seconds=.5*(step-capture_step) if safe and capture_step else None)


def summarize(records):
    result = {}
    for scene in ('mixed', 'coverage'):
        rows = [r for r in records if r['scene'] == scene]
        if not rows:
            continue
        count = len(rows)
        steps = sum(r['length'] for r in rows)
        agent_rows = sum(r['active_rows'] for r in rows)
        merged = {p: bucket() for p in (*PHASES, 'support')}
        for r in rows:
            for phase, b in merged.items():
                source = r['support'] if phase == 'support' else r['phases'][phase]
                for key in ('rows', 'nonzero', 'positive', 'total', 'absolute'):
                    b[key] += source[key]
                for k in COMPONENTS:
                    for key in ('nonzero', 'total'):
                        b['components'][k][key] += source['components'][k][key]
        result[scene] = dict(
            episodes=count, env_steps=steps, active_agent_rows=agent_rows,
            **{k+'_count': sum(r[k] for r in rows) for k in
               ('captured', 'normal_capture', 'stationary_capture', 'ce_success', 'safe_complete', 'collision', 'boundary')},
            ring2_episode_count=sum(r['ring2_steps'] > 0 for r in rows),
            ring3_episode_count=sum(r['ring3_steps'] > 0 for r in rows),
            ring2_step_fraction=sum(r['ring2_steps'] for r in rows)/steps,
            ring3_step_fraction=sum(r['ring3_steps'] for r in rows)/steps,
            ring3_max_hold_steps=stats([r['ring3_max_hold'] for r in rows]),
            phase_step_counts={p: sum(r['phase_steps'][p] for r in rows) for p in PHASES},
            phase_rewards={p: {**finish_bucket(b), 'active_row_fraction': b['rows']/agent_rows}
                           for p, b in merged.items()},
            episode_length=stats([r['length'] for r in rows]),
            ce_rms=stats([r['ce_rms'] for r in rows if r['ce_rms'] is not None]),
            area_cv=stats([r['area_cv'] for r in rows if r['area_cv'] is not None]),
            entropy_nats=sum(r['entropy_sum'] for r in rows)/agent_rows,
            action_histogram=np.sum([r['action_histogram'] for r in rows], axis=0).tolist(),
            failed_or_censored_count=sum(not r['safe_complete'] for r in rows),
            native_done_count=sum(r['native_done'] for r in rows))
    return result


def occupancy(c, out, seed, episodes):
    seed_all(seed)
    actor = make_actor(c, seed)
    before = tensor_hash(actor.state_dict())
    stream = FinalMissionStream(seed, out / 'stream')
    runtime = validate_stream(c, stream)
    records = []
    for i in range(episodes):
        scene = 'mixed' if stream.task == 'voradj' else 'coverage'
        row = observe_episode(actor, stream.env, stream.observations, stream.step, scene)
        row.update(episode=i, reset_source=stream.last_reset_source[stream.task], pool_before=len(stream.recovery_init_pool))
        assert row['native_done'], 'Occupancy must finish native episodes, never shortened horizons'
        stream.finish()
        row['pool_after'] = len(stream.recovery_init_pool)
        records.append(row)
        write_json(out / 'progress.json', dict(episodes_complete=i+1, episodes=episodes,
                   env_steps=sum(r['length'] for r in records), status='running'))
    assert tensor_hash(actor.state_dict()) == before
    summary = summarize(records)
    mixed = summary.get('mixed', {})
    report = dict(status='COMPLETE', formal_training='HOLD', policy='random-init categorical sample, frozen',
                  seed=seed, actor_sha256=before, actor_unchanged=True, optimizer_steps=0,
                  runtime=runtime, summary=summary, records=records, reset_counts=dict(stream.reset_counts),
                  final_pool_size=len(stream.recovery_init_pool),
                  interpretation='PHASE VISITATION BOTTLENECK' if mixed.get('phase_step_counts', {}).get('post_capture', 0) == 0
                  else 'Exposure observed; inspect density, not a learnability conclusion')
    write_json(out / 'report.json', report)
    return report


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device)


def assert_tree_equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_tree_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    elif torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b


def append_transition(rollout, row):
    for k, v in row.items():
        rollout[k].append(v)


def smoke(c, out, seed, device):
    seed_all(seed)
    trainer = make_trainer(c, seed, device)
    initial = tensor_hash(trainer.actor.state_dict())
    assert tensor_hash(make_actor(c, seed, device).state_dict()) == initial
    assert tensor_hash(make_actor(c, seed+1, device).state_dict()) != initial
    seed_all(seed)
    stream = FinalMissionStream(seed, out / 'stream')
    runtime = validate_stream(c, stream)
    assert_runtime(stream.env, trainer.actor, categorical=True)
    metrics = []
    for _ in range(2):
        rollout = empty_rollout()
        for _ in range(16):
            row, _ = collect_transition(trainer, stream)
            append_transition(rollout, row)
        batch = stack_rollout(rollout)
        before = rollout_log_probs(trainer, batch)
        result = trainer.update(batch, categorical=True)
        result.update(exact_update_diagnostics(before, rollout_log_probs(trainer, batch)))
        assert all(np.isfinite(v) for v in result.values())
        metrics.append(result)
    assert initial != tensor_hash(trainer.actor.state_dict())
    assert trainer.actor_optimizer.state and trainer.value_optimizer.state
    assert float(trainer.value_norm.debiasing_term) > 0
    partial = empty_rollout()
    for _ in range(4):
        row, _ = collect_transition(trainer, stream)
        append_transition(partial, row)
    payload = dict(schema=SCHEMA, initialization='random', config=c,
                   transition_semantics=c['transition_semantics'], step=36,
                   trainer=trainer.state_dict(), stream_state=stream.__dict__, rollout=partial,
                   rng=rng_state(trainer.device))
    checkpoint = out / 'scratch_resume_step36.pt'
    if checkpoint.exists():
        raise ValueError('Refuse checkpoint overwrite')
    torch.save(payload, checkpoint)
    expected = [collect_transition(trainer, stream)[0] for _ in range(4)]
    expected_rng = rng_state(trainer.device)
    restored = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert restored['schema'] == SCHEMA and restored['config'] == c
    clone = make_trainer(c, seed+2, device)
    clone.load_state_dict(restored['trainer'])
    stream2 = FinalMissionStream.__new__(FinalMissionStream)
    stream2.__dict__.update(restored['stream_state'])
    restore_rng(restored['rng'], clone.device)
    actual = [collect_transition(clone, stream2)[0] for _ in range(4)]
    assert_tree_equal(expected, actual)
    assert_tree_equal(expected_rng, rng_state(clone.device))
    assert_tree_equal(restored['rollout'], partial)
    assert_tree_equal(trainer.state_dict(), clone.state_dict())
    # Also prove resumed partial rollout can take the same next optimizer update.
    for row in actual:
        append_transition(partial, row)
    next_batch = stack_rollout(partial)
    update_rng = rng_state(trainer.device)
    trainer.update(next_batch, categorical=True)
    restore_rng(update_rng, clone.device)
    clone.update(next_batch, categorical=True)
    assert_tree_equal(trainer.state_dict(), clone.state_dict())
    eval_records = []
    for mode in ('argmax', 'sample'):
        for scene in ('mixed', 'coverage'):
            seed_all(seed+100)
            env, obs = make_env(scene, seed+100)
            apf = [ApfAgent(e.a, e.w) for e in env.evaders]
            def step_fn(actions):
                set_global_config(env.config)
                return env.step(actions, [a.act(e, env.pursuers, env.obstacles) for a, e in zip(apf, env.evaders)])
            # Reuse native Final APF method via a minimal host, avoiding evaluator IQN calls.
            host = type('APFHost', (), {})()
            host.envs = {'eval': env}
            host.apf_agents = {}
            def step_fn(actions):
                set_global_config(env.config)
                return env.step(actions, CoCapTrainer._evader_actions(host, 'eval'))
            r = observe_episode(clone.actor, env, obs, step_fn, scene, mode, max_steps=4)
            eval_records.append(r)
    report = dict(status='SMOKE_PASS', formal_training='HOLD', seed=seed, device=device,
                  unique_training_env_steps=40, replay_verification_steps=4,
                  initial_optimizer_empty=True, initial_valuenorm_zero=True,
                  initial_value_scale=1., random_init_same_seed_reproducible=True,
                  different_seed_differs=True, all_actor_parameters_trainable=True,
                  no_teacher_dependency_guard=True, initial_actor_sha256=initial,
                  runtime=runtime, metrics=metrics, resume_transition_rng_optimizer_bit_exact=True,
                  checkpoint=str(checkpoint), eval_records=eval_records,
                  caveat='16-step rollout / 3 tiny updates only; no performance or learnability claim')
    write_json(out / 'report.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('occupancy', 'smoke'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=2026091401)
    p.add_argument('--episodes', type=int, default=40)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    if not 2 <= args.episodes <= 100 or args.episodes % 2:
        p.error('Bounded random occupancy: 2..100 even number of native episodes')
    if args.mode == 'occupancy' and args.device != 'cpu':
        p.error('Occupancy is CPU only; CUDA is reserved for the short smoke')
    if args.output.exists():
        p.error('Fresh output directory required')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    c = load_contract()
    forbid_teacher_dependencies(args.output)
    start = time.monotonic()
    write_json(args.output / 'launch.json', dict(config=c, mode=args.mode, seed=args.seed,
               device=args.device, pid=os.getpid(), formal_training='HOLD',
               teacher_execution_blocked=True, external_artifact_reads_blocked=True,
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    result = occupancy(c, args.output, args.seed, args.episodes) if args.mode == 'occupancy' else smoke(c, args.output, args.seed, args.device)
    write_json(args.output / 'completion.json', dict(status=result['status'], elapsed_seconds=time.monotonic()-start))
    print(json.dumps(dict(status=result['status'], output=str(args.output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
