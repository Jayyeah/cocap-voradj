"""Batch01 contracts + actual 256/8/4 entry regression, without long training."""
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from terl_mappo.native import NativeStage1, MarineEnv, central_state, pack_local, stage1_schedule
from terl_mappo.run import build, collect, save_checkpoint, rng_state, restore_rng, file_hash
from terl_mappo.batch01 import contracts
from terl_mappo.batch01.contracts import create_lock, fingerprint, verify_base, resolve_config, validate_resources
from terl_mappo.batch01.interfaces import Hooks, ActionBackend, assemble, multiscale_state, multiscale_critic
from terl_mappo.batch01.checkpoints import save_bound, load_bound, load_anchor, retention_plan
from terl_mappo.batch01.evaluation import isolated_rng, seed_manifest, summarize_scene, performance_report
from terl_mappo.batch01.provenance import active_values
from terl_mappo.evaluate import summarize
from cocap_voradj.training.small_step_ac import tensor_tree

torch.set_num_threads(1)


def config():
    return json.loads((contracts.ROOT / contracts.ANCHOR_CONFIG).read_text())


def committed_lock():
    return json.loads((contracts.ROOT / contracts.LOCK_PATH).read_text())


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.detach().cpu(), b.detach().cpu())
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: same(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): same(x, y)
    elif isinstance(a, np.random.RandomState):
        same(a.get_state(), b.get_state())
    elif hasattr(a, 'query') and hasattr(a, 'data'):
        # scipy KDTree's pickle includes storage/alias details. Check every
        # geometric input and exact spatial queries rather than pickle bytes.
        same(a.data, b.data); same(a.leafsize, b.leafsize); same(a.boxsize, b.boxsize)
        same(a.query(a.data, k=len(a.data)), b.query(b.data, k=len(b.data)))
    elif hasattr(a, '__dict__'):
        assert type(a) is type(b)
        same(a.__dict__, b.__dict__)
    else: assert a == b


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_formal_entry_t0_full_update_and_exact_resume(device, tmp_path):
    if device.startswith('cuda') and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    c = config()
    assert (c['hidden_dim'], c['num_heads'], c['num_layers']) == (256, 8, 4)
    runtime = assemble(c, device)
    assert len(runtime.trainer.actor.transformer_encoder.layers) == 4
    assert runtime.trainer.actor.transformer_encoder.layers[0].self_attn.num_heads == 8
    actor_state = copy.deepcopy(runtime.trainer.actor.state_dict())
    critic_state = copy.deepcopy(runtime.trainer.value.state_dict())
    batch, episodes = runtime.collect(256)
    runtime_rng = rng_state()
    metrics = runtime.update(batch)
    expected_after = copy.deepcopy(runtime.trainer.state_dict())
    expected_rng = rng_state()
    raw = build(c, device); adapter = NativeStage1(c['seed'])
    same(actor_state, raw.actor.state_dict()); same(critic_state, raw.value.state_dict())
    reference_batch, reference_episodes = collect(raw, adapter, 256)
    same(batch, reference_batch); same(episodes, reference_episodes)
    same(runtime_rng, rng_state())
    reference_metrics = raw.update(reference_batch, True)
    same(metrics, reference_metrics); same(expected_after, raw.state_dict()); same(expected_rng, rng_state())
    assert all(np.isfinite(x) for x in metrics.values())
    active = active_values(runtime)
    assert active['actor_parameters'] > 2000000 and active['critic_parameters'] > 2000000
    manifest = {'base': {'candidate_sha': 'a' * 40, 'lock_sha256': 'b' * 64}, 'contract': active,
                'common_source_hashes': contracts.source_inventory()}
    path = tmp_path / 'formal.pt'
    try:
        save_bound(path, runtime, manifest, 256, 768, int(metrics['minibatch_updates']))
        expected, expected_episodes = runtime.collect(8)
        expected_metrics = runtime.update(expected)
        after = copy.deepcopy(runtime.trainer.state_dict())
        env_after = runtime.adapter.state_dict()
        rng_after = rng_state()
        restored = assemble(c, device)
        loaded = load_bound(path, restored, manifest)
        assert loaded['steps'] == 256 and loaded['agent_transitions'] == 768
        actual, actual_episodes = restored.collect(8)
        actual_metrics = restored.update(actual)
        same(expected, actual); same(expected_episodes, actual_episodes); same(expected_metrics, actual_metrics)
        same(after, restored.trainer.state_dict()); same(rng_after, rng_state())
        same(env_after, restored.adapter.state_dict())
        if device.startswith('cuda'): assert torch.cuda.max_memory_allocated() / 1024**2 < 2048
        bad = copy.deepcopy(manifest); bad['base']['candidate_sha'] = 'c' * 40
        with pytest.raises(ValueError, match='BASE/hash'): load_bound(path, restored, bad)
        bad = copy.deepcopy(manifest); bad['common_source_hashes']['src/terl_mappo/native.py'] = '0' * 64
        with pytest.raises(ValueError, match='stale/undeclared'): load_bound(path, restored, bad)
    finally:
        path.unlink(missing_ok=True); path.with_suffix('.batch01.json').unlink(missing_ok=True)


def delta(lock, line='T0'):
    return {'schema': 'terl.batch01.delta.v1', 'line': line,
            'base': {'parent_sha': contracts.PARENT_SHA, 'candidate_sha': 'a' * 40,
                     'lock_sha256': fingerprint(lock)},
            'config_changes': [], 'source_changes': [], 'extensions': {}}


def test_real_source_lock_matches_anchor_and_rejects_wrong_pin():
    lock = committed_lock(); d = delta(lock)
    assert verify_base(lock, d, fingerprint(lock)) == config()
    with pytest.raises(ValueError, match='pin mismatch'): verify_base(lock, d, '0' * 64)
    bad = copy.deepcopy(d); bad['base']['candidate_sha'] = 'HEAD'
    with pytest.raises(ValueError, match='explicit full'): verify_base(lock, bad, fingerprint(lock))


def test_fresh_process_all_transitive_runtime_imports_are_locked():
    code = ('from terl_mappo.batch01.provenance import assert_loaded_sources_covered; '
            'from terl_mappo.batch01.source_guard import SourceGuard; '
            'SourceGuard.committed().check()')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_added_removed_and_changed_source_require_exact_delta(monkeypatch):
    lock = committed_lock(); d = delta(lock)
    changed = dict(lock['common_sources'])
    path = 'src/terl_mappo/native.py'
    changed[path] = '1' * 64
    monkeypatch.setattr(contracts, 'source_inventory', lambda root: changed)
    with pytest.raises(ValueError, match='undeclared'): verify_base(lock, d, fingerprint(lock))
    d['source_changes'] = [{'path': path, 'before': lock['common_sources'][path], 'after': changed[path],
                            'reason': 'candidate only', 'science_impact': 'requires independent review'}]
    with pytest.raises(ValueError, match='candidate_fix'): verify_base(lock, d, fingerprint(lock))
    d['source_changes'][0]['candidate_fix'] = True
    verify_base(lock, d, fingerprint(lock))
    changed['src/terl_mappo/batch01/extensions/n1.py'] = '2' * 64
    with pytest.raises(ValueError, match='undeclared'): verify_base(lock, d, fingerprint(lock))
    changed.pop(path)
    with pytest.raises(ValueError, match='after hash'): verify_base(lock, d, fingerprint(lock))


def test_config_scope_before_values_and_no_silent_unknown_keys():
    lock = committed_lock(); d = delta(lock, 'P1')
    d['config_changes'] = [{'path': 'ppo.target_kl', 'before': 0.02, 'after': 0.01, 'reason': 'controlled PPO fork'}]
    assert resolve_config(lock, d)['ppo']['target_kl'] == 0.01
    d['line'] = 'T0'
    with pytest.raises(ValueError, match='outside'): resolve_config(lock, d)
    d['line'] = 'P1'; d['config_changes'][0]['before'] = 999
    with pytest.raises(ValueError, match='before'): resolve_config(lock, d)
    d = delta(lock); d['unused_setting'] = True
    with pytest.raises(ValueError, match='unknown delta fields'): resolve_config(lock, d)


def test_reward_identity_hook_preserves_all_rollout_fields_and_rng():
    c = config()
    expected = assemble(c); batch, episodes = expected.collect(4); rng = rng_state()
    actual = assemble(c, hooks=Hooks(reward_transform=lambda reward, info: reward), line='R1')
    changed, changed_episodes = actual.collect(4)
    same(batch, changed); same(episodes, changed_episodes); same(rng, rng_state())
    with pytest.raises(ValueError, match='outside'): Hooks(reward_transform=lambda r, i: r).validate_line('N1')


def test_formal_256_8_4_fully_masked_no_target_forward_backward():
    runtime = assemble(config())
    local = tensor_tree(pack_local(runtime.adapter.observations), 'cpu')
    local['masks'].zero_()
    logits = runtime.trainer.actor(local)
    distribution = runtime.trainer.actor.distribution(local)
    assert torch.isfinite(logits).all() and torch.isfinite(distribution.entropy()).all()
    assert not runtime.trainer.actor._last_target_weights.any()
    logits.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in runtime.trainer.actor.parameters() if p.grad is not None)


def test_runtime_assertions_reject_unapplied_optimizer_and_dropout():
    runtime = assemble(config())
    runtime.trainer.actor_optimizer.param_groups[0]['lr'] = .01
    with pytest.raises(ValueError, match='optimizer contract'): active_values(runtime)
    runtime.trainer.actor_optimizer.param_groups[0]['lr'] = config()['ppo']['actor_lr']
    runtime.trainer.actor.train()
    with pytest.raises(ValueError, match='eval mode'): active_values(runtime)


def test_c0_interface_rejects_continuous_coercion_and_requires_adapter():
    categorical = ActionBackend()
    categorical.validate(np.array([0, 4, 8]), 3)
    with pytest.raises(ValueError, match='integer'): categorical.validate(np.zeros((3, 2)), 3)
    with pytest.raises(ValueError, match='integer'): categorical.validate(np.array([0., 4., 8.]), 3)
    continuous = ActionBackend('tanh_gaussian_aw', 'physical_differential_mc_nats',
                               'tanh_and_scale_corrected_density', False)
    continuous.validate(np.zeros((3, 2)), 3)
    with pytest.raises(ValueError, match='physical action'):
        assemble(config(), hooks=Hooks(backend=continuous), line='C0')
    assert continuous.entropy_measure != categorical.entropy_measure


@pytest.mark.parametrize('p,e,o,k', [(3, 1, 0, 4), (4, 1, 1, 6), (7, 2, 2, 8)])
def test_multiscale_formal_actor_critic_observation_shapes(p, e, o, k):
    schedule = stage1_schedule()
    schedule.update(num_pursuers=[p], num_evaders=[e], num_obstacles=[o], num_cores=[k])
    env = MarineEnv(seed=9, schedule=schedule)
    (obs, _), _ = env.reset()
    local = pack_local(obs)
    assert local['self'].shape == (p, 4) and local['masks'].shape == (p, 19)
    state = multiscale_state(env, max_cores=k)
    if (p, e, o, k) == (3, 1, 0, 4): same(state, central_state(env))
    from terl_mappo.model import TERLActor
    actor = TERLActor(256, 8, 4)
    critic = multiscale_critic(agents=p, evaders=e, obstacles=o, max_cores=k)
    logits = actor(tensor_tree(local, 'cpu'))
    value = critic(tensor_tree({key: v[None] for key, v in state.items()}, 'cpu'))
    assert logits.shape == (p, 9) and value.shape == (1, p)
    (logits.square().mean() + value.square().mean()).backward()
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    assert all(torch.isfinite(v.grad).all() for v in list(actor.parameters()) + list(critic.parameters()) if v.grad is not None)


def test_evaluation_domains_and_all_rng_restored_on_exception():
    lock = committed_lock()
    seeds = seed_manifest(config(), lock['evaluation_protocol'])
    assert len(seeds['final']) == 50 and not set(seeds['screen']) & set(seeds['selection'])
    state = rng_state()
    with pytest.raises(RuntimeError):
        with isolated_rng():
            random.seed(123); np.random.seed(123); torch.manual_seed(123)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(123)
            raise RuntimeError('evaluator error')
    same(state, rng_state())
    protocol = copy.deepcopy(lock['evaluation_protocol']); protocol['final']['seed_base'] = protocol['selection']['seed_base']
    with pytest.raises(ValueError, match='overlap'): seed_manifest(config(), protocol)


def episode(mode='argmax', capture=True):
    return {'mode': mode, 'capture': capture, 'normal_capture': capture, 'collision': not capture,
            'ring2': True, 'ring3': capture, 'strict_geometry': capture, 'censored': not capture,
            'capture_time': 10. if capture else None, 'action_entropy': 1., 'episode_reward': 2.,
            'collision_types': [] if capture else ['pursuer_evader'], 'reward_components': {'time': -2., 'goal': 4.}}


def test_metrics_scene_labels_time_censoring_entropy_and_best_last():
    rows = [episode(), episode(capture=False)]
    scene = {'pursuers': 3, 'evaders': 1, 'obstacles': 0, 'cores': 4, 'map': [120, 120],
             'horizon_decisions': 3001, 'decision_seconds': .5}
    r = summarize_scene(rows, scene, ActionBackend())
    assert r['normal_capture_rate'] == .5 and r['success_n'] == r['censored_n'] == 1
    assert r['collision_failure_n'] == 1 and r['time_limit_censored_n'] == 0
    assert r['capture_time_fraction_of_horizon_mean'] == pytest.approx(10 / 1500.5)
    with pytest.raises(ValueError, match='never silently pool'): summarize_scene([{**rows[0], 'scene': {'pursuers': 4}}], scene, ActionBackend())
    point = lambda step, rate, domain='screen': {'steps': step, 'seed_domain': domain, 'checkpoint_sha256': str(step),
        'modes': {m: {'normal_capture_rate': rate, 'collision_rate': 1-rate, 'strict_geometry_rate': rate, 'ring3_rate': rate}
                  for m in ('argmax', 'sample')}}
    screens = [point(0, 0), point(775000, .9), point(1000000, .3)]
    selections = [point(775000, .85, 'selection')]; final = [point(775000, .9, 'final')]
    report = performance_report(screens, selections, final)
    assert report['best_screen']['steps'] == 775000 and report['last_screen']['steps'] == 1000000
    assert report['late_stage']['last_minus_best'] == pytest.approx(-.6)
    with pytest.raises(ValueError, match='held-out'): performance_report(screens, selections, [point(1000000, 1., 'final')])


def test_retention_does_not_touch_other_runs_symlinks_pending_best_or_latest(tmp_path):
    root = tmp_path / 'own'; directory = root / 'checkpoints'; directory.mkdir(parents=True)
    foreign = tmp_path / 'other'; foreign.mkdir()
    rows = [episode('argmax'), episode('sample')]
    def point(path, step):
        path.write_bytes(b'synthetic checkpoint for retention scope')
        digest = file_hash(path)
        path.with_suffix('.json').write_text(json.dumps({'steps': step, 'sha256': digest}))
        return {'checkpoint': str(path), 'steps': step, 'checkpoint_sha256': digest, 'seed_domain': 'screen',
                'episodes_per_mode': 1, 'episodes': rows, 'modes': {m: summarize([r for r in rows if r['mode'] == m]) for m in ('argmax', 'sample')}}
    a = point(directory / 'step_000025000.pt', 25000)
    b = point(foreign / 'step_000050000.pt', 50000)
    c = point(directory / 'step_000075000.pt', 75000)
    os.link(directory / 'step_000075000.pt', directory / 'best.pt')
    (directory / 'step_000050000.pt').symlink_to(b['checkpoint'])
    assert retention_plan(root, [a, b, c], keep_steps=set(), pending_steps={25000}) == []
    plan = retention_plan(root, [a, b, c], keep_steps=set())
    assert [x['steps'] for x in plan] == [25000]
    assert all(Path(r['checkpoint']).exists() for r in [a, b, c])  # planner never removes files
    bad = copy.deepcopy(a); bad['checkpoint_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='hash mismatch'): retention_plan(root, [bad], keep_steps=set())
    bad = copy.deepcopy(a); bad['episodes'].pop()
    with pytest.raises(ValueError, match='partial'): retention_plan(root, [bad], keep_steps=set())


def test_resource_paths_and_gpu_assignment_fail_closed(monkeypatch):
    with pytest.raises(ValueError, match='/home/yjq'): validate_resources('/tmp/run')
    with pytest.raises(ValueError, match='resource limits'): validate_resources('/home/yjq/run', threads=128)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    with pytest.raises(ValueError, match='single physical'): validate_resources('/home/yjq/run', device='cuda:0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    assert validate_resources('/home/yjq/run', device='cuda:0')['physical_gpu'] == '1'


def test_shared_selected_checkpoint_strict_resume_readonly():
    path = Path('/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/runs/terl_mappo_stage1_seed9_1m_continuation/checkpoints/step_000775000.pt')
    if not path.exists(): pytest.skip('shared original anchor not available on this host')
    digest = '590d486876d4fca1cd75311c2dcc1255f847a1f756d91eed202e737193b58db4'
    runtime = assemble(config())
    loaded = load_anchor(path, digest, runtime)
    assert loaded['steps'] == 775000 and loaded['trainer']['update_count'] > 0
    batch, _ = runtime.collect(4)
    assert runtime.trainer.assert_behavior_log_probs(batch) < 1e-5
    assert file_hash(path) == digest
