#!/usr/bin/env python3
"""Targeted CE hold and physical pre-reset bootstrap checks; no PPO update."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'tools')]
from verify_batch02_e1_20261011 import (CoCapAdapter, SEED, original_oracle, make_oracle,
                                      compare_transition, same, plain)
from terl_mappo.batch02.runtime import build, central_state, collect
from terl_mappo.run import atomic_json, rng_state, restore_rng
from cocap_voradj.training.small_step_ac import tensor_tree, compute_gae


def main():
    torch.set_num_threads(1)
    module, root = original_oracle()
    adapter = CoCapAdapter(SEED, 'coverage')
    env, _, agents = make_oracle(module, root, 'coverage', SEED)
    # Diagnostic physical state in the successful coverage scene. Do not
    # change task thresholds, grace, reward parameters, obstacle map or spawn.
    for target in (adapter.env, env):
        for robot, position in zip(target.pursuers, [(30,30),(90,30),(30,90),(90,90)]):
            robot.x, robot.y = position; robot.speed = 0.; robot.velocity = np.zeros(2)
        target._invalidate_voronoi_cache()
        target.distribution_hold_steps = 29
    adapter.observations = adapter.env.get_observations()
    end, row = compare_transition(adapter, env, agents, [4] * 4)
    assert adapter.env.distribution_hold_steps == 30 and end
    assert adapter.env.post_capture_coverage_success
    assert row['phase'] == ['pure_coverage'] * 4
    hold = {'hold_before': 29, 'hold_after': 30, 'exact_task_parity': True,
            'terminated': row['terminated'], 'CE_success': True,
            'thresholds_unchanged': True, 'input_positions': [[30,30],[90,30],[30,90],[90,90]]}

    base = json.loads((ROOT / 'configs/experiments/terl_mappo_20261008/stage1.json').read_text())
    config = {key:base[key] for key in ('actor_seed','hidden_dim','num_heads','num_layers','masked_pool','ppo')}
    adapter = CoCapAdapter(SEED, 'mixed'); trainer = build(config, adapter)
    adapter.env.episode_step = 2999
    initial = adapter.state_dict(); initial_rng = rng_state()
    local = adapter.packed_observations(); global_obs = central_state(adapter.env)
    actions, logp, latent, values = trainer.act(local, {k:v[None] for k,v in global_obs.items()})
    _, _, terminal, truncated, end, _ = adapter.step(actions)
    assert end and truncated.all() and not terminal.any()
    physical_global = central_state(adapter.env)
    with torch.no_grad():
        expected = trainer.value(tensor_tree({k:v[None] for k,v in physical_global.items()}, trainer.device)).numpy()[0]
    physical_fingerprint = adapter.fingerprint()
    adapter.reset()
    with torch.no_grad():
        reset_value = trainer.value(tensor_tree({k:v[None] for k,v in central_state(adapter.env).items()},trainer.device)).numpy()[0]
    assert not np.array_equal(expected, reset_value)
    adapter.load_state_dict(initial); restore_rng(initial_rng, 'cpu')
    batch, episodes, trace = collect(trainer, adapter, 2)
    same(batch['actions'][0], actions); same(batch['next_values'][0], expected)
    assert trace[0]['fingerprint_before_reset'] == physical_fingerprint
    assert batch['episode_end'][0].all() and batch['truncated'][0].all()
    assert not batch['terminated'][0].any()
    assert trainer.update_count == 0
    # Existing GAE must bootstrap at a sampling limit without propagating the
    # next episode's advantage through that boundary.
    rewards = torch.tensor(batch['rewards'], dtype=torch.float32)
    old = torch.tensor(batch['values']); next_values = torch.tensor(batch['next_values'])
    advantage, returns = compute_gae(rewards, old, next_values,
            torch.tensor(batch['terminated']), torch.tensor(batch['active_mask']),
            gamma=config['ppo']['gamma'], gae_lambda=config['ppo']['gae_lambda'],
            truncated=torch.tensor(batch['truncated']), episode_end=torch.tensor(batch['episode_end']))
    expected_delta = rewards[0] + config['ppo']['gamma'] * next_values[0] - old[0]
    torch.testing.assert_close(advantage[0], expected_delta, rtol=0, atol=0)
    result = {'status':'BOUNDARY_WIRING_PASS','CE_hold_29_to_30':hold,
              'timeout3000_uses_physical_pre_reset_next_value':True,
              'physical_next_values':expected.tolist(),'reset_values':reset_value.tolist(),
              'episode_advantage_does_not_cross_reset':True,'PPO_updates':0,
              'diagnostic_rollout_decisions':2,'independent_validation':False}
    atomic_json(ROOT/'artifacts/2026-10-11_batch02_e1/boundary_wiring.json', plain(result))
    print(json.dumps(result))


if __name__=='__main__':main()
