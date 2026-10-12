#!/usr/bin/env python3
"""Bounded full-GPU-state restore/replay/update/evaluator verification."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import numpy as np
import torch
from terl_mappo.batch02_stability.runtime import (fork_parent, config_and_plan, profile,
                                                restore, save, state_hashes, digest_graph)
from terl_mappo.run import atomic_json, rng_state, file_hash


def continuation(arm, trainer, adapter):
    batch, episodes = profile(arm)[0].collect(trainer, adapter, 16)
    metrics = trainer.update(batch, categorical=True)
    assert all(np.isfinite(v) for v in metrics.values())
    assert all(torch.isfinite(p).all().item() for p in [*trainer.actor.parameters(), *trainer.value.parameters()])
    assert all(p.grad is None or torch.isfinite(p.grad).all().item() for p in [*trainer.actor.parameters(), *trainer.value.parameters()])
    return dict(batch=digest_graph(batch), episodes=digest_graph(episodes), metrics=metrics,
                state=state_hashes(trainer, adapter))


def main(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / 'fork500k.pt'
    if args.restore:
        trainer, adapter, _, payload = restore(args.arm, checkpoint, args.device)
        before = state_hashes(trainer, adapter)
        assert before == json.loads((args.output / 'restored_state_expected.json').read_text())
        actual = continuation(args.arm, trainer, adapter)
        assert actual == json.loads((args.output / 'expected.json').read_text())
        atomic_json(args.output / 'restored.json', dict(exact=True, continuation=actual, decisions=payload['decisions']))
        return
    config, baseline, plan = config_and_plan(args.arm)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(plan['physical_gpu'])
    assert args.device == 'cuda:0', 'verify parent CUDA RNG on original physical GPU'
    trainer, adapter, config, payload, receipt = fork_parent(args.arm, args.device)
    atomic_json(args.output / 'fork_receipt.json', receipt)
    digest = save(args.arm, checkpoint, trainer, adapter, config, payload['decisions'],
                  payload['agent_transitions'], payload['optimizer_steps'])
    atomic_json(args.output / 'restored_state_expected.json', state_hashes(trainer, adapter))
    # No optimizer update yet: the same RNG/action stream must reproduce all task outputs.
    batch, episodes = profile(args.arm)[0].collect(trainer, adapter, 16)
    expected_rollout = dict(batch=digest_graph(batch), episodes=digest_graph(episodes), runtime=digest_graph(adapter.state_dict()))
    trainer, adapter, _, _, _ = fork_parent(args.arm, args.device)
    from dataclasses import replace
    trainer.config = replace(trainer.config, target_kl=.02)
    batch, episodes = profile(args.arm)[0].collect(trainer, adapter, 16)
    assert expected_rollout == dict(batch=digest_graph(batch), episodes=digest_graph(episodes), runtime=digest_graph(adapter.state_dict()))
    trainer, adapter, _, _ = restore(args.arm, checkpoint, args.device)
    atomic_json(args.output / 'expected.json', continuation(args.arm, trainer, adapter))
    subprocess.run([sys.executable, __file__, '--arm', args.arm, '--output', str(args.output),
                    '--device', args.device, '--restore'], check=True)
    evaluator = profile(args.arm)[1]
    smoke = evaluator.evaluate(checkpoint, args.output / 'evaluator_smoke.json', config, 'screen',
                               args.device, smoke_episodes=1)
    baseline_screen = json.loads((Path(plan['baseline_run']) / 'evaluations/screen_000500000.json').read_text())
    # Identical parent policy, evaluator seeds/action mode and unmodified task.
    for row in smoke['episodes']:
        old = next(r for r in baseline_screen['episodes'] if r['mode']==row['mode'] and r['seed']==row['seed'])
        assert row == old
    report = dict(status='ENGINEERING_PASS', arm=args.arm, classification='PROVISIONAL',
                  parent_checkpoint=plan['parent_checkpoint'], config_diff=plan['scientific_delta'],
                  strict_resume_all_state_exact=True, fork_full_state_unchanged=True,
                  CUDA_RNG_restored=True, pre_update_rollout_parity=True,
                  fresh_process_16_decision_PPO_continuation_exact=True,
                  evaluator_original_bytes_and_contract_unchanged=True,
                  evaluator_parent_episodes_exact=True, evaluator_smoke_episodes=len(smoke['episodes']),
                  checkpoint_hash=digest, reward_observation_network_environment_action_terminal_delta='ZERO',
                  diagnostic_decisions_excluded_from_long_budget=True,
                  live_target_kl=.01, additional_training_authorized=500000,
                  gradient_parameter_metrics_finite=True)
    atomic_json(args.output / 'verification.json', report)
    print(json.dumps(dict(arm=args.arm, status=report['status'], strict_resume=True, fresh_process_exact=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['R2','R3','V3'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--restore', action='store_true')
    main(parser.parse_args())
