"""Fixed CoCap task metrics; MAPPO argmax/sample replace IQN expected-Q only."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch

from terl_mappo.batch01.evaluation import isolated_rng
from terl_mappo.run import atomic_json, file_hash
from cocap_voradj.training.small_step_ac import tensor_tree
from .runtime import CapabilityAdapter, build, sources

EVALUATOR_VERSION = 'batch02-a-cocap-native-mixed-v1'


def summary(rows):
    n = len(rows)
    times = [r['capture_time_seconds'] for r in rows if r['capture_completion']]
    failures = Counter(r['failure_type'] for r in rows if not r['capture_completion'])
    collisions = Counter()
    for row in rows:
        collisions.update(row['collision_type_counts'])
    return dict(episodes=n,
                episode_reward_mean=float(np.mean([r['episode_reward_mean'] for r in rows])),
                capture_completion_rate=sum(r['capture_completion'] for r in rows) / n,
                fully_capture_rate=sum(r['fully_capture'] for r in rows) / n,
                episode_success_rate=sum(r['episode_success'] for r in rows) / n,
                collision_rate=sum(r['collision_event'] for r in rows) / n,
                boundary_rate=sum(r['boundary_collision_event'] for r in rows) / n,
                episode_length_mean=float(np.mean([r['length'] for r in rows])),
                capture_time_seconds_mean=float(np.mean(times)) if times else None,
                capture_time_n=len(times), noncapture_censored_n=n - len(times),
                failure_types=dict(failures), collision_type_counts=dict(collisions))


def evaluate(checkpoint, output, config, domain, device='cpu', *, smoke_episodes=None):
    output, checkpoint = Path(output), Path(checkpoint)
    digest = file_hash(checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload['sources'] != sources() or payload['config'] != config:
        raise ValueError('evaluator checkpoint/source/config mismatch')
    if output.exists():
        import json
        existing = json.loads(output.read_text())
        if existing['checkpoint_sha256'] != digest or existing['evaluator_version'] != EVALUATOR_VERSION:
            raise ValueError('immutable evaluation output collision')
        return existing
    protocol = config['evaluation']
    setting = protocol[domain]
    count = smoke_episodes or setting['episodes_per_mode']
    rows = []
    started = time.monotonic()
    with isolated_rng():
        initial = CapabilityAdapter(setting['seed_base'], config['arm'], training=False)
        trainer = build(config, initial, device)
        trainer.actor.load_state_dict(payload['trainer']['actor'])
        trainer.actor.eval()
        del initial
        for mode in protocol['modes']:
            for index in range(count):
                seed = setting['seed_base'] + index
                adapter = CapabilityAdapter(seed, config['arm'], training=False)
                torch.manual_seed(seed + protocol['sample_torch_offset'])
                # No fixed evaluator horizon substitutes for native dones.
                end = False
                while not end:
                    with torch.no_grad():
                        action = trainer.actor.sample(tensor_tree(adapter.packed_observations(), trainer.device),
                                                      deterministic=mode == 'argmax')[0].cpu().numpy()
                    _, _, _, _, end, _ = adapter.step(action)
                row = dict(seed=seed, action_seed=seed + protocol['sample_torch_offset'],
                           mode=mode, **adapter.last_episode)
                rows.append(row)
                atomic_json(output.with_suffix('.progress.json'),
                            dict(status='EVALUATING', domain=domain, decisions=payload['decisions'],
                                 completed_episodes=len(rows), total_episodes=count * len(protocol['modes'])))
    report = dict(evaluator_version=EVALUATOR_VERSION, evaluator_sha256=file_hash(Path(__file__)),
                  checkpoint=str(checkpoint), checkpoint_sha256=digest, decisions=payload['decisions'],
                  arm=config['arm'], classification='PROVISIONAL', seed_domain=domain,
                  independent_validation=False, engineering_smoke=smoke_episodes is not None,
                  task_reference=config['task_reference'],
                  protocol=protocol, modes={m: summary([r for r in rows if r['mode'] == m]) for m in protocol['modes']},
                  episodes=rows, duration_seconds=time.monotonic() - started)
    atomic_json(output, report)
    atomic_json(output.with_suffix('.progress.json'), dict(status='COMPLETE', domain=domain,
                decisions=payload['decisions'], completed_episodes=len(rows)))
    return report


def mean_metric(report, field):
    values = [row[field] for row in report['modes'].values() if row[field] is not None]
    return float(np.mean(values)) if values else None


def selection_key(report):
    # Predeclared descriptive choice; no task performance threshold or gate.
    capture_time = mean_metric(report, 'capture_time_seconds_mean')
    return (mean_metric(report, 'capture_completion_rate'), -mean_metric(report, 'collision_rate'),
            -mean_metric(report, 'boundary_rate'), -(capture_time if capture_time is not None else float('inf')),
            -report['decisions'])
