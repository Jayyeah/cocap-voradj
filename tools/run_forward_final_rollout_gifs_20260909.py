#!/usr/bin/env python3
"""Final BC/PPO: C3 transitions + historical renderer, 20 rollouts/10 GIFs per scene."""
import argparse, copy, html, json, multiprocessing, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLBACKEND', 'Agg')
import torch
import yaml
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.forward_final import preflight, CONTRACT, TEACHER
from tools.evaluate_forward_final_ppo_20260909 import load_frozen_actor
from tools.train_forward_final_ppo_20260909 import BC, BC_SHA
from tools.run_forward_final_bridge_20260908 import run_episode, atomic_json, summarize
from tools.rollout_voradj_visual import snapshot_env, render_gif
from tools.collect_iqn_aw_teacher_dataset_20260903 import sha256_file
PRESET = ROOT / 'configs/evaluation/forward_final_rollout_20260909/standard.yaml'


def capture_frame(env, scene, step):
    # The legacy snapshot helper writes label/cache fields. Isolate it from execution.
    copied = copy.deepcopy(env)
    phase = 'coverage' if scene == 'coverage' or all(e.deactivated for e in env.evaders) else 'capture'
    frame = snapshot_env(copied, 'mix' if scene == 'mixed' else scene, phase,
                         int(scene == 'mixed' and phase == 'coverage'), step, step)
    assert all(s['type'] == 'pursuer' for s in frame['sites'])
    assert frame['sensing']['enemy_sensing_radius'] == 20 and frame['sensing']['surface_sensing']
    return frame


def render_saved(path, display, label):
    torch.set_num_threads(1)
    path = Path(path); data = json.loads(path.read_text()); row = data['summary']
    gif = path.parent / f"rollout_{row['scenario']}_seed_{row['seed']}.gif"
    if gif.exists(): raise ValueError(f'Preserve existing GIF: {gif}')
    start = time.monotonic()
    record = render_gif(data['frames'], gif, display['max_gif_frames'], display['trail_window'],
        display['frame_duration_ms'], label, display['draw_neighbor_edges'], display['draw_trails'],
        display['draw_sensing_circles'], display['draw_ce_targets'])
    record.update(seed=row['seed'], scene=row['scene'], elapsed_seconds=time.monotonic()-start,
                  sha256=sha256_file(gif), episode_json=str(path))
    atomic_json(path.parent / f"summary_{row['scenario']}_seed_{row['seed']}.json", {**row, **record})
    return record


def assert_reference(row, reference):
    # Diagnostics-only floating-point entropy tolerances never mask task/seed mismatch.
    for key in ('initial_state_fingerprint', 'captured', 'normal_capture', 'stationary_capture',
                'collision', 'boundary', 'ce_success', 'safe_complete', 'length', 'capture_seconds',
                'recovery_seconds', 'mission_seconds'):
        assert row[key] == reference[key], f'GIF trajectory mismatch: {row["scene"]}/{row["seed"]}/{key}'
    # Tracker stores simultaneous events through sets; their list order is not semantic.
    a, b = row['mission_events'], reference['mission_events']
    assert {k:v for k,v in a.items() if k != 'events'} == {k:v for k,v in b.items() if k != 'events'}
    assert sorted(json.dumps(e, sort_keys=True) for e in a['events']) == sorted(json.dumps(e, sort_keys=True) for e in b['events'])
    assert row['policy_diagnostics']['action_histogram'] == reference['policy_diagnostics']['action_histogram']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--actor-checkpoint', type=Path)
    p.add_argument('--preset', type=Path, default=PRESET)
    p.add_argument('--mode', choices=('bc_sample', 'bc_argmax'), default='bc_sample')
    p.add_argument('--label', default='Final BC sample')
    p.add_argument('--reference-report', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--render-only', action='store_true')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args(); out = args.output_root
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    preset = yaml.safe_load(args.preset.read_text()); display = preset['display']
    assert 0 <= preset['gif_count'] <= preset['episodes'] and preset['episodes'] > 0
    assert preset['scenarios'] == ['mixed', 'coverage']
    if args.smoke:
        preset.update(episodes=1, gif_count=1); display['max_gif_frames'] = 2
    reference = {}
    if args.reference_report:
        raw = json.loads(args.reference_report.read_text())
        reference = {(r['scene'], r['seed']): r for r in raw['records']}
    start = time.monotonic(); records = []; jobs = {}; rendered = []
    if args.render_only:
        launch = json.loads((out / 'launch.json').read_text())
        display = launch['preset']['display']; args.label = launch['label']
        records = json.loads((out / 'rollout_report.json').read_text())['records']
        rendered = [r for r in json.loads((out / 'gif_manifest.json').read_text()) if Path(r['gif']).exists()] if (out / 'gif_manifest.json').exists() else []
    else:
        if (out / 'launch.json').exists(): raise ValueError('Fresh output required')
        atomic_json(out / 'runtime_preflight.json', preflight())
        actor, sha, step = load_frozen_actor(args.actor_checkpoint, args.device)
        if reference:
            assert raw['actor_sha256'] == sha and raw['mode'] == args.mode
        teacher = CoCapIQN.load(str(TEACHER), device=args.device).eval()
        launch = dict(contract=CONTRACT, actor_sha256=sha, bc_parent_sha256=BC_SHA,
            checkpoint=str(args.actor_checkpoint or BC), ppo_training_steps=step, mode=args.mode,
            label=args.label, preset=preset, smoke=args.smoke, gpu_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
            evaluator_sha256=sha256_file(ROOT / 'tools/run_forward_final_bridge_20260908.py'),
            renderer_sha256=sha256_file(ROOT / 'tools/rollout_voradj_visual.py'),
            runner_sha256=sha256_file(Path(__file__)),
            reference_report=str(args.reference_report) if args.reference_report else None,
            display_contract='historical Final display; actual Forward Final horizon3000/postcapture500; final-outcome flags shown throughout')
        atomic_json(out / 'launch.json', launch)
        atomic_json(out / 'run_args.json', {**display, 'policy_mode': args.mode, 'actor_sha256': sha})
    def progress():
        cards = ''.join(f'<figure><a href="{Path(r["gif"]).relative_to(out)}"><img loading="lazy" width="385" src="{Path(r["gif"]).relative_to(out)}"></a><figcaption>{r["scene"]} seed {r["seed"]}</figcaption></figure>' for r in sorted(rendered, key=lambda r:(r['scene'],r['seed'])))
        (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>'+html.escape(args.label)+'</title><h1>'+html.escape(args.label)+'</h1><p>Completed GIFs; refresh to see new outputs. Final outcome flags, 100 ms/frame; not realtime.</p><main style="display:flex;flex-wrap:wrap">'+cards+'</main>')
        atomic_json(out / 'progress.json', dict(status='rendering' if args.render_only or len(records)==preset['episodes']*2 else 'rollout_and_render',
            rollouts_completed=len(records), rollouts_total=launch['preset']['episodes']*2,
            gifs_completed=len(rendered), gifs_total=launch['preset']['gif_count']*2,
            elapsed_seconds=time.monotonic()-start))
    with ProcessPoolExecutor(max_workers=preset['render_workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
        if args.render_only:
            for scene in ('mix', 'coverage'):
                for path in sorted((out / scene).glob('episode_*_seed_*.json')):
                    data = json.loads(path.read_text())['summary']
                    gif = path.parent / f"rollout_{scene}_seed_{data['seed']}.gif"
                    if not gif.exists(): jobs[pool.submit(render_saved, str(path), display, args.label)] = path
        else:
            for index in range(preset['episodes']):
                for scene in preset['scenarios']:
                    seed = preset['seed'] + index + (preset['coverage_seed_offset'] if scene=='coverage' else 0)
                    frames = []; capture = index < preset['gif_count']
                    row = run_episode(teacher, scene, seed, args.device, actor=actor, policy_mode=args.mode,
                        max_steps=3 if args.smoke else None,
                        on_progress=lambda step: atomic_json(out / 'active_episode.json', dict(scene=scene, seed=seed, step=step)),
                        on_snapshot=(lambda env, step: frames.append(capture_frame(env, scene, step))) if capture else None)
                    if reference and not args.smoke: assert_reference(row, reference[(scene, seed)])
                    records.append(row)
                    atomic_json(out / 'rollout_report.json', dict(launch=launch, summary=summarize(records), records=records,
                        reference_parity='PASS' if reference and not args.smoke else 'NOT_CHECKED'))
                    if capture:
                        name = 'mix' if scene == 'mixed' else scene
                        final_status = dict(capture_success=str(row['captured']).lower() if scene=='mixed' else 'N/A',
                            coverage_success=str(row['ce_success']).lower(), episode_success=str(row['safe_complete']).lower())
                        for frame in frames: frame['display_status'] = final_status
                        path = out / name / f'episode_{name}_seed_{seed}.json'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(dict(summary={**row, 'scenario':name, **display}, frames=frames)))
                        jobs[pool.submit(render_saved, str(path), display, args.label)] = path
                    for future in list(jobs):
                        if future.done(): rendered.append(future.result()); del jobs[future]
                    atomic_json(out / 'gif_manifest.json', rendered); progress()
            del actor, teacher
            if str(args.device).startswith('cuda'): torch.cuda.empty_cache()
        for future in as_completed(jobs):
            rendered.append(future.result()); atomic_json(out / 'gif_manifest.json', rendered); progress()
    atomic_json(out / 'progress.json', dict(status='complete', rollouts_completed=len(records),
        gifs_present=len(list(out.glob('*/*.gif'))), eta_seconds=0, elapsed_seconds=time.monotonic()-start))

if __name__ == '__main__': main()
