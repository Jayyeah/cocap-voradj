"""Full-state checkpoints with BASE binding, and deletion plans scoped to one run."""
from __future__ import annotations

import json
from pathlib import Path

from ..run import save_checkpoint, load_checkpoint, atomic_json, file_hash
from .contracts import fingerprint, validate_resources
from .source_guard import SourceGuard


def assert_manifest_sources(manifest, runtime=None):
    guard = runtime.source_guard if runtime is not None else SourceGuard.committed()
    if manifest.get('common_source_hashes') != guard.expected:
        raise ValueError('checkpoint runtime manifest has stale/undeclared common sources')
    if manifest.get('generated_runtime_source_hashes', dict(guard.generated_sources)) != dict(guard.generated_sources):
        raise ValueError('checkpoint runtime manifest has stale generated runtime sources')
    guard.check()


def save_bound(path, runtime, manifest, steps, agent_steps, optimizer_steps):
    with runtime.source_guard.scope():
        return _save_bound(path, runtime, manifest, steps, agent_steps, optimizer_steps)


def _save_bound(path, runtime, manifest, steps, agent_steps, optimizer_steps):
    assert_manifest_sources(manifest, runtime)
    validate_resources(path, str(runtime.trainer.device))
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    try:
        digest = save_checkpoint(pending, runtime.trainer, runtime.adapter, runtime.config,
                                 steps, agent_steps, optimizer_steps)
        runtime.source_guard.check()
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)
        pending.with_suffix('.pt.tmp').unlink(missing_ok=True)
    atomic_json(Path(path).with_suffix('.batch01.json'),
                {'schema': 'terl.batch01.checkpoint.v1', 'checkpoint_sha256': digest,
                 'steps': steps, 'manifest_fingerprint': fingerprint(manifest), 'base': manifest['base']})
    return digest


def load_bound(path, runtime, manifest):
    with runtime.source_guard.scope(): return _load_bound(path, runtime, manifest)


def _load_bound(path, runtime, manifest):
    assert_manifest_sources(manifest, runtime)
    side = json.loads(Path(path).with_suffix('.batch01.json').read_text())
    if (side['checkpoint_sha256'] != file_hash(path) or side['base'] != manifest['base'] or
            side['manifest_fingerprint'] != fingerprint(manifest)):
        raise ValueError('checkpoint manifest/BASE/hash mismatch')
    return load_checkpoint(path, runtime.trainer, runtime.adapter, runtime.config,
                           runtime.trainer.device, strict=True)


def load_anchor(path, expected_sha256, runtime):
    with runtime.source_guard.scope(): return _load_anchor(path, expected_sha256, runtime)


def _load_anchor(path, expected_sha256, runtime):
    """Explicit import of the original full-state anchor, never strict=False."""
    if runtime.hooks != type(runtime.hooks)():
        raise ValueError('historical exact resume requires T0 hooks')
    if file_hash(path) != expected_sha256:
        raise ValueError('historical anchor hash mismatch')
    return load_checkpoint(path, runtime.trainer, runtime.adapter, runtime.config,
                           runtime.trainer.device, strict=True)


def retention_plan(run, completed_screens, *, keep_steps, pending_steps=()):
    """Read-only plan. No checkpoint is removed by this common-base delivery."""
    root = Path(run).resolve()
    validate_resources(root)
    directory = root / 'checkpoints'
    keep = set(keep_steps) | set(pending_steps)
    protected_inodes = set()
    for alias in ('latest.pt', 'best.pt'):
        p = directory / alias
        if p.exists(): protected_inodes.add((p.stat().st_dev, p.stat().st_ino))
    result = []
    for evaluation in completed_screens:
        if evaluation.get('seed_domain') != 'screen' or 'modes' not in evaluation or 'episodes' not in evaluation:
            raise ValueError('retention requires complete screening evidence')
        step = evaluation['steps']
        original = Path(evaluation['checkpoint'])
        if not original.is_absolute(): raise ValueError('absolute checkpoint provenance required')
        path = original.resolve()
        if original.is_symlink() or path.parent != directory or path.name != f'step_{step:09d}.pt':
            continue  # unrelated experiment, alias, directory escape or symlink
        if step in keep or not path.exists(): continue
        if (path.stat().st_dev, path.stat().st_ino) in protected_inodes: continue
        metadata = json.loads(path.with_suffix('.json').read_text())
        digest = file_hash(path)
        if metadata.get('steps') != step or metadata.get('sha256') != digest or evaluation['checkpoint_sha256'] != digest:
            raise ValueError('retention checkpoint/evaluation/metadata hash mismatch')
        n = evaluation['episodes_per_mode']
        if any(evaluation['modes'].get(m, {}).get('episodes') != n or
               len([r for r in evaluation['episodes'] if r['mode'] == m]) != n for m in ('argmax', 'sample')):
            raise ValueError('partial screening cannot authorize retention')
        result.append({'path': str(path), 'steps': step, 'sha256': digest})
    return result
