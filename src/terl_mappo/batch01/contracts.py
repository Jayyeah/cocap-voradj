"""Explicit BASE pins and fail-closed source/config delta verification."""
from __future__ import annotations

import copy
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

from ..native import ROOT, TERL_SHA
from ..run import file_hash

PARENT_SHA = 'bb794ca8435f06b9fa5693c0fed98f567320decf'
ANCHOR_CONFIG = 'configs/experiments/terl_mappo_20261008/stage1_1m.json'
LOCK_PATH = 'configs/experiments/terl_mappo_batch01_20261009/base_lock.json'
PROTOCOL_PATH = 'configs/experiments/terl_mappo_batch01_20261009/evaluation.json'
REGRESSION_SOURCES = (
    'test/test_terl_native_mappo_20261008.py',
    'test/test_small_step_ac_migration_contract.py',
    'test/test_mappo_terminal_rows_20260923.py',
    'test/test_terl_mappo_batch01_base_20261009.py',
    'test/test_terl_mappo_batch01_qa_repairs_20261009.py',
    'test/test_terl_mappo_batch01_v3_20261009.py',
)
COMMON_DEPENDENCIES = (
    'src/cocap_voradj/training/small_step_ac.py',
    'src/cocap_voradj/models/small_step_ac.py',
    'src/cocap_voradj/models/iqn.py',
    'src/cocap_voradj/models/continuous/local_entity_token_encoder.py',
    'src/cocap_voradj/__init__.py', 'src/cocap_voradj/models/__init__.py',
    'src/cocap_voradj/models/continuous/__init__.py',
    'src/cocap_voradj/training/__init__.py',
)
OPERATIONAL_KEYS = {'experiment', 'seed', 'actor_seed', 'budget', 'stage1_ceiling'}
LINE_KEYS = {
    'T0': set(), 'T1': set(), 'N1': set(),
    'R1': set(), 'P1': {'ppo.target_kl'}, 'C0': set(),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def safe_relative(value):
    p = Path(value)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('source paths must be project-relative')
    return p.as_posix()


def source_inventory(root=ROOT):
    root = Path(root)
    files = set(COMMON_DEPENDENCIES)
    for directory in ('src/terl_mappo', 'vendor/terl'):
        for p in (root / directory).rglob('*'):
            if p.is_file() and (p.suffix in {'.py', '.yaml', '.json'} or
                                p.name in {'LICENSE', 'requirements.txt'}):
                files.add(p.relative_to(root).as_posix())
    files.update({ANCHOR_CONFIG, 'configs/experiments/terl_mappo_20261008/stage1.json',
                  PROTOCOL_PATH, 'tools/batch01_base_20261009.py'})
    files.update(REGRESSION_SOURCES)
    files.add('src/sitecustomize.py')
    files.add('pyproject.toml')
    files.add('configs/experiments/terl_mappo_batch01_20261009/arm_evaluation.json')
    # Package __init__ files import historical trainer/dynamics/logging modules
    # even when only the MAPPO learner is requested. Freeze their import closure
    # without rewriting those packages or accepting unverified live dependencies.
    pending = [p for p in files if p.endswith('.py')]
    visited = set()

    def include_module(name):
        if not name.startswith(('cocap_voradj', 'terl_mappo', 'tools')):
            return
        parts = name.split('.')
        for length in range(1, len(parts) + 1):
            prefix = '/'.join(parts[:length])
            base = '' if parts[0] == 'tools' else 'src/'
            for candidate in (f'{base}{prefix}.py', f'{base}{prefix}/__init__.py'):
                if (root / candidate).is_file() and candidate not in files:
                    files.add(candidate); pending.append(candidate)

    while pending:
        path = pending.pop()
        if path in visited: continue
        visited.add(path)
        parts = (Path(path).relative_to('src') if path.startswith('src/') else Path(path)).with_suffix('').parts
        package = parts[:-1]
        for node in ast.walk(ast.parse((root / path).read_text())):
            if isinstance(node, ast.Import):
                for alias in node.names: include_module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                prefix = list(package[:len(package) - node.level + 1]) if node.level else []
                module = '.'.join(prefix + (node.module.split('.') if node.module else []))
                include_module(module)
                for alias in node.names:
                    if alias.name != '*': include_module(module + '.' + alias.name)
    return {p: file_hash(root / p) for p in sorted(files)}


def create_lock():
    """Offline engineering command; never a launch-time refresh of BASE."""
    if git('branch', '--show-current') not in {'experiment/terl-mappo-batch01-base-20261009', 'experiment/terl-mappo-batch01-v3-20261009'}:
        raise ValueError('create BASE only on the authorized Core branch')
    config = json.loads((ROOT / ANCHOR_CONFIG).read_text())
    # Compare the entire historical source list to the actual immutable Git objects.
    from ..run import source_hash
    historical = source_hash()
    historical.update({p: file_hash(ROOT / p) for p in
                       (ANCHOR_CONFIG, 'configs/experiments/terl_mappo_20261008/stage1.json')})
    for path, digest in historical.items():
        raw = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{PARENT_SHA}:{path}'])
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('T0 anchor source changed: ' + path)
    return {'schema': 'terl.batch01.base.v1', 'status': 'BATCH01_BASE_CANDIDATE_V3',
            'parent_sha': PARENT_SHA, 'terl_sha': TERL_SHA, 'anchor_config': config,
            'common_sources': source_inventory(),
            'generated_runtime_sources': torch_generated_sources(),
            'evaluation_protocol': json.loads((ROOT / PROTOCOL_PATH).read_text()),
            'arm_evaluation_protocol': json.loads((ROOT / 'configs/experiments/terl_mappo_batch01_20261009/arm_evaluation.json').read_text()),
            'scientific_delta_from_t0': [], 'independent_qa': 'PENDING_MASTER_QA'}


def torch_generated_sources():
    """Explicit content pin for Torch's Adam-import temporary Python template.

    Its per-process filename changes with TMPDIR. It is a framework dependency,
    never permission for ARM-generated code or arbitrary runtime modules.
    """
    from torch.distributed.nn.jit.instantiator import get_remote_module_template
    text = get_remote_module_template(True).format(
        assign_module_interface_cls='module_interface_cls = None', args='*args', kwargs='**kwargs',
        arg_types='*args, **kwargs', arrow_and_return_type='', arrow_and_future_return_type='',
        jit_script_decorator='')
    return {'torch.distributed.nn.non_scriptable_remote_template': hashlib.sha256(text.encode()).hexdigest()}


def resolve_config(lock, delta):
    if delta.get('schema') != 'terl.batch01.delta.v1' or delta.get('line') not in LINE_KEYS:
        raise ValueError('unknown delta schema/experiment line')
    if set(delta) - {'schema', 'line', 'base', 'config_changes', 'source_changes', 'extensions'}:
        raise ValueError('unknown delta fields; never silently ignore configuration')
    if delta['line'] == 'T0' and delta.get('extensions'):
        raise ValueError('T0 does not accept extensions')
    config = copy.deepcopy(lock['anchor_config'])
    seen = set()
    for entry in delta.get('config_changes', []):
        if set(entry) != {'path', 'before', 'after', 'reason'} or not entry['reason']:
            raise ValueError('config delta requires path/before/after/reason')
        keys = entry['path'].split('.')
        if entry['path'] in seen or (keys[0] not in OPERATIONAL_KEYS and entry['path'] not in LINE_KEYS[delta['line']]):
            raise ValueError('config delta outside experiment scope: ' + entry['path'])
        seen.add(entry['path'])
        node = config
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                raise ValueError('unknown config path')
            node = node[key]
        if keys[-1] not in node or node[keys[-1]] != entry['before']:
            raise ValueError('config delta before value mismatch')
        node[keys[-1]] = entry['after']
    if config['checkpoint_interval'] != 25000 or config['rollout_length'] != 256:
        raise ValueError('shared rollout/screening cuts changed')
    if config['terl_sha'] != TERL_SHA or not 0 < config['budget'] <= config['stage1_ceiling']:
        raise ValueError('invalid native provenance/budget')
    return config


def verify_base(lock, delta, expected_lock_sha256, root=ROOT):
    if lock.get('schema') != 'terl.batch01.base.v1' or lock.get('parent_sha') != PARENT_SHA:
        raise ValueError('wrong BASE lineage/schema')
    if fingerprint(lock) != expected_lock_sha256:
        raise ValueError('BASE lock pin mismatch; do not adopt a newer Core HEAD')
    base = delta.get('base', {})
    if base.get('lock_sha256') != expected_lock_sha256 or base.get('parent_sha') != PARENT_SHA:
        raise ValueError('delta BASE pin mismatch')
    sha = base.get('candidate_sha', '')
    if len(sha) != 40 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('an explicit full candidate SHA is required')
    actual = source_inventory(root)
    declared = {}
    for entry in delta.get('source_changes', []):
        path = safe_relative(entry['path'])
        if path in declared or not entry.get('reason') or not entry.get('science_impact'):
            raise ValueError('source delta requires unique path, reason and science_impact')
        if entry.get('before') != lock['common_sources'].get(path):
            raise ValueError('source delta before hash mismatch')
        # Dynamic ARM dependencies outside the static closure need an explicit
        # declaration too. Never grow the trusted set from live sys.modules.
        if path not in actual and entry.get('after') is not None:
            source = Path(root) / path
            if not source.resolve().is_relative_to(Path(root).resolve()):
                raise ValueError('source delta escapes project realpath')
            if source.is_file(): actual[path] = file_hash(source)
        if entry.get('after') != actual.get(path):
            raise ValueError('source delta after hash mismatch')
        if path in lock['common_sources'] and not entry.get('candidate_fix'):
            raise ValueError('common source edit needs explicit candidate_fix review')
        declared[path] = entry
    changed = {p for p in actual.keys() | lock['common_sources'].keys()
               if actual.get(p) != lock['common_sources'].get(p)}
    if changed != set(declared):
        raise ValueError('undeclared/stale common source delta: ' + ', '.join(sorted(changed ^ set(declared))))
    return resolve_config(lock, delta)


def declared_sources(lock, delta):
    """Use only after verify_base; immutable BASE plus the exact ARM delta."""
    result = dict(lock['common_sources'])
    for entry in delta.get('source_changes', []):
        if entry['after'] is None: result.pop(entry['path'], None)
        else: result[entry['path']] = entry['after']
    return result


def verify_committed_pin(lock, delta):
    sha = delta['base']['candidate_sha']
    committed = json.loads(git('show', f'{sha}:{LOCK_PATH}'))
    if committed != lock:
        raise ValueError('lock does not belong to the pinned candidate commit')


def validate_resources(output, device='cpu', threads=1, workers=2, max_gpu_mib=2048):
    path = Path(output).resolve()
    if not path.is_relative_to(Path('/home/yjq')):
        raise ValueError('all run/checkpoint/temp paths must resolve under /home/yjq')
    if not 1 <= threads <= 4 or not 1 <= workers <= 2 or not 0 < max_gpu_mib <= 2048:
        raise ValueError('Batch01 resource limits: 1..4 threads, 1..2 workers, <=2048 MiB')
    if device not in {'cpu', 'cuda:0'}:
        raise ValueError('use process-local cuda:0')
    if device == 'cuda:0' and os.environ.get('CUDA_VISIBLE_DEVICES') not in {'0', '1'}:
        raise ValueError('explicit single physical CUDA_VISIBLE_DEVICES=0 or 1 required')
    return {'output': str(path), 'device': device, 'threads': threads, 'evaluation_workers': workers,
            'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'max_gpu_allocated_mib': max_gpu_mib}


def environment_versions():
    from importlib.metadata import version
    return {'python': platform.python_version(),
            **{name: version(name) for name in ('torch', 'numpy', 'scipy', 'gym', 'PyYAML')}}
