"""Pinned source membership, execution-time imports, and latched run failures.

This is a provenance contract for cooperative Python extensions, not a sandbox
against arbitrary hostile Python. No live module can grant itself permission.
Full hashes are checked at rollout/update/checkpoint boundaries; hook checks
cache hashes by stat and imports invalidate the module cache. Audit events reject
undeclared importlib execution before the module body runs.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from contextvars import ContextVar
import importlib
import inspect
import json
from pathlib import Path
import sys
import os
import time
import textwrap
from types import MappingProxyType
import sysconfig
import site

from .contracts import ROOT, LOCK_PATH, declared_sources, verify_base, verify_committed_pin
from ..run import file_hash, atomic_json

_ACTIVE = ContextVar('batch01_source_guard', default=None)
_EXECUTED = {}
_IMPORT_EPOCH = 0
_NAMESPACES = {'terl_mappo', 'cocap_voradj', 'config_manager', 'environment',
               'policy', 'robots', 'thirdparty', 'utils'}
_EXTERNAL_SOURCE_ROOTS = {Path(sysconfig.get_path(key)).resolve()
                          for key in ('stdlib', 'purelib', 'platlib')}
_EXTERNAL_SOURCE_ROOTS.update(Path(p).resolve() for p in site.getsitepackages())
_EXTERNAL_SOURCE_ROOTS.add(Path(site.getusersitepackages()).resolve())
# The successful anchor uses an existing read-only Gym installation outside
# Python's standard installation. Permit these named runtime packages only.
for _package in ('gym', 'gym_notices'):
    _filename = getattr(sys.modules.get(_package), '__file__', None)
    if _filename: _EXTERNAL_SOURCE_ROOTS.add(Path(_filename).resolve().parent)


class SourceViolation(ValueError):
    pass


def _audit(event, args):
    global _IMPORT_EPOCH
    if event == 'import': _IMPORT_EPOCH += 1
    if event != 'exec': return
    filename = getattr(args[0], 'co_filename', '')
    if not filename or filename.startswith('<'): return
    p = Path(filename).absolute()
    guard = _ACTIVE.get()
    if guard is not None:
        guard.check_path(p, force=True)
    # Capture executed content separately from permissions, including unguarded
    # imports between boundaries. This does not add anything to the allow-list.
    real = p.resolve()
    if real.is_relative_to(ROOT) and real.suffix == '.py' and real.is_file():
        _EXECUTED[str(p)] = file_hash(real)
    _IMPORT_EPOCH += 1


sys.addaudithook(_audit)


class SourceGuard:
    def __init__(self, expected, *, root=ROOT, diagnostics=None, generated_sources=None):
        self.root = Path(root).resolve()
        self.expected = MappingProxyType(dict(expected))
        if generated_sources is None:
            generated_sources = json.loads((self.root / LOCK_PATH).read_text()).get('generated_runtime_sources', {})
        self.generated_sources = MappingProxyType(dict(generated_sources))
        self.diagnostics = Path(diagnostics or (self.root / 'runs/batch01_source_failures')).resolve()
        if not self.diagnostics.is_relative_to(Path('/home/yjq')):
            raise ValueError('source diagnostics must stay under /home/yjq')
        self.failure = None
        self.failure_path = None
        self._hash_cache = {}
        self._modules_key = None
        self._module_paths = set()

    @classmethod
    def from_base(cls, lock, delta, pin, **kwargs):
        verify_base(lock, delta, pin)
        verify_committed_pin(lock, delta)
        return cls(declared_sources(lock, delta), generated_sources=lock.get('generated_runtime_sources', {}), **kwargs)

    @classmethod
    def committed(cls):
        lock = json.loads((ROOT / LOCK_PATH).read_text())
        return cls(lock['common_sources'], generated_sources=lock.get('generated_runtime_sources', {}))

    def fail(self, message):
        if self.failure is None:
            self.failure = message
            self.diagnostics.mkdir(parents=True, exist_ok=True)
            self.failure_path = self.diagnostics / f'{os.getpid()}_{time.monotonic_ns()}.json'
            atomic_json(self.failure_path, {'schema': 'terl.batch01.source_failure.v1',
                        'status': 'RUN_FAILED_SOURCE_CONTRACT', 'reason': message,
                        'pid': os.getpid(), 'expected_sources': dict(self.expected),
                        'generated_runtime_sources': dict(self.generated_sources)})
        raise SourceViolation('latched source contract failure: ' + self.failure)

    def alive(self):
        if self.failure is not None: self.fail(self.failure)

    def framework_generated(self, path):
        module = sys.modules.get('torch.distributed.nn.jit.instantiator')
        directory = getattr(module, 'INSTANTIATED_TEMPLATE_DIR_PATH', None)
        digest = self.generated_sources.get('torch.distributed.nn.non_scriptable_remote_template')
        if not directory or path != Path(directory) / '_remote_module_non_scriptable.py': return False
        if not digest or path != path.resolve() or not path.is_file() or file_hash(path) != digest:
            self.fail('Torch generated template differs from explicit runtime content pin')
        if _EXECUTED.get(str(path), digest) != digest:
            self.fail('executed Torch generated template differs from runtime pin')
        return True

    def check_path(self, path, *, force=False, required=False):
        self.alive()
        lexical = Path(path).absolute()
        real = lexical.resolve()
        if self.framework_generated(lexical): return
        local = lexical.is_relative_to(self.root) or real.is_relative_to(self.root)
        if not local:
            if not required and any(real.is_relative_to(p) for p in _EXTERNAL_SOURCE_ROOTS): return
            self.fail('local module origin outside pinned project: ' + str(lexical))
        if not real.is_relative_to(self.root): self.fail('source symlink escapes project: ' + str(lexical))
        relative = lexical.relative_to(self.root).as_posix()
        # Symlink aliases must not silently substitute another declared file.
        if lexical != real: self.fail('source lexical/realpath mismatch: ' + relative)
        expected = self.expected.get(relative)
        if expected is None: self.fail('loaded local dependency absent from lock/delta: ' + relative)
        if not real.is_file(): self.fail('declared source missing: ' + relative)
        stat = real.stat()
        stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = self._hash_cache.get(relative)
        if force or cached is None or cached[0] != stamp:
            digest = file_hash(real)
            if digest != expected: self.fail('declared source hash mismatch: ' + relative)
            self._hash_cache[relative] = (stamp, digest)
        executed = _EXECUTED.get(str(lexical))
        if executed is not None and executed != expected:
            self.fail('executed module content differs from lock/delta: ' + relative)

    def check_loaded(self, *, force=False):
        self.alive()
        key = (len(sys.modules), _IMPORT_EPOCH)
        if force or key != self._modules_key:
            paths = set()
            for name, module in list(sys.modules.items()):
                filename = getattr(module, '__file__', None)
                required = name.split('.')[0] in _NAMESPACES
                if not filename:
                    if required and module is not None:
                        locations = list(getattr(getattr(module, '__spec__', None), 'submodule_search_locations', ()) or ())
                        if not locations: self.fail('local namespace has no source origin: ' + name)
                        for location in locations:
                            path = Path(location).absolute()
                            if path != path.resolve() or not path.is_relative_to(self.root):
                                self.fail('namespace package origin outside pinned project: ' + name)
                            prefix = path.relative_to(self.root).as_posix() + '/'
                            if not any(p.startswith(prefix) for p in self.expected):
                                self.fail('namespace package absent from lock/delta: ' + name)
                    continue
                if not filename.endswith('.py'): continue
                # torch.ops/classes expose synthetic relative __file__ names.
                # They have no filesystem origin and are not local source.
                origin = getattr(getattr(module, '__spec__', None), 'origin', None)
                if not required and not Path(filename).is_absolute() and not origin and not Path(filename).is_file():
                    continue
                lexical = Path(filename).absolute()
                real = lexical.resolve()
                if required or not any(real.is_relative_to(p) for p in _EXTERNAL_SOURCE_ROOTS):
                    if origin and Path(origin).absolute() != lexical:
                        self.fail('module __file__/spec origin mismatch: ' + name)
                    if required:
                        base = 'src' if name.split('.')[0] in {'terl_mappo', 'cocap_voradj'} else 'vendor/terl'
                        stem = self.root / base / name.replace('.', '/')
                        if lexical not in {stem.with_suffix('.py'), stem / '__init__.py'}:
                            self.fail('module name/source origin mismatch: ' + name)
                    self.check_path(lexical, force=force, required=required)
                    paths.add(lexical)
            self._module_paths = paths
            self._modules_key = key
        # With no import event, hook calls check their own code file only.
        # Full content checks remain mandatory before accepting a rollout,
        # every optimizer step, and checkpoint save/load.

    def check(self, *, force=True):
        self.alive()
        # Fixed, explicit inventory only; no recursive filesystem scan here.
        for relative in self.expected: self.check_path(self.root / relative, force=force)
        self.check_loaded(force=force)

    @contextmanager
    def scope(self, *, force=True):
        if force: self.check()
        else: self.check_loaded()
        token = _ACTIVE.set(self)
        try:
            yield
            if force: self.check()
            else: self.check_loaded()
        finally:
            _ACTIVE.reset(token)

    def call(self, function, *args, **kwargs):
        code = getattr(getattr(function, '__func__', function), '__code__', None)
        if code is not None: self.check_path(code.co_filename, required=True)
        with self.scope(force=False): return function(*args, **kwargs)

    def preflight(self, functions):
        """Resolve literal dynamic imports early; computed names stay guarded."""
        for function in functions:
            if function is None: continue
            try: path = inspect.getsourcefile(function)
            except TypeError: path = None
            if not path: continue
            self.check_path(path, force=True, required=True)
            tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not node.args: continue
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
                if name not in {'import_module', '__import__'}: continue
                argument = node.args[0]
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    target = argument.value
                    if target.split('.')[0] in _NAMESPACES:
                        with self.scope(): importlib.import_module(target)
