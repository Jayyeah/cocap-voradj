"""Pinned source membership, execution-time imports, and latched run failures.

This is a provenance contract for cooperative Python extensions, not a sandbox
against arbitrary hostile Python. No live module can grant itself permission.
Full hashes are checked at rollout/update/checkpoint boundaries; hook checks
cache hashes by stat and imports invalidate the module cache. Audit events reject
undeclared importlib execution before the module body runs.
"""
from __future__ import annotations

import ast
import hashlib
from types import CodeType
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
_CODE_RECORDS = getattr(sys, "_batch01_exec_records", {})
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


def code_signature(code):
    """Actual executable content including nested constants and exception tables.

    Filename alone never grants permission. Normalize immutable constants rather
    than marshal reference/interning layout; compare same Python optimization.
    """
    def constant(value):
        if isinstance(value, CodeType): return ('code', code_signature(value))
        if isinstance(value, tuple): return ('tuple', tuple(constant(x) for x in value))
        if isinstance(value, frozenset): return ('frozenset', tuple(sorted(map(repr, map(constant, value)))))
        return (type(value).__name__, repr(value))
    return (code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
            code.co_nlocals, code.co_stacksize, code.co_flags, code.co_code,
            tuple(constant(x) for x in code.co_consts), code.co_names, code.co_varnames,
            code.co_freevars, code.co_cellvars, code.co_firstlineno,
            code.co_linetable, code.co_exceptiontable)


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
        guard.check_code(args[0])
    # Capture executed content separately from permissions, including unguarded
    # imports between boundaries. This does not add anything to the allow-list.
    real = p.resolve()
    if real.is_relative_to(ROOT) and real.suffix == '.py' and real.is_file():
        _EXECUTED[str(p)] = file_hash(real)
        rows = _CODE_RECORDS.setdefault(str(p), [])
        if not any(code is args[0] for code in rows): rows.append(args[0])
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
        self._local_modules = []
        self._compiled = {}
        self._verified_codes = {}
        self._path_cache = {}
        policy=getattr(sys,'_batch01_verified_source_bootstrap',None)
        if not policy or policy.get('root')!=str(ROOT) or policy.get('bootstrap_sha256')!=json.loads((ROOT/LOCK_PATH).read_text())['common_sources'].get('tools/batch01_python.py'):
            self.fail('trusted source-first bootstrap required; invoke python -S tools/batch01_python.py')

    @classmethod
    def from_base(cls, lock, delta, pin, **kwargs):
        if getattr(sys,'_batch01_verified_source_bootstrap',{}).get('lock_sha256')!=pin:
            raise SourceViolation('bootstrap canonical lock differs from runtime BASE pin')
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
        bootstrap_failure=getattr(sys,'_batch01_source_bootstrap_failure',None)
        if bootstrap_failure and self.failure is None:self.fail(bootstrap_failure)
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
        cached_path = self._path_cache.get(str(lexical))
        real = lexical if cached_path else lexical.resolve()
        if cached_path and lexical.is_symlink(): self.fail("source became symlink: " + str(lexical))
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
            self._path_cache[str(lexical)] = True
        executed = _EXECUTED.get(str(lexical))
        if executed is not None and executed != expected:
            self.fail('executed module content differs from lock/delta: ' + relative)

    def compiled_codes(self, path):
        path = Path(path).absolute()
        self.check_path(path, required=True)
        key = str(path)
        relative = path.relative_to(self.root).as_posix()
        digest = self.expected[relative]
        if key not in self._compiled:
            source = path.read_bytes()
            if hashlib.sha256(source).hexdigest() != digest:
                self.fail('source changed during compilation: ' + relative)
            code = compile(source, key, 'exec', dont_inherit=True, optimize=sys.flags.optimize)
            allowed = set()
            def visit(item):
                allowed.add(code_signature(item))
                for constant in item.co_consts:
                    if isinstance(constant, CodeType): visit(constant)
            visit(code)
            self._compiled[key] = allowed
        return self._compiled[key]

    def check_code(self, code, *, required=False):
        self.alive()
        path = Path(code.co_filename).absolute()
        self.check_path(path, force=True, required=required)
        if self.framework_generated(path):
            reference = compile(path.read_bytes(), str(path), 'exec', dont_inherit=True,
                                optimize=sys.flags.optimize)
            if code_signature(code) != code_signature(reference):
                self.fail('generated template executed code differs from pinned source')
            return
        if not path.is_relative_to(self.root): return
        cache = self._verified_codes.setdefault(str(path), {})
        if cache.get(id(code)) is code: return
        if code_signature(code) not in self.compiled_codes(path):
            self.fail('executed code object differs from locked source: ' + str(path))
        cache[id(code)] = code

    def check_execution(self, filename):
        path = Path(filename).absolute()
        if not path.is_relative_to(self.root): return
        records = _CODE_RECORDS.get(str(path))
        if not records:
            self.fail('no startup execution witness; include pinned src in PYTHONPATH: ' + str(path))
        for code in records:
            cache = self._verified_codes.get(str(path), {})
            if cache.get(id(code)) is not code: self.check_code(code)

    def check_loaded(self, *, force=False):
        self.alive()
        key = (len(sys.modules), _IMPORT_EPOCH)
        if key != self._modules_key:
            paths = set()
            local_modules = []
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
                    local_modules.append((name, module, str(lexical), origin))
            self._module_paths = paths
            self._local_modules = local_modules
            self._modules_key = key
        for name, module, filename, origin in self._local_modules:
            if (sys.modules.get(name) is not module or getattr(module, "__file__", None) != filename or
                    getattr(getattr(module, "__spec__", None), "origin", None) != origin):
                self.fail("module __file__/spec origin mismatch or replacement: " + name)
            self.check_execution(filename)
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
        if code is None: code = getattr(getattr(function, '__call__', None), '__code__', None)
        if code is not None: self.check_code(code, required=True)
        with self.scope(force=False): return function(*args, **kwargs)

    def preflight(self, functions):
        """Resolve literal dynamic imports early; computed names stay guarded."""
        for function in functions:
            if function is None: continue
            try: path = inspect.getsourcefile(function)
            except TypeError: path = None
            if not path: continue
            self.check_path(path, force=True, required=True)
            code = getattr(getattr(function, "__func__", function), "__code__", None)
            if code is not None: self.check_code(code, required=True)
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
