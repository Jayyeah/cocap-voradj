#!/usr/bin/env python3
"""Trusted source-first entrypoint. Invoke with python -S, never import it."""
import os
import sys
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Keep project imports out of bootstrap's stdlib dependency resolution.
sys.path[:]=[p for p in sys.path if not os.path.abspath(p or os.getcwd()).startswith(ROOT+os.sep) and os.path.abspath(p or os.getcwd())!=ROOT]
import argparse
import hashlib
import importlib.machinery
import json
from pathlib import Path
import runpy
import shlex
from types import MappingProxyType

P=Path(ROOT);LOCK='configs/experiments/terl_mappo_batch01_20261009/base_lock.json'
if not sys.flags.no_site:raise RuntimeError('Batch01 source bootstrap requires python -S')
ap=argparse.ArgumentParser();ap.add_argument('--lock-sha',required=True);ap.add_argument('--delta');ap.add_argument('--python-args',nargs=argparse.REMAINDER,required=True)
a=ap.parse_args();lock=json.loads((P/LOCK).read_text());digest=hashlib.sha256(json.dumps(lock,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
if digest!=a.lock_sha:raise RuntimeError('source bootstrap canonical lock mismatch')
expected=dict(lock['common_sources']);delta_path=a.delta or os.environ.get('BATCH01_BOOTSTRAP_DELTA')
if delta_path:
 path=Path(delta_path)
 if not path.is_absolute():path=P/path
 if not path.resolve().is_relative_to(P) or not path.is_file():raise RuntimeError('bootstrap delta missing/outside project')
 delta=json.loads(path.read_text())
 if delta['base']['lock_sha256']!=a.lock_sha:raise RuntimeError('bootstrap ARM delta lock mismatch')
 for change in delta.get('source_changes',[]):
  relative=Path(change['path'])
  if relative.is_absolute() or '..' in relative.parts or change['before']!=expected.get(str(relative)):raise RuntimeError('bootstrap source delta before/path mismatch')
  if change['after'] is None:expected.pop(str(relative),None)
  else:expected[str(relative)]=change['after']
 os.environ['BATCH01_BOOTSTRAP_DELTA']=str(path.relative_to(P))
bootstrap_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
if expected.get('tools/batch01_python.py')!=bootstrap_sha:raise RuntimeError('source bootstrap itself differs from canonical pin')
sys._batch01_verified_source_bootstrap=MappingProxyType({'root':ROOT,'lock_sha256':a.lock_sha,'bootstrap_sha256':bootstrap_sha})
sys._batch01_source_bootstrap_failure=None
original=importlib.machinery.SourceFileLoader.get_code

def source_code(loader,fullname):
 path=Path(loader.get_filename(fullname)).absolute()
 if not path.resolve().is_relative_to(P):return original(loader,fullname)
 raw=loader.get_data(str(path));relative=str(path.relative_to(P));actual=hashlib.sha256(raw).hexdigest()
 module=sys.modules.get('terl_mappo.batch01.source_guard');active=getattr(module,'_ACTIVE',None);guard=active.get() if active is not None else None
 sources=guard.expected if guard is not None else expected
 declared=sources.get(relative)
 # One exact framework template is already pinned by the canonical contract.
 framework=sys.modules.get('torch.distributed.nn.jit.instantiator');directory=getattr(framework,'INSTANTIATED_TEMPLATE_DIR_PATH',None)
 generated=directory is not None and path==Path(directory)/'_remote_module_non_scriptable.py' and path==path.resolve()
 if declared is None and guard is None:
  # Cold source delta imports execute fresh source, never cached code. They get
  # no permission here: SourceGuard must later verify their locked declaration
  # and actual exec witness before any hook/update/checkpoint.
  return compile(raw,str(path),'exec',dont_inherit=True,optimize=sys.flags.optimize)
 if declared!=actual and not (generated and actual==lock['generated_runtime_sources'].get('torch.distributed.nn.non_scriptable_remote_template')):
  message='source-first loader rejected undeclared/changed source: '+relative
  if guard is not None:guard.fail(message)
  sys._batch01_source_bootstrap_failure=message
  raise RuntimeError(message)
 # Deliberately never call get_code/get_bytecode_path or read any local pyc.
 return compile(raw,str(path),'exec',dont_inherit=True,optimize=sys.flags.optimize)

importlib.machinery.SourceFileLoader.get_code=source_code
sys.path[:0]=[str(P/'src'),str(P)]
import site
site.main()
# sitecustomize cannot witness the bootstrap frame that started before it.
sys._batch01_exec_records.setdefault(str(Path(__file__).absolute()),[]).append(sys._getframe().f_code)
# Child evaluators/QA subprocesses inherit the same source-first policy, including
# archives with the same canonical lock. This is a shell shim, never a module.
real_python=sys.executable;directory=Path('/home/yjq/rl/CoCap1/batch01-commander-runtime/source_launchers');directory.mkdir(parents=True,exist_ok=True)
shim=directory/(a.lock_sha+'.sh')
script='#!/bin/bash\nset -euo pipefail\nscience_root=""\nIFS=: read -ra candidates <<< "${PYTHONPATH:-}"\nfor candidate in "${candidates[@]}"; do\n if [[ "$candidate" == */src && -f "${candidate%/src}/tools/batch01_python.py" ]]; then science_root="${candidate%/src}"; break; fi\ndone\nif [[ -z "$science_root" && -f "$PWD/tools/batch01_python.py" ]]; then science_root="$PWD"; fi\n[[ -n "$science_root" ]] || exit 91\nexec '+shlex.quote(real_python)+' -S "$science_root/tools/batch01_python.py" --lock-sha '+shlex.quote(a.lock_sha)+' --python-args "$@"\n'
if not shim.exists():
 temp=shim.with_name(shim.name+'.'+str(os.getpid()));temp.write_text(script);temp.chmod(0o755);temp.replace(shim)
elif shim.read_text()!=script:raise RuntimeError('existing child source launcher differs')
sys._batch01_real_python=real_python;sys.executable=str(shim)
args=a.python_args
while args and args[0] in ('-B','-u'):args=args[1:]
if not args:raise ValueError('missing Python module/script/inline program')
if args[0]=='-m':
 sys.argv=[args[1],*args[2:]];runpy.run_module(args[1],run_name='__main__',alter_sys=True)
elif args[0]=='-c':
 sys.argv=['-c',*args[2:]];exec(compile(args[1],'<string>','exec'),{'__name__':'__main__'})
else:
 sys.argv=args;runpy.run_path(args[0],run_name='__main__')
