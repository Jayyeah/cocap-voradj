"""Bootstrap itself must never execute old Guard/site bytecode."""
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
from terl_mappo.batch01.contracts import ROOT

def cache_for(source,old,tmp):
 old=old+b'\n'+b'#'*(source.stat().st_size-len(old)-1)
 assert len(old)==source.stat().st_size
 text=tmp/'previous_source.txt';text.write_bytes(old);s=source.stat();os.utime(text,ns=(s.st_atime_ns,s.st_mtime_ns))
 cache=Path(importlib.util.cache_from_source(str(source)));cache.parent.mkdir(exist_ok=True)
 previous=cache.read_bytes() if cache.exists() else None
 py_compile.compile(str(text),cfile=str(cache),dfile=str(source),doraise=True)
 return cache,previous

def restore(cache,previous):
 if previous is None:cache.unlink()
 else:cache.write_bytes(previous)

def test_same_size_mtime_V2_SourceGuard_cache_never_executes(tmp_path):
 source=ROOT/'src/terl_mappo/batch01/source_guard.py'
 old=subprocess.check_output(['git','show','863a0cf55aca0ace8a0aaab36d9166aa4c97268f:src/terl_mappo/batch01/source_guard.py'],cwd=ROOT)
 cache,previous=cache_for(source,old,tmp_path)
 try:
  p=subprocess.run([sys.executable,'-c','import terl_mappo.batch01.source_guard as m; g=m.SourceGuard.committed(); g.check(); assert hasattr(m,"code_signature"); print("CURRENT_GUARD_EXECUTED")'],cwd=ROOT,capture_output=True,text=True)
  assert p.returncode==0,p.stdout+p.stderr
  assert 'CURRENT_GUARD_EXECUTED' in p.stdout
 finally:restore(cache,previous)

def test_same_size_mtime_site_cache_never_executes(tmp_path):
 source=ROOT/'src/sitecustomize.py';cache,previous=cache_for(source,b'raise RuntimeError("OBSOLETE_SITE_EXECUTED")',tmp_path)
 try:
  p=subprocess.run([sys.executable,'-c','from terl_mappo.batch01.source_guard import SourceGuard; SourceGuard.committed().check(); print("SOURCE_SITE_EXECUTED")'],cwd=ROOT,capture_output=True,text=True)
  assert p.returncode==0,p.stdout+p.stderr
  assert 'OBSOLETE_SITE_EXECUTED' not in p.stderr and 'SOURCE_SITE_EXECUTED' in p.stdout
 finally:restore(cache,previous)

def test_legal_Guard_bytecode_positive(tmp_path):
 source=ROOT/'src/terl_mappo/batch01/source_guard.py';cache=Path(importlib.util.cache_from_source(str(source)));previous=cache.read_bytes() if cache.exists() else None
 try:
  py_compile.compile(str(source),cfile=str(cache),doraise=True)
  p=subprocess.run([sys.executable,'-c','from terl_mappo.batch01.source_guard import SourceGuard; SourceGuard.committed().check(); print("LEGAL_BYTECODE_SOURCE_PASS")'],cwd=ROOT,capture_output=True,text=True)
  assert p.returncode==0,p.stdout+p.stderr
 finally:restore(cache,previous)


def test_plain_python_without_trusted_bootstrap_fails_before_training():
 p=subprocess.run([sys._batch01_real_python,'-c','from terl_mappo.batch01.source_guard import SourceGuard; SourceGuard.committed().check()'],cwd=ROOT,capture_output=True,text=True)
 assert p.returncode!=0 and 'trusted source-first bootstrap required' in p.stderr,p.stdout+p.stderr
