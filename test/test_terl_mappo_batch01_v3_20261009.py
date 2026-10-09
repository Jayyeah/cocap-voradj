"""V3 actual-code and new paired-seed contracts, fresh process per cache case."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from terl_mappo.batch01.contracts import ROOT, LOCK_PATH
from terl_mappo.batch01.evaluation import seed_manifest, protocol_for_line
from test_terl_mappo_batch01_qa_repairs_20261009 import source_snapshot

PROBE = r'''
import importlib, json, os, py_compile, sys
from pathlib import Path
from terl_mappo.batch01.contracts import ROOT, LOCK_PATH
from terl_mappo.batch01.source_guard import SourceGuard, SourceViolation
from terl_mappo.run import file_hash
name='cocap_voradj.models.continuous.qa_v3_cache'
p=ROOT/('src/'+name.replace('.','/')+'.py')
marker=ROOT/'runs/body.txt'
source='from pathlib import Path\nPath('+repr(str(marker))+').write_text("ran")\nVALUE = 0.4\ndef reward(raw,info):\n    return raw + VALUE\n'
p.write_text(source)
expected=json.loads((ROOT/LOCK_PATH).read_text())['common_sources'];expected[str(p.relative_to(ROOT))]=file_hash(p)
case=sys.argv[1];stamp=int(p.stat().st_mtime)
if case in {'stale','preloaded'}:
 p.write_text(source.replace('VALUE = 0.4','VALUE = 0.9'));os.utime(p,(stamp,stamp))
 py_compile.compile(str(p),doraise=True)
 p.write_text(source);os.utime(p,(stamp,stamp))
elif case=='legal_cache':py_compile.compile(str(p),doraise=True)
if case=='preloaded':
 module=importlib.import_module(name)
 assert module.VALUE == 0.9
 marker.unlink()
guard=SourceGuard(expected)
try:
 with guard.scope():module=importlib.import_module(name)
 value=guard.call(module.reward,0,{})
 if case=='drift':
  p.write_text(source.replace('VALUE = 0.4','VALUE = 0.9'))
  guard.call(module.reward,0,{})
 assert case in {'legal_source','legal_cache'} and value == 0.4
 print('POSITIVE_PASS')
except SourceViolation:
 assert case in {'stale','preloaded','drift'} and guard.failure
 if case=='stale':assert not marker.exists() and name not in sys.modules
 try:guard.check()
 except SourceViolation:pass
 else:raise AssertionError('failure not latched')
 print('NEGATIVE_PASS')
'''

@pytest.mark.parametrize('case',['legal_source','legal_cache','stale','preloaded','drift'])
def test_actual_execution_matches_source(source_snapshot,case):
    (source_snapshot/'runs').mkdir()
    env=dict(os.environ,PYTHONPATH=str(source_snapshot/'src')+':'+str(source_snapshot)+':'+os.environ['PYTHONPATH'],TMPDIR=str(source_snapshot/'runs'))
    result=subprocess.run([sys.executable,'-c',PROBE,case],cwd=source_snapshot,env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'PASS' in result.stdout


def test_new_arm_seeds_are_locked_paired_and_disjoint():
    lock=json.loads((ROOT/LOCK_PATH).read_text());config=lock['anchor_config']
    legacy=seed_manifest(config,protocol_for_line(lock,'T0'))
    manifests=[seed_manifest(config,protocol_for_line(lock,line)) for line in ('T1','N1','R1','P1','C0')]
    assert all(m==manifests[0] for m in manifests)
    assert legacy['final']==list(range(2046101800,2046101850))
    assert manifests[0]['final']==list(range(2076100900,2076100950))
    assert manifests[0]['historical_reserved_seed_count']==180
    bad=protocol_for_line(lock,'R1');bad['final']['seed_base']=2046101800
    with pytest.raises(ValueError,match='overlap'):seed_manifest(config,bad)
    bad=protocol_for_line(lock,'R1');bad['scene_offsets']['4P1E1O6C']=1
    with pytest.raises(ValueError,match='overlap'):seed_manifest(config,bad)
