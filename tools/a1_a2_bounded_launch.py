#!/usr/bin/env python3
"""Run one user-authorized bounded diagnostic under the central resource gate."""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT), str(ROOT/'src')]
from tools.a1_a2_cuda_audit import OUT, resource, heartbeat, write_json
p=argparse.ArgumentParser(__doc__)
p.add_argument('--label',required=True)
p.add_argument('--timeout',type=int,default=180)
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args()
assert 1<=a.timeout<=300
assert a.command and a.command[0]=='--'
a.command=a.command[1:]
resource(SimpleNamespace(label=a.label+'_pre'))
pre=json.loads((OUT/f'resource_{a.label}_pre.json').read_text())
gpu=pre['gpu_summary']['1']
assert gpu['mean_util']<60 and gpu['peak_util']<90 and gpu['min_free_mib']>4096, gpu
assert all(v['pid_alive'] and v['mtime_age_seconds']<120 for v in pre['heartbeat_after'].values())
start=time.monotonic()
with (OUT/f'{a.label}.stdout.log').open('w') as log:
 proc=subprocess.Popen(a.command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 status={'pid':proc.pid,'command':a.command,'timeout':a.timeout,'authorization':'user_requested_bounded_cuda_audit','physical_gpu':1}
 resource(SimpleNamespace(label=a.label+'_post'))
 post=json.loads((OUT/f'resource_{a.label}_post.json').read_text())
 gpu=post['gpu_summary']['1']
 stop_reason=None
 if gpu['mean_util']>=60 or gpu['peak_util']>=90: stop_reason='post_launch_overload'
 if not all(v['pid_alive'] and v['mtime_age_seconds']<120 for v in post['heartbeat_after'].values()): stop_reason='heartbeat_unhealthy'
 while proc.poll() is None and stop_reason is None:
  if time.monotonic()-start>a.timeout: stop_reason='bounded_timeout'; break
  if not all(v['pid_alive'] and v['mtime_age_seconds']<120 for v in heartbeat().values()): stop_reason='heartbeat_unhealthy';break
  try: proc.wait(timeout=10)
  except subprocess.TimeoutExpired: pass
 if proc.poll() is None:
  os.killpg(proc.pid,signal.SIGTERM)
  try: proc.wait(timeout=5)
  except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);proc.wait()
 status.update(exit_code=proc.returncode,stop_reason=stop_reason,seconds=time.monotonic()-start,heartbeat_after=heartbeat())
 write_json(OUT/f'{a.label}.launch.json',status)
 print(json.dumps(status),flush=True)
sys.exit(0 if proc.returncode==0 and stop_reason is None else 1)
