#!/usr/bin/env python3
"""Bounded dual-GPU experiment ownership and low-frequency conditional orchestration."""
from __future__ import annotations
import argparse
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from tools import forward_final_single_task_20260915 as s
from tools import analyze_forward_final_single_task_20260915 as a
LEDGER=ROOT/'docs/FORWARD_FINAL_SINGLE_TASK_LEDGER_20260915_ZH.md'
BRANCH='experiment/small-step-ac-migration-20260828'


def write_ledger_snapshot(root,result,status):
    marker='<!-- AUTO_SINGLE_TASK_RESULTS -->'
    text=LEDGER.read_text().split(marker)[0]
    c=result['capture'];v=result['coverage'];rep=result['attribution'];r2=result['r2']
    lines=[marker,'','## 最新自动结果','',f"状态：`{result['status']}`。最终因果结论：`{result['causal_conclusion'] or 'PENDING — 尚无最终结论'}`。",'',
        '| 任务 | 当前 corrected baseline | 历史/归因对照 | R2 |','|---|---|---|---|',
        f"| Capture | {c['status']} / {c.get('decision')} | MAPPO-9-v2：best 10%/50%/5%；terminal 0%/5%/5%；完整持续窗口见 historical_reference.json | 不适用 |",
        f"| Coverage | {v['status']} / {v.get('decision')} | {(rep['representation']['decision'] if rep else 'Probes pending or not required')} | {r2['status']} |",'',
        '### Capture best / terminal / sustained window','',
        '| 模式 | 新线 best capture（step） | 新线 terminal capture | 最后3点 capture mean / min | 最后3点 collision mean |',
        '|---|---|---|---|---|']
    for mode,m in c.get('modes',{}).items():
        w=m['last_window'];b=m['best'];t=m['terminal']
        lines.append(f"| {mode} | {b['captured_rate']:.1%} ({b['step']}) | {t['captured_rate']:.1%} ({t['step']}) | "+
            (f"{w['capture_mean']:.1%} / {w['capture_min']:.1%} | {w['collision_mean']:.1%} |" if w else '尚不足3点 | — |'))
    lines+=['','### 运行交接','', '```json',json.dumps(status,indent=2,ensure_ascii=False),'```','',
        '机器可读完整结果：[MASTER](../artifacts/2026-09-15_single_task/MASTER.json)。',
        '历史完整曲线与窗口：[historical_reference](../artifacts/2026-09-15_single_task/historical_reference.json)。',
        '未经完整预算和所有必要 gate，不形成五类最终结论。']
    if result['r2_gate'] and result['r2_gate']['decision']=='START_R2' and r2.get('status')=='COMPLETE' and result['causal_conclusion']!='COVERAGE_REWARD_SCALE_CAUSALLY_LIMITING':
        lines+=['','下一实验仅提出：`Coverage-R3: explicit strict CE completion / hold achievement reward`。本轮未启动。']
    LEDGER.write_text(text+'\n'.join(lines)+'\n')


def checkpoint_git(root,final=False):
    # Only this experiment's small reports and its ledger. Never include foreign work or model tensors.
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if branch!=BRANCH:raise RuntimeError('Refuse automatic push after branch change')
    paths=[LEDGER]
    for p in root.rglob('*.json'):
        if p.stat().st_size<=4_000_000 and 'smoke' not in str(p.relative_to(root)):
            paths.append(p)
    rel=[str(p.relative_to(ROOT)) for p in paths]
    subprocess.run(['git','add','--',*rel],cwd=ROOT,check=True)
    diff=subprocess.run(['git','diff','--cached','--quiet','--',*rel],cwd=ROOT)
    if diff.returncode==1:
        # --only avoids committing any user's unrelated staged files.
        subprocess.run(['git','commit','--only','-m',
            'Record completed single-task causal experiment results' if final else 'Record single-task checkpoint trends and gated attribution',
            '--',*rel],cwd=ROOT,check=True)
        subprocess.run(['git','push','origin',f'HEAD:{BRANCH}'],cwd=ROOT,check=True)


def launch(root,kind,gpu):
    out=root/kind
    assert not out.exists(),f'Cannot overwrite {out}'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH='src:.')
    cmd=[sys.executable,'tools/train_forward_final_single_task_20260915.py','--task',kind,'--output',str(out),'--device','cuda:0']
    if kind=='coverage_r2':cmd+=['--r2-gate',str(root/'attribution/r2_gate.json')]
    log=(root/f'{kind}.log').open('a')
    process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close();return process


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--poll-seconds',type=int,default=300)
    parser.add_argument('--push-results',action='store_true');args=parser.parse_args()
    assert args.poll_seconds>=60
    root=args.root.resolve();root.mkdir(exist_ok=True)
    # Exclusive owner; stale lock is evidence to inspect, not auto-delete.
    lock=root/'supervisor.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).strip()
    assert not compute,'GPUs must be free; never stop unrelated processes'
    processes={};probe=None;last_signature=None
    try:
        processes['capture']=launch(root,'capture',0)
        processes['coverage']=launch(root,'coverage',1)
        while True:
            status={}
            for kind,process in processes.items():
                p=root/kind/'progress.json';progress=json.loads(p.read_text()) if p.exists() else {}
                status[kind]=dict(pid=process.pid,alive=process.poll() is None,returncode=process.poll(),
                    step=progress.get('step',0),checkpoint=str(max((root/kind).glob('step_*.pt'),default='NONE')),
                    throughput=progress.get('steps_per_second'),eta_seconds=progress.get('eta_seconds'),
                    status=progress.get('status','STARTING'))
                if process.poll() is not None and process.returncode!=0:
                    status[kind]['engineering_error']=True
            coverage=a.coverage_result(a.reports(root/'coverage'))
            if processes['coverage'].poll()==0 and coverage['decision']=='NO_MEANINGFUL_LEARNING' and probe is None and not (root/'attribution').exists():
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
                log=(root/'attribution.log').open('a')
                probe=subprocess.Popen([sys.executable,'tools/probe_forward_final_coverage_20260915.py','--root',str(root),'--device','cuda:0'],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                log.close()
            if probe:
                status['attribution']=dict(pid=probe.pid,alive=probe.poll() is None,returncode=probe.poll())
                report=root/'attribution/report.json'
                if probe.poll()==0 and report.exists() and 'coverage_r2' not in processes:
                    gate=json.loads((root/'attribution/r2_gate.json').read_text())
                    if gate['decision']=='START_R2':processes['coverage_r2']=launch(root,'coverage_r2',1)
            result=a.master(root)
            status['supervisor']=dict(pid=os.getpid(),next_wake_up=(datetime.now(timezone(timedelta(hours=8)))+timedelta(seconds=args.poll_seconds)).isoformat())
            status['causal_conclusion']=result['causal_conclusion']
            s.p.write_json(root/'SESSION_HANDOFF.json',status)
            write_ledger_snapshot(root,result,status)
            signature=([r['step'] for r in a.reports(root/'capture')],[r['step'] for r in a.reports(root/'coverage')],
                (root/'attribution/report.json').exists(),[r['step'] for r in a.reports(root/'coverage_r2')],result['status'])
            if signature!=last_signature and args.push_results:
                try:checkpoint_git(root,final=result['status']=='COMPLETE')
                except Exception:s.p.write_json(root/'push_failure.json',dict(traceback=traceback.format_exc()))
                last_signature=signature
            if result['status']=='COMPLETE':break
            alive=any(p.poll() is None for p in processes.values()) or (probe and probe.poll() is None)
            if not alive:
                s.p.write_json(root/'supervisor_failure.json',dict(status='INCOMPLETE_ENGINEERING_STOP',jobs=status))
                break
            time.sleep(args.poll_seconds)
    except BaseException:
        s.p.write_json(root/'supervisor_failure.json',dict(status='SUPERVISOR_ERROR',traceback=traceback.format_exc(),
            owned_pids={k:v.pid for k,v in processes.items()}))
        raise

if __name__=='__main__':main()
