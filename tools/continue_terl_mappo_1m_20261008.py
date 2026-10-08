#!/usr/bin/env python3
"""Unconditional 100k -> cumulative 1m continuation; gate outcomes never stop it.

The frozen native runner restores all learning/environment/RNG state. This tool
only orchestrates processes, independent evaluation, retention and reporting.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import torch
from terl_mappo.run import atomic_json, file_hash, source_hash
from terl_mappo.supervise import selection_score, classify

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'artifacts/2026-10-08_terl_mappo_1m'
DOC=ROOT/'docs/TERL_MAPPO_1M_LEARNING_20261008_ZH.md'
BRANCH='experiment/terl-backbone-mappo-20261008'
MILESTONES={100000,250000,500000,750000,1000000}
FINAL_SEED_BASE=2046101800

def now(): return datetime.now().astimezone().isoformat()

def identity(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text()
        return stat[stat.rfind(')')+2:].split()[19]
    except FileNotFoundError: return None

def read(path): return json.loads(Path(path).read_text())

def rows(path):
    if not path.exists(): return []
    result=[];lines=path.read_text().splitlines()
    for i,line in enumerate(lines):
        try: result.append(json.loads(line))
        except json.JSONDecodeError:
            if i==len(lines)-1: break
            raise
    return result

def state(out,status,**data):
    value={'status':status,'controller_pid':os.getpid(),'target_total_steps':1000000,
           'resume_at_steps':100000,'gate_ignored':True,'mode':'EXACT_CONTINUATION','updated_at':now(),**data}
    atomic_json(out/'continuation_status.json',value)
    return value

def verify_contract(parent,config,proof):
    old=read(parent/'manifest.json'); original=old['config']
    if original['budget']!=100000 or config['budget']!=1000000: raise ValueError('unexpected budgets')
    if {k:v for k,v in original.items() if k!='budget'}!={k:v for k,v in config.items() if k!='budget'}:
        raise ValueError('continuation must differ only in total budget')
    if old['sources']!=source_hash(): raise ValueError('frozen scientific sources changed')
    if not proof['passed'] or proof['config_only_difference']!={'budget':[100000,1000000]}:
        raise ValueError('exact CUDA budget-resume validation required')
    if file_hash(proof['reference_checkpoint'])!=proof['sha256']: raise ValueError('resume-test reference hash changed')
    return old

def wait_parent(parent,pid,out,config,proof):
    verify_contract(parent,config,proof)
    tick=identity(pid)
    if tick is not None:
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        if 'terl_mappo.run' not in cmd or parent.name not in cmd: raise ValueError('wrong parent trainer PID')
    state(out,'ARMED_WAITING_PARENT_100K',parent_run=str(parent),parent_training_pid=pid,
          parent_start_ticks=tick,exact_resume_validation=proof)
    render(parent,out)
    while tick is not None and identity(pid)==tick:
        if (parent/'failure.json').exists(): raise RuntimeError('parent training failure; not a performance gate')
        time.sleep(5)
    progress=read(parent/'progress.json')
    if progress['steps']!=100000 or progress['status']!='COMPLETE_BUDGET' or not progress['finite']:
        raise ValueError('parent trainer exited without healthy 100k completion')
    checkpoint=parent/'checkpoints/step_000100000.pt'
    meta=read(checkpoint.with_suffix('.json'))
    if meta['steps']!=100000 or file_hash(checkpoint)!=meta['sha256']:
        raise ValueError('immutable 100k checkpoint/hash mismatch')
    data=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if data['steps']!=100000 or data['sources']!=source_hash() or data['config']!=read(parent/'manifest.json')['config']:
        raise ValueError('100k checkpoint contract mismatch')
    return checkpoint,meta,data

def prepare(parent,out,config,checkpoint,meta,data):
    if (out/'manifest.json').exists(): raise ValueError('existing continuation; inspect, do not duplicate')
    out.mkdir(parents=True,exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True);(out/'evaluations').mkdir(exist_ok=True)
    anchor=out/'checkpoints/step_000100000.pt';os.link(checkpoint,anchor)
    atomic_json(anchor.with_suffix('.json'),meta)
    # Preserve the full cumulative learning curve/counters, with no duplicate row.
    for name in ('metrics.jsonl','episodes.jsonl'):
        shutil.copyfile(parent/name,out/name)
    for path in sorted((parent/'evaluations').glob('*.json')):
        if path.name.endswith('.partial.json'): continue
        result=read(path)
        if result['seed_domain']=='screen' and result['steps']<=100000:
            result['provenance_parent_run']=str(parent)
            atomic_json(out/'evaluations'/path.name,result)
    manifest=read(parent/'manifest.json')
    manifest.update(config=config,started_at=now(),device='cuda:0',physical_gpu='0',
        initialization='exact full-state continuation of random-init seed9 parent; no teacher/BC',
        parent_run=str(parent),parent_manifest_sha256=file_hash(parent/'manifest.json'),
        resume_checkpoint=str(anchor),resume_checkpoint_sha256=meta['sha256'],resume_steps=100000,
        initial_fingerprint_scope='original random-init parent; full 100k runtime is restored from checkpoint',
        counter_units=read(parent/'manifest.json')['counter_units'],
        resume_counters={'steps':data['steps'],'agent_transitions':data['agent_transitions'],
                         'optimizer_steps':data['optimizer_steps'],'updates':data['trainer']['update_count']},
        evaluation_protocol={'screen_regular_per_mode':10,'screen_milestone_per_mode':20,
            'selection_top_screen_candidates':3,'selection_per_mode':20,'final_per_mode':50,
            'screen_seed_base':config['screen_seed_base'],'selection_seed_base':config['heldout_seed_base'],
            'final_seed_base':FINAL_SEED_BASE})
    atomic_json(out/'manifest.json',manifest)
    atomic_json(out/'progress.json',{'status':'RESUME_READY','steps':100000,'budget':1000000,
        'finite':True,'agent_transitions':data['agent_transitions'],'optimizer_steps':data['optimizer_steps'],
        'updates':data['trainer']['update_count'],'updated_at':now()})
    return anchor

def evaluations(out):
    return [read(p) for p in sorted((out/'evaluations').glob('*.json')) if not p.name.endswith('.partial.json')]

def cleanup(out,pending,current):
    screens=[r for r in evaluations(out) if r['seed_domain']=='screen']
    keep=MILESTONES|{r['steps'] for r in sorted(screens,key=selection_score,reverse=True)[:3]}
    keep|={step for _,step in pending}
    if current is not None: keep.add(current)
    deleted=[]
    for result in screens:
        path=Path(result['checkpoint']).resolve()
        if path.parent!=(out/'checkpoints').resolve() or result['steps'] in keep or not path.exists(): continue
        if file_hash(path)!=result['checkpoint_sha256']: raise ValueError('retention hash mismatch')
        path.unlink();deleted.append({'steps':result['steps'],'sha256':result['checkpoint_sha256'],'deleted_at':now()})
    if deleted:
        with (out/'retention.jsonl').open('a') as f:
            for row in deleted: f.write(json.dumps(row)+'\n')

def render(parent,out,decision=None,selected=None):
    ARCHIVE.mkdir(parents=True,exist_ok=True)
    status=read(out/'continuation_status.json')
    progress=read(out/'progress.json') if (out/'progress.json').exists() else read(parent/'progress.json')
    ev=evaluations(out) if (out/'evaluations').exists() else []
    metrics={r['steps']:r for r in rows(out/'metrics.jsonl')}
    joined=[]
    for result in ev:
        joined.append({**{k:v for k,v in result.items() if k!='episodes'},
            'ppo_training_at_checkpoint':metrics.get(result['steps']) if result['steps'] else None})
    summary={'updated_at':now(),'continuation':status,'progress':progress,'decision':decision,
             'selected':selected,'checkpoint_evidence':joined}
    atomic_json(ARCHIVE/'summary.json',summary)
    for name in ('continuation_status.json','manifest.json','metrics.jsonl','episodes.jsonl','retention.jsonl',
                 'selection.json','continuation_failure.json','evaluation_failures.json'):
        if (out/name).exists(): shutil.copyfile(out/name,ARCHIVE/name)
    for result in ev:
        atomic_json(ARCHIVE/'evaluations'/f'{result["seed_domain"]}_{result["steps"]:09d}.json',result)
    lines=['# TERL-MAPPO 累计1m Stage1自动线','',f'更新时间：{now()}。',
        f'自动线状态：`{status["status"]}`；总目标1,000,000 joint environment decisions；已完成累计步数：{progress["steps"]:,}。',
        f'科学分类：`{decision or "PENDING"}`。100k gate结果不控制续训。',
        '', '100k原run为精确续训parent；从100k checkpoint恢复actor/critic、Adam、ValueNorm、环境、全部RNG与计数。'
        '只改变累计budget：100k→1m；原native Stage1、256/8/4 backbone、seed9/109、PPO超参及25k保存切点全部保持。',
        '', '| step | domain | mode | normal/capture/n | ring2/ring3/strict | collision | capture mean/median/p90 s | censored |',
        '|---:|---|---|---|---|---|---|---|']
    for result in ev:
        for mode,m in result['modes'].items():
            times='/'.join('—' if m[k] is None else f'{m[k]:.2f}' for k in ('capture_time_mean','capture_time_median','capture_time_p90'))
            lines.append(f'| {result["steps"]} | {result["seed_domain"]} | {mode} | {m["normal_capture_count"]}/{m["capture_count"]}/{m["episodes"]} | {m["ring2_count"]}/{m["ring3_count"]}/{m["strict_geometry_count"]} | {m["collision_count"]}/{m["episodes"]} | {times} | {m["censored_n"]} |')
    if not ev: lines.append('| — | — | pending | — | — | — | — | — |')
    lines += ['', '原100k的screen结果复用且记录来源，后续每25k regular各10局，100/250/500/750/1000k各20局。'
        'screen/selection seed沿用原配置；最终独立seed2046101800起，各50局，与原100k final隔离。',
        '最终selection候选为screen预声明排序的前三个checkpoint，held-out各20局，再按normal、低collision、strict、ring3、较早step排序。'
        '保留latest、固定milestones、当前前三候选和待评估点，其余仅在完整screen/hash验证后删除本自动线文件。',
        '',f'best checkpoint/hash：`{selected["checkpoint"] if selected else "PENDING"}` / `{selected["checkpoint_sha256"] if selected else "PENDING"}`。',
        f'最新数值健康：`{progress.get("finite","pending")}`；training PID：`{progress.get("pid","waiting parent")}`。',
        f'最新PPO指标：`{json.dumps(progress.get("last_metrics",{}),ensure_ascii=False)}`。',
        '', '每checkpoint完整reward/entropy/type/time统计及相同步数PPO指标见artifact summary/evaluation JSON。'
        '低collision不等于低boundary；训练episode ring仅是终止时刻，学习判断使用独立整局visitation。',
        '', '100k报告只描述早期gate，不是1m终态结论。1m自动线不会因no-capture/partial/no-convincing-signal或CPU评估失败而停止训练。'
        '真实训练异常保留诊断，不静默重启或拼接科学版本。中央DAG仍只读。']
    DOC.write_text('\n'.join(lines)+'\n')
    screens=[r for r in ev if r['seed_domain']=='screen']
    if len(screens)>1:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,4,figsize=(16,3))
        for mode in ('argmax','sample'):
            for ax,key in zip(axes,('normal_capture_rate','ring2_rate','ring3_rate','collision_rate')):
                ax.plot([r['steps'] for r in screens],[r['modes'][mode][key] for r in screens],marker='o',label=mode)
                ax.set_title(key);ax.set_xlabel('cumulative environment decisions');ax.set_ylim(-.02,1.02);ax.legend()
        fig.tight_layout();fig.savefig(ARCHIVE/'learning_curves.png',dpi=150);plt.close(fig)
    return summary

def old_reports_live(parent):
    super_path=parent/'supervisor.json'
    if super_path.exists() and owned_report_alive(read(super_path)['pid'],'terl_mappo.supervise'): return True
    final_path=parent/'evidence_finalizer_status.json'
    if final_path.exists() and owned_report_alive(read(final_path)['pid'],'finalize_terl_mappo_results_20261008.py'): return True
    return False

def owned_report_alive(pid,tag):
    try: return tag in Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    except FileNotFoundError: return False

def best_effort_sync(parent,out):
    try: return sync(parent)
    except Exception as exc:
        atomic_json(out/'report_sync_error.json',{'error':repr(exc),'recorded_at':now()})
        print('report sync deferred: '+repr(exc),flush=True)
        return None

def sync(parent):
    # Training starts immediately; Git reporting waits for the older writers.
    if old_reports_live(parent): return None
    lock_path=ROOT/'runs/terl_mappo_reporting/git-sync.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!=BRANCH:
            raise ValueError('wrong worktree branch')
        old_docs=[ROOT/'docs/TERL_MAPPO_STAGE1_LEARNING_20261008_ZH.md',
                  ROOT/'docs/TERL_MAPPO_FOLLOWUP_DECISION_20261008_ZH.md']
        marker='\n当前训练预算以最新用户授权的累计1m自动线为准；本文件仅是100k早期gate报告，其结果不限制续训。见 `TERL_MAPPO_1M_CONTINUATION_20261008_ZH.md` 与 `TERL_MAPPO_1M_LEARNING_20261008_ZH.md`。\n'
        for path in old_docs:
            if path.exists() and marker not in path.read_text(): path.write_text(path.read_text()+marker)
        allowed={str(DOC.relative_to(ROOT)),*[str(p.relative_to(ROOT)) for p in old_docs]}|{str(p.relative_to(ROOT)) for p in ARCHIVE.rglob('*') if p.is_file()}
        staged=set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines())
        if staged-allowed: raise ValueError('unrelated staged changes; preserve')
        subprocess.run(['git','add',str(DOC.relative_to(ROOT)),str(ARCHIVE.relative_to(ROOT)),
                        *[str(p.relative_to(ROOT)) for p in old_docs if p.exists()]],cwd=ROOT,check=True)
        if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
            subprocess.run(['git','commit','-m','report(terl-mappo): update unconditional cumulative 1m line'],cwd=ROOT,check=True)
        subprocess.run(['git','-c','http.proxy=http://127.0.0.1:17892','push','origin',f'HEAD:refs/heads/{BRANCH}'],cwd=ROOT,check=True)
        remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:17892','ls-remote','origin',f'refs/heads/{BRANCH}'],cwd=ROOT,text=True).split()[0]
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        if head!=remote: raise ValueError('remote HEAD mismatch')
        return head

def supervise(args):
    parent=Path(args.parent).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    config=read(args.config);proof=read(args.validation)
    with (out/'continuation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        checkpoint,meta,data=wait_parent(parent,args.parent_pid,out,config,proof)
        if shutil.disk_usage(out).free<5*1024**3: raise RuntimeError('less than 5GiB free')
        anchor=prepare(parent,out,config,checkpoint,meta,data);del data
        env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'.runtime-deps')+':'+str(ROOT/'src'),
            CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        command=[sys.executable,'-u','-m','terl_mappo.run','--config',str(Path(args.config).resolve()),
                 '--output',str(out),'--device','cuda:0','--resume',str(anchor)]
        log=(out/'train.log').open('a');train=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state(out,'RUNNING_TO_1M',parent_run=str(parent),parent_training_pid=args.parent_pid,
              training_pid=train.pid,command=command,resume_checkpoint_sha256=meta['sha256'])
        manifest=read(out/'manifest.json');manifest.update(pid=train.pid,parent_training_pid=args.parent_pid,controller_pid=os.getpid())
        atomic_json(out/'manifest.json',manifest)
        render(parent,out)
        pending=[];seen={r['steps'] for r in evaluations(out) if r['seed_domain']=='screen'}
        evaluator=None;evaluator_log=None;current=None;failures=[]
        while True:
            if (out/'failure.json').exists() or (train.poll() is not None and train.returncode!=0):
                raise RuntimeError('continuation training failed; preserve runtime/checkpoints')
            if evaluator is not None and evaluator.poll() is not None:
                if evaluator.returncode:
                    failures.append({'steps':current,'returncode':evaluator.returncode,'recorded_at':now()})
                    atomic_json(out/'evaluation_failures.json',failures)
                else:
                    result=read(out/f'evaluations/screen_{current:09d}.json')
                    if file_hash(result['checkpoint'])!=result['checkpoint_sha256']: raise ValueError('evaluation hash mismatch')
                evaluator=None;evaluator_log.close();current=None
                cleanup(out,pending,current);render(parent,out);best_effort_sync(parent,out)
            for path in sorted((out/'checkpoints').glob('step_*.pt')):
                step=int(path.stem.rsplit('_',1)[1])
                if step not in seen: pending.append((path,step));seen.add(step)
            if pending and evaluator is None:
                path,current=pending.pop(0);n=20 if current in MILESTONES else 10
                evaluator_log=(out/f'eval_{current:09d}.log').open('a')
                eval_env=env.copy();eval_env['CUDA_VISIBLE_DEVICES']=''
                evaluator=subprocess.Popen([sys.executable,'-u','-m','terl_mappo.evaluate','--checkpoint',str(path),
                    '--output',str(out/f'evaluations/screen_{current:09d}.json'),'--episodes',str(n),
                    '--seed-base',str(config['screen_seed_base']),'--workers',str(config['evaluation_workers'])],
                    cwd=ROOT,env=eval_env,stdout=evaluator_log,stderr=subprocess.STDOUT)
                atomic_json(out/'evaluation_status.json',{'pid':evaluator.pid,'checkpoint_step':current,
                    'queued_steps':[s for _,s in pending]})
            if train.poll() is not None and not pending and evaluator is None: break
            time.sleep(5)
        progress=read(out/'progress.json')
        if progress['steps']!=1000000 or not progress['finite']: raise ValueError('1m target incomplete')
        if failures:
            state(out,'TRAINING_1M_COMPLETE_EVALUATION_INCOMPLETE',training_pid=train.pid,evaluation_failures=failures)
            render(parent,out);best_effort_sync(parent,out);return
        os.environ['CUDA_VISIBLE_DEVICES']=''
        from terl_mappo.evaluate import evaluate
        screens=[r for r in evaluations(out) if r['seed_domain']=='screen']
        expected=set(range(100000,1000001,25000))
        if expected-{r['steps'] for r in screens}: raise ValueError('missing 1m checkpoint screening')
        candidates=sorted(screens,key=selection_score,reverse=True)[:3]
        for result in candidates:
            evaluate(result['checkpoint'],out/f'evaluations/selection_{result["steps"]:09d}.json',20,
                     config['heldout_seed_base'],config['evaluation_workers'],'selection')
        selected=max([r for r in evaluations(out) if r['seed_domain']=='selection'],key=selection_score)
        os.link(selected['checkpoint'],out/'checkpoints/best.pt')
        final=evaluate(selected['checkpoint'],out/f'evaluations/final_{selected["steps"]:09d}.json',50,
                       FINAL_SEED_BASE,config['evaluation_workers'],'final')
        decision=classify(evaluations(out),final)
        selected={k:v for k,v in selected.items() if k!='episodes'}
        atomic_json(out/'selection.json',{'decision':decision,'selected':selected,
            'final':{k:v for k,v in final.items() if k!='episodes'},'rule':'top3 screen; heldout normal,-collision,strict,ring3,earliest'})
        state(out,'COMPLETE_1M_AND_FINAL_TEST',training_pid=train.pid,decision=decision)
        render(parent,out,decision,selected);head=sync(parent)
        atomic_json(out/'sync_complete.json',{'head':head,'verified_at':now()})

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--parent-pid',type=int,required=True)
    p.add_argument('--output',required=True);p.add_argument('--config',required=True);p.add_argument('--validation',required=True)
    p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    try: supervise(a)
    except BaseException as exc:
        out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
        atomic_json(out/'continuation_failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'failed_at':now()})
        state(out,'FAILED',error=repr(exc))
        try: render(Path(a.parent).resolve(),out);sync(Path(a.parent).resolve())
        except BaseException: traceback.print_exc()
        raise

if __name__=='__main__': main()
