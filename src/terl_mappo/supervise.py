"""Detached 100k gate: train + asynchronous screening + bounded final reporting.

No other experiment processes are inspected or signalled. No performance early
stop and no automatic extension; checkpoints are immutable and few at this gate.
"""
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
from .native import ROOT
from .run import atomic_json, DEFAULT

DOC = ROOT/'docs/TERL_MAPPO_STAGE1_LEARNING_20261008_ZH.md'
ARCHIVE = ROOT/'artifacts/2026-10-08_terl_mappo'

def read_evals(out):
    return [json.loads(p.read_text()) for p in sorted((out/'evaluations').glob('*.json'))
            if not p.name.endswith('.partial.json')]

def selection_score(result):
    m=list(result['modes'].values())
    return (sum(x['normal_capture_rate'] for x in m)/2,
            -sum(x['collision_rate'] for x in m)/2,
            sum(x['strict_geometry_rate'] for x in m)/2,
            sum(x['ring3_rate'] for x in m)/2,
            -result['steps'])

def classify(evals,final):
    screens=[x for x in evals if x['seed_domain']=='screen']
    later=[x for x in screens if x['steps']>0]
    base=next((x for x in screens if x['steps']==0),None)
    positive=[x for x in later if any(m['normal_capture_count']>=2 for m in x['modes'].values())]
    if (len(positive)>=2 and any(m['normal_capture_rate']>=.2 and m['collision_rate']<.5
                               for m in final['modes'].values()) and base and
            max(m['normal_capture_rate'] for m in final['modes'].values())>
            max(m['normal_capture_rate'] for m in base['modes'].values())+.1):
        return 'TERL_MAPPO_STAGE1_LEARNABLE'
    if any(m['capture_count'] for x in later for m in x['modes'].values()):
        return 'TERL_MAPPO_STAGE1_PARTIAL'
    if base:
        base2=max(m['ring2_rate'] for m in base['modes'].values())
        base3=max(m['ring3_rate'] for m in base['modes'].values())
        if any(m['ring2_rate']>base2+.15 or m['ring3_rate']>base3+.1 for x in later for m in x['modes'].values()):
            return 'TERL_MAPPO_STAGE1_PARTIAL'
    return 'TERL_MAPPO_STAGE1_NO_CONVINCING_SIGNAL'

def render_report(out,status,decision=None,selected=None):
    ARCHIVE.mkdir(parents=True,exist_ok=True)
    progress=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else {}
    evals=read_evals(out)
    lines=['# TERL-MAPPO Stage1 学习验证','',f'更新时间：{datetime.now().astimezone().isoformat()}（Asia/Shanghai）。',
           f'状态：`{status}`；科学分类：`{decision or "PENDING：训练/独立评估未完成"}`。',
           '',f'预算：100,000 joint environment decisions（Stage1上限2M；不自动扩步）。实际进度：{progress.get("steps",0):,}。',
           f'运行目录：`{out}`。最新已落盘数值健康：`{progress.get("finite","pending")}`；训练PID `{progress.get("pid","pending")}`。',
           '', '## Checkpoint screening / held-out', '',
           '| steps | domain | mode | normal/capture n | ring2/ring3 | strict | collision | capture mean/median/p90 s | censored |',
           '|---:|---|---|---|---|---|---|---|---|']
    for x in evals:
        for mode,m in x['modes'].items():
            times='/'.join('—' if m[k] is None else f'{m[k]:.2f}' for k in ('capture_time_mean','capture_time_median','capture_time_p90'))
            lines.append(f'| {x["steps"]} | {x["seed_domain"]} | {mode} | {m["normal_capture_count"]}/{m["capture_count"]}/{m["episodes"]} | {m["ring2_count"]}/{m["ring3_count"]} | {m["strict_geometry_count"]} | {m["collision_count"]}/{m["episodes"]} | {times} | {m["censored_n"]} |')
    if not evals: lines += ['| — | screen | pending | — | — | — | — | — | — |']
    lines += ['', '0/25k/50k/100k 每模式20局，75k每模式10局。screen同组seed对齐初态，sample有独立动作RNG。'
              '训练seed9；screen2026100800起；selection-heldout2036100800起；final2046100800起。',
              '', '预声明selection：screen pooled normal rate最高，其次collision最低、strict最高、ring3最高、较早checkpoint；'
              'screen capture候选和最终screen-best扩展selection-heldout各20局，再按同规则选取；final各50局不参与selection。',
              '',f'最佳checkpoint：`{selected["checkpoint"] if selected else "PENDING"}`；SHA256：`{selected["checkpoint_sha256"] if selected else "PENDING"}`。',
              '', '## 数值健康与解释', '',
              f'最新PPO telemetry：`{json.dumps(progress.get("last_metrics",{}),ensure_ascii=False)}`。',
              '', '仅凭reward上升不认定学会围捕。Strong要求多个checkpoint重复normal capture、相对初始改善且碰撞不主导；'
              'partial包括孤立capture或ring改善。100k无信号不代表MAPPO不可学习；2M内是否扩步需审查环形几何、exploration、critic和原生驻留reward。',
              '', '## 后续决定', '',
              ('当前只等待已启动的bounded训练/独立评估；不启动CoCap-backbone对照或7M课程。' if not decision else
               '保留本次单seed、有限预算证据；先复核最佳checkpoint与final独立测试，再决定Stage1扩展或同任务backbone对照。'),
              '', '中央DAG/state只读；handoff在本分支。模型文件留在服务器，不提交大checkpoint。']
    DOC.write_text('\n'.join(lines)+'\n')
    summary={'status':status,'decision':decision,'progress':progress,
             'evaluations':[{k:v for k,v in x.items() if k!='episodes'} for x in evals], 'selected':selected}
    atomic_json(ARCHIVE/'summary.json',summary)
    if (out/'manifest.json').exists(): shutil.copyfile(out/'manifest.json',ARCHIVE/'manifest.json')
    # Lightweight complete curves, plus failed diagnostics only from this experiment.
    for name in ('metrics.jsonl','episodes.jsonl','failure.json','supervisor.json','latest_checkpoint.json'):
        if (out/name).exists(): shutil.copyfile(out/name,ARCHIVE/name)
    for x in evals:
        dest=ARCHIVE/'evaluations'/f'{x["seed_domain"]}_{x["steps"]:09d}.json'
        atomic_json(dest,x)
    if evals:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        screens=[x for x in evals if x['seed_domain']=='screen']
        if screens:
            fig,axes=plt.subplots(1,3,figsize=(12,3))
            for mode in ('argmax','sample'):
                for ax,key in zip(axes,('normal_capture_rate','ring3_rate','collision_rate')):
                    ax.plot([x['steps'] for x in screens],[x['modes'][mode][key] for x in screens],marker='o',label=mode)
                    ax.set_title(key); ax.set_xlabel('environment decisions'); ax.set_ylim(-.02,1.02); ax.legend()
            fig.tight_layout(); fig.savefig(ARCHIVE/'learning_curves.png',dpi=150); plt.close(fig)
    return summary

def sync_report():
    # Only explicitly enumerated experimental reports/artifacts are staged.
    subprocess.run(['git','add',str(DOC.relative_to(ROOT)),str(ARCHIVE.relative_to(ROOT))],cwd=ROOT,check=True)
    changed=subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode
    if changed:
        subprocess.run(['git','commit','-m','report(terl-mappo): update bounded Stage1 evidence'],cwd=ROOT,check=True)
    subprocess.run(['git','-c','http.proxy=http://127.0.0.1:17892','push','origin','HEAD:refs/heads/experiment/terl-backbone-mappo-20261008'],cwd=ROOT,check=True)
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:17892','ls-remote','origin','refs/heads/experiment/terl-backbone-mappo-20261008'],cwd=ROOT,text=True).split()[0]
    local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if remote!=local: raise ValueError('remote HEAD mismatch')
    return local

def supervise(args):
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'supervisor.lock','w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if shutil.disk_usage(out).free<5*1024**3: raise RuntimeError('less than 5GiB free; no launch')
    env=os.environ.copy(); env['PYTHONPATH']=str(ROOT/'.runtime-deps')+':'+str(ROOT/'src')
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=str(args.gpu))
    config=json.loads(Path(args.config).read_text()); train_log=open(out/'train.log','a')
    command=[sys.executable,'-u','-m','terl_mappo.run','--config',args.config,'--output',str(out),'--device','cuda:0']
    train=subprocess.Popen(command,cwd=ROOT,env=env,stdout=train_log,stderr=subprocess.STDOUT)
    eval_process=None; pending=[]; screened=set(); evaluator_log=None
    atomic_json(out/'supervisor.json',{'status':'RUNNING','pid':os.getpid(),'training_pid':train.pid,'gpu':args.gpu,'command':command})
    checkpoints=[0,25000,50000,75000,100000]
    while True:
        if (out/'failure.json').exists() or (train.poll() is not None and train.returncode!=0):
            render_report(out,'IMPLEMENTATION_OR_CONTRACT_BLOCKED','IMPLEMENTATION_OR_CONTRACT_BLOCKED')
            sync_report(); raise RuntimeError('training failure; preserved diagnostics and no restart')
        if eval_process is not None and eval_process.poll() is not None:
            if eval_process.returncode: raise RuntimeError('independent evaluation failed; inspect eval log')
            eval_process=None; evaluator_log.close()
        for step in checkpoints:
            path=out/f'checkpoints/step_{step:09d}.pt'
            if path.exists() and step not in screened:
                pending.append((path,step)); screened.add(step)
        if pending and eval_process is None:
            path,step=pending.pop(0); n=10 if step==75000 else 20
            evaluator_log=open(out/f'eval_{step:09d}.log','a')
            eval_env=env.copy(); eval_env['CUDA_VISIBLE_DEVICES']=''
            eval_process=subprocess.Popen([sys.executable,'-u','-m','terl_mappo.evaluate','--checkpoint',str(path),
                '--output',str(out/f'evaluations/screen_{step:09d}.json'),'--episodes',str(n),
                '--seed-base',str(config['screen_seed_base']),'--workers',str(config['evaluation_workers'])],
                cwd=ROOT,env=eval_env,stdout=evaluator_log,stderr=subprocess.STDOUT)
            atomic_json(out/'evaluation_status.json',{'pid':eval_process.pid,'checkpoint_step':step,'queued_steps':[s for _,s in pending]})
        if train.poll() is not None and not pending and eval_process is None and len(screened)==len(checkpoints): break
        time.sleep(5)
    evals=read_evals(out); screens=[x for x in evals if x['seed_domain']=='screen']
    best=max(screens,key=selection_score)
    candidates=[x for x in screens if any(m['capture_count'] for m in x['modes'].values())]
    if best not in candidates: candidates.append(best)
    from .evaluate import evaluate
    for x in candidates:
        evaluate(x['checkpoint'],out/f'evaluations/selection_{x["steps"]:09d}.json',20,
                 config['heldout_seed_base'],config['evaluation_workers'],'selection')
    selected=max([x for x in read_evals(out) if x['seed_domain']=='selection'],key=selection_score)
    # best.pt aliases immutable candidate; latest remains independently resumable.
    best_path=out/'checkpoints/best.pt'
    if not best_path.exists(): os.link(selected['checkpoint'],best_path)
    final=evaluate(selected['checkpoint'],out/f'evaluations/final_{selected["steps"]:09d}.json',50,
                   2046100800,config['evaluation_workers'],'final')
    decision=classify(read_evals(out),final)
    atomic_json(out/'selection.json',{'rule':'normal_rate, -collision, strict, ring3, earliest',
        'selected':{k:v for k,v in selected.items() if k!='episodes'},'final':{k:v for k,v in final.items() if k!='episodes'},'decision':decision})
    atomic_json(out/'supervisor.json',{'status':'COMPLETE','decision':decision,'pid':os.getpid(), 'training_pid':train.pid})
    render_report(out,'COMPLETE_BOUNDED_GATE',decision,{k:v for k,v in selected.items() if k!='episodes'})
    sync_report()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--config',default=str(DEFAULT)); p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args()
    try: supervise(a)
    except BaseException as exc:
        atomic_json(Path(a.output)/'supervisor_failure.json',{'error':repr(exc),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__': main()
