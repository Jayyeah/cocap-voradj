#!/usr/bin/env python3
"""Read-only experiment watcher and bounded morning ledger writer; never trains."""
import argparse
from datetime import datetime, timezone, timedelta
import gzip
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/2026-09-14_overnight'
BRANCH='experiment/small-step-ac-migration-20260828'
LABEL='EXPLORATORY_NON_GATE'


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(temp,path)


def snapshot():
    now=datetime.now(timezone(timedelta(hours=8))).isoformat()
    runs={}
    for gpu,kind in [(0,'scratch'),(1,'bc_ppo')]:
        out=BASE/(kind+'_seed1')
        launch=read(out/'launch.json',{})
        progress=read(out/'progress.json',{})
        report=read(out/'report.json')
        resume=read(out/'resume.json',{})
        pid=resume.get('pid',launch.get('pid'))
        proc=Path(f'/proc/{pid}/cmdline')
        try: alive=proc.exists() and str(out).encode() in proc.read_bytes()
        except (FileNotFoundError,ProcessLookupError): alive=False
        status=report['status'] if report else progress.get('status','NOT_STARTED')
        if not alive and not report and status!='STOP_IMPLEMENTATION': status='PROCESS_EXITED_WITHOUT_REPORT'
        stages={}
        for step in ([0,25000,50000,75000,100000] if kind=='scratch' else [0,5000,10000]):
            ev=read(out/f'eval_step_{step:06d}.json')
            ck=read(out/f'step_{step:06d}.json')
            stages[str(step)]=dict(status='EVALUATED' if ev else ('CHECKPOINT_SAVED_EVAL_PENDING' if ck else 'NOT_REACHED'),
                checkpoint=ck,eval_summary=ev.get('summary') if ev else None,
                eval_env_steps=ev.get('eval_env_steps') if ev else None,
                matched_delta=read(out/f'matched_step_{step:06d}.json'),
                training=read(out/f'training_step_{step:06d}.json'))
        runs[kind]=dict(gpu=gpu,pid=pid,alive=alive,tmux=f'cocap_overnight_{kind}_20260914',
            run=str(out.relative_to(ROOT)),target_step=100000 if kind=='scratch' else 10000,
            next_gate_step=25000 if kind=='scratch' else 5000,
            status=status,classification=LABEL,decision=report.get('decision') if report else 'PENDING',
            launch_head=launch.get('git_head'),progress=progress,eval_progress=read(out/'eval_progress.json'),
            report=report,stages=stages)
        if not alive:
            for name in ('learning.jsonl','episodes.jsonl'):
                path=out/name
                if path.exists():
                    with gzip.GzipFile(filename=str(path)+'.gz',mode='wb',mtime=0) as f: f.write(path.read_bytes())
    summary=dict(timestamp=now,classification=LABEL,runs=runs,
        p1_transition_return='PASS_AFTER_FOUR_CONFIRMED_BUG_FIXES',p1_central_v_state_aliasing='UNRESOLVED',
        p2='FAIL_HOLD',p3='INCONCLUSIVE_HOLD',formal_ppo='HOLD',formal_scratch='HOLD',
        automatic_25k_ppo=False,automatic_scratch_over_100k=False,
        next_check='BC 5k matched evaluation / Scratch 25k checkpoint; morning final report; no budget extension')
    write(BASE/'MASTER_SUMMARY.json',summary)
    gatepath=ROOT/'artifacts/2026-09-09_root_cause/stage_gate.json'
    gate=read(gatepath,{})
    gate.update(updated_at=now,p1_transition_return=summary['p1_transition_return'],
        p1_central_v_state_aliasing='UNRESOLVED',p2='FAIL_HOLD',p3='INCONCLUSIVE_HOLD',
        formal_ppo='HOLD',formal_scratch='HOLD',ppo_25k='HOLD',
        overnight_diagnostic={'classification':LABEL,'summary':str((BASE/'MASTER_SUMMARY.json').relative_to(ROOT)),
            'bc_ppo':runs['bc_ppo']['status'],'scratch':runs['scratch']['status'],
            'user_authorized_bounded_exception_only':True},
        next_single_proposal='MASTER reviews bounded overnight diagnostic; no automatic formal gate promotion, 25k PPO, 200k scratch or extra seeds')
    write(gatepath,gate)
    handpath=ROOT/'artifacts/2026-09-09_root_cause/SESSION_HANDOFF.json'
    hand=read(handpath,{})
    hand.update(timestamp=now,project_jobs_running=any(r['alive'] for r in runs.values()),
        gpu0=runs['scratch'],gpu1=runs['bc_ppo'],p2='FAIL_HOLD',p3='INCONCLUSIVE_HOLD',ppo_25k='HOLD',
        formal_ppo='HOLD',formal_scratch='HOLD',overnight_summary=str((BASE/'MASTER_SUMMARY.json').relative_to(ROOT)),
        next_action=summary['next_check'],next_wake_up='BC 5k / Scratch 25k, then morning 100k review' if any(r['alive'] for r in runs.values()) else None)
    write(handpath,hand)
    write(BASE/'SESSION_HANDOFF.json',summary)
    block='\n<!-- OVERNIGHT_20260914_BEGIN -->\n\n## Overnight bounded diagnostic（最新自动快照）\n\n'
    block+=f'快照：{now}。两条均为 `{LABEL}`，用户显式授权的非正式预算例外；不覆盖既有科学裁决。\n\n'
    block+='P1 transition/return：四类修复后 PASS；central-V state aliasing：UNRESOLVED；P2 FAIL/HOLD；P3 INCONCLUSIVE/HOLD；formal PPO / formal Scratch：HOLD。\n\n'
    for kind,r in runs.items():
        block+=f'- {kind}: `{r["status"]}`，decision=`{r["decision"]}`，step={r["progress"].get("step",0)}；PID {r["pid"]}，tmux `{r["tmux"]}`；[run](../{r["run"]})。\n'
    block+='\n[逐 checkpoint 指标与 matched delta](../artifacts/2026-09-14_overnight/MASTER_SUMMARY.json)；[启动协议及 morning handoff](FORWARD_FINAL_OVERNIGHT_DIAGNOSTIC_20260914_ZH.md)。BC 5k 仅在全部 operational checks 成立时续至 10k；Scratch 100k 硬停，25k/50k 零 capture 不早停。禁止自动 25k PPO、>100k Scratch、200k 或新 seeds。任何正结果只提出下一 Gate，负结果不证明算法不可行。\n\n<!-- OVERNIGHT_20260914_END -->\n'
    for name in ['FORWARD_FINAL_PPO_ROOT_CAUSE_20260909_ZH.md','FORWARD_FINAL_SCRATCH_MAPPO_PREFLIGHT_20260914_ZH.md','FORWARD_FINAL_CORRECTED_PPO_LEDGER_20260909_ZH.md']:
        path=ROOT/'docs'/name
        content=path.read_text(); marker='\n<!-- OVERNIGHT_20260914_BEGIN -->'
        if marker in content: content=content[:content.index(marker)]
        path.write_text(content.rstrip()+'\n'+block)
    return summary


def commit_results():
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!=BRANCH:
        raise RuntimeError('Refuse commit on another branch')
    if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Index has unrelated staged work; result files retained; do not commit it')
    paths=[str(p.relative_to(ROOT)) for p in BASE.rglob('*') if p.is_file() and
           (p.suffix in ('.json','.txt') or p.name.endswith('.jsonl.gz'))]
    paths += ['artifacts/2026-09-09_root_cause/SESSION_HANDOFF.json','artifacts/2026-09-09_root_cause/stage_gate.json']
    paths += ['docs/'+name for name in ['FORWARD_FINAL_PPO_ROOT_CAUSE_20260909_ZH.md','FORWARD_FINAL_SCRATCH_MAPPO_PREFLIGHT_20260914_ZH.md','FORWARD_FINAL_CORRECTED_PPO_LEDGER_20260909_ZH.md']]
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
        subprocess.run(['git','commit','-m','Record bounded overnight diagnostic checkpoint results; retain formal HOLD'],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin',BRANCH],cwd=ROOT,check=True,timeout=180)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--commit',action='store_true')
    args=parser.parse_args()
    while True:
        result=snapshot()
        done=all(not r['alive'] for r in result['runs'].values())
        print(json.dumps({k:{f:r[f] for f in ('status','decision','alive')} for k,r in result['runs'].items()}),flush=True)
        if not args.watch or done:
            if args.commit: commit_results()
            break
        time.sleep(300)
