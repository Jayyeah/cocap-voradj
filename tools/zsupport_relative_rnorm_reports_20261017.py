"""Persistent measured training status; never turn a launch into acceptance."""
from __future__ import annotations
import csv,json,time,sqlite3
from datetime import datetime,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/2026-10-17_relative_z_rnorm'
DOC=ROOT/'docs/experiments/relative_z_rnorm_20261017'
STAGES=('stage1','stage2','stage3');BUDGETS=(2000000,700000,700000)
def read(p,default=None):
    return json.loads(p.read_text()) if p.is_file() else default

def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.reports-tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def generate(runtime):
    DOC.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    now=datetime.now().astimezone();stages={};samples=read(runtime/'measured_throughput_samples.json',[])
    for s,b in zip(STAGES,BUDGETS):
        sd=runtime/'stages'/s;p=read(sd/'training/progress.json',{});done=read(sd/'training_complete.json',{})
        n=done.get('environment_decision_steps',p.get('environment_decision_steps',0))
        stages[s]=dict(budget=b,step=n,optimizer_updates=done.get('optimizer_updates',p.get('optimizer_updates',0)),progress=p,training_complete=bool(done),selection=read(sd/'selection_report.json'))
        if p and not any(x['stage']==s and x['step']==n for x in samples):
            samples.append(dict(stage=s,step=n,time=p['updated_at']))
    write(runtime/'measured_throughput_samples.json',samples)
    active=next((s for s in STAGES if not stages[s]['training_complete']), 'stage3')
    seq=[x for x in samples if x['stage']==active];windows=[]
    for end in seq:
        starts=[x for x in seq if end['step']-x['step']>=50000]
        if starts:
            start=max(starts,key=lambda x:x['step'])
            dt=(datetime.strptime(end['time'],'%Y-%m-%dT%H:%M:%S%z')-datetime.strptime(start['time'],'%Y-%m-%dT%H:%M:%S%z')).total_seconds()
            if dt>0:windows.append(dict(start_step=start['step'],end_step=end['step'],start_time=start['time'],end_time=end['time'],wall_seconds=dt,steps_per_second=(end['step']-start['step'])/dt))
    independent=[]
    if windows:
        independent=[windows[-1]]
        prior=[x for x in windows if x['end_step']<=independent[0]['start_step']]
        if prior:independent.insert(0,prior[-1])
    provisional=False
    if not independent and len(seq)>1 and seq[-1]['step']-seq[0]['step']>=10000:
        a,b=seq[0],seq[-1];dt=(datetime.strptime(b['time'],'%Y-%m-%dT%H:%M:%S%z')-datetime.strptime(a['time'],'%Y-%m-%dT%H:%M:%S%z')).total_seconds()
        if dt>0:independent=[dict(start_step=a['step'],end_step=b['step'],wall_seconds=dt,steps_per_second=(b['step']-a['step'])/dt)];provisional=True
    rate=independent[-1]['steps_per_second'] if independent else None
    def eta(steps):
        if not rate:return dict(remaining_decisions=steps,remaining_hours=None,completion_local=None)
        seconds=max(steps,0)/rate
        return dict(remaining_decisions=steps,remaining_hours=seconds/3600,completion_local=(now+timedelta(seconds=seconds)).isoformat())
    n1=stages['stage1']['step'];remaining=sum(max(v['budget']-v['step'],0) for v in stages.values())
    queue=[];timings=read(runtime/'queue_service_timings.json',[])
    if (runtime/'queue.sqlite3').is_file():
        with sqlite3.connect(runtime/'queue.sqlite3',timeout=30) as c:
            c.row_factory=sqlite3.Row;queue=[dict(x) for x in c.execute('SELECT id,kind,status FROM jobs')]
    services={k:[x['seconds'] for x in timings if x['kind']==k] for k in ('selection','formal','reference','render')}
    mean=lambda xs:sum(xs)/len(xs) if xs else None
    selection_service=mean(services['selection']);render_service=mean(services['render'])
    # Full service is unknown before real jobs run; do not infer a final-GIF date from training speed.
    closure=None
    if selection_service:
        rollout_seconds=selection_service/60
        formal_remaining=540-sum(60 for x in queue if x['kind'] in ('formal','reference') and x['status']=='complete')
        gif_remaining=180-sum(1 for x in queue if x['kind']=='render' and x['status']=='complete')
        formal_service=mean(services['formal']) or 60*rollout_seconds*1.5
        reference_service=mean(services['reference']) or 60*rollout_seconds*1.5
        extra=formal_remaining*max(formal_service,reference_service)/60+(gif_remaining*(render_service or 30))
        closure=dict(estimated_tail_hours=extra/3600,completion_local=(now+timedelta(seconds=(remaining/rate if rate else 0)+extra)).isoformat() if rate else None,uncertainty='rough conservative upper estimate; formal cost extrapolated from actual selection at factor 1.5 until measured; unmeasured rendering assumed 30 seconds/GIF; queue overlaps training',includes_evaluation_and_rendering=True)
    status=dict(updated_at=now.isoformat(),timezone='Asia/Shanghai',active_stage=active,global_step=stages[active]['step'],steps_per_second=rate,windows=independent,has_two_nonoverlapping_50k_windows=len(independent)>=2 and not provisional,provisional=provisional,stages={s:{k:v[k] for k in ('step','optimizer_updates','budget','training_complete')} for s,v in stages.items()},stage1_300k=eta(max(300000-n1,0)),stage1_budget=eta(max(2000000-n1,0)),all_training_budgets=eta(remaining),stage1_selection_handoff=dict(estimated_additional_hours=selection_service/3600 if selection_service else None,assumption='one observed selection service duration if queue remains current; actual handoff waits all 20 selections'),stage2_budget=eta(max(2000000-n1,0)+max(700000-stages['stage2']['step'],0)),stage3_budget=eta(remaining),formal_evaluation_and_gif_closure=closure,uncertainty='Stage2/3 extrapolate latest active-stage measured rate until those stages run; training ETAs exclude evaluation/rendering and stage-boundary waiting; checkpoint stalls are included in measured windows',queue_counts={k:sum(x['status']==k for x in queue) for k in ('pending','running','complete','failed')},queue_service_seconds={k:mean(v) for k,v in services.items()})
    write(DOC/'TRAINING_ETA_STATUS.json',status);write(ART/'training_eta_status.json',status)
    rows=[]
    if (runtime/'queue.sqlite3').is_file():
        with sqlite3.connect(runtime/'queue.sqlite3',timeout=30) as c:
            c.row_factory=sqlite3.Row
            for x in c.execute('SELECT * FROM checkpoints'):rows.append(dict(stage=x['stage'],kind='model_checkpoint',path=x['path'],sha256=x['sha256'],bytes=x['bytes'],step=x['step']))
            for x in c.execute('SELECT * FROM artifacts'):rows.append(dict(stage=x['stage'],kind=x['kind'],path=x['path'],sha256=x['sha256'],bytes=x['bytes'],step=''))
    with (DOC/'CHECKPOINT_AND_ROLLOUT_REGISTRY.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['stage','kind','path','sha256','bytes','step']);w.writeheader();w.writerows(rows)
    p=stages['stage1']['progress'];acceptance=read(runtime/'stage1_300k_active_acceptance.json')
    lines=['# Stage1 300k主动监督记录','',f'更新时间：{now.isoformat()}（Asia/Shanghai）。',f'状态：{("PASS，主Agent完成签收" if acceptance else "未签收；不能视为300k验收成功")}。',f'实际环境决策：{n1:,}；optimizer updates：{stages["stage1"]["optimizer_updates"]:,}。','',f'持久化runtime：`{runtime}`。','', '## 实际健康数据','', '```json',json.dumps(p,ensure_ascii=False,indent=2),'```','', '## 吞吐与ETA','', '```json',json.dumps(status,ensure_ascii=False,indent=2),'```','', '训练频率、epsilon/LR、四类Replay配额与课程保持原合同。Support增长不单独证明方法改善；捕获/碰撞/CE必须同时报告。有限样本和单seed不构成非劣性或优越性结论。']
    if acceptance:lines+=['','## 主Agent验收证据','','```json',json.dumps(acceptance,ensure_ascii=False,indent=2),'```']
    (DOC/'STAGE1_300K_MONITOR_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    for i,s in enumerate(STAGES,1):
        sd=runtime/'stages'/s;v=stages[s]
        summary=dict(stage=s,**v,primary=read(sd/'independent/argmax/report.json'),secondary=read(sd/'independent/sample/report.json'))
        (DOC/f'STAGE{i}_RESULT_ZH.md').write_text(f'# {s}实际执行状态\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n')
