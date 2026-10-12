"""Read-only experiment snapshot/ETA/signal analysis; write delivery documents only."""
from datetime import datetime,timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import csv
import fcntl
import hashlib
import json
import math
import shutil
import statistics as st
import subprocess

BASE=Path('/home/yjq/rl/CoCap1')
CENTRAL=BASE/'ac-master-dag-20260921'
DELIVERY=BASE/'batch02-status-20261012-1655'
REL=Path('artifacts/2026-10-12_batch02_status_1655')
OUT=DELIVERY/REL
OUT.mkdir(parents=True,exist_ok=True)
NOW=datetime.now(ZoneInfo('Asia/Shanghai'))
NEXT=(NOW+timedelta(minutes=30)).isoformat()
ARCHIVE=[]


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def archive(path,relative):
    path=Path(path);raw=path.read_bytes();json.loads(raw)
    destination=OUT/relative;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
    ARCHIVE.append(dict(path=str(relative),source=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    return json.loads(raw)


def records(path):
    if not Path(path).exists():return []
    raw=Path(path).read_text()
    if not raw.endswith('\n'):raw=raw.rsplit('\n',1)[0]
    return [json.loads(line) for line in raw.splitlines() if line]


def alive(pid):
    p=Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().rsplit(') ',1)[1].split()[0]!='Z'


def mean(report,key):
    values=[m[key] for m in report['modes'].values() if m[key] is not None]
    return st.mean(values) if values else None


def selection_key(report):
    t=mean(report,'capture_time_seconds_mean')
    return mean(report,'capture_completion_rate'),-mean(report,'collision_rate'),-mean(report,'boundary_rate'),-(t if t is not None else float('inf')),-report['decisions']


def eta_training(info):
    steps=info['checkpoints'];screens=info['_screens'];progress=info['progress']
    samples=[]
    for a,b in list(zip(steps,steps[1:]))[-3:]:
        elapsed=(datetime.fromisoformat(b['created_at'])-datetime.fromisoformat(a['created_at'])).total_seconds()
        evaluation=screens.get(a['decisions'],{}).get('duration_seconds',0)
        training=elapsed-evaluation
        if training>0:samples.append((b['decisions']-a['decisions'])/training)
    rate=st.median(samples) if samples else progress.get('throughput',progress.get('training_and_evaluation_decisions_per_second'))
    durations=[r['duration_seconds'] for _,r in sorted(screens.items())[-3:]]
    evaluation=st.median(durations)
    remaining=1000000-progress['decisions']
    current=next((d for d in info['_active_evaluations'] if d['domain']=='screen' and d['decisions']==progress['decisions']),None)
    current_seconds=(max(0,current['total_episodes']-current['completed_episodes'])/current['total_episodes']*evaluation) if current else 0
    future=math.ceil(remaining/25000)
    seconds=remaining/rate+future*evaluation+current_seconds
    return dict(kind='remaining_training_and_fixed_screens',seconds=seconds,estimated_finish=(NOW+timedelta(seconds=seconds)).isoformat(),
                train_only_rate=rate,train_rate_samples=samples,screen_median_seconds=evaluation,future_screens=future,
                current_eval_estimated_remaining_seconds=current_seconds,runner_average_eta_seconds=progress.get('eta_seconds'),
                recent_rate_includes_plot_overhead=True,
                method='last3 checkpoint gaps minus previous completed screen duration; median training rate + remaining25k screens',
                warning='estimate from recent runtime; remaining evaluator trajectories and competing jobs may vary')


def eta_closeout(info):
    screens=info['_screens'];config=info['manifest']['config'];complete=info['_evaluations']
    ordered=sorted(screens.values(),key=selection_key,reverse=True)
    candidates={r['decisions'] for r in ordered[:config['evaluation']['top_screen_candidates']]}|{0,1000000}
    selections={step:r for (domain,step),r in complete.items() if domain=='selection'}
    selected=max([selections[s] for s in candidates],key=selection_key)['decisions'] if candidates<=selections.keys() else None
    per_mode=config['evaluation']['screen']['episodes_per_mode']
    scale=lambda step,domain:screens[step]['duration_seconds']*config['evaluation'][domain]['episodes_per_mode']/per_mode
    def pending(step,domain):
        if (domain,step) in complete:return 0.
        estimate=scale(step,domain)
        active=next((e for e in info['_active_evaluations'] if e['domain']==domain and e['decisions']==step),None)
        if active:estimate*=max(0,active['total_episodes']-active['completed_episodes'])/active['total_episodes']
        return estimate
    selection_seconds=sum(pending(s,'selection') for s in candidates)
    mandatory=sum(pending(s,'final') for s in {0,1000000})
    if selected is not None:
        extra=pending(selected,'final') if selected not in {0,1000000} else 0.
        lo=hi=selection_seconds+mandatory+extra
    else:
        lo=selection_seconds+mandatory
        hi=lo+max([pending(s,'final') for s in candidates-{0,1000000}] or [0])
    # A runtime sensitivity range, not a statistical confidence interval.
    seconds=(lo+hi)/2
    return dict(kind='selection_and_final_only_training_complete',seconds=seconds,
                estimated_finish=(NOW+timedelta(seconds=seconds)).isoformat(),
                rough_range_seconds=[.5*lo,1.5*hi],selection_candidates=sorted(candidates),
                completed_selections=sorted(selections),selected_step_known=selected,
                method='same-checkpoint completed screen duration scaled by registered selection/final episode counts; subtract completed work',
                warning='coarse runtime sensitivity range, not confidence interval; final seeds may have different episode lengths')


uuid_to_index={r.split(', ')[1]:int(r.split(', ')[0]) for r in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()}
compute={int(r.split(', ')[0]):uuid_to_index[r.split(', ')[1]] for r in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines()}
run_infos={}
screen_rows=[]
optimization_rows=[]
for condition,arms in [('baseline',['N2','R2','N3','R3','V3']),('stability',['R2','R3','V3'])]:
    for arm in arms:
        if condition=='stability':
            wt=BASE/'batch02-mappo-stability-20261012';root=wt/'runs/batch02_stability_gpu01_20261012'/(arm+'-stability')
        else:
            old=arm in ['N2','R2'];wt=BASE/('terl-mappo-batch02-a-long-20261011' if old else 'terl-mappo-batch02-native-20261011')
            root=wt/'runs'/('batch02_a_capability_20261011' if old else 'batch02_native_capability_20261011')/arm
        key=arm+('-stability' if condition=='stability' else '-baseline')
        manifest=archive(root/'manifest.json',Path('evidence')/key/'manifest.json')
        progress=archive(root/'progress.json',Path('evidence')/key/'progress.json')
        checkpoints=archive(root/'checkpoint_list.json',Path('evidence')/key/'checkpoint_list.json')
        last=archive(root/'last_checkpoint.json',Path('evidence')/key/'last_checkpoint.json')
        sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=wt,text=True).strip()
        assert sha==manifest['experiment_SHA']
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=wt,text=True).strip()
        screens={};evaluations={};active=[]
        for f in sorted((root/'evaluations').glob('*.json')):
            if f.name.endswith('.progress.json'):
                d=read(f)
                if d.get('status')=='EVALUATING':active.append(d)
                continue
            r=archive(f,Path('evidence')/key/'evaluations'/f.name)
            evaluations[(r['seed_domain'],r['decisions'])]=r
            if r['seed_domain']=='screen':
                screens[r['decisions']]=r
                for mode,m in r['modes'].items():
                    screen_rows.append(dict(line=key,condition=condition,arm=arm,decisions=r['decisions'],mode=mode,
                                            **{k:m[k] for k in ['capture_completion_rate','collision_rate','episode_reward_mean',
                                               'episode_success_rate','episode_length_mean','capture_time_seconds_mean']}))
        raw_metrics=records(root/'metrics.jsonl')
        fields=['decisions','entropy','approx_kl','approx_kl_max','actor_loss','value_loss','explained_variance',
                'kl_early_stop','minibatch_updates','gradient_finite','parameter_finite','reward_mean_active']
        optimization_rows += [dict(line=key,**{k:r.get(k) for k in fields}) for r in raw_metrics]
        pid=progress.get('pid');historical_pid=manifest['pid'];is_alive=alive(pid) if pid else False
        failure=read(root/'failure.json') if (root/'failure.json').exists() else None
        info=dict(line=key,arm=arm,condition=condition,run_directory=str(root),source_SHA=sha,manifest=manifest,progress=progress,
                  pid=pid,historical_pid=historical_pid,pid_alive=is_alive,expected_gpu=int(manifest['physical_gpu']),actual_gpu=compute.get(pid),
                  last_checkpoint=last,checkpoints=checkpoints,latest_complete_screen=max(screens),
                  active_evaluations=active,failure=failure,finite_metrics_all=all(r['gradient_finite'] and r['parameter_finite'] for r in raw_metrics),
                  _screens=screens,_evaluations=evaluations,_metrics=raw_metrics,_active_evaluations=active)
        if is_alive:assert compute[pid]==int(manifest['physical_gpu'])
        if progress['status']=='COMPLETE_BUDGET_AND_EVALUATION':
            result=archive(root/'capability_report.json',Path('evidence')/key/'capability_report.json')
            info['capability_summary']={k:v for k,v in result.items() if k!='final_reports'}
            info['final_metrics']={name:r['modes'] for name,r in result['final_reports'].items()}
            info['eta']=dict(kind='complete',seconds=0,estimated_finish=progress['updated_at'])
        elif progress['status']=='SELECTION_AND_FINAL':info['eta']=eta_closeout(info)
        elif condition=='stability':info['eta']=eta_training(info)
        else:info['eta']=dict(kind='unknown',seconds=None)
        info['next_wake_up']=NEXT
        run_infos[key]=info

pairs={}
for arm in ['R2','R3','V3']:
    a=run_infos[arm+'-baseline'];b=run_infos[arm+'-stability']
    common=sorted(a['_screens'].keys() & b['_screens'].keys())
    common=[s for s in common if s>=500000];end=max(common)
    late=[s for s in common if s>=800000]
    variants={}
    for label,info in [('baseline',a),('stability',b)]:
        rows=[r for r in info['_metrics'] if 500000<r['decisions']<=end]
        late_values=[m for step in late for m in info['_screens'][step]['modes'].values()]
        variants[label]=dict(late_capture=st.mean(m['capture_completion_rate'] for m in late_values),
            late_collision=st.mean(m['collision_rate'] for m in late_values),
            late_reward=st.mean(m['episode_reward_mean'] for m in late_values),
            per_mode={mode:{metric:st.mean(info['_screens'][s]['modes'][mode][metric] for s in late)
                for metric in ['capture_completion_rate','collision_rate','episode_reward_mean']} for mode in ['argmax','sample']},
            updates=len(rows),KL_stop_count=sum(r['kl_early_stop'] for r in rows),
            KL_stop_fraction=st.mean(r['kl_early_stop'] for r in rows),
            entropy_last100=st.mean(r['entropy'] for r in rows[-100:]),
            KL_last100=st.mean(r['approx_kl'] for r in rows[-100:]),
            value_loss_last100=st.mean(r['value_loss'] for r in rows[-100:]),
            explained_variance_last100=st.mean(r['explained_variance'] for r in rows[-100:]))
    pairs[arm]=dict(common_end_step=end,late_common_steps=late,conditions=variants,
                   latest_aligned_screen={label:info['_screens'][end]['modes'] for label,info in [('baseline',a),('stability',b)]},
                   interpretation='No consistent capture/safety recovery in the available aligned late window; incomplete PROVISIONAL result',
                   full_1m_pair_available=end==1000000)
pair_equivalence={}
for label in ['baseline','stability']:
    a=run_infos['R2-'+label]['_screens'];b=run_infos['R3-'+label]['_screens']
    common=sorted(a.keys() & b.keys())
    pair_equivalence[label]=dict(common_screen_steps=len(common),all_mode_summaries_exact=all(a[s]['modes']==b[s]['modes'] for s in common),
                                 independent_replication=False)
snapshot=dict(schema='batch02.quick_status.v1',observed_at=NOW.isoformat(),classification='PROVISIONAL',
              runs={k:{field:v for field,v in info.items() if not field.startswith('_')} for k,info in run_infos.items()},
              paired_signal=pairs,R2_R3_equivalence=pair_equivalence,
              scientific_interpretation='C / incomplete mixed evidence; historical KL setting activated but has not established stable recovery.',
              no_reward_network_causal_attribution=True,next_wake_up=NEXT,
              history_correction='R2/R3 baseline screen rebounded at925k; selected550k final62% vs last1M36%; not monotonic collapse.',
              experiment_code_untouched=True,training_processes_modified=False,next_stage_started=False)
write(OUT/'status_snapshot.json',snapshot)
write(OUT/'archive_manifest.json',dict(observed_at=NOW.isoformat(),files=ARCHIVE,checkpoint_weights_included=False))
for name,rows in [('screen_curves.csv',screen_rows),('optimization_curves.csv',optimization_rows)]:
    with (OUT/name).open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for arm in ['R2','R3','V3']:
    fig,axes=plt.subplots(2,3,figsize=(13,6),sharex=True)
    for i,mode in enumerate(['argmax','sample']):
        for label in ['baseline','stability']:
            curves=run_infos[arm+'-'+label]['_screens']
            steps=sorted(s for s in curves if s>=500000)
            for j,metric in enumerate(['capture_completion_rate','collision_rate','episode_reward_mean']):
                axes[i,j].plot([s/1000 for s in steps],[curves[s]['modes'][mode][metric] for s in steps],label=label)
                axes[i,j].set_title(mode+' '+metric);axes[i,j].set_xlabel('decisions(k)');axes[i,j].legend()
    fig.suptitle(arm+' paired screen / PROVISIONAL');fig.tight_layout();fig.savefig(OUT/f'{arm}_paired_screen.png',dpi=130);plt.close(fig)

lines=['# Batch02 并行线状态与历史信号','',f'观测时间：{NOW.isoformat()}。PROVISIONAL；只读检查，不修改/停止/重启训练。','',
       '| 线 | GPU | PID | 状态 | decisions | 完整 checkpoint | 最新完成screen | 剩余ETA | 预计结束 |',
       '|---|---:|---:|---|---:|---:|---:|---|---|']
for key,info in run_infos.items():
    eta=info['eta'];seconds=eta['seconds'];label='完成' if seconds==0 else f'{seconds/3600:.2f}h'
    if eta.get('rough_range_seconds'):label+=f"（粗估{eta['rough_range_seconds'][0]/3600:.1f}–{eta['rough_range_seconds'][1]/3600:.1f}h）"
    finish=eta.get('estimated_finish','')
    if finish:finish=datetime.fromisoformat(finish).strftime('%m-%d %H:%M')
    lines.append(f"| {key} | {info['actual_gpu'] if info['actual_gpu'] is not None else info['expected_gpu']} | {info['pid'] or '已退出'} | {info['progress']['status']} | {info['progress']['decisions']} | {info['last_checkpoint']['decisions']} | {info['latest_complete_screen']} | {label} | {finish} |")
lines+=['','三条 stability GPU0/0/1 绑定符合用户指定，进程存活、无 failure，累计记录的梯度/参数均有限。N2/R2/R3 baseline 已完成1M和selection/final；N3/V3 baseline训练预算已完成，仍在selection/final，不再消耗训练decision。','',
        'Stability ETA按最后3个checkpoint区间扣除前一screen耗时，取近期训练速度中位数，再加所有剩余25k screen；涵盖末点screen。Baseline selection/final ETA用相同checkpoint的screen耗时按登记episode数量换算并扣除已完成部分，属于较粗估计；所列敏感范围不是统计置信区间。硬件竞争和不同评估seed的episode长度会改变耗时。','',
        '| 完成的final（argmax/sample各50局） | selected | selected capture | last capture | selected collision | last collision | full episode_success |',
        '|---|---:|---|---|---|---|---|']
for arm in ['N2','R2','R3']:
    info=run_infos[arm+'-baseline'];m=info['final_metrics'];fmt=lambda name,metric:'/'.join(f"{m[name][mode][metric]:.0%}" for mode in ['argmax','sample'])
    lines.append(f"| {arm} | {info['capability_summary']['selected_checkpoint_decisions']} | {fmt('selected','capture_completion_rate')} | {fmt('last','capture_completion_rate')} | {fmt('selected','collision_rate')} | {fmt('last','collision_rate')} | {fmt('last','episode_success_rate')} |")
lines+=['','R2/R3 final：selected550k两模式capture均62%，last1M为32%/40%，两模式平均62%→36%；保持退化已有held-out final描述性信号。完整screen在925k反弹到90%/70%，因此历史叙述更新为明显波动及末点退化，不能简化成550k后单调collapse。N2末点capture为10%/24%，存在弱学习信号，但collision为90%/78%。上述所有last full episode_success均0，capture信号不能升级为完整任务成功。','',
        '| 线 | 同一晚期窗口 | baseline capture/collision | stability capture/collision | baseline/stability KL-stop占比 |',
        '|---|---|---|---|---|']
for arm,pair in pairs.items():
    a=pair['conditions']['baseline'];b=pair['conditions']['stability'];points=pair['late_common_steps']
    lines.append(f"| {arm} | {min(points)//1000}k–{max(points)//1000}k（{len(points)}点） | {a['late_capture']:.1%}/{a['late_collision']:.1%} | {b['late_capture']:.1%}/{b['late_collision']:.1%} | {a['KL_stop_fraction']:.1%}/{b['KL_stop_fraction']:.1%} |")
lines+=['','KL-stop占比也裁剪到同一绝对decision范围500k后→共同末点，避免baseline1M与未完续训比较不同区间。晚期均值先覆盖相同checkpoint，再平均argmax/sample；所有screen重复固定10个seed/模式，不把跨checkpoint样本当独立seed。','',
        '当前较低target_kl确实触发更多early-stop，但未建立一致的capture/safety恢复：R2/R3晚期capture下降、collision升高；V3窗口较短、capture无提升且collision略降。临时判断C（mixed/incomplete），不足以支持A。也不能根据无改善断言reward/network故障。等待1M完整配对，不做reward/architecture tuning。','',
        'R2/R3的baseline及stability在共同完成点的mode summary均完全相同，与reward telemetry-only关系一致，不能作为独立replication。source hashes/config/seed/checkpoint与原始已完成评估见status_snapshot.json、archive_manifest.json及evidence/。完整训练与screen曲线分开保存CSV，并保留每条baseline/stability曲线图。','',
        f'NEXT WAKE-UP：{NEXT}，或R2/R3达到1M并完成末点screen。所有预算不延长，不自动启动下一批。','']
doc='\n'.join(lines)
(OUT/'STATUS_20261012_1655_ZH.md').write_text(doc)
docs=DELIVERY/'docs/ops';docs.mkdir(parents=True,exist_ok=True)
(docs/'BATCH02_STATUS_20261012_1655_ZH.md').write_text(doc)
shutil.copytree(OUT,CENTRAL/REL,dirs_exist_ok=True)
(CENTRAL/'docs/ops/BATCH02_STATUS_20261012_1655_ZH.md').write_text(doc)

registry_root=BASE/'batch01-commander-runtime'
with (registry_root/'registry.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    stability_path=registry_root/'batch02_stability_runtime_20261012.json'
    stability=read(stability_path)
    for arm,line in stability['runs'].items():
        info=run_infos[arm+'-stability'];line.update(status=info['progress']['status'],decisions=info['progress']['decisions'],
            continuation_decisions=info['progress']['continuation_decisions'],pid_alive=info['pid_alive'],actual_physical_gpu=info['actual_gpu'],
            live_progress=info['progress'],last_checkpoint=info['last_checkpoint'],checkpoint_list=info['checkpoints'],
            eta_recomputed=info['eta'],eta_seconds=info['eta']['seconds'],observed_at=NOW.isoformat(),next_wake_up=NEXT)
    stability.update(status='RUNNING_INTERIM_PAIRED_SIGNAL_RECORDED',observed_at=NOW.isoformat(),next_wake_up=NEXT,
                     interim_signal=pairs,scientific_conclusion='C_INTERIM_INCOMPLETE_PROVISIONAL')
    write(stability_path,stability)
    write(CENTRAL/'artifacts/2026-10-12_batch02_stability/run_registry.json',stability)
    for filename,arms in [('batch02_a_long_runtime_20261011.json',['N2','R2']),('batch02_native_runtime_20261011.json',['N3','R3','V3'])]:
        path=registry_root/filename;reg=read(path)
        for arm in arms:
            info=run_infos[arm+'-baseline'];reg.setdefault('runs',{}).setdefault(arm,{}).update(
                status=info['progress']['status'],pid=info['pid'],last_pid=info['historical_pid'],pid_alive=info['pid_alive'],
                decisions=info['progress']['decisions'],live_progress=info['progress'],last_checkpoint=info['last_checkpoint'],
                eta_recomputed=info['eta'],observed_at=NOW.isoformat(),next_wake_up=NEXT)
        reg.update(status='COMPLETE_BUDGET_AND_EVALUATION' if all(run_infos[a+'-baseline']['progress']['status']=='COMPLETE_BUDGET_AND_EVALUATION' for a in arms) else 'TRAINING_COMPLETE_SELECTION_FINAL_PENDING',updated_at=NOW.isoformat(),next_wake_up=NEXT)
        write(path,reg)
    write(registry_root/'batch02_status_snapshot_20261012_1655.json',snapshot)
    (registry_root/'BATCH02_STATUS_20261012_1655_ZH.md').write_text(doc)

path=CENTRAL/'artifacts/2026-09-21_ac_master_dag/state.json';dag=read(path)
dag['batch02'].update(status='CAPABILITY_CLOSEOUT_AND_STABILITY_RUNNING_PROVISIONAL',updated_at=NOW.isoformat(),stability=stability,
                      latest_status_document='docs/ops/BATCH02_STATUS_20261012_1655_ZH.md')
dag['batch02']['A']['capability']['observed_runs']={a:run_infos[a+'-baseline']['progress'] for a in ['N2','R2']}
dag['batch02']['native'].update(status='TRAINING_COMPLETE_SELECTION_FINAL_PENDING',observed_runs={a:run_infos[a+'-baseline']['progress'] for a in ['N3','R3','V3']})
for task in dag['tasks']:
    key=task['task_id'].removeprefix('BATCH02-')
    identifier=key.lower().endswith('-stability')
    label=(key.removesuffix('-STABILITY')+'-stability') if identifier else key+'-baseline'
    if task['task_id'].startswith('BATCH02-') and label in run_infos:
        info=run_infos[label];task.update(status=info['progress']['status'],pid=info['pid'],current_step=info['progress']['decisions'],
            last_checkpoint=info['last_checkpoint'],eta_recomputed=info['eta'],next_wake_up=NEXT,updated_at=NOW.isoformat(),result_classification='PROVISIONAL')
        if 'capability_summary' in info:task['capability_result']=info['capability_summary']['learning_assessment']
write(path,dag)
for name in ['COCAP_MAPPO_BATCH02_DAG_20261011_ZH.md','COCAP_MAPPO_BATCH02_STABILITY_DAG_20261012_ZH.md']:
    p=CENTRAL/'docs/ops'/name
    p.write_text(p.read_text()+'\n最新状态/ETA/完整历史信号见 [2026-10-12 快速检查](BATCH02_STATUS_20261012_1655_ZH.md)。旧启动快照保留其观测时间。\n')
print(json.dumps(dict(observed_at=NOW.isoformat(),status='SNAPSHOT_DOCUMENTS_READY',
    eta={k:{'step':i['progress']['decisions'],'seconds':i['eta']['seconds'],'finish':i['eta'].get('estimated_finish')} for k,i in run_infos.items()},
    paired_signal={arm:{'steps':p['late_common_steps'],'baseline':p['conditions']['baseline']['late_capture'],'stability':p['conditions']['stability']['late_capture']} for arm,p in pairs.items()},
    archive_files=len(ARCHIVE),next_wake_up=NEXT),ensure_ascii=False))
