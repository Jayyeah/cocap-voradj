#!/usr/bin/env python3
"""Summarize immutable evidence after the owned supervisor exits; never train.

This reporting helper is outside the frozen scientific source tree. It preserves
the supervisor's predeclared selection/classification and only writes reports.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback

from audit_terl_mappo_results_20261008 import audit, digest, read_metrics

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'artifacts/2026-10-08_terl_mappo'
DOC=ROOT/'docs/TERL_MAPPO_FOLLOWUP_DECISION_20261008_ZH.md'
LEARNING=ROOT/'docs/TERL_MAPPO_STAGE1_LEARNING_20261008_ZH.md'
BRANCH='experiment/terl-backbone-mappo-20261008'
HEALTH_KEYS=('reward_mean','actor_loss','value_loss','entropy','explained_variance',
             'approx_kl','post_update_kl','actor_grad_norm','value_grad_norm',
             'clip_fraction','post_update_clip_fraction','post_update_ratio_min',
             'post_update_ratio_max','actor_update_l2','value_norm_mean','value_norm_std')

def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)

def stats(values):
    return {'n':len(values),'min':min(values),'median':statistics.median(values),
            'mean':statistics.mean(values),'max':max(values)} if values else None

def identity(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text()
        return stat[stat.rfind(')')+2:].split()[19]
    except FileNotFoundError: return None

def wait_owned_supervisor(out,pid):
    start=identity(pid)
    if start is not None:
        command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        if 'terl_mappo.supervise' not in command or out.name not in command:
            raise ValueError('PID is not the requested experiment supervisor')
    print(json.dumps({'waiting_for_supervisor_pid':pid,'start_ticks':start}),flush=True)
    while start is not None and identity(pid)==start:
        if (out/'failure.json').exists() or (out/'supervisor_failure.json').exists():
            raise RuntimeError('owned experiment reports failure; no restart or reclassification')
        time.sleep(30)

def summarize(out,final=False):
    proof=audit(out); progress=proof['progress']
    supervisor=json.loads((out/'supervisor.json').read_text())
    selection=json.loads((out/'selection.json').read_text()) if (out/'selection.json').exists() else None
    if final:
        if any((out/name).exists() for name in ('failure.json','supervisor_failure.json')):
            raise ValueError('failure artifact prevents a successful closeout')
        if not proof['scope_complete'] or supervisor['status']!='COMPLETE' or not selection:
            raise ValueError('complete screening/selection/final evidence required')
        if progress['status']!='COMPLETE_BUDGET' or progress['steps']!=progress['budget'] or not progress['finite']:
            raise ValueError('bounded training budget not completed')
        if selection['decision']!=supervisor['decision']:
            raise ValueError('authoritative decision mismatch')
        if selection['decision'] not in ('TERL_MAPPO_STAGE1_LEARNABLE','TERL_MAPPO_STAGE1_PARTIAL',
                                        'TERL_MAPPO_STAGE1_NO_CONVINCING_SIGNAL'):
            raise ValueError('unexpected healthy-gate classification')
        if selection['selected']['checkpoint_sha256']!=selection['final']['checkpoint_sha256']:
            raise ValueError('final evaluated a different checkpoint')
        if digest(out/'checkpoints/best.pt')!=selection['selected']['checkpoint_sha256']:
            raise ValueError('selected best alias hash mismatch')
        finals=[x for x in proof['checkpoint_evidence'] if x['seed_domain']=='final']
        if len(finals)!=1 or any(m['evaluation']['episodes']!=50 for m in finals[0]['modes'].values()):
            raise ValueError('isolated final must contain 50 episodes per mode')
    metrics,trailing=read_metrics(out/'metrics.jsonl')
    if final and trailing: raise ValueError('incomplete final telemetry line')
    if final and max(metrics)!=progress['steps']: raise ValueError('final telemetry/decision count mismatch')
    rows=list(metrics.values())
    windows=[]
    for low,high in ((0,25000),(25000,50000),(50000,75000),(75000,100000)):
        chunk=[r for r in rows if low<r['steps']<=high]
        if chunk: windows.append({'steps_low_exclusive':low,'steps_high_inclusive':high,
            'observed_last_step':chunk[-1]['steps'],
            'statistics':{k:stats([r[k] for r in chunk]) for k in HEALTH_KEYS}})
    episodes=[json.loads(line) for line in (out/'episodes.jsonl').read_text().splitlines()]
    evaluated=[]
    for path in sorted((out/'evaluations').glob('*.json')):
        if path.name.endswith('.partial.json'): continue
        result=json.loads(path.read_text())
        for mode,summary in result['modes'].items():
            ep=[r for r in result['episodes'] if r['mode']==mode]
            steps=sum(r['steps'] for r in ep)
            components={k:sum(r['reward_components'][k] for r in ep)/steps
                        for k in summary['reward_components']}
            # Stage1 cooperation is nonnegative; native boundary penalty is -5.
            # Global includes cooperation, so this is a lower bound, not an exact
            # reconstruction of boundary events or a collision statistic.
            lower=max(0,-components['global']/5)
            if lower>1+1e-6: raise ValueError('Stage1 boundary lower bound exceeds one')
            evaluated.append({'steps':result['steps'],'domain':result['seed_domain'],'mode':mode,
                'episodes':len(ep),'normal_capture':summary['normal_capture_count'],
                'collision':summary['collision_count'],'ring2':summary['ring2_count'],
                'ring3':summary['ring3_count'],'strict':summary['strict_geometry_count'],
                'reward_per_agent_decision':components,
                'boundary_agent_time_fraction_lower_bound':min(1,lower),
                'entropy_fraction_of_aw9_max':summary['action_entropy']/math.log(9)})
    report={'updated_at':datetime.now().astimezone().isoformat(),'scope_complete':proof['scope_complete'],
        'decision':selection['decision'] if final else 'PENDING',
        'training_windows':windows,'evaluated_diagnostics':evaluated,
        'training_completed_episodes':len(episodes),'training_capture_episodes':sum(e['capture'] for e in episodes),
        'training_collision_episodes':sum(e['collision'] for e in episodes),
        'training_episode_ring_fields':'terminal step only; not cumulative visitation; do not use for learning classification',
        'gradient_norms':'pre-clipping minibatch norms, averaged per rollout; threshold=0.5',
        'actor_loss':'includes -0.01*entropy; a near-constant negative loss does not mean zero policy updates',
        'boundary_bound':'max(0,-sum(global)/(5*sum(episode decisions))); includes native cooperation in global; boundary is not collision',
        'selection':selection,'scientific_sources_match_launch':proof['scientific_sources_match_launch']}
    atomic(ARCHIVE/'checkpoint_evidence_audit.json',proof)
    atomic(ARCHIVE/'training_diagnostics.json',report)
    plot(rows)
    write_decision(out,proof,report,final)
    return report

def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=[r['steps'] for r in rows]
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    panels=(('entropy',),('explained_variance',),('actor_loss','value_loss'),
            ('approx_kl','post_update_kl'),('actor_grad_norm','value_grad_norm'),
            ('post_update_ratio_min','post_update_ratio_max'))
    for ax,keys in zip(axes.flat,panels):
        for key in keys: ax.plot(x,[r[key] for r in rows],label=key,linewidth=1)
        ax.set_xlabel('environment decisions'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    axes[0,0].axhline(math.log(9),color='gray',linestyle='--')
    axes[1,1].axhline(.5,color='gray',linestyle='--')
    fig.tight_layout();fig.savefig(ARCHIVE/'training_diagnostics.png',dpi=150);plt.close(fig)

def write_decision(out,proof,report,final):
    p=proof['progress']; selected=report['selection']['selected'] if final else None
    lines=['# TERL-MAPPO Stage1 后续决定与诊断','',f'更新时间：{report["updated_at"]}。',
        f'当前分类：`{report["decision"]}`；实际预算进度：{p["steps"]:,}/{p["budget"]:,}。',
        '', '分类和best保持启动前预声明的supervisor规则；诊断不参与重选，不改训练合同或超参。',
        '', '## Checkpoint诊断', '',
        '| step | domain | mode | n | normal | collision | ring2/ring3/strict | entropy / log9 | boundary驻留占比下界 |',
        '|---:|---|---|---:|---:|---:|---|---:|---:|']
    for d in report['evaluated_diagnostics']:
        lines.append(f'| {d["steps"]} | {d["domain"]} | {d["mode"]} | {d["episodes"]} | {d["normal_capture"]} | {d["collision"]} | {d["ring2"]}/{d["ring3"]}/{d["strict"]} | {d["entropy_fraction_of_aw9_max"]:.4f} | {d["boundary_agent_time_fraction_lower_bound"]:.2%} |')
    lines += ['', 'boundary下界由原生Stage1 global reward推导：合作项非负、越界每agent每decision罚-5。'
        '它统计agent时间占比下界，不等同于越界episode率或碰撞；未增加终止或安全shaping。',
        '', '## Reward / PPO / critic / exploration / representation / safety', '',
        'Reward：逐episode分量与原reward总和审计一致。distance是距离驻留收益，不是进度差；global负值可由反复越界主导。'
        '完整每decision分量在training_diagnostics.json。不能用reward上升替代capture/ring证据。',
        'PPO：记录KL、更新后ratio范围和clip统计；actor_loss含entropy项，更新是否发生需结合actor_update_l2与梯度。'
        '梯度norm是裁剪前的minibatch值取平均，大于0.5本身不意味着裁剪失效。',
        'Critic：按0–25k/25–50k/50–75k/75–100k窗口报告EV、value_loss和ValueNorm尺度的min/median/mean/max。'
        'EV只解释采样rollout的return，不能证明长期围捕几何已学会。',
        'Exploration：独立sample/argmax分开，entropy/log9给出相对均匀AW9的随机程度。高entropy仍可能缺少联合几何探索；'
        '若argmax和sample行为差异大，应先诊断动作分布与轨迹，不能直接称探索已解决。',
        'Representation：保留公开TERL未mask的max pooling与完整evader tokens；必要finite mask修复已测试。'
        '本轮没有pooling或backbone消融，不能将表现因果归于representation。',
        'Safety：collision与boundary分开。低collision可能伴随高越界驻留，不能据此宣称安全或有效追逃。',
        '',f'训练完成episode {report["training_completed_episodes"]}；capture {report["training_capture_episodes"]}；collision {report["training_collision_episodes"]}。'
        '这些是探索训练记录，不能并入独立评估成功率；episode日志ring标志仅为终止时刻。',
        '', '完整窗口统计和图见本分支artifact：`training_diagnostics.json`、`training_diagnostics.png`；'
        '每checkpoint相同步数PPO指标与完整capture-time/censor/type/reward证据见 `checkpoint_evidence_audit.json`。',
        '', '## 最佳checkpoint与下一步', '']
    if not final:
        lines += ['后续screen、selection-heldout和final尚未完整；best/hash及科学结论保持PENDING。当前只继续已启动100k gate。']
    else:
        lines += [f'best：`{selected["checkpoint"]}`；SHA256：`{selected["checkpoint_sha256"]}`。',
            '最终隔离测试argmax/sample各50局，未参与selection；不得将最优screen成绩作为最终测试成功率。']
        if report['decision']=='TERL_MAPPO_STAGE1_LEARNABLE':
            lines += ['建议下一阶段先做同一TERL环境的backbone对照与额外训练seed复核；完整课程需单独预算与Stage1稳定性证据。']
        elif report['decision']=='TERL_MAPPO_STAGE1_PARTIAL':
            lines += ['建议保留同合同、同超参，下一轮先扩至500k bounded Stage1 gate，再据ring与capture稳定性决定是否向2M延长。'
                '扩步需建立清楚的resume/config lineage；当前不自动启动，不加入新shaping，也不把partial写成可靠捕获。']
        else:
            lines += ['本次100k单seed gate没有足够证据判定Stage1可学习；这不证明MAPPO无法围捕。建议在相同合同与超参下先扩至500k，'
                '同时保留boundary、几何与探索诊断，再决定2M内后续预算。算法或奖励变更应作为单独版本，不与当前曲线拼接。']
    lines += ['', '未解决：单训练seed与有限预算；公开APF/current/padding的设计意图；没有backbone/pooling消融；'
        '高entropy和critic EV无法单独归因围捕瓶颈。上述公开行为已保留，不静默修正任务难度。',
        '', '原始合同、迁移及43项测试见Migration/Implementation报告；中央DAG只读，handoff由中央owner接收。']
    DOC.write_text('\n'.join(lines)+'\n')
    if final:
        text=LEARNING.read_text()
        marker='\n补充诊断与后续决定：见 `TERL_MAPPO_FOLLOWUP_DECISION_20261008_ZH.md`；完整训练窗口统计和checkpoint证据审计已归档。\n'
        if marker not in text: LEARNING.write_text(text+marker)

def sync():
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!=BRANCH:
        raise ValueError('unexpected worktree branch; do not push')
    files=[DOC.relative_to(ROOT),LEARNING.relative_to(ROOT),
           *(p.relative_to(ROOT) for p in ARCHIVE.glob('training_diagnostics.*')),
           (ARCHIVE/'checkpoint_evidence_audit.json').relative_to(ROOT)]
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
    if set(staged)-{str(p) for p in files}: raise ValueError('unrelated staged changes; preserve them without committing')
    subprocess.run(['git','add',*[str(p) for p in files]],cwd=ROOT,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
        subprocess.run(['git','commit','-m','report(terl-mappo): finalize learning diagnostics and follow-up'],cwd=ROOT,check=True)
    subprocess.run(['git','-c','http.proxy=http://127.0.0.1:17892','push','origin',f'HEAD:refs/heads/{BRANCH}'],cwd=ROOT,check=True)
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:17892','ls-remote','origin',f'refs/heads/{BRANCH}'],cwd=ROOT,text=True).split()[0]
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if head!=remote: raise ValueError('remote HEAD mismatch')
    return head

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--snapshot',action='store_true');parser.add_argument('--supervisor-pid',type=int)
    parser.add_argument('--sync',action='store_true');args=parser.parse_args()
    out=Path(args.run).resolve();ARCHIVE.mkdir(parents=True,exist_ok=True)
    if not args.snapshot and args.supervisor_pid is None: parser.error('final closeout requires --supervisor-pid')
    with (out/'evidence_finalizer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if not args.snapshot:
                atomic(out/'evidence_finalizer_status.json',{'status':'WAITING_FOR_SUPERVISOR','pid':os.getpid(),
                    'supervisor_pid':args.supervisor_pid,'started_at':datetime.now().astimezone().isoformat()})
                wait_owned_supervisor(out,args.supervisor_pid)
            result=summarize(out,final=not args.snapshot)
            head=sync() if args.sync else None
            status={'status':'COMPLETE' if not args.snapshot else 'SNAPSHOT','pid':os.getpid(),
                    'decision':result['decision'],'scope_complete':result['scope_complete'],'synced_head':head,
                    'updated_at':datetime.now().astimezone().isoformat()}
            if not args.snapshot: atomic(out/'evidence_finalizer_status.json',status)
            print(json.dumps(status),flush=True)
        except BaseException as exc:
            if not args.snapshot: atomic(out/'evidence_finalizer_status.json',{'status':'FAILED','pid':os.getpid(),
                'error':repr(exc),'traceback':traceback.format_exc()})
            raise

if __name__=='__main__': main()
