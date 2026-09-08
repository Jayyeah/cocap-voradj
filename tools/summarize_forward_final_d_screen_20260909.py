#!/usr/bin/env python3
"""Paired D smoke health Gate; never starts training or selects a winner."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from tools.evaluate_forward_final_actor_20260908 import paired_delta
from tools.run_forward_final_bridge_20260908 import atomic_json
from tools.train_forward_final_ppo_20260909 import BC_SHA
GATE={'min_per_scene':20,'min_safe_success':.9,'max_reliability_drop':.1,'max_collision_increase':.1,'max_mission_p90_ratio':1.25,'interpretation':'short update-health screen; point estimates only; 256-step smoke warm-up cannot establish the benefit of planned4096-step warm-up'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    reports={m:json.loads((args.root/m/'report.json').read_text()) for m in ('bc_parent','direct512','warmup512')}
    base=reports['bc_parent'];assert base['actor_sha256']==BC_SHA and base['training_steps']==0
    output={'gate':GATE,'branches':{},'summary':{m:r['summary'] for m,r in reports.items()}}
    for name in ('direct512','warmup512'):
        r=reports[name]
        assert r['bc_parent_sha256']==BC_SHA and r['evaluator_sha256']==base['evaluator_sha256'] and r['training_steps']==512
        paired=paired_delta(base['records'],r['records']);failures=[]
        for scene in ('mixed','coverage'):
            a=base['summary'][scene];b=r['summary'][scene]
            assert a['episodes']>=20 and b['episodes']>=20
            for k in ('safe_complete_rate','ce_success_rate')+(('captured_rate',) if scene=='mixed' else ()):
                if b[k]<.9-1e-9 or a[k]-b[k]>.1+1e-9:failures.append(scene+'/'+k)
            if b['collision_rate']-a['collision_rate']>.1+1e-9:failures.append(scene+'/collision')
            if b['full_mission_seconds_completed']['p90'] is None or b['full_mission_seconds_completed']['p90']>1.25*a['full_mission_seconds_completed']['p90']:failures.append(scene+'/mission_p90')
        output['branches'][name]={'decision':'HOLD_PPO_SCALE' if failures else 'SMOKE_HEALTH_PASS','failed_checks':failures,'paired':paired}
    output['decision']='HOLD_PPO_SCALE' if any(r['failed_checks'] for r in output['branches'].values()) else 'ELIGIBLE_FOR_25K_NOT_AUTOSTARTED'
    atomic_json(args.root/'comparison.json',output);print(json.dumps(output,indent=2))

if __name__=='__main__':main()
