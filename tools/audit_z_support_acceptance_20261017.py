#!/usr/bin/env python3
"""Read-only pilot and provenance audit; never rewrites historical artifacts."""
from __future__ import annotations
import csv, gzip, hashlib, json, sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
from tools import iqn_token_matched_20260919 as formal
from tools import z_guided_support_20261016 as pilot
from tools.collect_iqn_aw_teacher_dataset_20260903 import sha256_file


def audit(output: Path):
    torch.set_num_threads(1)
    output.mkdir(parents=True,exist_ok=True)
    report={'status':'RUNNING','source_head':'4d7005d2bc80c21d1798b04ba711d258be2b8956','pairs':0,'checkpoints':[], 'windows':{},'alpha_100k':{}}
    cfgs={a:pilot.config(a) for a in ('control','treatment')}
    differences=formal._diff_rows(cfgs['control'],cfgs['treatment'])
    assert all(d['path'] in ('run_name','voradj.z_guided_support_enabled') for d in differences),differences
    report['pilot_config_differences']=differences
    registry=list(csv.DictReader((pilot.ART/'checkpoint_registry.csv').open()))
    for row in registry:
        path=Path(row['path']);actual=sha256_file(path)
        assert actual==row['sha256'],path
        report['checkpoints'].append({'path':str(path),'sha256':actual,'bytes':path.stat().st_size})
    for step,updates in ((25000,0),(50000,4225),(100000,16725)):
        paths={a:pilot.RUNTIME/a/'evaluations'/f'step_{step:06d}' for a in cfgs}
        # Accept archival directory widths, without trusting report paths blindly.
        for a in paths:
            if not paths[a].is_dir():
                matches=[p for p in (pilot.RUNTIME/a/'evaluations').glob('step_*') if int(p.name.split('_')[1])==step]
                assert len(matches)==1,(a,step);paths[a]=matches[0]
        report['windows'][str(step)]={}
        for a in cfgs:
            cp=pilot.RUNTIME/a/'checkpoints'/f'final_step_{step}.pt'
            payload=torch.load(cp,map_location='cpu',weights_only=False)
            assert payload['extra']['global_step']==step
            del payload
            metrics=[json.loads(l) for l in (pilot.RUNTIME/a/'metrics.jsonl').read_text().splitlines()]
            row=next(r for r in reversed(metrics) if r['global_step']==step)
            assert row['update_steps']==updates,(a,step,row['update_steps'])
            rp=json.loads((paths[a]/'report.json').read_text())
            assert rp['sha256']==sha256_file(cp)
            assert rp['episodes_per_scene_mode']==50 and len(rp['records'])==300
            report['windows'][str(step)][a]={'updates':updates,'checkpoint':str(cp),'sha256':rp['sha256'],'summary':rp['summary']}
        for mode in pilot.MODES:
            for scene in pilot.SCENES:
                rows={a:[json.loads((paths[a]/f'{mode}_{scene}_{i:03d}.json').read_text()) for i in range(50)] for a in cfgs}
                for c,t in zip(rows['control'],rows['treatment'],strict=True):
                    assert c['seed']==t['seed'] and c['initial_state_fingerprint']==t['initial_state_fingerprint']
                    for a,r in (('control',c),('treatment',t)):
                        assert r['checkpoint_sha256']==report['windows'][str(step)][a]['sha256']
                        assert r['scene']==scene and r['mode']==mode
                    report['pairs']+=1
                for a,rr in rows.items():
                    summary=report['windows'][str(step)][a]['summary'][mode][scene]
                    for key in ('normal_capture','collision','post_capture_ce','captured'):
                        assert summary[key]['count']==sum(r[key] for r in rr)
                    expected=np.mean([r['capture_seconds'] for r in rr if r['capture_seconds'] is not None]) if scene!='coverage' else None
                    if expected is not None:assert np.isclose(summary['capture_seconds']['mean'],expected)
                if step==100000:
                    hist=Counter()
                    for i in range(50):
                        with gzip.open(paths['treatment']/f'{mode}_{scene}_{i:03d}_support.json.gz','rt') as f:
                            for r in json.load(f):hist[str(r['alpha'])]+=1
                    report['alpha_100k'][f'{mode}/{scene}']=dict(hist)
    report['full_resume']={}
    for a in cfgs:
        path=pilot.RUNTIME/a/'checkpoints/resume_latest.pt'
        p=torch.load(path,map_location='cpu',weights_only=False)
        assert p['runtime']['global_step']==100000 and p['runtime']['update_steps']==16725
        cp=torch.load(pilot.RUNTIME/a/'checkpoints/final_step_100000.pt',map_location='cpu',weights_only=False)
        for k,v in cp['state_dict'].items():assert torch.equal(v,p['model'][k]),(a,k)
        report['full_resume'][a]={'path':str(path),'step':100000,'updates':16725,'sha256':sha256_file(path),'model_equal_final':True,'contract':p['contract'],'contract_hash':p['contract_hash']}
        del p,cp
    report['status']='PASS_ARCHIVE_900_PAIRS';report['completed_at']=pilot.now()
    pilot.write(output/'pilot_independent_audit.json',report)
    print(json.dumps({'status':report['status'],'pairs':report['pairs'],'checkpoints':len(report['checkpoints']),'alpha':report['alpha_100k']},ensure_ascii=False))
    return report

if __name__=='__main__':audit(Path(sys.argv[1]))
