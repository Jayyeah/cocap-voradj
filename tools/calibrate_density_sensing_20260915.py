#!/usr/bin/env python3
"""Offline multi-seed Final free-mask / CVT sweep; never trains a policy."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
from scipy.spatial.distance import cdist
from tools.forward_final_single_task_20260915 import task_config,p
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.envs.density_sensing import enable_v2,runtime_metadata
OUT=ROOT/'artifacts/2026-09-15_normsense_v2';OUT.mkdir(exist_ok=True,parents=True)

def main():
    layouts=[]
    for n in (4,8,12):
        for seed in range(2026091501,2026091513):
            cfg=task_config('coverage');cfg['env']['num_pursuers']=n;cfg['env']['num_obstacles']=n//4
            p.set_global_config(cfg);env=VorAdjEnv(cfg,seed=seed);env.reset()
            grid=env._coverage_voronoi_map();pts=grid['points'][grid['free_mask']]
            area=14400*float(grid['free_mask'].mean());rng=np.random.default_rng(seed)
            sites=pts[rng.choice(len(pts),n,replace=False)].copy()
            for iteration in range(100):
                labels=cdist(pts,sites).argmin(1)
                nxt=np.array([pts[labels==i].mean(0) if np.any(labels==i) else sites[i] for i in range(n)])
                if np.max(np.linalg.norm(nxt-sites,axis=1))<.001: break
                sites=nxt
            # Converged and mildly perturbed layouts, not only a perfect lattice.
            for variant in ('cvt','perturbed'):
                ss=sites.copy()
                if variant=='perturbed':ss+=rng.uniform(-.02,.02,ss.shape)*np.sqrt(area/n)
                # Runtime surface semantics on zero-radius free-space point targets.
                distances=cdist(pts,ss)-env.pursuers[0].r
                np.testing.assert_allclose(distances[0,0],env._vct_ls_surface_clearance(ss[0],env.pursuers[0].r,pts[0],0))
                threshold=float(np.quantile(distances.min(1),.999,method='higher')/np.sqrt(area/n))
                layouts.append(dict(n=n,num_obstacles=n//4,seed=seed,variant=variant,area=area,sites=ss.tolist(),obstacles=[[o.x,o.y,o.r] for o in env.obstacles],threshold=threshold,distances=distances))
        print('geometry',n,'complete',flush=True)
    sweep=[]
    for k in np.arange(.65,1.101,.0025):
        rows=[]
        for l in layouts:
            radius=k*np.sqrt(l['area']/l['n']);d=l['distances'];counts=(d<=radius).sum(1)
            adjacency=cdist(l['sites'],l['sites'])<=2*(radius+np.sqrt(1.25));np.fill_diagonal(adjacency,False)
            seen={0}
            while True:
                nxt=seen|{j for i in seen for j in np.flatnonzero(adjacency[i])}
                if nxt==seen:break
                seen=nxt
            rows.append(dict(blind=float(np.mean(counts==0)),isolates=int(np.sum(adjacency.sum(1)==0)),connected=len(seen)==l['n']))
        sweep.append(dict(k=float(round(k,6)),worst_blind=max(r['blind'] for r in rows),isolates=sum(r['isolates'] for r in rows),all_connected=all(r['connected'] for r in rows)))
    passes=[r for r in sweep if r['worst_blind']<=.001 and not r['isolates'] and r['all_connected']]
    minimum=passes[0]['k'] if passes else None
    if minimum is None:raise RuntimeError('No geometry pass')
    selected=round(minimum*1.05,6)
    for l in layouts:
        d=l.pop('distances');radius=selected*np.sqrt(l['area']/l['n']);counts=(d<=radius).sum(1)
        l.update(radius=radius,normalized_radius=selected,blind_fraction=float(np.mean(counts==0)),detector_count_distribution={str(i):float(np.mean(counts==i)) for i in range(l['n']+1)})
    report=dict(policy='forward-final-density-normalized-sensing-v2',minimum_k=minimum,calibrated_k=selected,safety_margin=.05,margin_source='predeclared 5% beyond minimum all-layout pass for discretization/layout perturbations; not a universal coverage proof',blind_tolerance=.001,layouts=layouts,sweep=sweep)
    (OUT/'geometry.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('layouts','sweep')}),flush=True)
if __name__=='__main__':main()
