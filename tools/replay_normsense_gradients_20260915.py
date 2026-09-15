import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
from tools.calibrate_reward_balance_20260915 import st,gradients,OUT,OLD
r=json.loads((OUT/'reward_balance.json').read_text());torch.set_num_threads(1)
t=st.p.make_trainer(st.contract()['scratch'],r['seed'],'cpu')
x=torch.load(OLD/'artifacts/2026-09-08_forward_final/c2_distillation/actor_epoch_030.pt',weights_only=True,map_location='cpu');t.actor.load_state_dict(x['actor_state_dict']);t.actor.eval()
data=torch.load(OUT/'frozen_reward_rollouts.pt',weights_only=False,map_location='cpu');vectors=None
for i,(batch,roles,comps) in enumerate(zip(data['batches'],data['roles'],data['capture_components'])):
 report,v=gradients(t,batch,roles,comps)
 for p in ('capture','support','coverage'):assert abs(report['phases'][p]['gradient_norm']-r['windows'][i]['phases'][p]['gradient_norm'])<1e-5
 if vectors is None:vectors=[g.clone() for g in v]
 else:
  for a,b in zip(vectors,v):a.add_(b)
print({p:float(v.norm()/len(data['batches'])) for p,v in zip(('capture','support','coverage'),vectors)},flush=True)
r['aggregate_mean_gradient_norms']={p:float(v.norm()/len(data['batches'])) for p,v in zip(('capture','support','coverage'),vectors)}
r['aggregate_capture_to_coverage_gradient_norm_ratio']=r['aggregate_mean_gradient_norms']['capture']/r['aggregate_mean_gradient_norms']['coverage']
r['frozen_gradient_replay_norms_match']=True
(OUT/'reward_balance.json').write_text(json.dumps(r,indent=2)+'\n')
