"""Export descriptive fixed-trajectory plots; no training performance claims."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/2026-10-16_z_guided_support'
rows=list(csv.DictReader((ART/'support_transitions.csv').open()))
get=lambda k:np.asarray([float(r[k]) for r in rows])
x=get('ally_progress');y=get('enemy_progress')
wrong=(x>1e-8)&(y< -1e-8)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10.5,4.0),layout='constrained')
ax=axes[0];ax.scatter(x[~wrong],y[~wrong],s=14,alpha=.55,color='#3178b9',label='Other Support transitions')
ax.scatter(x[wrong],y[wrong],s=30,color='#c84c38',label='Toward ally, away from enemy')
lo=min(x.min(),y.min(),0);hi=max(x.max(),y.max());ax.plot([lo,hi],[lo,hi],':',color='gray',lw=1)
ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7)
ax.set(xlabel='Own progress toward fixed ally anchor (m)',ylabel='Own progress toward fixed enemy anchor (m)',title=f'Frozen Z05 policy, {len(rows)} Support transitions')
ax.legend(loc='lower right',fontsize=8)
ax=axes[1]
old=get('old_capture')+get('old_ce');new=get('new_capture')+get('new_ce')
bins=np.linspace(min(old.min(),new.min()),max(old.max(),new.max()),25)
ax.hist(old,bins=bins,histtype='step',linewidth=1.8,label='Z05: 0.5 enemy approach + 0.5 CE',color='#707070')
ax.hist(new,bins=bins,histtype='step',linewidth=1.8,label='Candidate v0: ally/CE blend',color='#3178b9')
ax.set(xlabel='Routed Support task reward',ylabel='Transition count',title='Same states and actions; alpha = 1 in this sample')
ax.legend(fontsize=8)
fig.suptitle('Offline reward diagnostics: descriptive association, no learned efficiency claim',fontsize=11)
for ext in ('png','svg'):fig.savefig(ART/f'offline_reward_diagnostics.{ext}',dpi=160)
plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(10.5,3),layout='constrained')
cases=[([(.3,0,.5),(1.,.4,1.)],.25,1,'Higher Z wins'),([(.3,0,1.),(1.,.4,1.)],.25,0,'Equal friend Z: nearest'),([(.3,0,1.),(1.,.4,1.)],1.,None,'Self ties: full CE')]
for ax,(friends,zself,chosen,title) in zip(axes,cases):
 ax.scatter([0],[0],s=100,c='#333333');ax.text(0,-.16,f'Self Z={zself:g}',ha='center')
 for i,(x,y,z) in enumerate(friends):
  ax.scatter([x],[y],s=95,c='#3178b9');ax.text(x,y+(.23 if y==0 else .13),f'Friend {i+1}: Z={z:g}',ha='center')
 if chosen is not None:
  x,y,_=friends[chosen];ax.annotate('',xy=(x*.9,y*.9),xytext=(.03,0),arrowprops=dict(arrowstyle='->',color='#c84c38',lw=2))
 else:ax.text(.55,.32,'alpha=0',color='#c84c38')
 ax.set(xlim=(-.2,1.35),ylim=(-.3,.85),aspect='equal',title=title);ax.axis('off')
for ext in ('png','svg'):fig.savefig(ART/f'greedy_selection_examples.{ext}',dpi=160)

# Normalize generated SVG path-line whitespace for a clean review diff.
for output in ART.glob("*.svg"):
    output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines())+"\n")
