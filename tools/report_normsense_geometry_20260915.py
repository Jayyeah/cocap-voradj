#!/usr/bin/env python3
"""Render the actual calibrated layouts and their runtime strict-CE metrics."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/2026-09-15_normsense_v2'
r=json.loads((OUT/'geometry.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
for ax,n in zip(axes,(4,8,12)):
    rows=[x for x in r['layouts'] if x['n']==n]
    row=max(rows,key=lambda x:x['threshold'])
    for xy in row['sites']:
        ax.add_patch(Circle(xy,row['radius']+np.sqrt(1.25),alpha=.1,color='tab:blue'))
        ax.plot(*xy,'o',color='tab:blue',ms=4)
    for x,y,rad in row['obstacles']:ax.add_patch(Circle((x,y),rad,color='black'))
    ax.set(xlim=(0,120),ylim=(0,120),aspect='equal',xlabel='x (m)',ylabel='y (m)',title=f"N={n}, obstacles={n//4}\nsurface R={row['radius']:.3f} m; blind={row['fine_grid_blind_fraction']:.4%}")
fig.suptitle(f"Final V2: k={r['calibrated_k']} | hardest representative CVT layout per N | 5% margin")
fig.savefig(OUT/'geometry.png',dpi=160);plt.close(fig)
print('geometry.png written')
