"""Is policy churn a CAUSE of lower loss, or a marker of the harness condition?

Across the eight harness arms, mean churn and mean loss correlate at r = -0.82. Read alone that
invites "make the agent revise more and it will improve", which the same data does not support.

This script decomposes the 8 x 20 (arm x seed) panel four ways. The two-way fixed-effects figure —
both arm and seed means removed — is the one that answers the question, and it comes out at -0.11:
indistinguishable from zero at this sample size. Within a harness condition, a run that happened to
revise more did not thereby do better.

The arm-demeaned-only figure is +0.203, the opposite sign, and it is the artifact one expects from
seed difficulty: a harsher epidemic raises loss AND makes the agent flail. It is reported here
precisely because it is the number that would have been quoted by accident.

Conclusion: churn is a DIAGNOSTIC of what the harness did to the institution, not a target to
maximize. Lock-in is real and costly; revision is not itself the good.

Run: ``uv run python scripts/churn_loss_panel.py``
"""

import sys,itertools,statistics as st
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from govsim.core.result_store import ResultStore
from policy_churn import churn_by_seed, arm_name

s=ResultStore('logs/runs_v3')
data={}   # (arm, seed) -> (churn, loss)
for cell in itertools.product((False,True),repeat=3):
    name=arm_name('epidemic',cell)
    ch=churn_by_seed(s,name,carry_forward=True)
    loss={}
    for r in s.query(experiment=name):
        c=(r.get('components') or {}).get('regent:0',{})
        if 'loss' in c: loss[int(r['seed'])]=float(c['loss'])
    for k in set(ch)&set(loss): data[(name,k)]=(ch[k],loss[k])

arms=sorted({a for a,_ in data}); seeds=sorted({sd for _,sd in data})
C=np.full((len(arms),len(seeds)),np.nan); L=np.full_like(C,np.nan)
for (a,sd),(c,l) in data.items():
    C[arms.index(a),seeds.index(sd)]=c; L[arms.index(a),seeds.index(sd)]=l
ok=~np.isnan(C).any(axis=0)&~np.isnan(L).any(axis=0)
C,L=C[:,ok],L[:,ok]
print(f'balanced panel: {C.shape[0]} arms x {C.shape[1]} seeds')

def demean(M, arm=True, seed=True):
    X=M.copy()
    if arm:  X=X-X.mean(axis=1,keepdims=True)
    if seed: X=X-X.mean(axis=0,keepdims=True)
    return X

for lbl,(a,sd) in (('raw',(False,False)),('arm-demeaned',(True,False)),
                   ('seed-demeaned',(False,True)),('BOTH (two-way FE)',(True,True))):
    x=demean(C,a,sd).ravel(); y=demean(L,a,sd).ravel()
    r=np.corrcoef(x,y)[0,1]
    print(f'  {lbl:<20} corr(churn, loss) = {r:+.3f}   n={x.size}')
print()
xa=C.mean(axis=1); ya=L.mean(axis=1)
print(f'BETWEEN arms (arm means, n={len(xa)}): corr = {np.corrcoef(xa,ya)[0,1]:+.3f}')
