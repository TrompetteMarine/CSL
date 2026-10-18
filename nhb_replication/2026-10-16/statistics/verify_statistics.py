"""Independent raw-array spot reconstruction and exact finite-noise enumeration."""
from pathlib import Path
from itertools import product
import json,math
import numpy as np
import pandas as pd
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
BASE=ROOT/'baseline/nhb_smallworld_20261012/results';DATA=ROOT/'data/results'
table=pd.read_csv(OUT/'Main_Table1.csv');labels={}
for p in BASE.glob('c????.json'):
    c=json.loads(p.read_text())
    if c['group']=='primary' and c['n']==100 and c['initial']=='misleading_random' and c['geometry'] in ['ring','broadcast']:
        labels[c['geometry'],c['graph_id']]=p.stem
raw={}
for geometry in ['ring','broadcast']:
    Y=[]
    for gid in range(4):
        with np.load(DATA/(labels[geometry,gid]+'.npz')) as z:
            p=z['path'];q=z['final_Q']
            Y.append(np.stack([p[-100:,:,0].mean(0),(q[:,:,0]>q[:,:,1]).mean(1),p[:,:,5].sum(0)],axis=1))
    raw[geometry]=np.stack(Y)
max_error=0.
for j,metric in enumerate(['optimal_fraction','private_correct','regret']):
    r=table[(table.contrast_id=='broadcast_minus_ring')&(table.metric==metric)].iloc[0]
    diff=raw['broadcast'][:,:,j]-raw['ring'][:,:,j];batch=diff.mean(1)
    effect=batch.mean();se=batch.std(ddof=1)/2;half=t.ppf(.975,3)*se
    max_error=max(max_error,abs(effect-r.effect),abs(se-r.mcse),abs(effect-half-r.ci95_low),abs(effect+half-r.ci95_high))
assert max_error<1e-12

# All outcomes are enumerated, so this test has no Monte Carlo sampling error.
p=np.array([.1,.3,.7,.9]);d=np.maximum(p,1-p);sizes=np.array([2,2]);N=4
C=np.array([[.85,.15],[.15,.85]])
values=[];weights=[]
for bit in product([0.,1.],repeat=4):
    x=np.array(bit);prob=float(np.prod(np.where(x==1,p,1-p)))
    psi=np.array([(d[:2]*(2*x[:2]-1)).mean(),(d[2:]*(2*x[2:]-1)).mean()])
    values.append(np.r_[x.mean(),psi,C@psi]);weights.append(prob)
values=np.array(values);weights=np.array(weights);mu=weights@values
cov=(values-mu).T@(weights[:,None]*(values-mu))
v=4*(d*d*p*(1-p)).reshape(2,2).sum(1)/sizes**2
public_var=float((p*(1-p)).sum()/N**2)
public_psi=2*(d*p*(1-p)).reshape(2,2).sum(1)/(N*sizes)
enumeration_error=max(float(np.max(abs(cov[1:3,1:3]-np.diag(v)))),float(np.max(abs(cov[3:,3:]-C@np.diag(v)@C.T))),abs(cov[0,0]-public_var),float(np.max(abs(cov[0,1:3]-public_psi))))
assert enumeration_error<1e-14

intervals=pd.read_csv(OUT/'endpoint_intervals.csv')
assert len(intervals)==1328 and (intervals.ci95_low<=intervals['mean']).all() and (intervals.ci95_high>=intervals['mean']).all()
assert set(intervals.batches)=={2,4,12,16} # 12-batch additions are retained separately as exploratory follow-up records.
assert set(intervals.df)=={1,3,11,15}
curves=pd.read_csv(OUT/'trajectory_intervals.csv');assert len(curves)==9000
assert (curves.df==3).all() and (curves.smoothing=='10-trial trailing average within population; min_periods=1').all()
paired=pd.read_csv(OUT/'paired_contrasts.csv')
assert len(paired)==42 and (paired.populations==paired.batches*paired.trajectories_per_batch).all()
out=dict(all_passed=True,raw_array_main_table_max_error=max_error,
    exact_four_agent_enumeration_covariance_max_error=enumeration_error,
    endpoint_rows=len(intervals),paired_rows=len(paired),pointwise_curve_rows=len(curves),
    original_arrays_changed=False,main_table_comparison='Independent reconstruction directly from raw paths and terminal scores, without using audit CSV values.')
(OUT/'Statistical_Verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
