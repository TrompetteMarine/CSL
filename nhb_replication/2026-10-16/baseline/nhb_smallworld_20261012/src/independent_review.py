"""Recompute all recorded outcomes and audit complete model configurations."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from study import ROOT,RESULT
df=pd.read_csv(RESULT/'replications.csv');errors=dict(choice=0.,private=0.,gap=0.,regret=0.,tail=0.);dtypes={};num=0
for label,rows in df.groupby('label'):
 c=json.loads((RESULT/(label+'.json')).read_text());z=np.load(RESULT/(label+'.npz'));p=z['path'];q=z['final_Q'];tail=z['tail'];dtypes[str(q.dtype)]=dtypes.get(str(q.dtype),0)+1
 assert p.shape==(c['T'],c['reps'],6) and q.shape==(c['reps'],c['N'],2)
 assert np.isfinite(p).all() and np.isfinite(q).all();assert q.min()>=0 and q.max()<=1;assert c['audit']['clips']==0
 assert c['audit']['gain_max']<=.550001
 for matrix in [c['B'],c.get('choice_B',c['B'])]:
  for row in matrix:assert abs(sum(v for _,v in row)-1)<1e-12 and all(v>=0 for _,v in row)
 for _,r in rows.iterrows():
  rep=int(r.rep);choice=p[-100:,rep,0].mean();private=(q[rep,:,0]>q[rep,:,1]).mean();gap=(q[rep,:,0]-q[rep,:,1]).mean();regret=p[:,rep,5].sum()
  errors['choice']=max(errors['choice'],abs(choice-r.optimal_fraction));errors['private']=max(errors['private'],abs(private-r.private_correct));errors['gap']=max(errors['gap'],abs(gap-r.private_gap));errors['regret']=max(errors['regret'],abs(regret-r.regret));errors['tail']=max(errors['tail'],abs((tail[rep]<.5).mean()-r.wrong_communities));assert int(choice<.5)==r.wrong
 num+=1
errors={k:float(v) for k,v in errors.items()}
print(json.dumps(errors),flush=True)
assert errors['choice']<1e-12 and errors['private']<1e-12 and errors['gap']<1e-7 and errors['regret']<1e-8 and errors['tail']<1e-12
equiv=[]
for geo in ['ring','ws0.1','regular','broadcast']:
 for gid in range(4):
  configs=[json.loads(p.read_text()) for p in RESULT.glob('c[0-9][0-9][0-9][0-9].json')]
  one=next(c for c in configs if c['group']=='controls' and c['geometry']==geo and c['initial']=='misleading_random' and c['mode']==2 and c['graph_id']==gid)
  half=next(c for c in configs if c['group']=='controls' and c['geometry']==geo and c['initial']=='misleading_random' and c['mode']==1 and c['lam']==1.2 and c['graph_id']==gid)
  a=np.load(RESULT/(one['label']+'.npz'));b=np.load(RESULT/(half['label']+'.npz'))
  e=float(max(np.max(abs(a['path']-b['path'])),np.max(abs(a['final_Q']-b['final_Q']))));assert e<1e-12;equiv.append(dict(geometry=geo,graph_id=gid,error=e))
w=pd.DataFrame(json.loads((RESULT/'withdrawal.json').read_text()))
for gid in range(4):
 for lam,index in [(.6,gid+4),(0.,gid+8)]:
  z=np.load(RESULT/(f'w{index:04d}.npz'))
  for rep in range(4):
   r=w[(w.graph_id==gid)&(w.rep==rep)&(w.lam==lam)].iloc[0];assert abs(z['path'][-100:,rep,0].mean()-r.optimal_fraction)<1e-12
out=dict(configurations_recomputed=num,replications=len(df),max_summary_errors=errors,terminal_storage_dtypes=dtypes,constant_equivalence=equiv,withdrawal_recomputed=True,complete_model_state_bounds=True,source_sha256=hashlib.sha256((ROOT/'src/sim.cpp').read_bytes()).hexdigest())
(RESULT/'Independent_Review.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='constant_equivalence'}),flush=True)
