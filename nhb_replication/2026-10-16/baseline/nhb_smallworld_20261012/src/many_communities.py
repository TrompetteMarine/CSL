"""Literal 10,000-community sparse-network experiment, separate from N=10,000 agents."""
import json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from study import config,run,ROOT
cs=[]
for n in [10,20,50]:
 for geo in ['ring','ws0.01','ws0.1','broadcast']:
  for initial in ['misleading_random','misleading_contiguous']:
   for gid in range(2):
    c=config(10000*n,n,geo,initial,T=500,graph_id=gid,group='M10000');c['reps']=2;cs.append(c)
for j,c in enumerate(cs):c['label']=f'm{j:04d}'
(ROOT/'Many_Communities_Freeze.json').write_text(json.dumps(dict(interpretation='Exactly M=10,000 communities; total agents N=M*n. This is a separate exploratory scale extension, with 2 graph/initialisation seed batches and 2 innovation replications per cell.',configurations=cs),indent=2))
t=time.monotonic()
with ProcessPoolExecutor(max_workers=6) as pool:
 futures=[pool.submit(run,c) for c in cs]
 for j,f in enumerate(as_completed(futures)):
  result=f.result();print(json.dumps(dict(completed=j+1,total=len(cs),last=result,elapsed=time.monotonic()-t)),flush=True)
print(json.dumps(dict(done=True,configurations=len(cs),replications=len(cs)*2,seconds=time.monotonic()-t)),flush=True)
