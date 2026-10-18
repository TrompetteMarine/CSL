"""Exploratory follow-up frozen after initial geometry outcomes were inspected."""
import json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from study import config,run,ROOT
cs=[]
for n in [50,100,200]:
 for geo in ['ring','ws0.01','ws0.1']:
  for gid in range(4,16):cs.append(config(10000,n,geo,'misleading_contiguous',graph_id=gid,group='followup'))
for n in [20,100]:
 for geo in ['ring','ws0.01','ws0.1','ws1']:
  for gid in range(4):cs.append(config(10000,n,geo,'misleading_contiguous',graph_id=gid,group='degree',k=8))
for j,c in enumerate(cs):c['label']=f'f{j:04d}'
(ROOT/'Followup_Freeze.json').write_text(json.dumps(dict(reason='Nonmonotone clustered-initialisation outcomes in the primary sweep; increase graph/innovation sampling and vary degree. This is exploratory, not an independently preregistered confirmation.',configurations=cs),indent=2))
t=time.monotonic()
with ProcessPoolExecutor(max_workers=3) as pool:
 futures=[pool.submit(run,c) for c in cs]
 for j,f in enumerate(as_completed(futures)):
  result=f.result()
  if j%12==0:print(json.dumps(dict(completed=j+1,total=len(cs),last=result,elapsed=time.monotonic()-t)),flush=True)
print(json.dumps(dict(done=True,configurations=len(cs),seconds=time.monotonic()-t)),flush=True)
