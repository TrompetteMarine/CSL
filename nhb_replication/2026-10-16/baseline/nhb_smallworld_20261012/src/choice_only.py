"""Exploratory separation of choice geometry from peer reward exposure."""
import json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from study import config,run,ROOT
cs=[]
for n in [50,100]:
 for geo in ['ws0.01','ws0.1','ws1','regular']:
  for gid in range(4):
   c=config(10000,n,geo,'misleading_contiguous',graph_id=gid,group='choiceonly')
   peer=config(10000,n,'ring','misleading_contiguous',graph_id=gid)
   c['choice_B']=c['B'];c['B']=peer['B'];c['executable']='sim_v2';cs.append(c)
for j,c in enumerate(cs):c['label']=f'x{j:04d}'
(ROOT/'Choice_Only_Freeze.json').write_text(json.dumps(cs,indent=2))
t=time.monotonic()
with ProcessPoolExecutor(max_workers=2) as pool:
 for j,r in enumerate(pool.map(run,cs)):
  if j%8==0:print(json.dumps(dict(completed=j+1,total=len(cs),last=r,elapsed=time.monotonic()-t)),flush=True)
print(json.dumps(dict(done=True,configurations=len(cs),seconds=time.monotonic()-t)),flush=True)
