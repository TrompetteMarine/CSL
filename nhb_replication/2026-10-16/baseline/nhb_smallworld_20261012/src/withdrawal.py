"""Fresh high-precision entries and paired withdrawal continuations."""
import json,subprocess
import numpy as np
from study import config,run,decode,write_input,ROOT,RESULT
rows=[]
for gid in range(4):
 c=config(10000,100,'broadcast','misleading_random',graph_id=gid,group='withdrawal_entry');c['label']=f'w{gid:04d}'
 run(c)
 z=np.load(RESULT/(c['label']+'.npz'));Q=z['final_Q'];psi=z['final_psi']
 assert Q.dtype==np.float64
 # Entire entry ensemble is continued, rather than selecting wrong runs.
 st=RESULT/'withdraw_state.bin'
 np.concatenate([np.r_[Q[r].ravel(),psi[r]] for r in range(4)]).tofile(st)
 for lam in [.6,0.]:
  d={**c,'lam':lam,'T':1000,'seed':820000+gid*100,'group':'withdrawal','label':f'w{gid+4+(0 if lam else 4):04d}'}
  inp=RESULT/(d['label']+'.txt');out=RESULT/(d['label']+'.bin');write_input(d,inp)
  pr=subprocess.run([str(ROOT/'src/sim'),str(inp),str(out),'-',str(st)],capture_output=True,text=True,check=True)
  path,tail,q,pp=decode(d,out)
  np.savez_compressed(RESULT/(d['label']+'.npz'),path=path,tail=tail,final_Q=q,final_psi=pp,sizes=np.full(d['M'],d['n']))
  d['audit']=json.loads(pr.stdout);d['entry_label']=c['label'];d['initial_state_source']='complete high-precision saved ensemble, no outcome selection';(RESULT/(d['label']+'.json')).write_text(json.dumps(d,indent=2))
  for r in range(4):
   rows.append(dict(graph_id=gid,rep=r,lam=lam,entry_wrong=int(z['path'][-100:,r,0].mean()<.5),optimal_fraction=float(path[-100:,r,0].mean()),wrong=int(path[-100:,r,0].mean()<.5),private_correct=float((q[r,:,0]>q[r,:,1]).mean()),first_step_probability=float(path[0,r,4])))
  inp.unlink();out.unlink()
 st.unlink();print(json.dumps(dict(graph_id=gid,done=True)),flush=True)
(RESULT/'withdrawal.json').write_text(json.dumps(rows,indent=2))
