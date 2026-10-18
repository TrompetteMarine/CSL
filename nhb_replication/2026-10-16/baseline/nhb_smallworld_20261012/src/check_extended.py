import importlib.util,json,sys,subprocess
import numpy as np
from study import config,write_input,decode,ROOT,RESULT,graph
source=ROOT/'baseline/extended_lean.py'
sp=importlib.util.spec_from_file_location('verified',source);base=importlib.util.module_from_spec(sp);sys.modules['verified']=base;sp.loader.exec_module(base)
checks=[]
for mode in [0,1]:
 for theta in [0.,.5,1.]:
  c=config(18,3,'ws0.1',mode=mode,theta=theta,T=30);c['reps']=3;c['window']=10
  inp=RESULT/'extended.txt';out=RESULT/'extended.bin';write_input(c,inp)
  seed=778;cr=np.random.default_rng(np.random.SeedSequence([seed,101]));rr=np.random.default_rng(np.random.SeedSequence([seed,102]));U=cr.random((30,3,18));V=rr.random((30,3,18));tap=RESULT/'extended_tape.bin';np.r_[U.ravel(),V.ravel()].tofile(tap)
  subprocess.run([str(ROOT/'src/sim'),str(inp),str(out),str(tap)],check=True,capture_output=True)
  path,tail,q,psi=decode(c,out);B=np.zeros((6,6))
  for i,row in enumerate(c['B']):
   for j,v in row:B[i,j]=v
  C=(1-theta)*B+theta/6
  p=base.LeanParams(sizes=(3,)*6,T=30,B=tuple(map(tuple,B)),q_init=tuple(map(tuple,c['q_init'])),score='preference' if mode==0 else 'constant',window=10)
  ref=base.simulate(p,3,seed,record=True,choice_B=C,engine='dense')
  e=max(np.max(abs(q-ref['final_Q'])),np.max(abs(tail-ref['window_m'])),np.max(abs(path[:,:,0]-ref['mass_path'].mean(2))))
  assert e<1e-12;checks.append(dict(name='six-community dense reference',mode=mode,theta=theta,error=float(e)))
for M in [20,50,100,200,500]:
 for geo in ['ring','ws0.01','ws0.1','ws1','regular']:
  adj=graph(M,geo,2200)
  assert all(i not in a for i,a in enumerate(adj));assert all(i in adj[j] for i,a in enumerate(adj) for j in a);assert sum(map(len,adj))==4*M
  if geo in ['ring','regular']:assert all(len(x)==4 for x in adj)
  checks.append(dict(name='graph invariants',M=M,geometry=geo,passed=True))
for f in ['extended.txt','extended.bin','extended_tape.bin']:(RESULT/f).unlink()
(RESULT/'extended_validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps(dict(checks=len(checks),max_dense_error=max(x.get('error',0) for x in checks))),flush=True)
