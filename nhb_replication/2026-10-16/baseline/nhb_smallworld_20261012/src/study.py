"""Frozen design and executable large-population numerical experiment."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import csv, hashlib, json, subprocess, time, sys, importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results'; RESULT.mkdir(exist_ok=True)
def graph(M,geometry,seed,k=4):
 rng=np.random.default_rng(seed)
 if M==2:return [{1},{0}]
 adj=[set() for _ in range(M)]
 for c in range(M):
  for j in range(1,k//2+1):d=(c+j)%M;adj[c].add(d);adj[d].add(c)
 if geometry.startswith('ws'):
  prob=float(geometry[2:])
  for c in range(M):
   for j in range(1,k//2+1):
    d=(c+j)%M
    if rng.random()<prob:
     options=[v for v in range(M) if v!=c and v not in adj[c]]
     if options:
      v=int(rng.choice(options));adj[c].remove(d);adj[d].remove(c);adj[c].add(v);adj[v].add(c)
 elif geometry=='regular':
  edges=[(c,d) for c in range(M) for d in sorted(adj[c]) if c<d]
  for _ in range(20*len(edges)):
   x,y=rng.choice(len(edges),2,replace=False);a,b=edges[x];c,d=edges[y]
   if rng.random()<.5:c,d=d,c
   if len({a,b,c,d})<4 or d in adj[a] or b in adj[c]:continue
   adj[a].remove(b);adj[b].remove(a);adj[c].remove(d);adj[d].remove(c)
   adj[a].add(d);adj[d].add(a);adj[c].add(b);adj[b].add(c);edges[x]=(a,d);edges[y]=(c,b)
 elif geometry not in ('ring','broadcast'):raise ValueError(geometry)
 return adj
def metrics(adj,seed):
 M=len(adj);r=[];c=[];cl=[]
 for i,nb in enumerate(adj):
  nb=sorted(nb);d=len(nb);tri=sum(v in adj[u] for j,u in enumerate(nb) for v in nb[j+1:])
  cl.append(2*tri/(d*(d-1)) if d>1 else 0)
  for j in nb:r.append(i);c.append(j)
 mat=csr_matrix((np.ones(len(r)),(r,c)),shape=(M,M))
 rng=np.random.default_rng(seed);sources=np.sort(rng.choice(M,min(M,64),replace=False))
 dist=shortest_path(mat,directed=False,indices=sources);mask=dist>0
 return dict(clustering=float(np.mean(cl)),mean_path=float(dist[mask].mean()) if np.isfinite(dist[mask]).all() else None,
  global_efficiency_sample=float(np.where(mask,1/np.maximum(dist,1),0).sum()/(len(sources)*(M-1))),
  components=int(connected_components(mat,directed=False)[0]),mean_degree=float(np.mean([len(x) for x in adj])),
  degree_sd=float(np.std([len(x) for x in adj])),path_sources=sources.tolist())
def config(N,n,geometry,initial='misleading_random',mode=0,lam=.6,theta=None,T=1000,graph_id=0,group='primary',gamma=.15,k=4):
 M=N//n;assert M*n==N
 theta=1. if geometry=='broadcast' else 0. if theta is None else theta
 seed=202610120+graph_id
 adj=graph(M,geometry,seed,k=k)
 rng=np.random.default_rng(44000+graph_id)
 ids=np.arange(M)
 if initial.endswith('random'):rng.shuffle(ids)
 correct=set(ids[:M//2]);q=[]
 for c in range(M):
  if initial.startswith('misleading'):v=(.52,.50) if c in correct else (.42,.62)
  elif initial=='neutral':v=(.5,.5)
  elif initial=='correct':v=(.62,.42)
  elif initial=='reversed':v=(.62,.42) if c in correct else (.50,.52)
  else:raise ValueError(initial)
  q.append(v)
 B=[[(c,1-gamma)]+[(d,gamma/len(adj[c])) for d in sorted(adj[c])] for c in range(M)]
 return dict(N=N,n=n,M=M,geometry=geometry,initial=initial,mode=mode,lam=lam,theta=theta,T=T,window=100,k=k,
  graph_id=graph_id,group=group,gamma=gamma,B=B,q_init=q,graph=metrics(adj,seed),reps=4,
  seed=702000+graph_id*100,parameters=dict(h=12.,peer_gain=.15,alpha_min=.05,alpha_max=.4,eps=.001,mu=[.55,.45]))
def write_input(c,path):
 p=c['parameters'];header=[c['M'],c['T'],c['reps'],c['mode'],c['window'],c['seed'],c['lam'],c['theta'],p['h'],p['peer_gain'],p['alpha_min'],p['alpha_max'],p['eps'],*p['mu']]
 lines=[' '.join(map(str,header)),' '.join([str(c['n'])]*c['M']),' '.join(str(x) for q in c['q_init'] for x in q)]
 for row in c['B']:lines.append(' '.join(map(str,[len(row),*[x for edge in row for x in edge]])))
 if 'choice_B' in c:
  lines.append(str(c['M']))
  for row in c['choice_B']:lines.append(' '.join(map(str,[len(row),*[x for edge in row for x in edge]])))
 path.write_text('\n'.join(lines)+'\n')
def decode(c,path):
 raw=np.fromfile(path,dtype=np.float64);T,R,N,M=c['T'],c['reps'],c['N'],c['M'];o=0
 def take(shape):
  nonlocal o
  count=int(np.prod(shape));a=raw[o:o+count].reshape(shape);o+=count;return a
 patharr=take((T,R,6));tail=take((R,M));q=take((R,N,2));psi=take((R,M));assert o==len(raw)
 return patharr,tail,q,psi
def run(c):
 label=c['label'];inp=RESULT/(label+'.txt');binary=RESULT/(label+'.bin');write_input(c,inp)
 t=time.monotonic();proc=subprocess.run([str(ROOT/'src'/c.get('executable','sim')),str(inp),str(binary)],capture_output=True,text=True,check=True)
 audit=json.loads(proc.stdout);path,tail,q,psi=decode(c,binary)
 np.savez_compressed(RESULT/(label+'.npz'),path=path,tail=tail,final_Q=q,final_psi=psi,sizes=np.full(c['M'],c['n']))
 rows=[]
 for r in range(c['reps']):
  mean=path[-c['window']:,r,0].mean();private=(q[r,:,0]>q[r,:,1]).mean()
  rows.append(dict(label=label,**{k:c[k] for k in ['N','n','M','geometry','initial','mode','lam','theta','group','graph_id']},rep=r,
   optimal_fraction=float(mean),wrong=int(mean<.5),private_correct=float(private),private_gap=float((q[r,:,0]-q[r,:,1]).mean()),
   latent_public_gap=float(private-mean),mean_removed_probability=float(path[-100:,r,3].mean()),
   regret=float(path[:,r,5].sum()),wrong_communities=float((tail[r]<.5).mean()),
   community_variance=float(tail[r].var()),**{k:c['graph'][k] for k in ['clustering','mean_path','components','degree_sd']}))
 c={**c,'audit':audit,'runtime_seconds':time.monotonic()-t}
 (RESULT/(label+'.json')).write_text(json.dumps(c,indent=2));(RESULT/(label+'_rows.json')).write_text(json.dumps(rows))
 binary.unlink();inp.unlink()
 return label,c['runtime_seconds'],audit
def design():
 cs=[]
 for n in [20,50,100,200,500]:
  for geo in ['ring','ws0.01','ws0.1','ws1','regular','broadcast']:
   for initial in ['misleading_random','misleading_contiguous']:
    for gid in range(4):cs.append(config(10000,n,geo,initial,graph_id=gid))
 # Independent initialisation and fixed-dose controls at 100 communities.
 for geo in ['ring','ws0.1','regular','broadcast']:
  for initial,mode,lam in [('neutral',0,.6),('correct',0,.6),('reversed',0,.6),('misleading_random',1,.6),('misleading_random',2,.6),('misleading_random',1,1.2)]:
   for gid in range(4):cs.append(config(10000,100,geo,initial,mode,lam,graph_id=gid,group='controls'))
 for gamma in [.0,.15,.5]:
  for initial in ['misleading_random','neutral','correct','reversed']:
   for gid in range(4):cs.append(config(10000,5000,'ring',initial,graph_id=gid,group='two',gamma=gamma))
 # Two distinct population scalings, sharing an intermediate design point.
 for N,n in [(1000,50),(10000,500),(40000,2000),(1000,20),(10000,20),(40000,20)]:
  for geo in ['ring','ws0.1']:
   for gid in range(4):cs.append(config(N,n,geo,'misleading_random',graph_id=gid,group='scaling',T=500))
 for idx,c in enumerate(cs):c['label']=f'c{idx:04d}'
 return cs
def validate():
 source=ROOT/'baseline/extended_lean.py'
 spec=importlib.util.spec_from_file_location('baseline',source);base=importlib.util.module_from_spec(spec);sys.modules['baseline']=base;spec.loader.exec_module(base)
 checks=[]
 for mode in [0,1]:
  for theta in [0.,.5,1.]:
   c=config(12,6,'ring',mode=mode,theta=theta,T=30);c['window']=10;c['reps']=3;c['seed']=123
   inp=RESULT/'check.txt';out=RESULT/'check.bin';write_input(c,inp)
   seed=321;cr=np.random.default_rng(np.random.SeedSequence([seed,101]));rr=np.random.default_rng(np.random.SeedSequence([seed,102]));U=cr.random((30,3,12));V=rr.random((30,3,12));tap=RESULT/'tape.bin';np.r_[U.ravel(),V.ravel()].tofile(tap)
   subprocess.run([str(ROOT/'src/sim'),str(inp),str(out),str(tap)],check=True,capture_output=True)
   path,tail,q,psi=decode(c,out);B=np.array([[.85,.15],[.15,.85]]);C=(1-theta)*B+theta*.5
   p=base.LeanParams(sizes=(6,6),T=30,B=tuple(map(tuple,B)),q_init=tuple(map(tuple,c['q_init'])),score='preference' if mode==0 else 'constant',window=10)
   ref=base.simulate(p,3,seed,record=True,choice_B=C)
   checks.append(dict(mode=mode,theta=theta,q_error=float(np.max(np.abs(q-ref['final_Q']))),path_error=float(np.max(np.abs(path[:,:,0]-ref['mass_path'].mean(2)))),tail_error=float(np.max(np.abs(tail-ref['window_m'])))))
   assert max(checks[-1][k] for k in ['q_error','path_error','tail_error'])<1e-12
 for f in ['check.txt','check.bin','tape.bin']:(RESULT/f).unlink()
 (RESULT/'validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks),flush=True)
if __name__=='__main__':
 if '--validate' in sys.argv:validate()
 else:
  cs=design();(ROOT/'Design_Freeze.json').write_text(json.dumps(cs,indent=2))
  start=time.monotonic()
  with ProcessPoolExecutor(max_workers=6) as pool:
   futures=[pool.submit(run,c) for c in cs if not (RESULT/(c['label']+'.npz')).exists()]
   for j,f in enumerate(as_completed(futures)):
    result=f.result()
    if j%12==0:print(json.dumps(dict(completed=j+1,total=len(futures),last=result,elapsed=time.monotonic()-start)),flush=True)
  rows=[]
  for c in cs:rows.extend(json.loads((RESULT/(c['label']+'_rows.json')).read_text()))
  with (RESULT/'replications.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  print(json.dumps(dict(done=True,configurations=len(cs),replications=len(rows),seconds=time.monotonic()-start)),flush=True)
