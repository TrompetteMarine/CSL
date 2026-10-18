"""Independently reconstruct reported diagnostics from stored adjacency rows."""
from pathlib import Path
import hashlib,json,re
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components,shortest_path
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
META=ROOT/'baseline/nhb_smallworld_20261012/results'
cache={};errors={};checks=[]
for f in sorted(META.glob('*.json')):
  if not re.fullmatch('[cfmxw][0-9]{4}',f.stem):continue
  c=json.loads(f.read_text());M=c['M'];rows=c.get('choice_B',c['B']) if c['group']=='choiceonly' else c['B']
  adj=[{j for j,v in row if j!=i} for i,row in enumerate(rows)]
  sources=c['graph']['path_sources']
  key=hashlib.sha256(json.dumps([list(map(sorted,adj)),sources]).encode()).hexdigest()
  if key not in cache:
    uu=[];vv=[]
    for i,nb in enumerate(adj):
      uu.extend([i]*len(nb));vv.extend(sorted(nb))
    A=csr_matrix((np.ones(len(uu)),(uu,vv)),shape=(M,M))
    deg=np.array([len(nb) for nb in adj]);cl=[]
    for i,nb in enumerate(adj):
      cl.append(sum(len(nb&adj[j]) for j in nb)/(len(nb)*(len(nb)-1)) if len(nb)>1 else 0.)
    dist=shortest_path(A,directed=False,indices=sources);positive=dist>0
    cache[key]=dict(clustering=float(np.mean(cl)),mean_degree=float(deg.mean()),degree_sd=float(deg.std()),
       components=int(connected_components(A,directed=False)[0]),mean_path=float(dist[positive].mean()) if np.isfinite(dist[positive]).all() else None,
       global_efficiency_sample=float(np.where(positive,1/np.maximum(dist,1),0).sum()/(len(sources)*(M-1))))
  result=cache[key]
  for name,value in result.items():
    recorded=c['graph'][name]
    if value is None:assert recorded is None
    else:errors[name]=max(errors.get(name,0.),abs(value-recorded))
  checks.append(dict(label=f.stem,diagnostic_graph='choice' if c['group']=='choiceonly' else 'peer',cache_key=key))
assert max(errors.values())<1e-12
out=dict(configurations=len(checks),distinct_graph_source_combinations=len(cache),max_absolute_errors=errors,all_passed=True)
(OUT/'graph_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
