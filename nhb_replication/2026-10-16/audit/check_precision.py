"""Resolve the sole rounded terminal tie without replacing historical arrays."""
from pathlib import Path
import json,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
BASE=ROOT/'baseline/nhb_smallworld_20261012/results';DATA=ROOT/'data/results'
label='f0008';c=json.loads((BASE/(label+'.json')).read_text());p=c['parameters']
header=[c['M'],c['T'],c['reps'],c['mode'],c['window'],c['seed'],c['lam'],c['theta'],p['h'],p['peer_gain'],p['alpha_min'],p['alpha_max'],p['eps'],*p['mu']]
lines=[' '.join(map(str,header)),' '.join([str(c['n'])]*c['M']),' '.join(str(x) for q in c['q_init'] for x in q)]
for row in c['B']:lines.append(' '.join(map(str,[len(row),*[x for edge in row for x in edge]])))
inp=OUT/'precision_input.txt';binary=OUT/'precision_output.bin';inp.write_text('\n'.join(lines)+'\n')
proc=subprocess.run([str(OUT/'sim_audit'),str(inp),str(binary)],check=True,capture_output=True,text=True)
raw=np.fromfile(binary,dtype=np.float64);T,R,N,M=c['T'],c['reps'],c['N'],c['M'];offset=0
def take(shape):
    global offset
    n=int(np.prod(shape));a=raw[offset:offset+n].reshape(shape);offset+=n;return a
path,tail,q,psi=[take(shape) for shape in [(T,R,6),(R,M),(R,N,2),(R,M)]]
with np.load(DATA/(label+'.npz')) as old:
    path_error=float(np.max(abs(path-old['path'])));tail_error=float(np.max(abs(tail-old['tail'])))
    original_q=old['final_Q'];tied=np.argwhere(original_q[:,:,0]==original_q[:,:,1]);resolved=[]
    for rep,i in tied:resolved.append(dict(rep=int(rep),agent=int(i),saved_Q=original_q[rep,i].tolist(),replayed_Q=q[rep,i].tolist(),exact_gap=float(q[rep,i,0]-q[rep,i,1])))
    rank_difference=int(np.sum((q[:,:,0]>q[:,:,1])!=(original_q[:,:,0]>original_q[:,:,1])))
    assert path_error==0 and tail_error==0 and rank_difference==0
    max_storage_error=float(np.max(abs(q-original_q)))
np.savez_compressed(OUT/(label+'_precision_replay.npz'),path=path,tail=tail,final_Q=q,final_psi=psi,sizes=np.full(M,c['n']))
out=dict(label=label,QA_agent_trials=N*T*R,path_error=path_error,tail_error=tail_error,ranking_differences=rank_difference,
    max_terminal_storage_error=max_storage_error,resolved_ties=resolved,engine_audit=json.loads(proc.stdout),
    historical_arrays_unchanged=True)
(OUT/'precision_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
inp.unlink();binary.unlink()
